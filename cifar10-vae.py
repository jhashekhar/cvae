import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

dataset = datasets.MNIST(root='./mnist', train=True, download=True,
                         transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


class VAE(nn.Module):
  def __init__(self, ngf, ndf):
    super().__init__()

    # Encoder
    self.conv1_en = nn.Conv2d(3, ndf, 4, stride=2, padding=1)
    self.bn1_en = nn.BatchNorm2d(ndf)

    self.conv2_en = nn.Conv2d(ndf, 2*ndf, 4, stride=2, padding=1)
    self.bn2_en = nn.BatchNorm2d(2*ndf)

    self.conv3_en = nn.Conv2d(2*ndf, 4*ndf, 4, stride=2, padding=1)
    self.bn3_en = nn.BatchNorm2d(4*ndf)

    self.fc1 = nn.Linear(4*4*4*ndf, 500)
    self.fc2 = nn.Linear(4*4*4*ndf, 500)

    # Decoder
    self.de1 = nn.Linear(500, 4*4*4*ndf)

    self.conv1_de = nn.ConvTranspose2d(4*ngf, 2*ngf, 4, stride=2, padding=1)
    self.bn1_de = nn.BatchNorm2d(2*ngf)

    self.conv2_de = nn.ConvTranspose2d(2*ngf, ngf, 4, stride=2, padding=1)
    self.bn2_de = nn.BatchNorm2d(ngf)

    self.conv3_de = nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1)
    self.bn3_de = nn.BatchNorm2d(3)

    # Activations
    self.leakyrelu = nn.LeakyReLU(0.2)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def encoder(self, x):
    h1 = self.leakyrelu(self.bn1_en(self.conv1_en(x)))
    h2 = self.leakyrelu(self.bn2_en(self.conv2_en(h1)))
    h3 = self.leakyrelu(self.bn3_en(self.conv3_en(h2)))
    out = h3.view(h3.size(0), -1)
    mean = self.relu(self.fc1(out))
    logvar = self.relu(self.fc2(out))
    return mean, logvar

  def reparameterize(self, mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.rand_like(std)
    z = mean + eps * std
    return z

  def decoder(self, z):
    h1 = self.relu(self.de1(z))
    out = h1.view(16, 4*128, 4, 4)
    out = self.leakyrelu(self.bn1_de(self.conv1_de(out)))
    out = self.leakyrelu(self.bn2_de(self.conv2_de(out)))
    out = self.sigmoid(self.bn3_de(self.conv3_de(out)))
    return out

  def forward(self, x):
    mean, logvar = self.encoder(x)
    z = self.reparameterize(mean, logvar)
    return self.decoder(z), mean, logvar


def loss_function(x, recon_x, mu, logvar):
  # KLD + BCE
  BCE = torch.nn.functional.binary_cross_entropy(recon_x, x)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return KLD + BCE


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sample_size = 24
latent_dim = 20
fixed_z = np.random.normal(0, 1, size=(sample_size, latent_dim))
fixed_z = torch.from_numpy(fixed_z).float().to(device)

num_epochs = 200
train_loss = 0.0
for epoch in range(num_epochs):
  model.train()
  for data in trainloader:
    img, _ = data
    img = img.to(device)
    optimizer.zero_grad()
    recon_x, mu, logvar = model(img)
    loss = loss_function(img, recon_x, mu, logvar)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  model.eval()
  zx = model.decode(fixed_z)
  torchvision.utils.save_image(zx.view(24, 1, 28, 28),
  'sample_{}.png'.format(epoch), nrow=4, padding=2)
  print("Epoch: {}/{}  loss: {}".format(epoch, num_epochs, loss))
