import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

dataset = datasets.MNIST(root='./mnist', train=True, download=True,
                         transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


class VAE(nn.Module):
  def __init__(self, c_in=28*28, c_out=256, f_out=20):
    super().__init__()
    self.fc1 = nn.Linear(c_in, c_out)
    self.fc2 = nn.Linear(c_out, 2*c_out)
    self.fc31 = nn.Linear(2*c_out, f_out)
    self.fc32 = nn.Linear(2*c_out, f_out)
    self.fc4 = nn.Linear(20, 400)
    self.fc5 = nn.Linear(400, 28 * 28)
    self.relu = nn.ReLU()

  def encoder(self, x):
    out = self.relu(self.fc1(x))
    out = self.relu(self.fc2(out))
    mean = self.fc31(out)
    var = self.fc32(out)
    return mean, var

  def reparameterize(self, mean, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.rand_like(std)
    z = mean + eps * std
    return z

  def decoder(self, z):
    h3 = self.relu(self.fc4(z))
    return torch.sigmoid(self.fc5(h3))

  def forward(self, x):
    mean, logvar = self.encoder(x.view(-1, 784))
    z = self.reparameterize(mean, logvar)
    return self.decoder(z), mean, logvar


def loss_function(x, recon_x, mu, logvar):
  # KLD + BCE
  BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784))
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return KLD + BCE


device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sample_size = 24
latent_dim = 20
fixed_z = np.random.normal(0, 1, size=(sample_size, latent_dim))
fixed_z = torch.from_numpy(fixed_z).float().to(device)

num_epochs = 25
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
