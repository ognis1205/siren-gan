import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_size = 128, channels = 3):
        super().__init__()
        self.main = nn.Sequential(
            # in: latent_size x 1 x 1
            nn.ConvTranspose2d(
                in_channels=latent_size,
                out_channels=512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # state: 512 x 4 x 4
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # state: 256 x 8 x 8
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            # state: 128 x 16 x 16
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            # state: 64 x 32 x 32
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            # out: 64 x 64 x channel
            nn.Tanh())

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        self.main = nn.Sequential(
            # in: 64 x 64 x 3
            nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 32 x 32 x 64
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 16 x 16 x 128
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 8 x 8 x 256
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 4 x 4 x 512
            nn.Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False),
            # out: 1 x 1 x 1
            nn.Flatten(),
            nn.Sigmoid())

    def forward(self, x):
        return self.main(x)


class Model:
    def __init__(
        self,
        cuda_enabled = False,
        cuda_index = 0,
        latent_size = 128,
        dim = 64,
        channels = 3
    ):
        self.cuda_enabled = cuda_enabled
        self.cuda_index = cuda_index
        self.latent_size = latent_size
        self.dim = 64
        self.channels = channels
        self.G = Generator(self.latent_size, self.channels)
        self.D = Discriminator(self.channels)
        if self.cuda_enabled:
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)

    def train_g(self, device, batch_size, optimizer):
        optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_size, 1, 1).to(device)
        i = self.G(z).to(device)
        p = self.D(i).to(device)
        t = torch.ones(batch_size, 1).to(device)
        loss = F.binary_cross_entropy(p, t)
        loss.backward()
        optimizer.step()
        return loss.item(), z

    def train_d(self, device, batch_size, optimizer, x):
        optimizer.zero_grad()
        rp = self.D(x).to(device)
        rt = torch.ones(x.size(0), 1).to(device)
        real_loss = F.binary_cross_entropy(rp, rt)
        real_score = torch.mean(rp).item()
        z = torch.randn(batch_size, self.latent_size, 1, 1).to(device)
        fx = self.G(z).to(device)
        fp = self.D(fx).to(device)
        ft = torch.zeros(fx.size(0), 1).to(device)
        fake_loss = F.binary_cross_entropy(fp, ft)
        fake_score = torch.mean(fp).item()
        loss = real_loss + fake_loss
        loss.backward()
        optimizer.step()
        return loss.item(), real_score, fake_score

    def save(self, path):
        torch.save(self.G.state_dict(), path / 'g.pkl')
        torch.save(self.D.state_dict(), path / 'd.pkl')

    def load(self, path):
        self.G.load_state_dict(torch.load(path / 'g.pkl'))
        self.D.load_state_dict(torch.load(path / 'd.pkl'))
