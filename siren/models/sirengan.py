import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Sine(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features, 
                     1 / self.in_features)      
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, 
                     np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Generator(nn.Module):
    def __init__(
        self,
        dim = 64,
        latent_size = 2,
        channels = 3,
        hidden_features = 64,
        hidden_layers = 2,
        outermost_linear = True,
        omega = 30,
    ):
        super().__init__()
        self.dim = dim
        self.channels = channels
        self.omega = omega
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.main = []

        grid = np.zeros((self.dim * self.dim, 2))
        for i in range(self.dim):
            for j in range(self.dim):
                grid[i * self.dim + j][0] = -1 + (2 / self.dim) * i 
                grid[i * self.dim + j][1] = -1 + (2 / self.dim) * j
        self.grid = torch.from_numpy(grid).float()

        first_layer = Sine(
            2 + latent_size,
            hidden_features,
            is_first=True,
            omega_0=omega)
        self.main.append(first_layer)

        for i in range(hidden_layers):
            hidden_layer = Sine(
                hidden_features,
                hidden_features, 
                is_first=False,
                omega_0=omega)
            self.main.append(hidden_layer)

        if outermost_linear:
            linear = nn.Linear(
                hidden_features,
                channels)
#            with torch.no_grad():
#                linear.weight.uniform_(
#                    -np.sqrt(6 / hidden_features) / omega, 
#                    np.sqrt(6 / hidden_features) / omega)
            self.main.append(linear)
            self.main.append(nn.Tanh())
        else:
            final_layer = Sine(
                hidden_features,
                channels, 
                is_first=False,
                omega_0=omega)
            self.main.append(final_layer)

        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.repeat_interleave(x, self.dim * self.dim, dim = 0)
        grid = self.grid.repeat((batch_size, 1))
        x = self.main(torch.cat((grid, x), 1))
        x = x.view(batch_size, self.dim, self.dim, self.channels)
        x = x.permute(0, 3, 1, 2)
        return x


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
        latent_size = 2,
        dim = 64,
        channels = 3
    ):
        self.cuda_enabled = cuda_enabled
        self.cuda_index = cuda_index
        self.latent_size = latent_size
        self.dim = dim
        self.channels = channels
        self.G = Generator(
            latent_size = self.latent_size,
            dim = self.dim,
            channels = self.channels)
        self.D = Discriminator(
            channels = self.channels)
        if self.cuda_enabled:
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)

    def train_g(self, device, batch_size, optimizer):
        optimizer.zero_grad()
        z = torch.randn(batch_size, self.latent_size).to(device)
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
        z = torch.randn(batch_size, self.latent_size).to(device)
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
