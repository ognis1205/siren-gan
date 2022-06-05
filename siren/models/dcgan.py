import torch
import torch.nn as nn


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
                bisa=False),
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
            # out: channel x 64 x 64
            nn.Tanh())

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, channels = 3):
        super().__init__()
        self.main = nn.Sequential(
            # in: 3 x 64 x 64
            nn.Conv2d(
                in_channels=channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 32 x 32
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_feautures=128),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 16 x 16
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_fueatures=256),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 8 x 8
            nn.Conv2d(
                in_channles=256,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 512 x 4 x 4
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
