import torch
import torch.nn as nn
import numpy as np


class Sine(nn.Module)
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
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Generator(nn.Module):
    def __init__(
        self,
        dim = 64,
        latent_size = 2,
        channels = 3,
        hidden_features = 64,
        hidden_layers = 4,
        outermost_linear=True, 
        omega=30
    ):
        super().__init__()
        self.dim = dim
        self.omega = omega
        self.hidden_layers = hidden_layers
        self.hidden_features = hidden_features
        self.main = []

        grid = np.zeros((self.dim * self.dim, 2))
        for i in range(self.dim):
            for j in range(self.dim):
                grid[i * self.dim + j][0] = -1 + (2 / self.dim) * i 
                grid[i * self.dim + j][1] = -1 + (2 / self.dim) * j
        self.grid = torch.from_numpy(grid).float().cuda()

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
            final_layer = nn.Linear(
                hidden_features,
                channels)
            with torch.no_grad():
                final_layer.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / omega, 
                    np.sqrt(6 / hidden_features) / omega)
            self.main.append(final_layer)
        else:
            final_layer = Sine(
                hidden_features,
                channels, 
                is_first=False,
                omega_0=omega)
            self.main.append(final_layer)

        self.main = nn.Sequential(*self.main)
        self.output = nn.Tanh()

    def forward(self, x):
        batch_size = x.shape[0]
        x = torch.repeat_interleave(x, self.dim * self.dim, dim = 0)
        grid = self.grid.repeat((batch_size, 1))
        x = self.main(torch.cat((grid, x), 1))
        x = x.view(batch_size, self.dim, self.dim, self.channels)
        x = x.permute(0, 3, 1, 2)
        return self.output(x)
