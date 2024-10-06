import torch
import torch.nn as nn
from typing import Callable

class SimpleMLP(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 hidden_dim: int,
                 n_layers: int,
                 activator: Callable[[], nn.Module] = nn.ReLU) -> None:
        """
        :param in_channel: input channels
        :param out_channel: output channels
        :param hidden_dim: hidden dims
        :param n_layers: total layers
        """
        super().__init__()
        assert n_layers >= 1, 'n_layers should be positive'
        if n_layers == 1:
            self.network = nn.Linear(in_channel, out_channel)
        else:
            layers = []  # think: why not use nn.ModuleList?
            layers.append(nn.Linear(in_channel, hidden_dim))
            layers.append(activator())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activator())
            layers.append(nn.Linear(hidden_dim, out_channel))
            self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
