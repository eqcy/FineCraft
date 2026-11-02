"""Public-observation-only stall classifier and causal temporal descriptor."""
import torch
from torch import nn

WINDOW = 16  # inference calls every two actions: 30-step observed span
PERSISTENCE = 6
DIMENSION = 815


def descriptor(latent, public):
    """Both arguments contain the last WINDOW observations, including now."""
    return torch.cat((latent[:, -1], latent[:, -1]-latent[:, 0],
        public[:, -1], public[:, -1]-public[:, 0], public.std(1, unbiased=False)), -1)


class StallGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(DIMENSION))
        self.register_buffer('std', torch.ones(DIMENSION))
        self.net = nn.Sequential(nn.Linear(DIMENSION, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(((x-self.mean)/self.std.clamp_min(.02)).clamp(-8, 8)).flatten()
