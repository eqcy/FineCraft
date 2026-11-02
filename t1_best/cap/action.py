"""Retain the teammate's learned action architecture with expanded conditioning."""
from .._vendor.model import T1DiT
from .schema import VISUAL_DIM


class T1CapDiT(T1DiT):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('visual_dim', VISUAL_DIM)
        super().__init__(*args, **kwargs)

    def forward(self, features, actions, mask):
        return self.loss(features, actions, mask)
