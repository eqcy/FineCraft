"""Resolve only asset paths for the supplied frozen StallRepairPolicy."""
from pathlib import Path
import torch
from .t5_m0_generalize.runtime import StallRepairPolicy


class ComboPolicyAdapter(StallRepairPolicy):
    def __init__(self, checkpoint, terminal, branch, gate, protection, legacy_branch, o_branch, vision_path, device='auto'):
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super().__init__(checkpoint=Path(checkpoint), terminal=Path(terminal), branch=Path(branch),
            gate=Path(gate), protection=Path(protection), legacy_branch=Path(legacy_branch),
            o_branch=Path(o_branch), device=device, vision_path=Path(vision_path))
