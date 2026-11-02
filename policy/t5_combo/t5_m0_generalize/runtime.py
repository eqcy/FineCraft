"""No simulator imports, geometry rules or evaluation labels at deployment."""
from collections import deque
from pathlib import Path
import torch
from ..t5_m0_endgame.runtime import EndgamePolicy
from ..t5_m0_endgame.model import load
from ..t5_m0_stall_branch.model import StallGate, descriptor, WINDOW, PERSISTENCE


class StallRepairPolicy(EndgamePolicy):
    def __init__(self, checkpoint, terminal, branch, gate, protection, legacy_branch, o_branch, device='cuda:0', *, shadow=False, vision_path=None):
        self.shadow = shadow
        super().__init__(checkpoint, terminal, device, vision_path=vision_path)
        self.repair_model, _ = load(branch, device)
        self.repair_model.requires_grad_(False)
        self.legacy_repair, _ = load(legacy_branch, device)
        self.legacy_repair.requires_grad_(False)
        self.o_repair, _ = load(o_branch, device)
        self.o_repair.requires_grad_(False)
        finish = torch.load(protection, map_location='cpu', weights_only=False)
        assert finish['format'] == 't5-finish-protection-v1'
        self.protection = StallGate().to(device).eval().requires_grad_(False)
        self.protection.load_state_dict(finish['model'])
        self.protection_threshold = float(finish['threshold'])
        payload = torch.load(gate, map_location='cpu', weights_only=False)
        if payload['format'] != 't5-stall-gate-v1':
            raise ValueError('Wrong gate format')
        self.stall_gate = StallGate().to(device).eval().requires_grad_(False)
        self.stall_gate.load_state_dict(payload['model'])
        self.stall_threshold = float(payload['threshold'])
        self.stall_latents = deque(maxlen=WINDOW); self.stall_public = deque(maxlen=WINDOW)
        baseline_sample = self.model.sample_from_features

        def sample(features, *, generator=None, **kwargs):
            if self.letter == 2:  # R instruction: original 70.28 throughout.
                return baseline_sample(features, generator=generator, **kwargs)
            # Additional encoders and classifier are deterministic and use no
            # random generator; the original sampling stream remains intact.
            with torch.autocast(device_type=self.device.type, enabled=False):
                x = self.last_x.half().float()
                self.stall_latents.append(self.terminal.encode(x)[0])
                self.stall_public.append(((x[0, -1, 240:341]-self.terminal.mean[240:341]) /
                    self.terminal.std[240:341].clamp_min(.01)).clamp(-10, 10))
            eligible = (self.active and self.takeover_step is not None
                and self.next_step >= self.takeover_step+2*(WINDOW-1)
                and len(self.stall_latents) == WINDOW and self.next_step <= 352)
            if not self.repair_active and eligible:
                with torch.autocast(device_type=self.device.type, enabled=False):
                    vector = descriptor(torch.stack(tuple(self.stall_latents))[None],
                                        torch.stack(tuple(self.stall_public))[None])
                    probability = float(self.stall_gate(vector).sigmoid().item())
                self.max_stall_probability = max(self.max_stall_probability, probability)
                self.stall_count = self.stall_count+1 if probability >= self.stall_threshold else 0
                permitted = True
                if self.letter in (0, 1) and self.stall_count >= PERSISTENCE:
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        finish_probability = float(self.protection(vector).sigmoid().item())
                    self.max_finish_probability = max(self.max_finish_probability, finish_probability)
                    permitted = finish_probability < self.protection_threshold
                    if not permitted:
                        self.protected_calls += 1
                        if self.first_protection_step is None: self.first_protection_step = self.next_step
                if self.stall_count >= PERSISTENCE and permitted:
                    self.repair_active = True; self.repair_step = self.next_step
                    self.calls_at_repair = (self.m0_calls, self.terminal_calls)
                    print(dict(event='learned_stall_repair_takeover', letter=self.letter,
                        step=self.repair_step, probability=probability,
                        threshold=self.stall_threshold, consecutive=self.stall_count), flush=True)
            elif not self.repair_active:
                self.stall_count = 0
            if self.repair_active and not self.shadow:
                assert (self.m0_calls, self.terminal_calls) == self.calls_at_repair
                actor = self.legacy_repair if self.letter == 3 else self.o_repair if self.letter == 1 else self.repair_model
                chunk = actor.sample(self.last_x, self.repair_generator)
                self.repair_calls += 1
                return torch.cat((chunk, chunk[:, -1:].expand(-1, self.model.horizon-chunk.shape[1], -1)), 1)
            return baseline_sample(features, generator=generator, **kwargs)

        self.model.sample_from_features = sample

    def reset(self, *args):
        super().reset(*args)
        self.stall_latents.clear(); self.stall_public.clear()
        self.stall_count = 0; self.repair_active = False; self.repair_step = None
        self.repair_calls = 0; self.calls_at_repair = None; self.max_stall_probability = 0.
        self.protected_calls = 0; self.first_protection_step = None; self.max_finish_probability = 0.
        self.repair_generator = torch.Generator(device=self.device).manual_seed(20261020)
