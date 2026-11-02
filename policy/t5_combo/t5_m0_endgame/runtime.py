"""Deployment boundary: images, qpos, instruction and executed-action history only.

Frozen M0 acts until the learned phase head fires twice. The latch is permanent
until the next episode reset. Neither geometry teachers nor simulator code are
imported here. Feature hooks do not change M0 outputs, parameters or buffers.
"""
from collections import deque
import numpy as np
import torch
from ..t5_target_dp.runtime import TargetDPPolicy
from ..t5_b94_next.model import pool_regions
from .model import load, HISTORY, FEATURES


class EndgamePolicy(TargetDPPolicy):
    name = 't5-m0-learned-endgame'

    def __init__(self, checkpoint, terminal=None, device='cuda:0', *, vision_path=None):
        super().__init__(checkpoint, device, vision_path=vision_path)
        self.model.requires_grad_(False)
        self.terminal = load(terminal, device)[0] if terminal else None
        if self.terminal is not None:
            self.terminal.requires_grad_(False)
        self.feature_history = deque(maxlen=HISTORY)
        self._hook = self.model.perception.backbone.register_forward_hook(self._maps)
        original_raw, original_sample = self.model.raw_features, self.model.sample_from_features

        def raw(*args, **kwargs):
            value = original_raw(*args, **kwargs)
            target = value[..., 240:300].reshape(1, self.history, 2, 10, 3)
            local = pool_regions(self.maps, target[..., :2].reshape(-1, 10, 2),
                                 target[..., 2].reshape(-1, 10)).reshape(1, self.history, -1)
            relative = (target[..., :5, :2]-target[..., 5:, :2]).flatten(2)
            time = value.new_tensor([self.next_step/400, (400-self.next_step)/400])
            x = torch.cat((value[:, -1], relative[:, -1], local[:, -1], time[None]), -1)[0]
            if x.shape != (FEATURES,) or not torch.isfinite(x).all():
                raise ValueError('Invalid public terminal features')
            if not self.feature_history:
                self.feature_history.extend([x]*(HISTORY-1))
            self.feature_history.append(x)
            self.last_x = torch.stack(tuple(self.feature_history))[None]
            self.last_features = value
            return value

        def sample(features, *, generator=None, **kwargs):
            if self.terminal is not None:
                # No elapsed-step/coordinate eligibility rule at deployment.
                self.last_gate = float(self.terminal.gate(self.terminal.encode(self.last_x)).sigmoid().item())
                self.ready_count = self.ready_count+1 if self.last_gate >= .5 else 0
                if not self.active and self.ready_count >= 2:
                    self.active, self.takeover_step = True, self.next_step
                if self.active:
                    chunk = self.terminal.sample(self.last_x, self.terminal_generator)
                    self.terminal_calls += 1
                    # Base runtime consumes only two actions, preserving its history bookkeeping.
                    return torch.cat((chunk, chunk[:, -1:].expand(-1, self.model.horizon-chunk.shape[1], -1)), 1)
            self.m0_calls += 1
            return original_sample(features, generator=generator, **kwargs)

        self.model.raw_features, self.model.sample_from_features = raw, sample

    def _maps(self, module, args, result):
        self.maps = result

    def reset(self, task_id, instruction, action_dim):
        super().reset(task_id, instruction, action_dim)
        self.feature_history.clear()
        self.active, self.takeover_step, self.ready_count = False, None, 0
        self.m0_calls = self.terminal_calls = 0
        self.last_gate = 0.
        self.terminal_generator = torch.Generator(device=self.device).manual_seed(20261017)

    @torch.inference_mode()
    def observe_teacher(self, state, images, step, action):
        """Public feature extraction for offline executed teacher actions; no DiT call."""
        self.next_step = step
        observations = self.observations(state, images)
        if step % 2 == 0:
            self.model.raw_features(*observations)
        self.previous = np.asarray(action, np.float32).copy()
        self.next_step = step+1
