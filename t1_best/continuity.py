"""Choose one sampled plan using arm continuity; never average candidate actions."""
from __future__ import annotations

import torch


class PlanContinuity:
    def __init__(self, overlap=4, decay=.8, max_cost=.5):
        if overlap < 1 or not 0 < decay <= 1 or max_cost <= 0:
            raise ValueError("Invalid continuity configuration")
        self.overlap, self.decay, self.max_cost = overlap, decay, max_cost
        self.reset()

    def reset(self):
        self.reference = None
        self.reference_step = None

    def choose(self, candidates, step, action_scale):
        if candidates.ndim != 3 or candidates.shape[-1] != 8 or not torch.isfinite(candidates).all():
            raise ValueError("Expected finite candidate plans [N,H,8]")
        chosen, costs, reason = 0, None, "first_plan"
        if len(candidates) == 1:
            reason = "disabled_or_single"
        elif self.reference is not None:
            elapsed = int(step) - self.reference_step
            if elapsed <= 0:
                raise ValueError("Replanning steps must increase")
            length = min(self.overlap, candidates.shape[1], len(self.reference) - elapsed)
            if length > 0:
                # Keep gripper transitions free: only the seven arm channels
                # contribute, scaled in the checkpoint's own physical units.
                scale = action_scale[:7].to(candidates).clamp_min(.01)
                error = (candidates[:, :length, :7] - self.reference[elapsed:elapsed + length, :7]) / scale
                weight = self.decay ** torch.arange(length, device=candidates.device, dtype=candidates.dtype)
                cost = (error.square().mean(-1) * weight).sum(-1) / weight.sum()
                best = int(cost.argmin())
                costs = cost.detach().cpu().tolist()
                if costs[best] <= self.max_cost:
                    chosen, reason = best, "overlap_continuity"
                else:
                    reason = "reference_diverged_use_first"
            else:
                reason = "reference_expired"
        self.reference = candidates[chosen].detach().clone()
        self.reference_step = int(step)
        return chosen, {"selected": chosen, "costs": costs, "reason": reason}

