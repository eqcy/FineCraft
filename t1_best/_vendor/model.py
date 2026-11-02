"""T1 action DiT conditioned only on learned RGB keypoints and robot history."""

from __future__ import annotations

import math

from diffusers import DDIMScheduler, DDPMScheduler
import torch
from torch import nn


FEATURE_DIM = 48 + 9 + 8 + 2
HISTORY = 2
HORIZON = 16


def timestep_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(
        half, device=time.device) / max(half - 1, 1))
    phase = time.float()[..., None] * frequency
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition).chunk(6, -1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None].tanh() * self.attention(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None].tanh() * self.mlp(h)


class ActionDiT(nn.Module):
    def __init__(self, condition_dim: int = HISTORY * FEATURE_DIM,
                 horizon: int = HORIZON, width: int = 384,
                 layers: int = 6, heads: int = 6) -> None:
        super().__init__()
        self.width = width
        self.input = nn.Linear(8, width)
        self.position = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(),
                                       nn.Linear(width, width))
        self.blocks = nn.ModuleList(DiTBlock(width, heads) for _ in range(layers))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8))

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        timestep = torch.as_tensor(timestep, device=sample.device).reshape(-1).expand(len(sample))
        condition = (self.condition(global_cond)
                     + self.time(timestep_embedding(timestep, self.width)))
        x = self.input(sample) + self.position
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x)


class T1DiT(nn.Module):
    def __init__(self, feature_mean: torch.Tensor, feature_std: torch.Tensor,
                 action_min: torch.Tensor, action_max: torch.Tensor,
                 visual_dim: int = 48, horizon: int = HORIZON,
                 width: int = 384, layers: int = 6, heads: int = 6) -> None:
        super().__init__()
        self.feature_dim = visual_dim + 9 + 8 + 2
        self.horizon = horizon
        if (feature_mean.shape != (self.feature_dim,)
                or feature_std.shape != (self.feature_dim,)
                or action_min.shape != (8,) or action_max.shape != (8,)):
            raise ValueError("T1 DiT normalization dimensions differ")
        self.denoiser = ActionDiT(condition_dim=HISTORY * self.feature_dim,
                                  horizon=horizon, width=width,
                                  layers=layers, heads=heads)
        self.register_buffer("feature_mean", feature_mean.float())
        self.register_buffer("feature_std", feature_std.float().clamp_min(.01))
        self.register_buffer("action_center", ((action_min + action_max) / 2).float())
        self.register_buffer("action_scale", ((action_max - action_min) / 2).float().clamp_min(.01))
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")

    def condition(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1:] != (HISTORY, self.feature_dim):
            raise ValueError(f"expected two frames of {self.feature_dim}-D T1 features")
        return ((features - self.feature_mean) / self.feature_std).clamp(-10, 10).flatten(1)

    def loss(self, features: torch.Tensor, actions: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        if actions.shape[1:] != (self.horizon, 8) or mask.shape != actions.shape[:2]:
            raise ValueError("future action horizon and mask differ")
        normalized = ((actions - self.action_center) / self.action_scale).clamp(-1, 1)
        normalized = normalized * mask[..., None]
        noise = torch.randn_like(normalized)
        time = torch.randint(100, (len(actions),), device=actions.device)
        noisy = self.noise_scheduler.add_noise(normalized, noise, time)
        predicted = self.denoiser(noisy, time, self.condition(features))
        return (((predicted.float() - noise.float()).square().mean(-1) * mask).sum()
                / mask.sum().clamp_min(1))

    @torch.inference_mode()
    def sample(self, features: torch.Tensor, *, steps: int = 20,
               generator: torch.Generator | None = None) -> torch.Tensor:
        sampler = DDIMScheduler.from_config(self.noise_scheduler.config)
        sampler.set_timesteps(steps, device=features.device)
        action = torch.randn((len(features), self.horizon, 8),
                             device=features.device, generator=generator)
        condition = self.condition(features)
        for timestep in sampler.timesteps:
            noise = self.denoiser(action, timestep, condition)
            action = sampler.step(noise.float(), timestep, action, eta=0).prev_sample
        return (action * self.action_scale + self.action_center).clamp(-1, 1)
