"""Learned local DINO residual conditions; only RGB predictions locate regions."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from ..t5_target_dp.model import (
    TargetKeypointPolicy, ActionDiT, spatial_expectation, timestep_embedding, FORMAT)


def pool_regions(dense, uv, visibility):
    """Pool peg/slot regions from 5+5 predicted keypoints, with global fallback.

    dense [N,D,H,W], uv [N,10,2], visibility [N,10]. No teacher inputs.
    The minimum radius covers two patch cells; low visibility uses the mean map.
    """
    if uv.shape[1:] != (10, 2) or visibility.shape != uv.shape[:2]:
        raise ValueError('Ten predicted peg/slot points required')
    pooled = []
    for begin in (0, 5):
        points, weights = uv[:, begin:begin+5], visibility[:, begin:begin+5]
        valid = torch.isfinite(points).all(-1) & (weights > .5) & (points.abs() <= 1).all(-1)
        safe = torch.nan_to_num(points).clamp(-1, 1)
        w = weights * valid
        center = (safe*w[..., None]).sum(1)/w.sum(1, keepdim=True).clamp_min(1e-6)
        extent = ((safe-center[:, None]).abs()*valid[..., None]).amax(1)
        radius = (extent*1.5).clamp(min=2/min(dense.shape[-2:]), max=.8)
        axis = torch.linspace(-1, 1, 3, device=dense.device, dtype=dense.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        grid = center[:, None, None]+torch.stack((xx, yy), -1)[None]*radius[:, None, None]
        values = F.grid_sample(dense, grid.clamp(-1, 1), mode='bilinear',
                               padding_mode='border', align_corners=False).mean((-2, -1))
        values = torch.where((valid.sum(1) >= 2)[:, None], values, dense.mean((-2, -1)))
        pooled.append(values)
    return torch.cat(pooled, dim=-1)


@torch.no_grad()
def predicted_features_and_local(model, rgb, qpos, letter, previous):
    """One frozen DINO forward feeds original heads and dense local descriptors."""
    p = model.perception
    b, hist, views, channels, height, width = rgb.shape
    backbone = p.backbone
    pixels = (rgb.reshape(-1, channels, height, width)-p.image_mean)/p.image_std
    hp, wp = height//backbone.patch, width//backbone.patch
    tokens = backbone.encoder(pixel_values=pixels).last_hidden_state[:, -(hp*wp):]
    dense = tokens.transpose(1, 2).reshape(len(pixels), -1, hp, wp)
    maps = backbone.smooth(F.interpolate(backbone.project(dense), size=(height//4, width//4),
                                        mode='bilinear', align_corners=False))
    logits = p.heatmap_head(maps)
    prefix = (b, hist, views, 4, p.keypoints)
    uv = spatial_expectation(logits).reshape(*prefix, 2)
    vis_logits = p.visibility_head(maps.mean((-2, -1))).reshape(prefix)
    predicted = dict(uv=uv, visibility_logits=vis_logits)
    old = TargetKeypointPolicy.raw_features(model, rgb, qpos, letter, previous, predictions=predicted)
    flat_letters = letter[:, None, None].expand(b, hist, views).reshape(-1)
    idx = torch.arange(len(pixels), device=rgb.device)
    target_uv = uv.reshape(-1, 4, p.keypoints, 2)[idx, flat_letters]
    target_vis = vis_logits.sigmoid().reshape(-1, 4, p.keypoints)[idx, flat_letters]
    local = pool_regions(dense, target_uv, target_vis).reshape(b, hist, -1)
    return old, local


class LocalDiT(ActionDiT):
    def __init__(self, condition_dim, local_dim, horizon, width, layers, heads):
        super().__init__(condition_dim, horizon, width, layers, heads)
        self.original_condition_dim = condition_dim
        self.local_adapter = nn.Sequential(nn.Linear(local_dim, 256), nn.SiLU(), nn.Linear(256, width))
        nn.init.zeros_(self.local_adapter[-1].weight)
        nn.init.zeros_(self.local_adapter[-1].bias)

    def forward(self, sample, timestep, global_cond):
        original = global_cond[:, :self.original_condition_dim]
        local = global_cond[:, self.original_condition_dim:]
        timestep = torch.as_tensor(timestep, device=sample.device).reshape(-1).expand(len(sample))
        cond = self.condition(original)+self.time(timestep_embedding(timestep, self.width))
        cond = cond+self.local_adapter(local)
        x = self.input(sample)+self.position
        for block in self.blocks:
            x = block(x, cond)
        return self.output(x)


class NextPolicy(TargetKeypointPolicy):
    def __init__(self, cfg, pretrained=False):
        super().__init__(cfg, pretrained=pretrained)
        self.use_local = bool(cfg.get('local_dino', False))
        self.local_dim = int(cfg.get('local_feature_dim', 4096))
        if self.use_local:
            self.denoiser = LocalDiT(self.history*self.feature_dim, self.history*self.local_dim,
                self.horizon, int(cfg.get('dit_width', 384)), int(cfg.get('dit_layers', 6)),
                int(cfg.get('dit_heads', 6)))
            self.register_buffer('local_mean', torch.tensor(cfg.get('local_mean', [0.]*self.local_dim)))
            self.register_buffer('local_std', torch.tensor(cfg.get('local_std', [1.]*self.local_dim)).clamp_min(.01))

    def raw_features(self, rgb, qpos, letter, previous_action, predictions=None):
        if not self.use_local:
            return super().raw_features(rgb, qpos, letter, previous_action, predictions)
        old, local = predicted_features_and_local(self, rgb, qpos, letter, previous_action)
        return torch.cat((old, local), dim=-1)

    def condition_from_features(self, features):
        if not self.use_local:
            return super().condition_from_features(features)
        if features.shape[1:] != (self.history, self.feature_dim+self.local_dim):
            raise ValueError('Local feature history/dim mismatch')
        original = super().condition_from_features(features[..., :self.feature_dim])
        local = ((features[..., self.feature_dim:]-self.local_mean)/self.local_std).clamp(-10, 10).flatten(1)
        return torch.cat((original, local), dim=-1)


def load_policy(path, device='cpu', use_ema=True):
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if payload['format'] != FORMAT:
        raise ValueError('Unexpected checkpoint format')
    from transformers import modeling_utils
    with torch.device('meta'):
        model = NextPolicy(payload['config'])
    state = dict(payload['model'])
    if use_ema:
        state.update(payload.get('ema', {}))
    model.load_state_dict(state, assign=True, strict=True)
    return model.to(device).eval(), payload
