"""Two-camera DINOv2 body/gripper keypoint predictor."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import K


class KeypointPerception(nn.Module):
    def __init__(self, vision_path: str | Path | None = None, *, pretrained: bool = True,
                 backbone: nn.Module | None = None, width: int = 128) -> None:
        super().__init__()
        self.vision_path = str(vision_path) if vision_path is not None else None
        self.width = width
        if backbone is None:
            if vision_path is None:
                raise ValueError("local DINOv2-L config/weights directory required")
            from transformers import AutoConfig, AutoModel
            config = AutoConfig.from_pretrained(vision_path, local_files_only=True)
            if config.model_type != "dinov2" or config.hidden_size != 1024 or config.patch_size != 14:
                raise ValueError("expected local official DINOv2-L configuration")
            self.backbone = (AutoModel.from_pretrained(vision_path, local_files_only=True)
                             if pretrained else AutoModel.from_config(config))
            self.patch_size = config.patch_size
            self.project = nn.Sequential(nn.Conv2d(config.hidden_size, width, 1),
                                         nn.GroupNorm(8, width), nn.GELU())
            self.backbone.requires_grad_(False)
            for block in self.backbone.encoder.layer[-2:]:
                block.requires_grad_(True)
        else:
            self.backbone = backbone
            self.patch_size = 14
            self.project = nn.Identity()
        self.heatmap = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                     nn.GroupNorm(8, width), nn.GELU(),
                                     nn.Conv2d(width, K, 1))
        self.visibility = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                        nn.Linear(width, K))
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vision_path is not None:
            self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3):
            raise ValueError("RGB must have shape [B,2,3,H,W]")
        batch, views, channels, height, width = rgb.shape
        if height != 224 or width != 224:
            raise ValueError("T1 DINO input must be 224x224")
        pixels = rgb.reshape(batch * views, channels, height, width)
        if self.vision_path is not None:
            tokens = self.backbone(pixel_values=(pixels - self.image_mean) / self.image_std).last_hidden_state
            hp, wp = height // self.patch_size, width // self.patch_size
            maps = tokens[:, -(hp * wp):].transpose(1, 2).reshape(batch * views, -1, hp, wp)
            maps = F.interpolate(self.project(maps), size=(56, 56), mode="bilinear", align_corners=False)
        else:
            maps = self.project(self.backbone(pixels))
        logits = self.heatmap(maps).reshape(batch, views, K, 56, 56)
        probability = logits.float().flatten(-2).softmax(-1).reshape_as(logits)
        coordinate = (torch.arange(56, device=rgb.device, dtype=torch.float32) + .5) / 28 - 1
        u = (probability.sum(-2) * coordinate).sum(-1)
        v = (probability.sum(-1) * coordinate).sum(-1)
        valid_logits = self.visibility(maps.mean((-2, -1))).reshape(batch, views, K)
        return {"uv": torch.stack((u, v), -1), "visibility_logits": valid_logits,
                "heatmap_logits": logits,
                "context": maps.mean((-2, -1)).reshape(batch, views, -1)}

    def features(self, rgb: torch.Tensor) -> torch.Tensor:
        prediction = self(rgb)
        visibility = prediction["visibility_logits"].sigmoid()[..., None]
        # Invalid/off-screen landmarks should contribute no arbitrary UV value.
        return torch.cat((prediction["uv"] * visibility, visibility), -1).flatten(1)

    def features_with_context(self, rgb: torch.Tensor) -> torch.Tensor:
        """Keep DINO spatial context that 2D keypoint coordinates discard."""
        prediction = self(rgb)
        visibility = prediction["visibility_logits"].sigmoid()[..., None]
        keypoints = torch.cat((prediction["uv"] * visibility, visibility), -1).flatten(1)
        return torch.cat((keypoints, prediction["context"].flatten(1)), dim=1)

    def losses(self, rgb: torch.Tensor, truth: torch.Tensor,
               valid: torch.Tensor) -> dict[str, torch.Tensor]:
        predicted = self(rgb)
        valid = valid.float()
        coordinate_loss = (((predicted["uv"] - truth).square().mean(-1) * valid).sum()
                           / valid.sum().clamp_min(1))
        coordinate = (torch.arange(56, device=rgb.device, dtype=torch.float32) + .5) / 28 - 1
        sigma = 1.5 / 28
        distance = ((coordinate - truth[..., 0, None])[..., None, :] / sigma).square()
        distance = distance + ((coordinate - truth[..., 1, None])[..., :, None] / sigma).square()
        target = (-.5 * distance).flatten(-2).softmax(-1)
        heatmap_loss = (-(target * predicted["heatmap_logits"].float().flatten(-2)
                         .log_softmax(-1)).sum(-1) * valid).sum() / valid.sum().clamp_min(1)
        valid_loss = F.binary_cross_entropy_with_logits(predicted["visibility_logits"].float(), valid)
        return {"loss": 10 * coordinate_loss + heatmap_loss + .1 * valid_loss,
                "coordinate": coordinate_loss, "heatmap": heatmap_loss, "valid": valid_loss}


def load_perception(checkpoint: str | Path, device: str | torch.device = "cpu") -> KeypointPerception:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    bundled_vision_path = Path(checkpoint).resolve().parent.parent / "models/dinov2-large"
    vision_path = bundled_vision_path
    if not (vision_path / "config.json").is_file():raise FileNotFoundError(vision_path / "config.json")
    model = KeypointPerception(vision_path, pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()
