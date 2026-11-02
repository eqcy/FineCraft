"""Supervised full-frame keypoints and a learned, conditional action policy.

Both visual backbones predict the same named keypoints for every letter. The
instruction selects predicted keypoints; labels are used only by the auxiliary
training loss. Both action backbones receive identical numerical conditions and
use the original DP noise schedule, per-channel normalization and DDIM sampler.
The first predicted action is the current action t, not t - history + 1.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn
from torch.nn import functional as F

FORMAT = "t5_target_dp_v1"
LETTERS = ("C", "o", "R", "L")


def _unet_class():
    """Preserved upstream DP implementation, with source provenance in vendor/."""
    from .vendor.conditional_unet1d import ConditionalUnet1D
    return ConditionalUnet1D


def _group_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(32, child.num_features))
        else:
            _group_norm(child)


class CNNFeatures(nn.Module):
    """Shared ResNet18/FPN retaining a quarter-resolution spatial feature map."""
    def __init__(self, width=128):
        super().__init__()
        from torchvision.models import resnet18
        self.encoder = resnet18(weights=None)
        _group_norm(self.encoder)
        self.encoder.fc = nn.Identity()
        self.laterals = nn.ModuleList(nn.Conv2d(c, width, 1) for c in (64, 128, 256, 512))
        self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                    nn.GroupNorm(8, width), nn.GELU())

    def forward(self, pixels):
        net = self.encoder
        x = net.maxpool(net.relu(net.bn1(net.conv1(pixels))))
        maps = []
        for block in (net.layer1, net.layer2, net.layer3, net.layer4):
            x = block(x)
            maps.append(x)
        x = self.laterals[-1](maps[-1])
        for lateral, feature in zip(reversed(self.laterals[:-1]), reversed(maps[:-1])):
            x = F.interpolate(x, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            x = x + lateral(feature)
        return self.smooth(x)


class DINOFeatures(nn.Module):
    """Official pretrained dense patches with the same downstream keypoint head."""
    def __init__(self, cfg, width=128, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        path = Path(cfg["vision_path"])
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        if config.model_type not in ("dinov2", "dinov2_with_registers"):
            raise ValueError("This controlled visual ablation requires official DINOv2")
        self.encoder = (AutoModel.from_pretrained(path, local_files_only=True) if pretrained
                        else AutoModel.from_config(config))
        self.patch = int(config.patch_size)
        self.project = nn.Sequential(nn.Conv2d(config.hidden_size, width, 1),
                                     nn.GroupNorm(8, width), nn.GELU())
        self.smooth = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                    nn.GroupNorm(8, width), nn.GELU())
        self.unfreeze_blocks = int(cfg.get("unfreeze_blocks", 2))
        self.set_trainable(True)

    def set_trainable(self, enabled):
        self.requires_grad_(enabled)
        self.encoder.requires_grad_(False)
        if enabled and self.unfreeze_blocks:
            for block in self.encoder.encoder.layer[-self.unfreeze_blocks:]:
                block.requires_grad_(True)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, pixels):
        height, width = pixels.shape[-2:]
        if height % self.patch or width % self.patch:
            raise ValueError("DINO input dimensions must be divisible by patch size")
        hp, wp = height // self.patch, width // self.patch
        tokens = self.encoder(pixel_values=pixels).last_hidden_state[:, -(hp * wp):]
        maps = tokens.transpose(1, 2).reshape(len(pixels), -1, hp, wp)
        maps = F.interpolate(self.project(maps), size=(height // 4, width // 4),
                             mode="bilinear", align_corners=False)
        return self.smooth(maps)


class TinyFeatures(nn.Module):
    """CPU tests only. Production configs cannot select this implicitly."""
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, width, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(width, width, 3, 2, 1), nn.GELU())

    def forward(self, x):
        return self.net(x)


def spatial_expectation(logits):
    """Pixel centers normalized to [-1, 1], x then y, align_corners=False."""
    height, width = logits.shape[-2:]
    y = (torch.arange(height, device=logits.device, dtype=torch.float32) + .5) * (2 / height) - 1
    x = (torch.arange(width, device=logits.device, dtype=torch.float32) + .5) * (2 / width) - 1
    probability = logits.float().flatten(-2).softmax(-1).reshape(logits.shape)
    u = (probability.sum(-2) * x).sum(-1)
    v = (probability.sum(-1) * y).sum(-1)
    return torch.stack((u, v), dim=-1)


class KeypointPerception(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        self.cfg = dict(cfg)
        self.keypoints = int(cfg.get("keypoints_per_letter", 10))
        self.width = width = int(cfg.get("perception_width", 128))
        backbone = cfg.get("vision_backbone", "cnn")
        if cfg.get("test_only_tiny_vision", False):
            self.backbone = TinyFeatures(width)
        elif backbone == "cnn":
            self.backbone = CNNFeatures(width)
        elif backbone == "dinov2":
            self.backbone = DINOFeatures(cfg, width, pretrained)
        else:
            raise ValueError(f"Unknown visual backbone: {backbone}")
        self.heatmap_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                          nn.GroupNorm(8, width), nn.GELU(),
                                          nn.Conv2d(width, 4 * self.keypoints, 1))
        self.visibility_head = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                             nn.Linear(width, 4 * self.keypoints))
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([.229, .224, .225])[None, :, None, None])

    def set_trainable(self, enabled):
        self.requires_grad_(enabled)
        if isinstance(self.backbone, DINOFeatures):
            self.backbone.set_trainable(enabled)

    def forward(self, rgb):
        if rgb.ndim != 6 or rgb.shape[2:4] != (2, 3):
            raise ValueError("RGB must have shape [B,history,2,3,height,width]")
        batch, history, views, channels, height, width = rgb.shape
        if not torch.is_floating_point(rgb):
            raise ValueError("RGB must be floating point in [0,1]")
        pixels = rgb.reshape(-1, channels, height, width)
        maps = self.backbone((pixels - self.image_mean) / self.image_std)
        logits = self.heatmap_head(maps)
        prefix = (batch, history, views, 4, self.keypoints)
        uv = spatial_expectation(logits).reshape(*prefix, 2)
        visibility = self.visibility_head(maps.mean((-2, -1))).reshape(prefix)
        return {"uv": uv, "visibility_logits": visibility,
                "heatmap_logits": logits.reshape(*prefix, *logits.shape[-2:])}

    def losses(self, predicted, uv, visible):
        if predicted["uv"].shape != uv.shape or predicted["visibility_logits"].shape != visible.shape:
            raise ValueError("Keypoint labels must match [B,history,2,4,K,2] / [B,history,2,4,K]")
        visible = visible.float()
        finite = torch.isfinite(uv).all(-1)
        valid = visible * finite
        safe_uv = torch.nan_to_num(uv.float()).clamp(-1, 1)
        keypoint = ((predicted["uv"] - safe_uv).square().mean(-1) * valid).sum() / valid.sum().clamp_min(1)
        logits = predicted["heatmap_logits"].float()
        height, width = logits.shape[-2:]
        y = (torch.arange(height, device=uv.device) + .5) * (2 / height) - 1
        x = (torch.arange(width, device=uv.device) + .5) * (2 / width) - 1
        sigma = float(self.cfg.get("heatmap_sigma_pixels", 1.5))
        distance = ((x - safe_uv[..., 0, None])[..., None, :] / (2 * sigma / width)).square()
        distance = distance + ((y - safe_uv[..., 1, None])[..., :, None] / (2 * sigma / height)).square()
        distribution = (-.5 * distance).flatten(-2).softmax(-1)
        entropy = -(distribution * logits.flatten(-2).log_softmax(-1)).sum(-1)
        heatmap = (entropy * valid).sum() / valid.sum().clamp_min(1)
        visibility = F.binary_cross_entropy_with_logits(predicted["visibility_logits"].float(), visible)
        return {"keypoint_loss": keypoint, "heatmap_loss": heatmap, "visibility_loss": visibility}


def timestep_embedding(time, width):
    half = width // 2
    frequency = torch.exp(-math.log(10000) * torch.arange(half, device=time.device) / max(half - 1, 1))
    phase = time.float()[..., None] * frequency
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))

    def forward(self, x, cond):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(6, -1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None].tanh() * self.attention(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None].tanh() * self.mlp(h)


class ActionDiT(nn.Module):
    """~17M default parameters; no visual tokens or extra privileged inputs."""
    def __init__(self, condition_dim, horizon, width=384, layers=6, heads=6):
        super().__init__()
        self.width = width
        self.input = nn.Linear(8, width)
        self.position = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.condition = nn.Sequential(nn.Linear(condition_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(DiTBlock(width, heads) for _ in range(layers))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 8))

    def forward(self, sample, timestep, global_cond):
        timestep = torch.as_tensor(timestep, device=sample.device).reshape(-1).expand(len(sample))
        cond = self.condition(global_cond) + self.time(timestep_embedding(timestep, self.width))
        x = self.input(sample) + self.position
        for block in self.blocks:
            x = block(x, cond)
        return self.output(x)


class TargetKeypointPolicy(nn.Module):
    def __init__(self, cfg, *, pretrained=True):
        super().__init__()
        self.cfg = dict(cfg)
        self.history = int(cfg.get("history", 2))
        self.horizon = int(cfg.get("horizon", 16))
        self.inference_steps = int(cfg.get("inference_steps", 20))
        self.perception = KeypointPerception(cfg, pretrained)
        k = self.perception.keypoints
        # All objects preserve scene context; explicit target indexing prevents
        # the action policy from needing to rediscover letter-to-head alignment.
        self.feature_dim = 2 * 4 * k * 3 + 2 * k * 3 + 9 + 8 + 4
        condition_dim = self.history * self.feature_dim
        # Initial U-Net weights must remain identical when only the vision
        # architecture changes, regardless of RNG consumed by that architecture.
        self.cfg.setdefault("action_init_seed", int(cfg.get("seed", 20261007)))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.cfg["action_init_seed"]))
            if cfg.get("action_backbone", "unet") == "unet":
                self.denoiser = _unet_class()(
                    input_dim=8, global_cond_dim=condition_dim,
                    diffusion_step_embed_dim=int(cfg.get("diffusion_step_embed_dim", 256)),
                    down_dims=tuple(cfg.get("down_dims", (128, 256, 512))),
                    kernel_size=5, n_groups=8, cond_predict_scale=True)
            elif cfg["action_backbone"] == "dit":
                self.denoiser = ActionDiT(condition_dim, self.horizon,
                                         int(cfg.get("dit_width", 384)), int(cfg.get("dit_layers", 6)),
                                         int(cfg.get("dit_heads", 6)))
            else:
                raise ValueError("action_backbone must be unet or dit")
        # Schedulers are not nn.Modules; retain their real CPU tensors even when
        # a checkpoint is initialized on meta before assigning mmap weights.
        with torch.device("cpu"):
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=int(cfg.get("diffusion_steps", 100)),
                                                 beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                                 prediction_type="epsilon")
        self.register_buffer("feature_mean", torch.tensor(cfg.get("feature_mean", [0.] * self.feature_dim), dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(cfg.get("feature_std", [1.] * self.feature_dim), dtype=torch.float32).clamp_min(.01))
        self.register_buffer("qpos_mean", torch.tensor(cfg.get("qpos_mean", [0.] * 9), dtype=torch.float32))
        self.register_buffer("qpos_std", torch.tensor(cfg.get("qpos_std", [1.] * 9), dtype=torch.float32).clamp_min(.01))
        low = torch.tensor(cfg.get("action_min", [-1.] * 8), dtype=torch.float32)
        high = torch.tensor(cfg.get("action_max", [1.] * 8), dtype=torch.float32)
        if low.shape != (8,) or high.shape != (8,):
            raise ValueError("action_min/max must contain eight finite values")
        if low.device.type != "meta":
            if not torch.isfinite(low).all() or not torch.isfinite(high).all():
                raise ValueError("action_min/max must contain eight finite values")
            if (high < low).any():
                raise ValueError("action_max cannot be smaller than action_min")
        self.register_buffer("action_center", (low + high) / 2)
        self.register_buffer("action_scale", ((high - low) / 2).clamp_min(.01))

    def set_perception_trainable(self, enabled):
        self.perception.set_trainable(enabled)

    def normalize_actions(self, actions):
        return (actions - self.action_center) / self.action_scale

    def denormalize_actions(self, actions):
        return (actions * self.action_scale + self.action_center).clamp(-1, 1)

    def raw_features(self, rgb, qpos, letter, previous_action, predictions=None):
        batch = len(rgb)
        history = rgb.shape[1]
        if qpos.shape != (batch, history, 9) or previous_action.shape != (batch, history, 8):
            raise ValueError("qpos/previous_action shapes must be [B,history,9/8]")
        if letter.shape != (batch,) or ((letter < 0) | (letter > 3)).any():
            raise ValueError("letter must be a B-vector of C,o,R,L indices 0,1,2,3")
        predicted = self.perception(rgb) if predictions is None else predictions
        kp = torch.cat((predicted["uv"], predicted["visibility_logits"].sigmoid()[..., None]), dim=-1)
        k = self.perception.keypoints
        index = letter.long()[:, None, None, None, None, None].expand(-1, history, 2, 1, k, 3)
        target = kp.gather(3, index).squeeze(3).flatten(2)
        all_objects = kp.flatten(2)
        one_hot = F.one_hot(letter.long(), 4).float()[:, None].expand(-1, history, -1)
        features = torch.cat((all_objects, target, qpos, previous_action, one_hot), dim=-1)
        return features

    def condition_from_features(self, features):
        if features.ndim != 3 or features.shape[1:] != (self.history, self.feature_dim):
            raise ValueError(f"Expected predicted features [B,{self.history},{self.feature_dim}]")
        return ((features - self.feature_mean) / self.feature_std.clamp_min(.01)).clamp(-10, 10).flatten(1)

    def encode_features(self, rgb, qpos, letter, previous_action, *, predictions=None):
        return self.condition_from_features(self.raw_features(rgb, qpos, letter, previous_action, predictions))

    def perception_loss(self, batch, predictions=None):
        predicted = self.perception(batch["rgb"]) if predictions is None else predictions
        visible = batch["keypoint_valid"] if "keypoint_valid" in batch else batch["keypoint_visibility"]
        losses = self.perception.losses(predicted, batch["keypoints"], visible)
        losses["loss"] = (float(self.cfg.get("keypoint_loss_weight", 10.)) * losses["keypoint_loss"] +
                          float(self.cfg.get("heatmap_loss_weight", 1.)) * losses["heatmap_loss"] +
                          float(self.cfg.get("visibility_loss_weight", .1)) * losses["visibility_loss"])
        return losses

    def action_loss_from_features(self, features, actions, mask):
        actions = self.normalize_actions(actions)
        mask = mask.to(actions.dtype)
        if actions.shape != (len(actions), self.horizon, 8) or mask.shape != actions.shape[:2]:
            raise ValueError("Expected actions [B,horizon,8] and action_mask [B,horizon]")
        # Canonical padding prevents changes in placeholder labels affecting
        # supervised tokens through either convolution or attention.
        actions = actions * mask[..., None]
        noise = torch.randn_like(actions)
        time = torch.randint(self.noise_scheduler.config.num_train_timesteps, (len(actions),), device=actions.device)
        noisy = self.noise_scheduler.add_noise(actions, noise, time)
        condition = self.condition_from_features(features)
        estimated = self.denoiser(noisy, time, global_cond=condition)
        return (((estimated.float() - noise.float()).square().mean(-1) * mask).sum()
                / mask.sum().clamp_min(1))

    def forward(self, batch, action_weight=1., perception_weight=1.):
        predicted = self.perception(batch["rgb"])
        losses = self.perception_loss(batch, predicted)
        perception_loss = losses["loss"]
        action_loss = perception_loss.new_zeros(())
        if action_weight:
            features = self.raw_features(batch["rgb"], batch["qpos"], batch["letter"],
                                          batch["previous_action"], predictions=predicted)
            action_loss = self.action_loss_from_features(features, batch["actions"], batch["action_mask"])
        losses.update(action_loss=action_loss, perception_loss=perception_loss,
                      loss=action_weight * action_loss + perception_weight * perception_loss)
        return losses

    @torch.no_grad()
    def sample(self, rgb, qpos, letter, previous_action, *, steps=None, generator=None):
        features = self.raw_features(rgb, qpos, letter, previous_action)
        return self.sample_from_features(features, steps=steps, generator=generator)

    @torch.no_grad()
    def sample_from_features(self, features, *, steps=None, generator=None):
        count = self.inference_steps if steps is None else int(steps)
        if not 1 <= count <= self.noise_scheduler.config.num_train_timesteps:
            raise ValueError("Invalid number of DDIM steps")
        condition = self.condition_from_features(features)
        # Local sampler avoids timestep mutation shared by concurrent requests.
        sampler = DDIMScheduler.from_config(self.noise_scheduler.config)
        sampler.set_timesteps(count, device=features.device)
        actions = torch.randn((len(features), self.horizon, 8), device=features.device, generator=generator)
        for time in sampler.timesteps:
            noise = self.denoiser(actions, time, global_cond=condition)
            actions = sampler.step(noise.float(), time, actions, eta=0).prev_sample
        return self.denormalize_actions(actions)

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "perception": sum(p.numel() for p in self.perception.parameters()),
                "action": sum(p.numel() for p in self.denoiser.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}


def load_policy(path, device="cpu", *, use_ema=True, include_training_payload=False, vision_path=None):
    """Assign mmap weights into meta tensors, including frozen DINO parameters."""
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if vision_path is not None:
        payload = dict(payload, config=dict(payload['config'], vision_path=str(Path(vision_path).resolve())))
    if payload.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} checkpoint")
    if (payload["config"].get("vision_backbone") == "dinov2"
            and not payload["config"].get("test_only_tiny_vision", False)):
        # HF lazy imports can initialize torch._dynamo with real-tensor probes.
        # Resolve them outside meta even in a fresh HTTP-server process.
        from transformers import modeling_utils as _modeling_utils
    with torch.device("meta"):
        model = TargetKeypointPolicy(payload["config"], pretrained=False)
    state = dict(payload["model"])
    if use_ema and payload.get("ema"):
        state.update(payload["ema"])
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device).eval()
    if include_training_payload:
        return model, payload
    return model, {key: value for key, value in payload.items()
                   if key not in ("model", "ema", "optimizer", "training_state")}
