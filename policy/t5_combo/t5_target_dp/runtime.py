"""Official RGB/qpos/instruction deployment; all eight actions are learned."""
from collections import deque
import contextlib
import re

import numpy as np
from PIL import Image
import torch

LETTERS = ('C', 'o', 'R', 'L')
CAMERAS = ('base_camera', 'hand_camera')


def parse_letter(instruction):
    match = re.search(r"\bletter\s+['\"]?([CoRL])(?:['\"]|\b)", instruction)
    if match is None:
        raise ValueError('Instruction must name one of C, o, R, L')
    return LETTERS.index(match.group(1))


def resize_rgb(rgb, image_size=224, cache_size=336):
    """Match the existing 512->336 cache and dataset's 336->224 resize exactly."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (512, 512, 3):
        raise ValueError('Expected official 512x512 uint8 RGB')
    image = Image.fromarray(rgb)
    if cache_size != 512:
        image = image.resize((cache_size, cache_size), Image.Resampling.BILINEAR)
    if image_size != cache_size:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image).copy()


class TargetDPPolicy:
    name = 't5-learned-target-diffusion'

    def __init__(self, checkpoint=None, device='cuda:0', *, model=None, vision_path=None):
        self.device = torch.device(device)
        if model is None:
            from .model import load_policy
            model, payload = load_policy(checkpoint, device=device, use_ema=True, vision_path=vision_path)
            self.checkpoint_step = payload['step']
            if payload.get('stage', payload['config'].get('stage')) != 'action':
                raise ValueError('Deployment requires a trained action checkpoint')
            del payload
        else:
            model = model.to(self.device).eval()
            self.checkpoint_step = None
        self.model = model
        self.cfg = dict(model.cfg)
        self.history = int(self.cfg.get('history', 2))
        self.execution_horizon = int(self.cfg.get('execution_horizon', 2))
        self.rgb_history = deque(maxlen=self.history)
        self.qpos_history = deque(maxlen=self.history)
        self.previous_history = deque(maxlen=self.history)
        self.pending_actions = deque()
        self.ready = False

    def reset(self, task_id, instruction, action_dim):
        if task_id != 'insert_letter' or int(action_dim) != 8:
            raise ValueError('This policy supports T5 and its eight official actions')
        self.letter = parse_letter(instruction)
        self.previous = np.zeros(8, np.float32)
        self.rgb_history.clear()
        self.qpos_history.clear()
        self.previous_history.clear()
        self.pending_actions.clear()
        self.generator = torch.Generator(device=self.device).manual_seed(
            int(self.cfg.get('inference_seed', 20261007)))
        self.next_step = 0
        self.ready = True

    def observations(self, state, images):
        qpos = np.asarray(state, np.float32).reshape(-1)
        if qpos.shape != (9,) or not np.isfinite(qpos).all():
            raise ValueError('Expected nine finite qpos coordinates')
        rgb = np.stack([resize_rgb(images[c], self.cfg.get('image_size', 224),
                                  self.cfg.get('source_image_size', 336)) for c in CAMERAS])
        pixels = torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()).float().div_(255)
        robot = torch.from_numpy(qpos.copy())
        previous = torch.from_numpy(self.previous.copy())
        if not self.rgb_history:
            for _ in range(self.history - 1):
                self.rgb_history.append(pixels)
                self.qpos_history.append(robot)
                self.previous_history.append(previous)
        self.rgb_history.append(pixels)
        self.qpos_history.append(robot)
        self.previous_history.append(previous)
        return (torch.stack(tuple(self.rgb_history))[None].to(self.device),
                torch.stack(tuple(self.qpos_history))[None].to(self.device),
                torch.tensor([self.letter], dtype=torch.long, device=self.device),
                torch.stack(tuple(self.previous_history))[None].to(self.device))

    @torch.inference_mode()
    def act(self, state, images, instruction, step):
        if not self.ready or step != self.next_step or parse_letter(instruction) != self.letter:
            raise ValueError('Reset and consecutive steps with one unchanged instruction required')
        rgb, qpos, letter, previous = self.observations(state, images)
        if not self.pending_actions:
            # Prediction caches use the frozen EMA perception model in FP32.
            # Keep the same feature distribution online; AMP only denoises actions.
            features = self.model.raw_features(rgb, qpos, letter, previous)
            cast = (torch.autocast('cuda', dtype=torch.bfloat16)
                    if self.device.type == 'cuda' and self.cfg.get('inference_bf16', False)
                    else contextlib.nullcontext())
            with cast:
                chunk = self.model.sample_from_features(features, generator=self.generator)
            chunk = chunk[0].float().cpu().numpy()
            if chunk.shape != (int(self.cfg.get('horizon', 16)), 8) or not np.isfinite(chunk).all():
                raise ValueError('Invalid learned action chunk')
            self.pending_actions.extend(chunk[:self.execution_horizon].copy())
        action = np.asarray(self.pending_actions.popleft(), np.float32)
        if action.shape != (8,) or not np.isfinite(action).all() or np.abs(action).max() > 1.00001:
            raise ValueError('Invalid official normalized learned action')
        self.previous = action.copy()
        self.next_step += 1
        return action
