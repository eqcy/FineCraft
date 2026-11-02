"""Input/output transforms for the MetaFine CoRL competition tasks."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def canonicalize_toggle_switch_prompt(prompt: object) -> object:
    """Make the T3 target colour explicit while leaving other tasks unchanged."""
    if isinstance(prompt, np.ndarray):
        if prompt.ndim != 0:
            return prompt
        prompt = prompt.item()
    if not isinstance(prompt, str):
        return prompt

    normalized = prompt.strip().lower()
    if "switch" not in normalized:
        return prompt
    if "red" in normalized:
        return "Target color: RED. Toggle the RED switch on the table. RED switch only."
    if "blue" in normalized:
        return "Target color: BLUE. Toggle the BLUE switch on the table. BLUE switch only."
    return prompt


def _parse_image(image) -> np.ndarray:
    """Convert LeRobot tensors or simulator arrays to HWC uint8 images."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255.0 * image, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected a rank-3 image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class MetaFineInputs(transforms.DataTransformFn):
    """Map MetaFine observations to the canonical OpenPI observation schema."""

    model_type: _model.ModelType
    explicit_toggle_color: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["base_camera"])
        hand_image = _parse_image(data["hand_camera"])
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": hand_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = (
                canonicalize_toggle_switch_prompt(data["prompt"])
                if self.explicit_toggle_color else data["prompt"]
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class MetaFineOutputs(transforms.DataTransformFn):
    """Return the eight-dimensional pd_joint_delta_pos action used by MetaFine."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :8], dtype=np.float32)}
