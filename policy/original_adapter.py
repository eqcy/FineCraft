"""Original local OpenPI adapter; the Policy backend is supplied by runtime.py."""
from __future__ import annotations
from collections import deque
import logging
from pathlib import Path
import numpy as np

class OpenPiMetaFinePolicy:
    """Adapt OpenPI action chunks to the competition's one-action HTTP API."""

    name = "openpi-pi05-metafine"

    def __init__(
        self,
        config_name: str,
        checkpoint: str | Path,
        *,
        replan_interval: int = 10,
        warmup: bool = True,
        backend=None,
    ) -> None:
        self.config_name = config_name
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.replan_interval = int(replan_interval)
        if self.replan_interval <= 0:
            raise ValueError("replan_interval must be positive")
        self.action_dim = 8
        self.task_id = ""
        self.instruction = ""
        self._action_queue: deque[np.ndarray] = deque()
        self._steps_since_replan = 0

        if backend is None:
            raise ValueError("The final-release router must supply its pretrained backend")
        self._backend = backend

        if warmup:
            logging.info("Compiling one π0.5 inference call...")
            self._predict_chunk(
                np.zeros(9, dtype=np.float32),
                {
                    "base_camera": np.zeros((512, 512, 3), dtype=np.uint8),
                    "hand_camera": np.zeros((512, 512, 3), dtype=np.uint8),
                },
                "Move the robot safely",
            )
            logging.info("π0.5 inference warmup finished")

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        if int(action_dim) != 8:
            raise ValueError(f"MetaFine π0.5 expects action_dim=8, got {action_dim}")
        self.task_id = task_id
        self.instruction = instruction
        self.action_dim = int(action_dim)
        self._action_queue.clear()
        self._steps_since_replan = 0

    def _predict_chunk(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        instruction: str,
    ) -> np.ndarray:
        output = self._backend.infer(
            {
                "state": np.asarray(state, dtype=np.float32),
                "base_camera": np.asarray(images["base_camera"], dtype=np.uint8),
                "hand_camera": np.asarray(images["hand_camera"], dtype=np.uint8),
                "prompt": instruction,
            }
        )
        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise ValueError(f"OpenPI returned invalid action chunk shape: {actions.shape}")
        return actions[:, : self.action_dim]

    def act(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        instruction: str,
        step: int,
    ) -> np.ndarray:
        del step
        if not self._action_queue or self._steps_since_replan >= self.replan_interval:
            chunk = self._predict_chunk(state, images, instruction or self.instruction)
            keep = min(self.replan_interval, len(chunk))
            self._action_queue = deque(chunk[:keep])
            self._steps_since_replan = 0
        self._steps_since_replan += 1
        return np.clip(self._action_queue.popleft(), -1.0, 1.0)
