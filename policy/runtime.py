"""Five-task routing: learned body/cap T1, pi0.5 T2/T3/T4, frozen combination T5.

The three pi0.5 checkpoints are cached in host RAM. Task changes release the
active model before loading the next backend, retaining each pi0.5 RNG stream.
"""
from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
import sys
import time

from policy.assets import model_root

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = model_root()
sys.path.insert(0, str(ROOT / "vendor"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The official self-test talks to this process over loopback. Cluster HTTP
# proxies must not intercept those requests (the wire protocol is unchanged).
for _proxy_key in ("NO_PROXY", "no_proxy"):
    _bypass = [part for part in os.environ.get(_proxy_key, "").split(",") if part]
    for _host in ("127.0.0.1", "localhost"):
        if _host not in _bypass:
            _bypass.append(_host)
    os.environ[_proxy_key] = ",".join(_bypass)

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.metafine_policy import MetaFineInputs, MetaFineOutputs
from openpi.policies.policy import Policy as OpenPiPolicy
from openpi.shared import normalize

from policy.original_adapter import OpenPiMetaFinePolicy

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

PI05_BACKENDS = ("pi05",)


def _restore_torch_settings(backend: str) -> None:
    """Restore each standalone policy's process-wide precision settings."""
    if backend != "t5_combo":
        return
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def make_backend(
    model, model_config, norm_stats, tokenizer_path: Path, *, task_id: str = "",
) -> OpenPiPolicy:
    """Equivalent to create_trained_policy for the original MetaFine pi0.5 recipe.

Inference does not use the training dataset/repack transform or training loader.
The tokenizer is read from the submission itself instead of a machine cache.
"""
    return OpenPiPolicy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(None),
            MetaFineInputs(
                model_type=model_config.model_type,
                explicit_toggle_color=task_id == "toggle_switch_table",
            ),
            transforms.Normalize(norm_stats, use_quantiles=True),
            transforms.InjectDefaultPrompt(None),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(
                PaligemmaTokenizer(model_config.max_token_len, model_path=str(tokenizer_path)),
                discrete_state_input=model_config.discrete_state_input,
            ),
            transforms.PadStatesAndActions(model_config.action_dim),
        ],
        output_transforms=[
            transforms.Unnormalize(norm_stats, use_quantiles=True),
            MetaFineOutputs(),
        ],
    )


class TaskRouter:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO)
        self.config = json.loads((ROOT / "configs/policy.json").read_text())
        self.model_config = Pi0Config(**self.config["model"])
        self.tokenizer_path = MODEL_ROOT / "tokenizer/paligemma_tokenizer.model"
        if not self.tokenizer_path.is_file():
            raise FileNotFoundError(self.tokenizer_path)
        self.host_params = {}
        self.norm_stats = {}
        self.rng_states = {}
        self.active_task = None
        self.adapter = None
        self.episode_ready = False
        for task, spec in self.config["tasks"].items():
            backend = spec.get("backend", "pi05")
            checkpoint = MODEL_ROOT / spec["checkpoint"]
            if backend == "t1_best" and task == "grasp_part":
                required = [checkpoint / name for name in (
                    "body/weights/action.pt", "body/weights/perception.pt",
                    "body/models/dinov2-large/config.json", "cap/weights/action.pt",
                    "cap/weights/perception.pt", "deployment_options.json")]
            elif backend == "t5_combo" and task == "insert_letter":
                required = [MODEL_ROOT / spec[key] for key in (
                    "checkpoint", "terminal", "branch", "gate", "protection", "legacy_branch", "o_branch")]
                required.append(MODEL_ROOT / spec["vision_path"] / "config.json")
            elif backend == "pi05" and task in ("grasp_move_mug", "toggle_switch_table", "put_blocks_into_boxes"):
                required = [checkpoint / "params", checkpoint / "assets" / spec["asset_id"] / "norm_stats.json"]
            else:
                raise ValueError(f"Unsupported final-release task/backend: {task}/{backend}")
            for path in required:
                if not path.exists():
                    raise FileNotFoundError(f"Missing {task} model asset: {path}")
        for task, spec in self.config["tasks"].items():
            if spec.get("backend", "pi05") not in PI05_BACKENDS:
                continue
            checkpoint = MODEL_ROOT / spec["checkpoint"]
            started = time.monotonic()
            self.host_params[task] = model_lib.restore_params(
                checkpoint / "params", restore_type=np.ndarray, dtype=jnp.bfloat16,
            )
            self.norm_stats[task] = normalize.load(checkpoint / "assets" / spec["asset_id"])
            LOGGER.info("Loaded %s into host RAM in %.1fs", task, time.monotonic() - started)
        self._activate("grasp_part")

    def _release_active(self) -> None:
        if self.adapter is None:
            return
        previous_backend = self.config["tasks"][self.active_task].get("backend", "pi05")
        if previous_backend in PI05_BACKENDS:
            # Keep each task's original RNG stream when tasks are interleaved.
            pi_adapter = self.adapter
            self.rng_states[self.active_task] = np.asarray(
                jax.device_get(jax.random.key_data(pi_adapter._backend._rng))
            ).copy()
            del pi_adapter
        self.adapter = None
        self.active_task = None
        self.episode_ready = False
        gc.collect()
        if previous_backend in PI05_BACKENDS:
            jax.clear_caches()
        if previous_backend != "pi05":
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def _activate(self, task: str) -> None:
        if task == self.active_task:
            return
        started = time.monotonic()
        self._release_active()
        spec = self.config["tasks"][task]
        backend_name = spec.get("backend", "pi05")
        _restore_torch_settings(backend_name)
        if backend_name == "t1_best":
            from policy.t1_best_adapter import T1BestPolicy
            self.adapter = T1BestPolicy(
                checkpoint=MODEL_ROOT / spec["checkpoint"], device=spec.get("device", "auto"),
            )
            self.active_task = task
            LOGGER.info("Activated %s (body r25 + cap r3 continuity v4) in %.1fs", task, time.monotonic() - started)
            return
        if backend_name == "t5_combo":
            from policy.t5_combo.backend import ComboPolicyAdapter
            self.adapter = ComboPolicyAdapter(
                **{key: MODEL_ROOT / spec[key] for key in ("checkpoint", "terminal", "branch", "gate", "protection", "legacy_branch", "o_branch", "vision_path")},
                device=spec.get("device", "auto"),
            )
            self.active_task = task
            LOGGER.info("Activated %s (frozen T5 75.83 combination) in %.1fs", task, time.monotonic() - started)
            return
        # Match restore_params' original one-device NamedSharding. Changing to
        # SingleDeviceSharding can select different bfloat16 compilation paths.
        sharding = jax.sharding.NamedSharding(
            jax.sharding.Mesh(np.array([jax.devices()[0]]), ("x",)),
            jax.sharding.PartitionSpec(),
        )
        device_params = jax.device_put(self.host_params[task], sharding)
        model = self.model_config.load(device_params)
        backend = make_backend(
            model, self.model_config, self.norm_stats[task], self.tokenizer_path, task_id=task,
        )
        adapter = OpenPiMetaFinePolicy(
            spec["config_name"], MODEL_ROOT / spec["checkpoint"],
            replan_interval=spec.get("replan_interval", self.config["replan_interval"]),
            backend=backend,
        )
        if task in self.rng_states:
            backend._rng = jax.random.wrap_key_data(jnp.asarray(self.rng_states[task]))
        self.adapter = adapter
        self.active_task = task
        LOGGER.info("Activated %s in %.1fs", task, time.monotonic() - started)

    def reset(self, task_id: str, instruction: str, action_dim: int) -> None:
        self.episode_ready = False
        if task_id not in self.config["tasks"]:
            raise ValueError(f"Unknown task_id: {task_id}")
        if int(action_dim) != 8:
            raise ValueError(f"Expected action_dim=8, got {action_dim}")
        self._activate(task_id)
        self.adapter.reset(task_id, instruction, action_dim)
        self.episode_ready = True

    def act(self, state, images, instruction, step):
        if not self.episode_ready:
            raise RuntimeError("Call /reset with a valid task before /act")
        action = self.adapter.act(state, images, instruction, step)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise RuntimeError("Policy produced an invalid action")
        return action
