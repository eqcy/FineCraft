"""Isolate the supplied learned T1 branches from other tasks' Torch settings."""
from contextlib import contextmanager
from pathlib import Path

import torch
import t1_best
from policy.assets import model_root


def snapshot_settings():
    return dict(
        threads=torch.get_num_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
    )


@contextmanager
def numerical_isolation():
    before = snapshot_settings()
    try:
        if before['threads'] != 2:
            torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = before['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = before['cudnn_allow_tf32']
        torch.backends.cudnn.benchmark = before['cudnn_benchmark']
        torch.backends.cudnn.deterministic = before['cudnn_deterministic']
        if torch.get_num_threads() != before['threads']:
            torch.set_num_threads(before['threads'])
        # The TF32 setter can collapse "medium" to "high"; restore precision last.
        torch.set_float32_matmul_precision(before['float32_matmul_precision'])


class T1BestPolicy:
    name = t1_best.T1Override.name

    def __init__(self, checkpoint, device='auto'):
        if (model_root() / "t1_best").resolve() != Path(checkpoint).resolve():
            raise ValueError("T1 checkpoint must belong to the configured model root")
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.policy = t1_best.T1Override(fallback=None, device=device)

    def reset(self, task_id, instruction, action_dim):
        with numerical_isolation():
            return self.policy.reset(task_id, instruction, action_dim)

    def act(self, state, images, instruction, step):
        with numerical_isolation():
            return self.policy.act(state, images, instruction, step)
