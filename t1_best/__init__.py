"""Portable learned T1 body/cap routing; other tasks delegate to the captain."""
from contextlib import contextmanager
from pathlib import Path
from policy.assets import model_root


@contextmanager
def numerical_settings(variant):
    import torch
    before=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32=variant=='cap'
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=variant=='cap'
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,torch.backends.cudnn.benchmark=before


class LocalPolicy:
    def __init__(self,backend,variant):self.backend=backend;self.variant=variant
    def reset(self,task_id,instruction,action_dim):
        with numerical_settings(self.variant):return self.backend.reset(task_id,instruction,action_dim)
    def act(self,state,images,instruction,step):
        with numerical_settings(self.variant):return self.backend.act(state,images,instruction,step)


def load_body_policy(device='cuda:0'):
    from .body.runtime import WaypointPolicy
    root=model_root()/'t1_best/body'
    with numerical_settings('body'):
        backend=WaypointPolicy(root/'weights/action.pt',root/'weights/perception.pt',device=device,style=1,continuity=True)
        assert backend.variant=='body'
    return LocalPolicy(backend,'body')


def load_cap_policy(device='cuda:0'):
    import json
    from .cap.runtime import DinoDiTPolicy
    root=model_root()/'t1_best/cap'
    options=json.loads((root.parent/'deployment_options.json').read_text())
    kwargs={'continuity':True} if options.get('cap_continuity',False) else {}
    with numerical_settings('cap'):backend=DinoDiTPolicy(root/'weights/action.pt',device=device,**kwargs)
    return LocalPolicy(backend,'cap')


class T1Override:
    name='t1-evaluated-body-cap-learned'
    def __init__(self,fallback=None,*,device='cuda:0',body=None,cap=None):
        self.fallback=fallback;self.device=device;self.body=body;self.cap=cap;self.active=None
    def reset(self,task_id,instruction,action_dim):
        self.active=None
        if task_id=='grasp_part' and 'body of the bottle' in instruction.lower():
            if self.body is None:self.body=load_body_policy(self.device)
            selected=self.body
        elif task_id=='grasp_part' and 'cap of the bottle' in instruction.lower():
            if self.cap is None:self.cap=load_cap_policy(self.device)
            selected=self.cap
        else:
            if self.fallback is None:raise ValueError('This package serves T1; provide the existing policy for other tasks')
            selected=self.fallback
        selected.reset(task_id,instruction,action_dim);self.active=selected
    def act(self,state,images,instruction,step):
        if self.active is None:raise RuntimeError('Call reset before act')
        return self.active.act(state,images,instruction,step)


def load_policy(device='cuda:0'):return T1Override(device=device)
