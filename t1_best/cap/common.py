import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
import torch.distributed as dist

ROOT=Path(__file__).resolve().parents[3]

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)

def save_new(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(value,tmp);tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def distributed(seed):
    local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if 'T1_SAFE_MEMORY_BUDGET_MIB' in os.environ:
        total=torch.cuda.get_device_properties(device).total_memory/1024**2
        torch.cuda.set_per_process_memory_fraction(min(1.,int(os.environ['T1_SAFE_MEMORY_BUDGET_MIB'])/total),device)
    print(f'DISTRIBUTED INIT rank={rank}/{world} device={device}',flush=True)
    if world>1:dist.init_process_group('nccl',device_id=device)
    print(f'DISTRIBUTED READY rank={rank}/{world} device={device}',flush=True)
    torch.set_num_threads(2);torch.manual_seed(seed+rank);np.random.seed(seed+rank);random.seed(seed+rank)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    return rank,world,device

def barrier():
    if dist.is_initialized():dist.barrier(device_ids=[torch.cuda.current_device()])

def finish():
    if dist.is_initialized():dist.destroy_process_group()

def provenance(config):
    return {'config':config,'commit':os.environ.get('T1_CAP_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'torch':torch.__version__,'time':time.time(),'pid':os.getpid()}

def ema_init(model):return {name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}

@torch.no_grad()
def ema_update(model,ema):
    for name,p in model.named_parameters():
        if name in ema:ema[name].lerp_(p.detach(),.001)

@contextlib.contextmanager
def use_ema(model,ema):
    parameters=dict(model.named_parameters());backup={name:parameters[name].detach().clone() for name in ema}
    try:
        with torch.no_grad():
            for name,value in ema.items():parameters[name].copy_(value)
        yield
    finally:
        with torch.no_grad():
            for name,value in backup.items():parameters[name].copy_(value)

def load_perception(path,device):
    from transformers import modeling_utils,Dinov2Config,AutoModel
    from .perception import KeypointPerception
    saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    assert saved['format']=='t1_cap_perception_v1'
    with torch.device('meta'):
        backbone=AutoModel.from_config(Dinov2Config.from_dict(saved['vision_config']))
        model=KeypointPerception(backbone=backbone,width=128)
        # Real DINO output needs its learned projection, unlike test backbones.
        from torch import nn
        model.project=nn.Sequential(nn.Conv2d(1024,128,1),nn.GroupNorm(8,128),nn.GELU())
        model.vision_path='checkpoint-config'
    state=dict(saved['model']);state.update(saved['ema'])
    model.load_state_dict(state,strict=True,assign=True)
    return model.to(device).eval(),saved
