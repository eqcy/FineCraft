"""Standalone learned T1 cap policy: RGB/qpos/instruction and own action history."""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .action import T1CapDiT
from .common import load_perception,digest
from .features import visual_features
from .schema import CAMERAS
from ..continuity import PlanContinuity

def color_index(instruction):
    if 'cap of the bottle' not in instruction.lower():
        raise ValueError('This independent candidate handles T1 cap only')
    return 0

def resize_rgb(images):
    values=[]
    for name in CAMERAS:
        image=np.asarray(images[name])
        if image.dtype!=np.uint8 or image.shape!=(512,512,3):raise ValueError('Expected official 512px uint8 RGB')
        pixels=np.asarray(Image.fromarray(image).resize((224,224),Image.Resampling.BILINEAR)).transpose(2,0,1).copy()
        values.append(pixels)
    return np.stack(values)

class DinoDiTPolicy:
    name='t1-cap-dino-multiview-local-context-dit'
    def __init__(self,checkpoint=None,*,device='cuda:0',audit=None,perception=None,action=None,continuity=False):
        self.device=torch.device(device);self.audit=Path(audit) if audit else None;self.context=None;self.pending_audit=None
        if action is None:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
            assert payload['format']=='t1_cap_multiview_action_v1'
            perception_path=Path(payload['perception_checkpoint'])
            # Deployment can package both weights without retaining training-machine paths.
            bundled=Path(checkpoint).parent/'perception.pt'
            if bundled.exists():perception_path=bundled
            if digest(perception_path)!=payload['perception_sha256']:raise ValueError('Perception and action checkpoints do not match')
            self.perception,_=load_perception(perception_path,self.device)
            norm={k:torch.tensor(v) for k,v in payload['normalization'].items()}
            self.action=T1CapDiT(**norm);state=dict(payload['model']);state.update(payload['ema'])
            self.action.load_state_dict(state,strict=True);self.action=self.action.to(self.device).eval()
            self.seed=int(payload['config']['seed']);self.execute_steps=int(payload['config']['execution_steps']);self.inference_steps=int(payload['config']['inference_steps'])
        else:
            self.perception=perception.to(self.device).eval();self.action=action.to(self.device).eval();self.seed=2026101407;self.execute_steps=2;self.inference_steps=20
        self.history=deque(maxlen=2);self.pending=deque();self.ready=False
        self.continuity=bool(continuity);self.selector=PlanContinuity()
    def reset(self,task_id,instruction,action_dim):
        if task_id!='grasp_part' or int(action_dim)!=8:raise ValueError('T1 cap learned policy requires 8 actions')
        self.color=color_index(instruction);self.previous=np.zeros(8,np.float32);self.history.clear();self.pending.clear()
        context,self.context=self.context,None
        seed=self.seed if context is None else int(context['seed'])
        self.generator=torch.Generator(device=self.device).manual_seed(seed);self.pending_audit=context;self.next_step=0;self.ready=True
        self.extra_generator=torch.Generator(device=self.device).manual_seed(seed+1)
        self.selector.reset()
    @torch.inference_mode()
    def act(self,state,images,instruction,step):
        if not self.ready or step!=self.next_step or color_index(instruction)!=self.color:raise ValueError('Reset and consecutive steps required')
        qpos=np.asarray(state,np.float32).reshape(-1)
        if qpos.shape!=(9,) or not np.isfinite(qpos).all():raise ValueError('Expected finite 9D qpos')
        if self.pending_audit is not None and self.audit:
            self.audit.parent.mkdir(parents=True,exist_ok=True)
            record={**self.pending_audit,'instruction':instruction,'state_sha256':hashlib.sha256(qpos.tobytes()).hexdigest(),
                    'image_sha256':{k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in images.items()}}
            with self.audit.open('a') as file:file.write(json.dumps(record)+'\n')
            self.pending_audit=None
        self.history.append((resize_rgb(images),qpos.copy(),self.previous.copy()))
        if not self.pending:
            past=list(self.history)
            if len(past)==1:past=[past[0],past[0]]
            pixels=torch.from_numpy(np.stack([v[0] for v in past])).to(self.device).float()/255
            colors=torch.full((2,),self.color,device=self.device,dtype=torch.long)
            visual=visual_features(self.perception,pixels)
            robot=torch.from_numpy(np.stack([v[1] for v in past])).to(self.device)
            previous=torch.from_numpy(np.stack([v[2] for v in past])).to(self.device)
            onehot=torch.nn.functional.one_hot(colors,2).float()
            features=torch.cat((visual,robot,previous,onehot),dim=1)[None]
            plans=self.action.sample(features,steps=self.inference_steps,generator=self.generator)
            if self.continuity:
                if self.selector.reference is not None:
                    extra=self.action.sample(features.expand(2,-1,-1).contiguous(),
                        steps=self.inference_steps,generator=self.extra_generator)
                    plans=torch.cat((plans,extra))
                chosen,selection=self.selector.choose(plans,step,self.action.action_scale)
                if self.audit:
                    path=self.audit.with_name(self.audit.stem+'.plans.jsonl')
                    with path.open('a') as stream:stream.write(json.dumps(dict(step=step,**selection))+'\n')
            else:chosen=0
            chunk=plans[chosen].float().cpu().numpy()
            if chunk.shape!=(16,8) or not np.isfinite(chunk).all():raise ValueError('Invalid learned action chunk')
            self.pending.extend(chunk[:self.execute_steps].copy())
        action=np.asarray(self.pending.popleft(),np.float32)
        if np.abs(action).max()>1.0001:raise ValueError('Action outside official bounds')
        self.previous=action.copy();self.next_step+=1;return action
