"""Deployment uses learned RGB/qpos actions only; waypoints are learned from demos."""
from collections import deque
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import torch
from .model import StyleDiT,pack_style
from .._vendor.perception import load_perception
from ..continuity import PlanContinuity


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class WaypointPolicy:
    name='t1-learned-upper-waypoint-body-dit'
    def __init__(self,checkpoint,keypoint_checkpoint,*,device='cuda:0',style=1,continuity=True):
        self.device=torch.device(device);torch.set_num_threads(2)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.perception_precision=saved.get('config',{}).get('perception_precision','float32')
        if self.perception_precision not in ('float32','bfloat16'):raise ValueError('Unsupported perception precision')
        if saved['format']!='t1_waypoint_body_v1':raise ValueError('Wrong learned waypoint format')
        old_hash=saved.get('config',{}).get('old_keypoint_checkpoint_sha256',saved['keypoint_checkpoint_sha256'])
        if int(style)==0 and old_hash!=saved['keypoint_checkpoint_sha256']:
            raise ValueError('Style 0 requires its original perception; this deployment serves the refined style 1')
        if digest(keypoint_checkpoint)!=saved['keypoint_checkpoint_sha256']:raise ValueError('DINO mismatch')
        state=dict(saved['model']);state.update(saved['ema'])
        visual_dim=state['feature_mean'].numel()-19
        self.variant='body' if visual_dim==49 else 'cap' if visual_dim==847 else None
        if self.variant is None:raise ValueError('Unknown learned visual schema')
        # Match the corresponding frozen-feature cache numerical settings.
        torch.backends.cuda.matmul.allow_tf32=self.variant=='cap'
        self.model=StyleDiT(state['feature_mean'],state['feature_std'],
            state['action_center']-state['action_scale'],state['action_center']+state['action_scale'],visual_dim=visual_dim,
            relative_adapter=saved.get('config',{}).get('relative_adapter',False))
        self.model.load_state_dict(state,strict=True);self.model=self.model.to(self.device).eval()
        self.gripper_head=None
        if 'learned_gripper_head' in saved:
            if self.variant!='body' or int(style)!=1:raise ValueError('Learned gripper serves body style1 only')
            from .gripper_head import LearnedGripper
            self.gripper_head=LearnedGripper(state['feature_mean'],state['feature_std'])
            self.gripper_head.load_state_dict(saved['learned_gripper_head'],strict=True)
            self.gripper_head=self.gripper_head.to(self.device).eval()
        if self.variant=='cap':
            from ..cap.common import load_perception as load_cap
            self.perception,_=load_cap(keypoint_checkpoint,self.device)
        else:self.perception=load_perception(keypoint_checkpoint,self.device)
        self.style=int(style);self.continuity=continuity
        self.history=deque(maxlen=2);self.pending=deque();self.selector=PlanContinuity()
        self.ready=False;self.context_seed=None
    def reset(self,task_id,instruction,action_dim):
        if task_id!='grasp_part' or int(action_dim)!=8 or f'{self.variant} of the bottle' not in instruction.lower():
            raise ValueError('Wrong learned T1 route')
        self.history.clear();self.pending.clear();self.selector.reset()
        self.previous=np.zeros(8,np.float32);self.next_step=0;self.ready=True
        seed=20261012 if self.context_seed is None else int(self.context_seed);self.context_seed=None
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.extra_generator=torch.Generator(device=self.device).manual_seed(seed+1)
    @torch.inference_mode()
    def act(self,state,images,instruction,step):
        if not self.ready or step!=self.next_step or f'{self.variant} of the bottle' not in instruction.lower():
            raise ValueError('Reset and consecutive T1 body steps required')
        qpos=np.asarray(state,np.float32).reshape(-1)
        if qpos.shape!=(9,) or not np.isfinite(qpos).all():raise ValueError('Expected finite 9D qpos')
        rgb=np.stack([np.asarray(Image.fromarray(np.asarray(images[c],np.uint8)).resize(
            (224,224),Image.Resampling.BILINEAR)) for c in ('base_camera','hand_camera')])
        self.history.append((rgb,qpos.copy(),self.previous.copy()))
        if not self.pending:
            past=list(self.history)
            if len(past)==1:past=[past[0],past[0]]
            pixels=torch.from_numpy(np.stack([v[0] for v in past])).to(self.device).permute(0,1,4,2,3).float()/255
            with torch.autocast(self.device.type,dtype=torch.bfloat16,
                                enabled=self.perception_precision=='bfloat16'):
                if self.variant=='cap':
                    from ..cap.features import visual_features
                    visual=visual_features(self.perception,pixels)
                else:visual=self.perception.features(pixels)
            visual=visual.float()
            robot=torch.from_numpy(np.stack([v[1] for v in past])).to(self.device)
            previous=torch.from_numpy(np.stack([v[2] for v in past])).to(self.device)
            onehot=torch.zeros((2,2),device=self.device);onehot[:,1 if self.variant=='body' else 0]=1
            features=pack_style(torch.cat((visual,robot,previous,onehot),-1)[None],self.style)
            plans=self.model.sample(features,generator=self.generator)
            if self.continuity and self.selector.reference is not None:
                extra=self.model.sample(features.expand(2,-1,-1).contiguous(),generator=self.extra_generator)
                plans=torch.cat((plans,extra))
            chosen,_=self.selector.choose(plans,step,self.model.action_scale)
            selected=plans[chosen,:2].float().cpu().numpy().copy()
            if self.gripper_head is not None:selected[:,7]=float(self.gripper_head.action(features).item())
            self.pending.extend(selected)
        action=np.asarray(self.pending.popleft(),np.float32)
        if action.shape!=(8,) or not np.isfinite(action).all() or np.abs(action).max()>1.0001:
            raise ValueError('Invalid learned action')
        self.previous=action.copy();self.next_step+=1;return action
