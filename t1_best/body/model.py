"""Two learned grasp styles, exact migration of both original history frames."""
import torch
from .._vendor.model import T1DiT

STYLE_INDEX=48
OLD_INDEX=torch.cat((torch.arange(48),torch.arange(49,68)))


class StyleDiT(T1DiT):
    def __init__(self,*args,**kwargs):
        self.relative_adapter_enabled=kwargs.pop('relative_adapter',False)
        kwargs.setdefault('visual_dim',49)
        super().__init__(*args,**kwargs)
        if self.relative_adapter_enabled:
            if self.feature_dim!=68:raise ValueError('Relative adapter requires the body keypoint schema')
            self.denoiser.relative_adapter=torch.nn.Sequential(torch.nn.Linear(56,128),torch.nn.SiLU(),
                torch.nn.Linear(128,2*self.feature_dim))
            torch.nn.init.zeros_(self.denoiser.relative_adapter[-1].weight)
            torch.nn.init.zeros_(self.denoiser.relative_adapter[-1].bias)

    def condition(self,features):
        base=super().condition(features)
        if not self.relative_adapter_enabled:return base
        points=features[...,:48].reshape(*features.shape[:-1],2,8,3)
        confidence=points[...,2:3].clamp(0,1)
        uv=points[...,:2]/confidence.clamp_min(1e-4)
        # Existing learned predictions only. Pair confidence removes arbitrary
        # off-screen UV; robot/world geometry and simulator state are not used.
        delta=(uv[...,:5,:]-uv[...,5:6,:])*(confidence[...,:5,:]*confidence[...,5:6,:]).sqrt()
        axis=(uv[...,2,:]-uv[...,0,:])*(confidence[...,2,:]*confidence[...,0,:]).sqrt()
        fingers=(uv[...,7,:]-uv[...,6,:])*(confidence[...,7,:]*confidence[...,6,:]).sqrt()
        relative=torch.cat((delta.flatten(-3),axis.flatten(-2),fingers.flatten(-2)),-1)
        return base+self.denoiser.relative_adapter((relative/.1).clamp(-10,10).flatten(1))


def load_style_state(model,state):
    """Only a new, zero-output relative adapter may be missing on continuation."""
    missing,unexpected=model.load_state_dict(state,strict=False)
    allowed={key for key in model.state_dict() if key.startswith('denoiser.relative_adapter.')}
    if unexpected or (set(missing) and (not model.relative_adapter_enabled or set(missing)!=allowed)):
        raise ValueError(f'Unexpected action checkpoint migration: missing={missing}, unexpected={unexpected}')


def pack_style(features,style):
    visual_dim=features.shape[-1]-19
    if visual_dim<48:raise ValueError('Expected original T1 feature order')
    flag=torch.full_like(features[...,:1],float(style))
    return torch.cat((features[...,:visual_dim],flag,features[...,visual_dim:]),-1)


def expand_original(state,device='cpu'):
    old_dim=state['feature_mean'].numel();visual_dim=old_dim-19
    indices=torch.cat((torch.arange(visual_dim),torch.arange(visual_dim+1,old_dim+1)))
    mean=torch.zeros(old_dim+1);std=torch.ones(old_dim+1)
    mean[indices]=state['feature_mean'];std[indices]=state['feature_std']
    model=StyleDiT(mean,std,state['action_center']-state['action_scale'],
                  state['action_center']+state['action_scale'],visual_dim=visual_dim+1)
    new=model.state_dict()
    for key,value in state.items():
        if key not in ('feature_mean','feature_std','denoiser.condition.0.weight'):
            new[key]=value
    weight=torch.zeros_like(new['denoiser.condition.0.weight'])
    for frame in range(2):
        weight[:,indices+frame*(old_dim+1)]=state['denoiser.condition.0.weight'][:,frame*old_dim:(frame+1)*old_dim]
    new['denoiser.condition.0.weight']=weight
    model.load_state_dict(new,strict=True)
    return model.to(device)


def objective(model,teacher,features,actions,mask,is_old,teacher_weight,*,anchor=None,
              anchor_mask=None,anchor_weight=0.,return_anchor=False):
    normalized=((actions-model.action_center)/model.action_scale).clamp(-1,1)*mask[...,None]
    noise=torch.randn_like(normalized);time=torch.randint(100,(len(features),),device=features.device)
    noisy=model.noise_scheduler.add_noise(normalized,noise,time)
    predicted=model.denoiser(noisy,time,model.condition(features))
    loss=((predicted.float()-noise.float()).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
    distillation=predicted.float().sum()*0
    if is_old.any():
        with torch.no_grad():
            d=teacher.feature_dim-19
            indices=torch.cat((torch.arange(d,device=features.device),torch.arange(d+1,teacher.feature_dim+1,device=features.device)))
            original_features=features[is_old][...,indices]
            reference=teacher.denoiser(noisy[is_old],time[is_old],teacher.condition(original_features))
        distillation=((predicted[is_old].float()-reference.float()).square().mean(-1)*mask[is_old]).sum()/mask[is_old].sum().clamp_min(1)
    retention=predicted.float().sum()*0
    if anchor is not None and anchor_mask is not None and anchor_mask.any():
        with torch.no_grad():
            reference=anchor.denoiser(noisy[anchor_mask],time[anchor_mask],anchor.condition(features[anchor_mask]))
        retention=((predicted[anchor_mask].float()-reference.float()).square().mean(-1)*mask[anchor_mask]).sum()/mask[anchor_mask].sum().clamp_min(1)
    total=loss+teacher_weight*distillation+anchor_weight*retention
    return (total,loss,distillation,retention) if return_anchor else (total,loss,distillation)
