"""Small shared terminal diffusion actor and phase head; no simulator imports."""
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler
from ..t5_target_dp.model import ActionDiT

FEATURES = 855  # original 321 + predicted relative points 20 + local maps 512 + time 2
HISTORY = 4
HORIZON = 8


class Endgame(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(FEATURES))
        self.register_buffer('std', torch.ones(FEATURES))
        self.encoder = nn.Sequential(nn.Linear(FEATURES, 256), nn.LayerNorm(256), nn.SiLU())
        self.temporal = nn.GRU(256, 256, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
        self.actor = ActionDiT(256, HORIZON, width=256, layers=4, heads=4)
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True, prediction_type='epsilon')

    def encode(self, x):
        x = ((x.float() - self.mean) / self.std.clamp_min(.01)).clamp(-10, 10)
        return self.temporal(self.encoder(x))[0][:, -1]

    def loss(self, actor_x, actions, mask, gate_x, labels):
        cond = self.encode(actor_x)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(100, (len(actions),), device=actions.device)
        noisy = self.scheduler.add_noise(actions * mask[..., None], noise, timesteps)
        estimate = self.actor(noisy, timesteps, cond)
        bc = ((estimate.float()-noise).square().mean(-1)*mask).sum()/mask.sum().clamp_min(1)
        gate = F.binary_cross_entropy_with_logits(self.gate(self.encode(gate_x)).flatten(), labels)
        return bc + gate, bc.detach(), gate.detach()

    @torch.inference_mode()
    def sample(self, x, generator):
        cond = self.encode(x)
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(12, device=x.device)
        action = torch.randn((len(x), HORIZON, 8), device=x.device, generator=generator)
        for t in scheduler.timesteps:
            noise = self.actor(action, t, cond)
            action = scheduler.step(noise.float(), t, action, eta=0).prev_sample
        return action.clamp(-1, 1)


def load(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['format'] != 't5-learned-endgame-v1':
        raise ValueError('Wrong terminal model format')
    model = Endgame()
    model.load_state_dict(payload['ema'])
    return model.to(device).eval(), payload
