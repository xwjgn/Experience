"""Three 3-D wrist waypoints conditioned on observed motion and two as-of RGB thumbnails."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WristDiffusion(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.vision = nn.Sequential(nn.Conv2d(6,16,5,stride=2,padding=2),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Conv2d(24,32,3,stride=2,padding=1),nn.SiLU(),nn.AvgPool2d((3,4)),
            nn.Flatten(),nn.Linear(128,32))
        self.motion = nn.GRU(dims,32,batch_first=True)
        self.context = nn.Sequential(nn.Linear(64,64),nn.LayerNorm(64),nn.SiLU())
        self.mean = nn.Linear(64,9)
        self.noise = nn.Sequential(nn.Linear(64+9+4,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,9))

    def condition(self, history, rgb):
        _, h = self.motion(history)
        visual = self.vision(rgb)
        context = self.context(torch.cat((h[-1],visual),1))
        return visual, context, self.mean(context)

    def denoise(self, ctx, value, step):
        phase = step.float()*np.pi/16
        embedding = torch.stack((phase.sin(),phase.cos(),(phase*2).sin(),(phase*2).cos()),1)
        return self.noise(torch.cat((ctx,value,embedding),1))


def draw_knots(net,ctx,mean,noise,schedule,alpha):
    k=len(noise)
    ctx=ctx.repeat_interleave(k,0)
    value=noise.repeat(len(mean),1)
    for cur,prev in zip((16,13,10,7,4,1),(13,10,7,4,1,0)):
        ts=torch.full((len(value),),cur,device=value.device)
        eps=net.denoise(ctx,value,ts)
        clean=schedule._predict_xstart_from_eps(value,ts-1,eps).clamp(-8,8)
        value=alpha[prev].sqrt()*clean+(1-alpha[prev]).sqrt()*eps
    return (.20*(mean[:,None]+value.reshape(-1,k,9))).reshape(-1,k,3,3)


def fit_and_sample(sequence, left, video_index, images, truth, mask, train, out, schedule, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = WristDiffusion(sequence.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.002)
    x, start, vi, rgb = [torch.as_tensor(v,device=device) for v in (sequence,left,video_index,images)]
    offsets = torch.tensor([18,12,8,5,3,1,0],device=device)
    def condition(ids):
        seq = x[torch.maximum(ids[:,None]-offsets,start[ids,None])]
        pair = rgb[vi[ids].long()].permute(0,1,4,2,3).reshape(-1,6,48,64)
        return net.condition(seq,pair.float()/127.5-1.)
    target = torch.as_tensor(truth[:,[9,19,29]],device=device).reshape(-1,9)/.20
    eligible = np.flatnonzero(train & mask[:,[9,19,29]].all(1))
    alpha = torch.as_tensor(np.r_[1.,schedule.alphas_cumprod],dtype=torch.float32,device=device)
    for epoch in range(22):
        for ix in np.array_split(rng.choice(eligible,6400,replace=False),25):
            ids = torch.as_tensor(ix,device=device)
            _,ctx,mean = condition(ids)
            ts = torch.randint(1,17,(len(ix),),device=device)
            eps = torch.randn_like(mean)
            resid = target[ids]-mean.detach()
            noisy = alpha[ts,None].sqrt()*resid+(1-alpha[ts,None]).sqrt()*eps
            loss = F.smooth_l1_loss(mean,target[ids],beta=.3)+.55*F.mse_loss(net.denoise(ctx,noisy,ts),eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
    torch.save(net.state_dict(),out/"qsm_visual_diffusion_scratch.pt")
    print("New wrist/video diffusion trained; sampling eight hypotheses",flush=True)
    net.eval()
    half = torch.randn(4,9,device=device,generator=torch.Generator(device=device).manual_seed(seed+1))
    noise = torch.cat((half,-half))
    np.save(out/"q1_sampler_noise.npy",noise.cpu().numpy())
    vision = np.empty((len(x),32),np.float32)
    samples = np.empty((len(x),8,3,3),np.float32)
    with torch.inference_mode():
        for first in range(0,len(x),384):
            ids=torch.arange(first,min(first+384,len(x)),device=device)
            vis,ctx,mean=condition(ids)
            vision[first:first+len(ids)]=vis.cpu()
            samples[first:first+len(ids)]=draw_knots(net,ctx,mean,noise,schedule,alpha).cpu()
    return vision,samples
