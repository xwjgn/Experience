"""A variational neural position/velocity gain and a positive-definite 3D forecast covariance."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KNOTS=(1,3,7,12,20,30)
KINDS=("constant","deterministic","bayesian")
INFERENCE_DRAWS=16


class GainForecaster(nn.Module):
    def __init__(self,n_context,kind):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(kind)
        self.kind=kind
        self.gain_encoder=nn.Sequential(nn.Linear(20,24),nn.Tanh())
        self.weight=nn.Parameter(torch.empty(6,24))
        self.bias=nn.Parameter(torch.tensor([.35]*3+[-.65]*3))
        nn.init.normal_(self.weight,std=.06)
        if kind=="bayesian":
            self.weight_rho=nn.Parameter(torch.full((6,24),-2.8))
            self.bias_rho=nn.Parameter(torch.full((6,),-2.8))
        self.context=nn.Sequential(nn.Linear(n_context,80),nn.LayerNorm(80),nn.SiLU(),
                                   nn.Linear(80,56),nn.LayerNorm(56),nn.SiLU())
        self.decoder=nn.Sequential(nn.Linear(56+12,72),nn.LayerNorm(72),nn.SiLU(),nn.Linear(72,64),nn.SiLU())
        self.time_constant=nn.Linear(64,1)
        self.motion=nn.Linear(64,3*len(KNOTS))
        self.chol=nn.Linear(64,6*len(KNOTS))
        nn.init.normal_(self.motion.weight,std=.012)
        nn.init.zeros_(self.motion.bias)
        nn.init.normal_(self.chol.weight,std=.02)
        nn.init.zeros_(self.chol.bias)
        interp=np.stack([np.interp(np.arange(1,31),KNOTS,y) for y in np.eye(len(KNOTS))],axis=1)
        self.register_buffer("interp",torch.tensor(interp,dtype=torch.float32))

    def sample_gain(self,draws,generator=None,mean_only=False):
        if self.kind!="bayesian" or mean_only:
            return self.weight[None],self.bias[None]
        def sample(mu,rho):
            eps=torch.randn((draws,*mu.shape),device=mu.device,generator=generator)
            return mu[None]+F.softplus(rho)[None]*eps
        return sample(self.weight,self.weight_rho),sample(self.bias,self.bias_rho)

    def kl(self):
        if self.kind!="bayesian":
            return self.weight.new_zeros(())
        prior_sd=.35
        def gaussian_kl(mu,rho):
            sd=F.softplus(rho)
            return .5*((sd/prior_sd)**2+(mu/prior_sd)**2-1-2*torch.log(sd/prior_sd)).sum()
        return gaussian_kl(self.weight,self.weight_rho)+gaussian_kl(self.bias,self.bias_rho)

    def forward(self,hist,deltas,available,flag,context,cad,sampled):
        w,b=sampled
        d,n=len(w),len(hist)
        pos=hist.new_zeros((d,n,3))
        vel=torch.zeros_like(pos)
        pp=torch.full_like(pos,.025**2)
        pv=torch.zeros_like(pos)
        vv=torch.full_like(pos,.45**2)
        seen=torch.zeros((n,),device=hist.device,dtype=torch.bool)
        correction=torch.zeros_like(pos)
        last_obs=hist.new_zeros((n,3))
        gain_p=torch.zeros_like(pos)
        gain_v=torch.zeros_like(pos)
        for j in range(hist.shape[1]):
            dt=deltas[None,:,j,None]
            valid=available[:,j]
            known=seen&valid
            first=(~seen&valid)[None,:,None]
            prior=pos+dt*vel
            acc_var=6.25
            a=pp+2*dt*pv+dt.square()*vv+.25*acc_var*dt**4
            c=pv+dt*vv+.5*acc_var*dt**3
            e=vv+acc_var*dt.square()
            measurement=hist[:,j]
            noisy=flag[None,:,j,None]
            r=torch.where(noisy,.045**2,.004**2)
            innovation=torch.where(valid[None,:,None],measurement[None]-prior,0.)
            diff=torch.where(known[:,None],measurement-last_obs,0.)[None].expand(d,-1,-1)
            if self.kind=="constant":
                logits=b[:,None].expand(-1,n,-1)
            else:
                inputs=torch.cat(((innovation/.025).clamp(-7,7),
                    (innovation/(a+r).clamp_min(1.e-9).sqrt()).clamp(-8,8),vel.clamp(-4,4),
                    (correction/.025).clamp(-7,7),((a+r).sqrt()*100).clamp(0,9),
                    (diff/.025).clamp(-7,7),noisy.expand(d,-1,-1),dt.expand(d,-1,-1)*30),dim=2)
                logits=self.gain_encoder(inputs)@w.transpose(1,2)+b[:,None]
            alpha,beta=logits.chunk(2,dim=2)
            trust=torch.where(noisy,.20,1.)*known[None,:,None]
            kp=(.04+.94*alpha.sigmoid())*trust
            kv=(.008+.65*beta.sigmoid())*trust/dt.clamp_min(.012)
            correction=kp*innovation
            pos=torch.where(first,measurement[None],prior+correction)
            vel=torch.where(first,0.,vel+kv*innovation)
            # Per-axis Joseph update for a position measurement and state (position,velocity).
            pp=torch.where(first,r,(1-kp).square()*a+kp.square()*r)
            pv=torch.where(first,0.,(1-kp)*(c-kv*a)+kp*kv*r)
            vv=torch.where(first,.45**2,e-2*kv*c+kv.square()*(a+r)).clamp_min(1.e-8)
            last_obs=torch.where(valid[:,None],measurement,last_obs)
            gain_p=torch.where(known[None,:,None],kp,gain_p)
            gain_v=torch.where(known[None,:,None],kv,gain_v)
            seen=seen|valid
        ctx=self.context(context)[None].expand(d,-1,-1)
        encoded=self.decoder(torch.cat((ctx,vel,(20*pos).clamp(-5,5),
                                        (pp.clamp_min(1.e-10).sqrt()*100).clamp(0,6),vv.sqrt().clamp(0,6)),dim=2))
        sec=cad[None,:,None]*torch.arange(1,31,device=hist.device)
        tau=.035+.78*self.time_constant(encoded).sigmoid()
        lead=tau*(-torch.expm1(-sec/tau))
        knots=self.motion(encoded).reshape(d,n,len(KNOTS),3)
        residual=torch.einsum("hk,dnkj->dnhj",self.interp,knots)
        mean=pos[:,:,None]+lead[:,:,:,None]*vel[:,:,None]+sec[:,:,:,None]*residual
        raw=self.chol(encoded).reshape(d,n,len(KNOTS),6)
        raw=torch.einsum("hk,dnkj->dnhj",self.interp,raw)
        diag=.0007+(.0025+.19*sec[:,:,:,None])*F.softplus(raw[:,:,:,:3])
        off=raw[:,:,:,3:].tanh()
        x,y,z=diag.unbind(-1)
        zeros=torch.zeros_like(x)
        chol=torch.stack((x,zeros,zeros,y*off[...,0],y,zeros,z*off[...,1],z*off[...,2],z),-1).reshape(d,n,30,3,3)
        var=(pp[:,:,None]+2*lead[:,:,:,None]*pv[:,:,None]+lead[:,:,:,None].square()*vv[:,:,None]).clamp_min(1.e-9)
        cov=chol@chol.transpose(-1,-2)+torch.diag_embed(var)
        return dict(mean=mean,cov=cov,pos=pos,vel=vel,pp=pp,pv=pv,vv=vv,kp=gain_p,kv=gain_v)


def gaussian_nll(error,cov):
    # Scalar Cholesky for a 3 x 3 matrix, avoiding a device synchronization at every training step.
    a=cov[...,0,0].clamp_min(1.e-12).sqrt()
    b=cov[...,1,0]/a
    c=cov[...,2,0]/a
    e=(cov[...,1,1]-b.square()).clamp_min(1.e-12).sqrt()
    f=(cov[...,2,1]-b*c)/e
    g=(cov[...,2,2]-c.square()-f.square()).clamp_min(1.e-12).sqrt()
    u=error[...,0]/a
    v=(error[...,1]-b*u)/e
    w=(error[...,2]-c*u-f*v)/g
    return 1.5*math.log(2*math.pi)+a.log()+e.log()+g.log()+.5*(u.square()+v.square()+w.square())


def loss_for(model,result,target,cad,turn,n_fit):
    avg=result["mean"].mean(0)
    err=(avg-target).norm(dim=2)
    unit=.015+.17*cad[:,None]*torch.arange(1,31,device=cad.device)
    position=((err/unit).mean(1)+.12*err[:,-1]/unit[:,-1])*(1+.30*turn)
    nll=gaussian_nll(target[None]-result["mean"],result["cov"]).mean()
    return position.mean()+.105*(nll+model.kl()/n_fit)
