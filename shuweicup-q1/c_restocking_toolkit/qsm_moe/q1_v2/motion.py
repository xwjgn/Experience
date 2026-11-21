"""Equivariant multi-scale motion experts and a bounded visual correction for the Q1 revision."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
import joblib

from settings import WORK, OUT, SEED, array, equal_env, write_json
from features import KNOTS
from qsm_data import reproducible_torch
from qsm_train import time_basis

SEEDS=(SEED+7,SEED+29)
RIDGES=(.006,.032)
MIXES=(0.,.25,.5,.75,1.)
VSTRENGTH=(0.,.25,.5,1.)


def interpolation():
    anchors=np.r_[0,KNOTS]
    return np.array([np.interp(np.arange(1,31),anchors,np.eye(len(anchors))[j])
                     for j in range(1,len(anchors))],np.float32).T


INTERP=interpolation()


def phase_inputs():
    frozen=json.loads((OUT/"phase_selection.json").read_text(encoding="utf8"))
    with np.load(WORK/(frozen["model"]+"_phase.npz")) as f:
        return f["p"]


def motion_design(v,p,state,q):
    speed,cos=state.T
    context=np.column_stack((np.ones(len(q)),p[:,0]-.35,p[:,1]-.15,speed/(speed+.35)-.5,
                            (1-cos)*.5-.3,np.sqrt(q.clip(0,1))))
    return (v[:,None]*context[:,:,None,None]).reshape(len(v),-1,3).astype("float32")


def base_path(physics,cad,tau):
    age=cad[:,None]*np.arange(1,31,dtype="float32")
    return physics[:,None,:3]+physics[:,None,3:6]*(-tau*np.expm1(-age/tau))[:,:,None]


def tensor(a,dev):
    return torch.as_tensor(np.ascontiguousarray(a).copy(),device=dev,dtype=torch.float32)


class Inputs:
    def __init__(self):
        self.meta=pd.read_parquet(WORK/"cache"/"meta.parquet")
        self.spec=json.loads((WORK/"cache"/"ready.json").read_text(encoding="utf8"))
        self.env=self.meta.env.to_numpy()
        self.roles=self.spec["environments"]
        self.fit=self.meta.env.isin(self.roles["fit"]).to_numpy()
        self.gate=self.meta.env.isin(self.roles["gate"]).to_numpy()
        self.cal=self.meta.env.isin(self.roles["calibration"]).to_numpy()
        self.val=self.meta.split.eq("validation").to_numpy()
        self.test=self.meta.split.eq("test").to_numpy()
        self.mask=np.asarray(array("mask"))
        self.grid=array("grid") & self.mask.all(1)
        self.y=array("y")
        self.v=np.load(WORK/"vectors.npy",mmap_mode="r")
        self.phase=phase_inputs()
        self.state=np.asarray(array("state"))
        self.q=np.asarray(array("prob"))
        self.cad=np.asarray(array("cadence"))
        self.phys=np.asarray(array("physics")[:,1])
        self.base=base_path(self.phys,self.cad,self.spec["tau"])
        self.visual=np.load(WORK/"local_video.npy",mmap_mode="r")
        with np.load(WORK/"v2_normalization.npz") as f:
            self.x=np.clip((np.load(WORK/"kine.npy",mmap_mode="r")-f["mu"])/f["sd"],-6,6)
            self.vm=f["video_mu"]; self.vs=f["video_sd"]


def fit_ridges(d,dev):
    ids=np.flatnonzero(d.fit & (d.meta.origin.to_numpy()%2==0) & d.mask[:,-1])
    features=motion_design(d.v[ids],d.phase[ids],d.state[ids],d.q[ids])
    scale=np.maximum(np.sqrt(np.square(features).mean((0,2))),.035).astype("float32")
    x=tensor(features.transpose(0,2,1)/scale,dev)
    age=d.cad[ids,None]*KNOTS
    y=tensor((d.y[ids][:,KNOTS-1]-d.base[ids][:,KNOTS-1])/age[:,:,None],dev)
    env=np.asarray(d.env[ids])
    _,offset,count=np.unique(env,return_counts=True,return_inverse=True)
    envw=tensor(1/count[offset],dev)
    result={}
    for lam in RIDGES:
        coefs=[]
        for j,h in enumerate(KNOTS):
            use=torch.as_tensor(d.mask[ids,h-1],device=dev)
            Z=x[use]; Y=y[use,j]
            base_w=envw[use]; w=base_w/base_w.sum()
            identity=torch.eye(Z.shape[-1],device=dev)
            bound=.045+.06/(1+h/5)
            flat=Z.flatten(0,1)
            for _ in range(4):
                weighted=(Z*w[:,None,None]).flatten(0,1)
                theta=torch.linalg.solve(weighted.T@flat+lam*identity,weighted.T@Y.flatten())
                error=torch.linalg.vector_norm(Z@theta-Y,dim=1)
                w=base_w/error.clamp_min(bound)
                w/=w.sum()
            coefs.append(theta.cpu().numpy())
        result[lam]=np.stack(coefs)
    np.savez(WORK/"motion_ridges.npz",scale=scale,**{f"c{j}":result[l] for j,l in enumerate(RIDGES)})
    return scale,result


def ridge_prediction(v,p,state,q,cad,base,scale,coef):
    x=motion_design(v,p,state,q)/scale[None,:,None]
    residual=np.einsum("nkd,hk->nhd",x,coef,optimize=True)
    knot_delta=residual*(cad[:,None]*KNOTS)[...,None]
    return (base+INTERP@knot_delta).astype("float32")


class VelocityResidual(nn.Module):
    def __init__(self,dims,nvectors):
        super().__init__()
        self.nvectors=nvectors
        self.encode=nn.Sequential(nn.Linear(dims+3,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.16),
                                  nn.Linear(48,40),nn.SiLU())
        self.coef=nn.Linear(40,3*4*nvectors)
        nn.init.zeros_(self.coef.weight)
        nn.init.zeros_(self.coef.bias)

    def forward(self,x,v,stage,basis):
        z=self.encode(torch.cat((x,stage),1))
        c=.09*self.coef(z).tanh().reshape(-1,3,4,self.nvectors)
        coefficients=(c*stage[:,:,None,None]).sum(1)
        velocity=coefficients@v
        return basis@velocity


def env_pool(groups,rng,count):
    return np.concatenate([rng.choice(ids,count,replace=True) for ids in groups])


def fit_residuals(d,base,dev):
    ids=np.flatnonzero(d.fit & d.mask[:,-1])
    vscale=np.maximum(np.sqrt(np.square(d.v[ids]).mean((0,2))),.10).astype("float32")
    np.save(WORK/"vector_scale.npy",vscale)
    x,v,p=tensor(d.x,dev),tensor(np.asarray(d.v)/vscale[None,:,None],dev),tensor(d.phase,dev)
    age=tensor(d.cad[:,None]*np.arange(1,31),dev)
    basis=time_basis(age)
    ref=tensor(base,dev); target=tensor(d.y,dev)
    masks=torch.as_tensor(np.array(d.mask),device=dev)
    groups=[ids[d.env[ids]==env] for env in sorted(set(d.env[ids]))]
    predictions=np.zeros_like(base)
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        net=VelocityResidual(x.shape[1],v.shape[1]).to(dev)
        opt=torch.optim.AdamW(net.parameters(),lr=.0014,weight_decay=.008)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,15,eta_min=.00015)
        for _ in range(15):
            for _ in range(26):
                ix=torch.as_tensor(env_pool(groups,rng,24),device=dev)
                delta=net(x[ix],v[ix],p[ix],basis[ix])
                err=((delta+ref[ix]-target[ix]).square().sum(2)+1e-7).sqrt()
                value=err/(.017+.16*age[ix])
                loss=.58*value[masks[ix]].mean()+.42*value[:,-1].mean()
                loss+=.06*(delta/(.025+.20*age[ix,:,None])).square().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),2.)
                opt.step()
            sched.step()
        torch.save(net.state_dict(),WORK/f"velocity_{seed}.pt")
        net.eval()
        with torch.inference_mode():
            for i in range(0,len(base),512):
                ix=slice(i,i+512)
                predictions[ix]+=net(x[ix],v[ix],p[ix],basis[ix]).cpu().numpy()/len(SEEDS)
    return predictions


def val_metrics(d,pred):
    # Only validation targets enter the selection criterion.
    ix=np.flatnonzero(d.val)
    err=np.linalg.norm(pred[ix]-d.y[ix],axis=2)*100
    endpoint=d.mask[ix,-1]; full=d.mask[ix].all(1)
    ade=equal_env(err[full].mean(1),d.env[ix][full])
    fde=equal_env(err[endpoint,-1],d.env[ix][endpoint])
    by=pd.Series(err[full].mean(1)).groupby(d.env[ix][full]).mean()
    return dict(ade=ade,fde=fde,objective=ade+.35*fde),by


def visual_regressors(video,v,vm,vs):
    signed=np.clip((video[:,:-1]-vm)/vs,-3,3)
    # Only temporal image evidence, relative to the background, can shift the point.
    # Features are multiplied by visible kinematic directions; no absolute scene colour or identity is fed.
    signals=np.column_stack((signed[:,0:5],signed[:,6],signed[:,7]))
    dirs=v[:,[0,3,6,7,13],:]
    return (signals[:,:,None,None]*dirs[:,None]).reshape(len(v),-1,3).astype("float32")


def fit_visual(d,base):
    ix=np.flatnonzero(d.gate & (d.meta.origin.to_numpy()%4==0) & d.mask.all(1) & (d.visual[:,-1]>0))
    z=visual_regressors(d.visual[ix],d.v[ix],d.vm,d.vs)
    sd=np.maximum(np.sqrt((z*z).mean((0,2))),.08)
    x=(z/sd[None,:,None]).transpose(0,2,1).reshape(len(ix)*3,-1)
    target=(d.y[ix][:,KNOTS-1]-base[ix][:,KNOTS-1])/(d.cad[ix,None,None]*KNOTS[None,:,None])
    y=target.transpose(0,2,1).reshape(len(ix)*3,len(KNOTS))
    _,inverse,count=np.unique(d.env[ix],return_inverse=True,return_counts=True)
    weights=1/count[inverse]; weights/=weights.mean()
    regression=Ridge(alpha=len(ix)*.35,fit_intercept=False).fit(x,y,sample_weight=np.repeat(weights,3))
    coef=regression.coef_.astype("float32")
    np.savez(WORK/"visual_adapter.npz",sd=sd,coef=coef)
    return visual_shift(d.visual,d.v,d.vm,d.vs,d.cad,sd,coef)


def visual_shift(video,v,vm,vs,cad,sd,coef):
    features=visual_regressors(video,v,vm,vs)/sd[None,:,None]
    velocity=np.einsum("nkd,hk->nhd",features,coef,optimize=True)
    velocity*=np.minimum(1.,.045/np.maximum(np.linalg.norm(velocity,axis=2),1e-9))[:,:,None]
    return (INTERP@(velocity*(cad[:,None,None]*KNOTS[None,:,None]))*video[:,-1,None,None]).astype("float32")


def fit_motion():
    reproducible_torch()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=Inputs()
    scale,coeffs=fit_ridges(d,dev)
    rows=[]; best=None; base=None
    for lam,coef in coeffs.items():
        pred=ridge_prediction(d.v,d.phase,d.state,d.q,d.cad,d.base,scale,coef)
        result,_=val_metrics(d,pred)
        rows.append(dict(candidate="kinematic_ridge",ridge=lam,**result))
        if best is None or result["objective"]<best[0]:
            best=(result["objective"],lam); base=pred
    ridge=best[1]
    print("Validation-selected motion ridge",ridge,flush=True)
    correction=fit_residuals(d,base,dev)
    blends=[]
    for w in MIXES:
        result,_=val_metrics(d,base+w*correction)
        row=dict(candidate="motion_experts",ridge=ridge,neural_weight=w,**result)
        blends.append(row);rows.append(row)
    chosen=min(blends,key=lambda r:r["objective"])
    motion=base+chosen["neural_weight"]*correction
    shift=fit_visual(d,motion)
    first,refenv=val_metrics(d,motion)
    visual=[]
    for w in VSTRENGTH:
        result,by=val_metrics(d,motion+w*shift)
        improved=int((by<refenv).sum())
        eligible=(w==0 or (improved>=3 and result["ade"]<=first["ade"]-.02 and result["fde"]<=first["fde"]+.05))
        row=dict(candidate="local_visual_residual",ridge=ridge,neural_weight=chosen["neural_weight"],visual_weight=w,
                 improved_envs=improved,eligible=eligible,**result)
        rows.append(row)
        if eligible:
            visual.append(row)
    selected=min(visual,key=lambda r:r["objective"])
    np.save(WORK/"motion_only.npy",motion)
    np.save(WORK/"visual_shift.npy",shift)
    np.save(WORK/"motion_center.npy",motion+selected["visual_weight"]*shift)
    pd.DataFrame(rows).to_csv(OUT/"motion_validation_candidates.csv",index=False)
    write_json(OUT/"motion_selection.json",dict(ridge=ridge,ridge_index=RIDGES.index(ridge),neural_weight=chosen["neural_weight"],
            visual_weight=selected["visual_weight"],visual_velocity_cap_mps=.045,seeds=SEEDS,knots=KNOTS.tolist(),
            validation_objective="environment-macro ADE + 0.35 FDE, measured in cm",
            visual_acceptance="3/4 validation environments with a smaller ADE, macro ADE gain >=0.02cm and FDE increase <=0.05cm"))
    print("Locked point expert",selected,flush=True)


if __name__=="__main__":
    fit_motion()
