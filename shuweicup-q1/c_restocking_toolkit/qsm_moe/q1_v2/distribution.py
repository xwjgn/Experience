"""A new motion-conditioned residual diffusion, output selection and delayed-feedback calibration."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge

from settings import WORK, OUT, SEED, array, equal_env, write_json
from motion import Inputs
from phase import decide
from qsm_data import diffusion_schedule, reproducible_torch
from qsm_q1 import geometry, fit_gate, gate_value, medoid, feedback_scales, rank95


DK=np.array([3,8,15,22,30])
ANCHORS=np.r_[0,DK]
DMAT=np.array([np.interp(np.arange(1,31),ANCHORS,np.eye(len(ANCHORS))[j]) for j in range(1,len(ANCHORS))],np.float32).T
DTYPE=np.float32


class ResidualDiffusion(nn.Module):
    def __init__(self,dims):
        super().__init__()
        self.ctx=nn.Sequential(nn.Linear(dims,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,32))
        self.eps=nn.Sequential(nn.Linear(32+len(DK)*3+6,96),nn.SiLU(),nn.Linear(96,80),nn.SiLU(),nn.Linear(80,len(DK)*3))

    def denoise(self,c,value,t):
        phase=t.float()[:,None]*torch.tensor([1.,2.,4.],device=t.device)*np.pi/16
        return self.eps(torch.cat((c,value,phase.sin(),phase.cos()),1))


def noise_scale(cad,state):
    return (.003+cad[:,None]*DK[None,:]*(.09+.12*np.sqrt(np.clip(state[:,0],0,2)))[:,None]).astype("float32")


def draw(net,x,scales,noise,schedule,alpha):
    c=net.ctx(x).repeat_interleave(len(noise),0)
    v=noise.repeat(len(x),1)
    for step,nxt in zip((16,13,10,7,4,1),(13,10,7,4,1,0)):
        ts=torch.full((len(v),),step,device=v.device)
        eps=net.denoise(c,v,ts)
        clean=schedule._predict_xstart_from_eps(v,ts-1,eps).clamp(-7,7)
        v=alpha[nxt].sqrt()*clean+(1-alpha[nxt]).sqrt()*eps
    return (v.reshape(len(x),len(noise),len(DK),3)*scales[:,None,:,None]).cpu().numpy()


def fit_generator(d,point,dev):
    seed=SEED+71
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    x=np.column_stack((d.x,d.phase)).astype(DTYPE)
    inputs=torch.tensor(x,device=dev)
    scales=noise_scale(d.cad,d.state)
    scale=torch.tensor(scales,device=dev)
    ids=np.flatnonzero(d.fit & d.mask[:,DK-1].all(1))
    target=((d.y[ids][:,DK-1]-point[ids][:,DK-1])/scales[ids,:,None]).reshape(len(ids),-1)
    y=torch.tensor(target,device=dev)
    source=torch.as_tensor(ids,device=dev)
    net=ResidualDiffusion(x.shape[1]).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=.0018,weight_decay=.005)
    schedule=diffusion_schedule()
    alpha=torch.tensor(np.r_[1.,schedule.alphas_cumprod],dtype=torch.float32,device=dev)
    groups=[np.flatnonzero(d.env[ids]==s) for s in sorted(set(d.env[ids]))]
    for epoch in range(26):
        for _ in range(26):
            local=np.concatenate([rng.choice(i,24,replace=True) for i in groups])
            local=torch.as_tensor(local,device=dev)
            ts=torch.randint(1,17,(len(local),),device=dev)
            eps=torch.randn_like(y[local])
            noisy=alpha[ts,None].sqrt()*y[local]+(1-alpha[ts,None]).sqrt()*eps
            pred=net.denoise(net.ctx(inputs[source[local]]),noisy,ts)
            loss=F.mse_loss(pred,eps)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
    torch.save(net.state_dict(),WORK/"residual_diffusion.pt")
    # Use fixed, antithetic paths independent of device RNG at deployment.
    half=np.random.default_rng(seed+1).normal(size=(4,len(DK)*3)).astype(DTYPE)
    noise=np.concatenate((half,-half),axis=0).astype(DTYPE)
    np.save(WORK/"residual_noise.npy",noise)
    noise=torch.as_tensor(noise,device=dev)
    result=np.empty((len(point),len(noise),len(DK),3),DTYPE)
    net.eval()
    with torch.inference_mode():
        for first in range(0,len(point),512):
            sl=slice(first,first+512)
            result[sl]=draw(net,inputs[sl],scale[sl],noise,schedule,alpha)
    np.save(WORK/"residual_knots.npy",result)
    return result


def supports(point,residual,gain):
    return point[:,None]+gain*(DMAT@residual)


def gate_features(p,q,state,physics):
    return np.column_stack((p[:,0],p[:,1],q,np.sqrt(q.clip(0,1)),state,
                            np.log(.00001+physics[:,6]),np.linalg.norm(physics[:,3:6],axis=1)))


def readouts(point,candidates,w):
    # Pairwise costs depend only on the issued supports. There is no reference path in this interface.
    ab,within,_=geometry(point,candidates,point)
    chosen,_,_=medoid(point,candidates,ab,within,w)
    mean=point*(1-w[:,None,None])+w[:,None,None]*candidates.mean(1)
    weights=np.column_stack((1-w,np.broadcast_to(w[:,None]/candidates.shape[1],(len(w),candidates.shape[1]))))
    sup=np.concatenate((point[:,None],candidates),1)
    central=mean.copy()
    for _ in range(10):
        dist=np.maximum(np.linalg.norm(sup-central[:,None],axis=-1),.0004)
        importance=weights[:,:,None]/dist
        central=(importance[:,:,:,None]*sup).sum(1)/importance.sum(1)[:,:,None]
    return {"motion":point,"medoid":chosen,"mean":mean.astype(DTYPE),"geometric_median":central.astype(DTYPE)}


def widths_x(point,center,candidates,w,p,state,q,cad):
    age=cad[:,None]*np.arange(1,31)
    spread=(1-w[:,None])*np.square(point-center).sum(2)+w[:,None]*np.square(candidates-center[:,None]).sum(3).mean(1)
    spread=np.sqrt(np.maximum(spread,0))
    static=np.column_stack((p[:,0],p[:,1],np.log1p(state[:,0]*3),state[:,1],np.sqrt(q.clip(0,1))))
    x=np.concatenate((np.broadcast_to(static[:,None],(len(q),30,static.shape[1])),
           np.log1p(spread/(age+.03)*4)[:,:,None],np.log1p(np.linalg.norm(center,axis=2)/(age+.03)*4)[:,:,None]),axis=2)
    return x.astype(DTYPE),age.astype(DTYPE)


def fit_width(x,age,err,env):
    _,idx,count=np.unique(env,return_inverse=True,return_counts=True)
    weight=1/count[idx]; weight/=weight.mean()
    params=[]
    for h in range(30):
        target=np.log((err[:,h]+.002)/(age[:,h]+.02))
        model=Ridge(alpha=.09*len(err)).fit(x[:,h],target,sample_weight=weight)
        params.append(np.r_[model.intercept_,model.coef_])
    return np.array(params,DTYPE)


def base_width(x,age,params):
    return np.maximum(.003,np.exp((x*params[None,:,1:]).sum(2)+params[None,:,0])*(age+.02)).astype(DTYPE)


def predict_block(point,residual,x,phase,state,q,cad,lock,params):
    b=supports(point,residual,lock["spread"])
    g=gate_value(x,lock["gate"]) if lock["distribution"]=="adaptive" else np.full(len(x),lock["gate"]["constant"])
    p=readouts(point,b,g)[lock["point"]]
    wx,age=widths_x(point,p,b,g,phase,state,q,cad)
    return p,b,g,base_width(wx,age,params)


def select():
    reproducible_torch()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=Inputs()
    point=np.load(WORK/"motion_center.npy",mmap_mode="r")
    residual=fit_generator(d,point,dev)
    gate=np.flatnonzero(d.gate & d.grid)
    val=np.flatnonzero(d.val & d.grid)
    x=gate_features(d.phase,d.q,d.state,d.phys)
    candidates=[]
    for gain in (.65,1.,1.4,1.8):
        gen=supports(point[gate],residual[gate],gain)
        _,_,co=geometry(point[gate],gen,d.y[gate])
        model=fit_gate(x[gate],co,d.env[gate])
        vb=supports(point[val],residual[val],gain)
        _,_,vco=geometry(point[val],vb,d.y[val])
        for name,w in (("constant",np.full(len(val),model["constant"])),("adaptive",gate_value(x[val],model))):
            score=(vco[:,0]+vco[:,1]*w+vco[:,2]*w*w)*100
            candidates.append(dict(spread=gain,distribution=name,energy_cm=equal_env(score,d.env[val]),gate=model))
    choice=min(candidates,key=lambda v:v["energy_cm"])
    lock=dict(spread=choice["spread"],distribution=choice["distribution"],gate=choice["gate"])
    pd.DataFrame([{k:v for k,v in row.items() if k!="gate"} for row in candidates]).to_csv(OUT/"distribution_validation.csv",index=False)
    vi=np.flatnonzero(d.val)
    raw_preds={k:[] for k in ("motion","medoid","mean","geometric_median")}
    for group in np.array_split(vi,int(np.ceil(len(vi)/512))):
        b=supports(point[group],residual[group],lock["spread"])
        w=gate_value(x[group],lock["gate"]) if lock["distribution"]=="adaptive" else np.full(len(group),lock["gate"]["constant"])
        for name,v in readouts(point[group],b,w).items():
            raw_preds[name].append(v)
    rows=[]
    for name,chunks in raw_preds.items():
        e=np.linalg.norm(np.concatenate(chunks)-d.y[vi],axis=2)*100
        full=d.mask[vi].all(1); end=d.mask[vi,-1]
        ade=equal_env(e[full].mean(1),d.env[vi][full]); fde=equal_env(e[end,-1],d.env[vi][end])
        rows.append(dict(readout=name,ade=ade,fde=fde,objective=ade+.35*fde))
    lock["point"]=min(rows,key=lambda v:v["objective"])["readout"]
    pd.DataFrame(rows).to_csv(OUT/"readout_validation.csv",index=False)
    # Widths, adaptation and model selection use only official training and validation environments.
    used=np.flatnonzero((d.gate | d.cal) & array("grid") | d.val)
    center=[]; wx=[]; ages=[]
    for group in np.array_split(used,int(np.ceil(len(used)/512))):
        b=supports(point[group],residual[group],lock["spread"])
        w=gate_value(x[group],lock["gate"]) if lock["distribution"]=="adaptive" else np.full(len(group),lock["gate"]["constant"])
        pred=readouts(point[group],b,w)[lock["point"]]
        z,age=widths_x(point[group],pred,b,w,d.phase[group],d.state[group],d.q[group],d.cad[group])
        center.append(pred);wx.append(z);ages.append(age)
    z=np.concatenate(wx); age=np.concatenate(ages)
    err=np.linalg.norm(np.concatenate(center)-d.y[used],axis=2)
    gf=d.grid[used]&d.gate[used]; cf=d.grid[used]&d.cal[used]; vf=d.grid[used]&d.val[used]
    param=fit_width(z[gf],age[gf],err[gf],d.env[used][gf])
    np.save(WORK/"width_parameters.npy",param)
    scale=base_width(z,age,param)
    reference=np.sort(err[gf]/scale[gf],axis=0)
    np.save(WORK/"feedback_reference.npy",reference)
    phase_cfg=json.loads((OUT/"phase_selection.json").read_text(encoding="utf8"))
    with np.load(WORK/(phase_cfg["model"]+"_phase.npz")) as f:
        mode=decide(f["raw"][used],d.phase[used],phase_cfg["rule"])
    feedback=feedback_scales(d.meta.iloc[used].reset_index(drop=True),mode,err,scale,array("ok")[used],reference)
    variants=[]
    for scheme,base in (("static",scale),("feedback",feedback)):
        mx=(err/base).max(1)
        pooled=rank95(mx[cf])
        worst=max([pooled]+[rank95(mx[cf&(d.env[used]==e)]) for e in d.roles["calibration"] if sum(cf&(d.env[used]==e))>=20])
        for rule,factor in (("pooled",pooled),("max_env",worst)):
            bands=base[vf]*factor
            coverage=float((err[vf]<=bands).all(1).mean())
            variants.append(dict(scheme=scheme,quantile=rule,factor=factor,coverage=coverage,radius_cm=float(100*bands.mean())))
    good=[v for v in variants if v["coverage"]>=.95]
    band=min(good,key=lambda v:v["radius_cm"]) if good else max(variants,key=lambda v:(v["coverage"],-v["radius_cm"]))
    pd.DataFrame(variants).to_csv(OUT/"band_validation.csv",index=False)
    lock.update(band=band,band_validation_target_met=bool(good),phase=phase_cfg,
                motion=json.loads((OUT/"motion_selection.json").read_text(encoding="utf8")),
                roles=d.roles,gate_windows=int(gf.sum()),calibration_windows=int(cf.sum()),validation_windows=int(vf.sum()),
                training_source="freshly fitted from raw canonical wrist/head observations and as-of pixels",
                previous_checkpoints_loaded=0,selection_splits=["train","validation"])
    write_json(OUT/"frozen_before_test.json",lock)
    print("Final lock",lock["point"],lock["spread"],lock["distribution"],band,flush=True)


if __name__=="__main__":
    select()
