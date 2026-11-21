"""Train the wrist dynamics experts with nested ablations and independent random initialization."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from qsm_data import OUT, CACHE, SEED, author_transformer, reproducible_torch, write_json

VARIANTS=("A0","A1","B1","B2","B3","B4")
AGE_KNOTS=(0.,.12,.32,.62)


def time_basis(age):
    spans=[(age-l).clip(0,r-l) for l,r in zip(AGE_KNOTS,AGE_KNOTS[1:])]
    spans.append((age-AGE_KNOTS[-1]).clip(0,None))
    return torch.stack(spans,-1) if isinstance(age,torch.Tensor) else np.stack(spans,-1)


class ResidualPath(nn.Module):
    def __init__(self, dims, kind):
        super().__init__()
        self.kind=kind
        self.has_stage=kind in ("B2","B3","B4")
        self.has_quality=kind in ("B3","B4")
        self.short=nn.Sequential(nn.Conv1d(dims,32,3),nn.SiLU(),nn.Conv1d(32,32,3),nn.SiLU())
        if kind=="A1":
            self.short=nn.GRU(dims,32,batch_first=True)
        else:
            self.project=nn.Linear(dims,32)
            self.position=nn.Parameter(torch.randn(1,1,13,32)*.025)
            self.long=author_transformer()(32,1,num_heads=4,dropout=.08)
        size=(32 if kind=="A1" else 64)+5+(2 if self.has_quality else 0)+(32 if kind=="B4" else 0)
        self.fuse=nn.Sequential(nn.Linear(size,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.07))
        self.stage=nn.Linear(64,3) if self.has_stage else None
        self.change=nn.Linear(64,1) if self.has_stage else None
        self.read=nn.Linear(64,12*(3 if self.has_stage else 1))
        self.scale=nn.Linear(64,4)

    def forward(self, short,long,physics,state,risk,visual,basis,base):
        if self.kind=="A1":
            parts=[self.short(short)[1][-1]]
        else:
            tokens=self.project(long)[:,None]+self.position
            parts=[self.short(short.transpose(1,2)).mean(2),self.long(tokens.contiguous())[:,0,-1]]
        parts.extend((physics[:,3:6],state))
        if self.has_quality:
            parts.extend((risk[:,None],(physics[:,6:7]*100)))
        if self.kind=="B4":
            parts.append(visual)
        h=self.fuse(torch.cat(parts,1))
        if self.has_stage:
            stage=self.stage(h)
            change=self.change(h).squeeze(1)
            vel=(stage.softmax(1)[:,:,None,None]*self.read(h).reshape(-1,3,4,3)).sum(1)
        else:
            stage=change=None
            vel=self.read(h).reshape(-1,4,3)
        scale=.003+(basis*(.25*F.softplus(self.scale(h))+.003)[:,None]).sum(-1)
        return base+basis@vel,scale,stage,change


class Data:
    def __init__(self, dev):
        self.dev=dev
        self.spec=json.loads((CACHE/"ready.json").read_text(encoding="utf8"))
        assert self.spec["version"]=="qsm-independent-2" and self.spec["source_weights_loaded"]==0
        self.meta=pd.read_parquet(CACHE/"meta.parquet")
        for k in ("sequence","start","physics","state","prob","vision","y","mask","mode","next_mode","cadence"):
            setattr(self,k,torch.as_tensor(np.load(CACHE/(k+".npy")),device=dev))
        self.short=torch.arange(8,-1,-1,device=dev)
        self.long=torch.arange(60,-1,-5,device=dev)
        self.age=self.cadence[:,None]*torch.arange(1,31,device=dev)
        self.basis=time_basis(self.age)

    def args(self, ids, kind):
        idx=torch.maximum(ids[:,None]-self.short,self.start[ids,None])
        ixlong=torch.maximum(ids[:,None]-self.long,self.start[ids,None])
        p=self.physics[ids,int(kind in ("B3","B4"))]
        tau=self.spec["tau"]
        base=p[:,:3,None].transpose(1,2)+p[:,None,3:6]*(-tau*torch.expm1(-self.age[ids]/tau))[:,:,None]
        return self.sequence[idx],self.sequence[ixlong],p,self.state[ids],self.prob[ids],self.vision[ids],self.basis[ids],base


def fit(data,kind,train):
    torch.manual_seed(SEED+17)
    rng=np.random.default_rng(SEED+17)
    net=ResidualPath(data.sequence.shape[1],kind).to(data.dev)
    opt=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.004)
    epochs=19
    lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00018)
    horizon=1/(.025+.16*torch.arange(1,31,device=data.dev)/30)
    for _ in range(epochs):
        for ix in np.array_split(rng.choice(train,6912,replace=False),27):
            ids=torch.as_tensor(ix,device=data.dev)
            pred,scale,stage,change=net(*data.args(ids,kind))
            err=((pred-data.y[ids]).square().sum(2)+1e-7).sqrt()
            valid=data.mask[ids]
            loss=.52*(err*horizon)[valid].mean()+.48*err[:,-1].mean()/.185
            loss+=.10*(err.detach()/scale+scale.log())[valid].mean()
            if net.has_stage:
                keep=valid[:,9]
                label=data.next_mode[ids][keep].long()
                transition=(label!=data.mode[ids][keep]).float()
                loss+=.13*F.cross_entropy(stage[keep],label)+.07*F.binary_cross_entropy_with_logits(change[keep],transition)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
        lr.step()
    torch.save(net.state_dict(),OUT/(kind+"_scratch.pt"))
    return net.eval()


@torch.inference_mode()
def issue(data,kind,net,ids):
    path=[]; scale=[]; stage=[]; change=[]
    for sub in np.array_split(ids,max(1,int(np.ceil(len(ids)/384)))):
        ix=torch.as_tensor(sub,device=data.dev)
        args=data.args(ix,kind)
        if net is None:
            p=args[-1]
            s=.003+.17*data.age[ix]
            pi=F.one_hot(data.mode[ix].long(),3).float()
            c=torch.zeros(len(ix),device=data.dev)
        else:
            p,s,pi,c=net(*args)
            pi=pi.softmax(1) if pi is not None else F.one_hot(data.mode[ix].long(),3).float()
            c=c.sigmoid() if c is not None else torch.zeros(len(ix),device=data.dev)
        for buf,tensor in zip((path,scale,stage,change),(p,s,pi,c)):
            buf.append(tensor.cpu().numpy())
    np.savez(CACHE/(kind+"_issues.npz"),ids=ids,path=np.concatenate(path),scale=np.concatenate(scale),
             stage=np.concatenate(stage),change=np.concatenate(change))


def run():
    reproducible_torch()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=Data(dev)
    tr=data.meta.split.eq("train").to_numpy()
    ids=np.flatnonzero(~tr | np.load(CACHE/"grid.npy"))
    train=np.flatnonzero(data.meta.env.isin(data.spec["environments"]["fit"]) & data.meta.score)
    write_json(OUT/"q1_protocol.json",dict(scratch=True,initial_weights="random, no outside checkpoints",
        version=data.spec["version"],tau=data.spec["tau"],quality_kalman=data.spec["quality_kalman"],
        code=["GeoX-Lab/LSTTN 68185aa8 TransformerLayers","IRMVLab/UniHand 7f3edd59 GaussianDiffusion"],
        training_roles=data.spec["environments"],epochs=19,minibatches_per_epoch=27,seed=SEED+17,
        issue="All 30 3D wrist displacements fixed at origin, time unit seconds, no future frames or task text",
        modes=["dwell","turn","translate"],mode_target_frames=10,point="geometric medoid on the nine trajectory supports for B5",
        gate="Energy score on non-overlapping paths of three gate-training environments",
        calibration="Three further official training environments; all candidates frozen before validation and retrospective test",
        comparison="all official wrist origins; endpoint, path, energy score and 95% complete-window coverage; test retrospective"))
    for kind in VARIANTS:
        net=None if kind=="A0" else fit(data,kind,train)
        issue(data,kind,net,ids)
        print(kind,"fitted and issued",len(ids),flush=True)


if __name__=="__main__":
    run()
