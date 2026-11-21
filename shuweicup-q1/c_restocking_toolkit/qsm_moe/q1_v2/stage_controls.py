"""2 × 2 diagnostic on the v1 architecture, changing only phase class weights and the direct mixture-gradient path."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from settings import OUT, WORK, array, equal_env
import qsm_train
from qsm_data import reproducible_torch
from phase import scores


class ControlledPath(qsm_train.ResidualPath):
    def __init__(self,dims,detach_gate):
        super().__init__(dims,"B4")
        self.detach_gate=detach_gate

    def forward(self,short,long,physics,state,risk,visual,basis,base):
        tokens=self.project(long)[:,None]+self.position
        short=self.short(short.transpose(1,2)).mean(2)
        long=self.long(tokens.contiguous())[:,0,-1]
        h=self.fuse(torch.cat((short,long,physics[:,3:6],state,risk[:,None],physics[:,6:7]*100,visual),1))
        stage=self.stage(h)
        weights=stage.softmax(1)
        if self.detach_gate:
            weights=weights.detach()
        velocity=(weights[:,:,None,None]*self.read(h).reshape(-1,3,4,3)).sum(1)
        scale=.003+(basis*(.25*F.softplus(self.scale(h))+.003)[:,None]).sum(-1)
        return base+basis@velocity,scale,stage,self.change(h).squeeze(1)


def experiment(data,train,seed,balanced,detach):
    torch.manual_seed(seed+17)
    rng=np.random.default_rng(seed+17)
    net=ControlledPath(data.sequence.shape[1],detach).to(data.dev)
    train_labels=data.next_mode[train][data.mask[train,9]].long()
    freq=torch.bincount(train_labels,minlength=3).float()
    weight=(freq.sum()/(3*freq)) if balanced else None
    opt=torch.optim.AdamW(net.parameters(),lr=.0013,weight_decay=.004)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,19,eta_min=.00018)
    horizon=1/(.025+.16*torch.arange(1,31,device=data.dev)/30)
    for _ in range(19):
        for ix in np.array_split(rng.choice(train,6912,replace=False),27):
            ids=torch.as_tensor(ix,device=data.dev)
            pred,width,stage,change=net(*data.args(ids,"B4"))
            err=((pred-data.y[ids]).square().sum(2)+1e-7).sqrt()
            valid=data.mask[ids]
            loss=.52*(err*horizon)[valid].mean()+.48*err[:,-1].mean()/.185
            loss+=.10*(err.detach()/width+width.log())[valid].mean()
            keep=valid[:,9]
            label=data.next_mode[ids][keep].long()
            transitioned=(label!=data.mode[ids][keep]).float()
            loss+=.13*F.cross_entropy(stage[keep],label,weight=weight)+.07*F.binary_cross_entropy_with_logits(change[keep],transitioned)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
        sched.step()
    return net.eval()


def run():
    reproducible_torch()
    qsm_train.CACHE=WORK/"cache"
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=qsm_train.Data(dev)
    meta=data.meta
    train=np.flatnonzero(meta.env.isin(data.spec["environments"]["fit"])&meta.score)
    ids=np.flatnonzero(array("grid")&array("mask").all(1))
    roles=meta.split.to_numpy()[ids].astype(object)
    for key,envs in data.spec["environments"].items():
        roles[np.isin(meta.env.to_numpy()[ids],envs)]=key
    ys=array("next_mode")[ids]
    env=meta.env.to_numpy()[ids]
    from visual_controls import issue
    results=[]
    for seed in (9317,9431):
        for balance,detach in ((False,False),(True,False),(False,True),(True,True)):
            net=experiment(data,train,seed,balance,detach)
            path,p=issue(net,data,ids)
            c=p.argmax(1)
            err=np.linalg.norm(path-array("y")[ids],axis=2).mean(1)*100
            for role in ("fit","gate","validation","test"):
                take=roles==role
                results.append(dict(seed=seed,balanced=balance,detached=detach,role=role,
                      path_cm=equal_env(err[take],env[take]),**scores(ys[take],p[take,1],c[take],env[take])))
        print("Direct gradient and class-frequency controls completed",seed,flush=True)
    pd.DataFrame(results).to_csv(OUT/"turn_loss_interventions.csv",index=False)


if __name__=="__main__":
    run()
