import argparse
import copy
import numpy as np
import pandas as pd
import torch

from paths_q1 import DATA,OUT,DEVICES,read_json,save_json
from lsttn_model import WristLSTTN,SequenceInputs,VARIANTS,deterministic,loss_parts


class Study:
    def __init__(self):
        self.meta = pd.read_parquet(DATA/"keys.parquet")
        self.y = np.load(DATA/"y.npy",mmap_mode="r")
        self.valid = np.load(DATA/"valid.npy",mmap_mode="r")
        self.mode = np.load(DATA/"mode.npy",mmap_mode="r")
        self.ok = np.load(DATA/"ok.npy",mmap_mode="r")
        self.roles = read_json(OUT/"data_protocol.json")["roles"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inputs = SequenceInputs.recorded(self.device)
        self.dims = (self.inputs.values["x"].shape[1],self.inputs.values["vec"].shape[1])

    def rows(self, role, stride=30, complete=True):
        env = self.meta.env.isin(self.roles[role]) if role in self.roles else self.meta.split.eq(role)
        o = self.meta.origin.to_numpy()
        keep = env.to_numpy() & (o>=10) & ((o-10)%stride==0)
        if complete:
            keep &= self.valid.all(1)
        return np.flatnonzero(keep)


@torch.inference_mode()
def predict(nets,inputs,ids,temp=1.,batch=192):
    acc = []
    for at in range(0,len(ids),batch):
        current = ids[at:at+batch]
        count = len(current)
        # GPU convolution kernels may change with batch size. Evaluate a fixed shape
        # so a single online prefix uses the same arithmetic as a recorded panel.
        padded = np.pad(current,(0,batch-count),mode="edge") if count<batch else current
        args = inputs.batch(padded)
        probs,w,means,scales = [],[],[],[]
        for net in nets:
            logits,mu,sd = net(*args)
            p = torch.softmax(logits/temp,dim=1)
            probs.append(p)
            w.append((p if mu.shape[1]>1 else torch.ones_like(p[:,:1]))/len(nets))
            means.append(mu)
            scales.append(sd)
        acc.append([a[:count].cpu().numpy() for a in (torch.stack(probs).mean(0),torch.cat(w,1),torch.cat(means,1),torch.cat(scales,1))])
    return tuple(np.concatenate(v) for v in zip(*acc))


def summarize(w,means,scales,kind):
    center = np.sum(w[:,:,None,None]*means,axis=1)
    if kind=="smooth_median":
        for _ in range(10):
            distance = np.sqrt(np.square(means-center[:,None]).sum(-1)+2*np.square(scales)).clip(.0002)
            importance = w[:,:,None]/distance
            center = (importance[:,:,:,None]*means).sum(1)/importance.sum(1)[:,:,None]
    dispersion = (w[:,:,None]*(3*scales**2+np.square(means-center[:,None]).sum(-1))).sum(1)
    return center,np.sqrt(np.maximum(dispersion,1e-8))


def point_stats(y,center,mode,env):
    err = np.linalg.norm(y-center,axis=-1)*100
    ade = err.mean(1)
    groups = np.unique(env)
    return dict(ade=float(ade.mean()),macro_ade=float(np.mean([ade[env==e].mean() for e in groups])),
                turn=float(ade[mode==1].mean()),fde=float(err[:,-1].mean()),
                macro_fde=float(np.mean([err[env==e,-1].mean() for e in groups])))


def criterion(m):
    return .45*m["turn"]+.4*m["macro_ade"]+.15*m["macro_fde"]


def checkpoint(variant,seed):
    return OUT/f"{variant}_{seed}.pt"


def load_nets(variant,seeds,dims,device):
    nets = []
    for seed in seeds:
        m = WristLSTTN(*dims,variant).to(device)
        m.load_state_dict(torch.load(checkpoint(variant,seed),map_location=device,weights_only=True))
        nets.append(m.eval())
    return nets


def train_one(d,variant,seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = WristLSTTN(*d.dims,variant).to(d.device)
    fit = d.rows("fit",stride=1)
    gate = d.rows("gate")
    _,ix,counts = np.unique(d.meta.env.to_numpy()[fit],return_inverse=True,return_counts=True)
    probability = 1/np.sqrt(counts[ix])
    probability /= probability.sum()
    truth = torch.tensor(np.asarray(d.y[fit]),device=d.device)
    modes = torch.tensor(np.asarray(d.mode[fit]),dtype=torch.long,device=d.device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=.0012,weight_decay=.006)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,26,eta_min=.00009)
    best_value = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1,27):
        net.train()
        picked = rng.choice(len(fit),size=min(4096,len(fit)),replace=False,p=probability)
        for local_idx in np.array_split(picked,26):
            ids = fit[local_idx]
            batch = d.inputs.batch(ids)
            idx = torch.as_tensor(local_idx,device=d.device)
            loss = loss_parts(net(*batch),truth[idx],modes[idx],batch[-1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            optimizer.step()
        scheduler.step()
        net.eval()
        _,w,mu,sd = predict([net],d.inputs,gate)
        center,_ = summarize(w,mu,sd,"smooth_median")
        stats = point_stats(d.y[gate],center,d.mode[gate],d.meta.env.to_numpy()[gate])
        value = criterion(stats)
        history.append(dict(epoch=epoch,objective=value,**stats))
        if value<best_value:
            best_epoch,best_value = epoch,value
            best = copy.deepcopy(net.state_dict())
        if epoch in (1,8,16,26):
            print(variant,seed,"epoch",epoch,"gate ADE/turn",round(stats["ade"],2),round(stats["turn"],2),flush=True)
        if epoch>=16 and epoch-best_epoch>=9:
            break
    torch.save(best,checkpoint(variant,seed))
    save_json(OUT/f"{variant}_{seed}_training.json",dict(seed=seed,variant=variant,parameters=sum(p.numel() for p in net.parameters()),
              initialized="random",checkpoint_role="train.gate",best_epoch=best_epoch,fit_rows=len(fit),epochs=history))


def fit_all(variants=VARIANTS):
    deterministic()
    d = Study()
    for variant in variants:
        seeds = DEVICES if variant=="quality_stage" else DEVICES[:1]
        for seed in seeds:
            train_one(d,variant,seed)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variants",nargs="*",default=list(VARIANTS))
    args = parser.parse_args()
    fit_all(args.variants)
