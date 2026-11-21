"""Fit every candidate from random initialization; checkpoints from other studies are never opened."""
import argparse
import copy
import hashlib
import numpy as np
import pandas as pd
import torch

from study_paths import DATA,OUT,SEEDS,read,save
from recurrent_network import BATCH,MODELS,HandLSTM,Histories,deterministic,objective


def matrix(name):
    return np.load(DATA/(name+".npy"),mmap_mode="r")


class Study:
    def __init__(self):
        self.meta = pd.read_parquet(DATA/"windows.parquet")
        self.roles = read(OUT/"release_protocol.json")["roles"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = {k:matrix(k) for k in ("nodes","global_x","node_mask","qinfo","basis","time","cad","start")}
        self.dims = [values["nodes"].shape[-1],values["global_x"].shape[-1],values["basis"].shape[1]]
        self.features = Histories(values,self.device)
        self.eligible,self.y,self.stage,self.flag = [matrix(k) for k in ("eligible","y","stage","flag")]
        self.flag_rate = float(self.flag[self.meta.env.isin(self.roles["fit"]).to_numpy()].mean())

    def rows(self,role,stride=30,eligible=True,start=10):
        env = self.meta.env.isin(self.roles[role]) if role in self.roles else self.meta.split.eq(role)
        o = self.meta.origin.to_numpy()
        keep = env.to_numpy() & (o>=start) & ((o-start)%stride==0)
        if eligible:
            keep &= self.eligible.all(1)
        return np.flatnonzero(keep)


def weights_file(kind,seed):
    return OUT/f"{kind}_new_{seed}.pt"


def load_own(kind,seeds,dims,rate,dev):
    nets = []
    for seed in seeds:
        m = HandLSTM(dims,kind,rate).to(dev)
        m.load_state_dict(torch.load(weights_file(kind,seed),map_location=dev,weights_only=True))
        nets.append(m.eval())
    return nets


@torch.inference_mode()
def issue_components(nets,features,rows,temperature=1.,quality_only=False):
    batches = []
    for start in range(0,len(rows),BATCH):
        ids = rows[start:start+BATCH]
        padded = np.pad(ids,(0,BATCH-len(ids)),mode="edge")
        args = features.get(padded)
        qs,st,weights,means,scale = [],[],[],[],[]
        for net in nets:
            if quality_only:
                _,q,_ = net.encode_quality(args[0],args[1],args[2],args[4])
                qs.append(q[torch.arange(len(padded),device=q.device),args[6]])
            else:
                logit,mu,sd,q = net(*args)
                p = (logit/temperature).softmax(1)
                st.append(p); weights.append(p/len(nets)); means.append(mu); scale.append(sd); qs.append(q)
        q = torch.stack(qs).mean(0)
        parts = [q] if quality_only else [torch.stack(st).mean(0),torch.cat(weights,1),torch.cat(means,1),torch.cat(scale,1),q]
        batches.append([v[:len(ids)].cpu().numpy() for v in parts])
    return tuple(np.concatenate(part) for part in zip(*batches))


def center_and_scale(weight,means,sd,method):
    p = (weight[:,:,None,None]*means).sum(1)
    if method=="median":
        for _ in range(9):
            dist = np.sqrt(np.square(means-p[:,None]).sum(-1)+2*np.square(sd)).clip(.0001)
            w = weight[:,:,None]/dist
            p = (w[:,:,:,None]*means).sum(1)/w.sum(1)[:,:,None]
    var = (weight[:,:,None]*(3*sd**2+np.square(means-p[:,None]).sum(-1))).sum(1)
    return p,np.sqrt(var.clip(.000001))


def point_metrics(y,center,t,env):
    err = 100*np.linalg.norm(y-center,axis=-1)
    pooled = err.mean(1)
    return dict(ade=float(pooled.mean()),turn=float(pooled[t==1].mean()),fde=float(err[:,-1].mean()),
                macro=float(np.mean([pooled[env==e].mean() for e in np.unique(env)])))


def performance(m):
    return .49*m["turn"]+.38*m["macro"]+.13*m["fde"]


def initial_fingerprint(net):
    h = hashlib.sha256()
    for v in net.state_dict().values():
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit_one(d,kind,seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = HandLSTM(d.dims,kind,d.flag_rate).to(d.device)
    signature = initial_fingerprint(net)
    train = d.rows("fit",stride=1)
    all_q = d.rows("fit",stride=1,eligible=False,start=0)
    positives,negatives = all_q[d.flag[all_q]],all_q[~d.flag[all_q]]
    _,idx,counts = np.unique(d.meta.env.to_numpy()[train],return_inverse=True,return_counts=True)
    weights = 1/np.sqrt(counts[idx])
    weights = weights/weights.sum()
    targets = torch.tensor(d.y[train],device=d.device)
    stages = torch.tensor(d.stage[train],device=d.device,dtype=torch.long)
    gate = d.rows("gate")
    optimizer = torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.004)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,26,eta_min=.00012)
    best,epoch_best = np.inf,0
    logs = []
    for epoch in range(1,27):
        net.train()
        sample = rng.choice(len(train),size=3072,replace=False,p=weights)
        for index in np.array_split(sample,24):
            qids = np.r_[rng.choice(positives,16),rng.choice(negatives,16)]
            qtruth = torch.as_tensor(d.flag[qids].astype("float32"),device=d.device)
            batch = d.features.get(np.r_[train[index],qids])
            choose = torch.as_tensor(index,device=d.device)
            loss = objective(net(*batch),targets[choose],stages[choose],batch[-1],qtruth)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            optimizer.step()
        scheduler.step()
        net.eval()
        _,w,mu,sd,_ = issue_components([net],d.features,gate)
        center,_ = center_and_scale(w,mu,sd,"median")
        stats = point_metrics(d.y[gate],center,d.stage[gate],d.meta.env.to_numpy()[gate])
        value = performance(stats)
        logs.append(dict(epoch=epoch,score=value,**stats))
        if value<best:
            best,epoch_best = value,epoch
            best_state = copy.deepcopy(net.state_dict())
        if epoch in (1,12,26):
            print(kind,seed,epoch,round(stats["ade"],2),round(stats["turn"],2),flush=True)
        if epoch>=18 and epoch-epoch_best>=10:
            break
    torch.save(best_state,weights_file(kind,seed))
    save(OUT/f"{kind}_{seed}_fit.json",dict(kind=kind,seed=seed,prior_weights_read=0,
            initial_parameter_hash=signature,parameter_count=sum(p.numel() for p in net.parameters()),
            fit_paths=len(train),fit_quality_frames=len(all_q),fit_quality_positive=len(positives),
            selected_epoch=epoch_best,selection_role="three official training environments",history=logs))


def train(kinds=MODELS):
    deterministic()
    d = Study()
    for kind in kinds:
        for seed in SEEDS:
            fit_one(d,kind,seed)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("models",nargs="*",default=list(MODELS))
    train(p.parse_args().models)
