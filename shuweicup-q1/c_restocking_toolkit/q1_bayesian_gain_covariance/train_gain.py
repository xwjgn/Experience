"""Train the constant, deterministic, and stochastic gain controls from fresh parameters."""
import copy
import json
import numpy as np
import torch
from torch import nn

from gain_records import OUT,SEEDS,dataset,digest,isolate,partition,save
from neural_gain import GainForecaster,INFERENCE_DRAWS,KINDS,loss_for

BLOCK=96


def configure():
    torch.set_num_threads(3)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensors(inp,stats,device):
    norm=((inp.context-stats["mean"])/stats["sd"]).clip(-7,7).astype("float32")
    return tuple(torch.tensor(a,device=device) for a in
        (inp.history,inp.delta,inp.available,inp.flag,norm,inp.cadence))


def seed_file(kind,seed):
    return OUT/f"{kind}_{seed}_fresh.pt"


@torch.inference_mode()
def posterior(net,args,seed,mean_only=False):
    gen=torch.Generator(device=args[0].device).manual_seed(seed+30191)
    sample=net.sample_gain(INFERENCE_DRAWS,gen,mean_only)
    collected={}
    n=len(args[0])
    for start in range(0,n,BLOCK):
        end=min(n,start+BLOCK)
        ids=torch.arange(start,start+BLOCK,device=args[0].device).clamp(max=end-1)
        raw=net(*(a[ids] for a in args),sample)
        mu=raw["mean"].mean(0)
        dx=raw["mean"]-mu[None]
        epi=(dx.unsqueeze(-1)*dx.unsqueeze(-2)).mean(0)
        state=torch.cat((raw["pos"],raw["vel"]),-1)
        sm=state.mean(0)
        dev=state-sm[None]
        pp,pv,vv=[torch.diag_embed(raw[k].mean(0)) for k in ("pp","pv","vv")]
        within=torch.cat((torch.cat((pp,pv),-1),torch.cat((pv,vv),-1)),-2)
        sc=within+(dev.unsqueeze(-1)*dev.unsqueeze(-2)).mean(0)
        gains=torch.stack((raw["kp"],raw["kv"]),dim=2)
        arrays=dict(mean=mu,conditional=raw["cov"].mean(0),weight_cov=epi,state=sm,
                    state_cov=sc,gain=gains.mean(0),gain_weight_sd=gains.std(0,correction=0))
        for name,v in arrays.items():
            collected.setdefault(name,[]).append(v[:end-start].cpu().numpy())
    return {name:np.concatenate(v) for name,v in collected.items()}


def load_stats():
    with np.load(OUT/"training_normalization.npz") as f:
        return {k:f[k] for k in f}


def objective_stats(pred,target,turn,env):
    error=100*np.linalg.norm(pred-target,axis=2).mean(1)
    pooled=float(error.mean())
    turns=float(error[turn].mean())
    macro=float(np.mean([error[env==e].mean() for e in np.unique(env)]))
    return .20*pooled+.45*turns+.35*macro,dict(ade=pooled,turn=turns,macro=macro)


def fit_one(kind,seed,fit,gate,fit_args,gate_args,stats,dev):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    net=GainForecaster(len(stats["mean"]),kind).to(dev)
    y_head=np.einsum("nhi,nij->nhj",fit.truth,fit.inputs.axes)
    target=torch.tensor(y_head,dtype=torch.float32,device=dev)
    turns=torch.tensor(fit.meta.turn.to_numpy(),device=dev,dtype=torch.float32)
    counts=fit.meta.groupby("env").turn.transform("size").to_numpy(float)
    weights=counts**-.5
    optimizer=torch.optim.AdamW(net.parameters(),lr=.00125,weight_decay=.003)
    lr=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,18,eta_min=.00012)
    best=np.inf
    record=[]
    for epoch in range(1,19):
        net.train()
        order=rng.choice(len(target),3840,replace=False,p=weights/weights.sum())
        for start in range(0,len(order),160):
            ix=torch.tensor(order[start:start+160],device=dev)
            draws=net.sample_gain(2)
            out=net(*(a[ix] for a in fit_args),draws)
            loss=loss_for(net,out,target[ix],fit_args[-1][ix],turns[ix],len(target))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss in {kind}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            optimizer.step()
        lr.step()
        net.eval()
        p=posterior(net,gate_args,seed)
        m=np.einsum("nhi,nji->nhj",p["mean"],gate.inputs.axes)
        score,metrics=objective_stats(m,gate.truth,gate.meta.turn.to_numpy(),gate.meta.env.to_numpy())
        record.append(dict(epoch=epoch,**metrics))
        if score<best:
            best,chosen=score,epoch
            state=copy.deepcopy(net.state_dict())
        if epoch in (1,9,18):
            print(kind,seed,epoch,"gate turn ADE",round(metrics["turn"],3),flush=True)
        if epoch>=12 and epoch-chosen>=6:
            break
    name=seed_file(kind,seed)
    torch.save(state,name)
    return dict(seed=seed,epoch=chosen,history=record,sha256=digest(name),fresh=True)


def fit_networks(training=None):
    OUT.mkdir(parents=True,exist_ok=True)
    dev=configure()
    if training is None:
        training=dataset("train")
    fit,gate=partition(training,"fit",False),partition(training,"gate")
    x=fit.inputs.context
    stats={"mean":x.mean(0,dtype=float).astype("float32"),
           "sd":x.std(0,dtype=float).clip(.0015).astype("float32")}
    np.savez(OUT/"training_normalization.npz",**stats)
    a,b=tensors(fit.inputs,stats,dev),tensors(gate.inputs,stats,dev)
    experiments={}
    for kind in KINDS:
        experiments[kind]=[fit_one(kind,s,fit,gate,a,b,stats,dev) for s in SEEDS]
        save(OUT/"fit_progress.json",experiments)
    save(OUT/"fresh_fit.json",dict(models=experiments,fit=len(fit.meta),gate=len(gate.meta),
         normalization_sha256=digest(OUT/"training_normalization.npz"),prior_parameter_reads=0,
         inference_per_seed=INFERENCE_DRAWS,bayesian_posterior="Variational Gaussian on the gain output-layer weights and biases; weights shared across the entire observation prefix per draw."))


def combine_members(items,axes):
    mean=np.mean([f["mean"] for f in items],axis=0)
    diff=np.stack([f["mean"]-mean for f in items])
    seed_cov=np.mean(diff[...,None]*diff[...,None,:],axis=0)
    conditional=np.mean([f["conditional"] for f in items],axis=0)
    weight=np.mean([f["weight_cov"] for f in items],axis=0)
    st=np.mean([f["state"] for f in items],axis=0)
    ds=np.stack([f["state"]-st for f in items])
    sc=np.mean([f["state_cov"] for f in items],axis=0)+np.mean(ds[...,None]*ds[...,None,:],axis=0)
    # The path tensors are returned in the common origin-head axes. State/gain diagnostics use heading axes.
    a=axes[:,None]
    def rotate(c):
        return a@c@a.swapaxes(-1,-2)
    return dict(mean=np.einsum("nhi,nji->nhj",mean,axes),cov=rotate(conditional+weight+seed_cov),
                conditional=rotate(conditional),weight_cov=rotate(weight),seed_cov=rotate(seed_cov),
                state=st,state_cov=sc,gain=np.mean([f["gain"] for f in items],axis=0),
                gain_weight_sd=np.mean([f["gain_weight_sd"] for f in items],axis=0))


class ModelBank:
    def __init__(self,kinds=KINDS):
        self.dev=configure()
        self.stats=load_stats()
        manifest=json.loads((OUT/"fresh_fit.json").read_text(encoding="utf8"))
        if digest(OUT/"training_normalization.npz")!=manifest["normalization_sha256"]:
            raise ValueError("Training normalization changed")
        self.nets={}
        for kind in kinds:
            members=[]
            for r in manifest["models"][kind]:
                path=seed_file(kind,r["seed"])
                if digest(path)!=r["sha256"]:
                    raise ValueError("Fresh gain checkpoint changed")
                net=GainForecaster(len(self.stats["mean"]),kind).to(self.dev)
                net.load_state_dict(torch.load(path,map_location=self.dev,weights_only=True))
                members.append((r["seed"],net.eval()))
            self.nets[kind]=members

    def issue(self,inp,kind="bayesian",mean_only=False):
        args=tensors(inp,self.stats,self.dev)
        return combine_members([posterior(net,args,seed,mean_only) for seed,net in self.nets[kind]],inp.axes)


if __name__=="__main__":
    isolate()
    fit_networks()
