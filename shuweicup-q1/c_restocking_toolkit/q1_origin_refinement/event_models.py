"""Cold fits for a phase-conditioned wrist warning, with optional future wrist and head supervision."""
import copy

import numpy as np
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F

from wrist_history import OUT, SEEDS


KINDS = ("single","wrist_aux","joint_aux")


class TurnNetwork(nn.Module):
    def __init__(self,nseq,ntab,nmovement):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(nseq,36),nn.LayerNorm(36),nn.SiLU())
        self.recurrent = nn.GRU(36,48,batch_first=True)
        self.summary = nn.Sequential(nn.Linear(ntab,56),nn.LayerNorm(56),nn.SiLU())
        self.context = nn.Sequential(nn.Linear(152,72),nn.LayerNorm(72),nn.SiLU(),nn.Dropout(.14),
                                     nn.Linear(72,48),nn.SiLU())
        self.common = nn.Linear(48,3)
        self.specialists = nn.Linear(48,9)
        self.future = nn.Linear(48,nmovement)

    def forward(self,seq,tab,phase):
        time,_ = self.recurrent(self.project(seq))
        ctx = self.context(torch.cat([time[:,-1],time.mean(1),self.summary(tab)],dim=1))
        experts = self.specialists(ctx).reshape(-1,3,3)
        logits = self.common(ctx)+.5*experts[torch.arange(len(ctx),device=phase.device),phase]
        return logits,self.future(ctx)


def configure():
    torch.set_num_threads(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def scale(a, axes):
    return np.mean(a,axis=axes,dtype=float).astype("float32"),np.maximum(
        np.std(a,axis=axes,dtype=float),.002).astype("float32")


def as_tensors(seq,tab,stage,stats,device):
    def make(a,prefix):
        mu,sd = stats[prefix]
        return torch.tensor(np.clip((a-mu)/sd,-7,7),device=device,dtype=torch.float32)
    return make(seq,"seq"),make(np.asarray(tab),"tab"),torch.tensor(stage,dtype=torch.long,device=device)


def predict_net(net,tensors):
    net.eval()
    with torch.inference_mode():
        portions = []
        batch_size = 128
        for start in range(0,len(tensors[0]),batch_size):
            count = min(len(tensors[0])-start,batch_size)
            ids = torch.arange(start,start+batch_size,device=tensors[0].device).clamp(max=start+count-1)
            args = (v[ids] for v in tensors)
            portions.append(net(*args)[0][:count,1].sigmoid().cpu().numpy())
    return np.concatenate(portions)


def rank_stats(y,score,fraction=.24):
    k = max(1,min(len(y),round(len(y)*fraction)))
    order = np.argsort(-np.asarray(score),kind="stable")
    return .5*average_precision_score(y,score)+.5*np.mean(np.asarray(y)[order[:k]])


def cold_fit(fit,gate,stats,kind,seed,device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = as_tensors(fit.seq,fit.features,fit.meta.stage.to_numpy(),stats,device)
    check = as_tensors(gate.seq,gate.features,gate.meta.stage.to_numpy(),stats,device)
    dims = (fit.seq.shape[-1],fit.features.shape[-1],fit.movement.shape[-1])
    model = TurnNetwork(*dims).to(device)
    evt = torch.tensor(fit.events,device=device)
    mu,sd = stats["movement"]
    targets = torch.tensor(((fit.movement-mu)/sd).clip(-6,6),device=device)
    counts = fit.meta.groupby("env").turn.transform("size").to_numpy()
    weight = counts.astype(float)**-.5
    weight /= weight.sum()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0013,weight_decay=.009)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,18,eta_min=.00015)
    pw = torch.tensor([2.3,2.3,2.3],device=device)
    best = -np.inf
    for epoch in range(1,19):
        model.train()
        order = rng.choice(len(fit.meta),min(9600,len(fit.meta)),p=weight,replace=False)
        for start in range(0,len(order),192):
            ids = torch.as_tensor(order[start:start+192],dtype=torch.long,device=device)
            pred,motion = model(*(x[ids] for x in train))
            loss = F.binary_cross_entropy_with_logits(pred[:,1],evt[ids,1],pos_weight=pw[1])
            if kind != "single":
                loss = loss+.25*F.binary_cross_entropy_with_logits(pred[:,[0,2]],evt[ids][:,[0,2]],pos_weight=pw[[0,2]])
                loss = loss+.25*F.smooth_l1_loss(motion[:,:9],targets[ids,:9])
            if kind == "joint_aux":
                loss = loss+.15*F.smooth_l1_loss(motion[:,9:],targets[ids,9:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
        scheduler.step()
        score = rank_stats(gate.meta.turn.to_numpy(),predict_net(model,check))
        if score > best:
            best, saved, best_epoch = score,copy.deepcopy(model.state_dict()),epoch
    path = OUT/f"{kind}_{seed}_fresh.pt"
    torch.save(saved,path)
    print(kind,seed,"training-only selection",best_epoch,round(float(best),3),flush=True)
    return dict(file=path.name,selected_epoch=best_epoch,gate_score=float(best))


def train_models(train,roles):
    configure()
    fit = train.take(train.meta.env.isin(roles["fit"]))
    def fixed_grid(role):
        return train.take(train.meta.env.isin(roles[role]) & ((train.meta.origin-10)%30==0))
    gate,cal = fixed_grid("gate"),fixed_grid("cal")
    stats = dict(seq=scale(fit.seq,(0,1)),tab=scale(fit.features.to_numpy(),0),movement=scale(fit.movement,0))
    params = dict(max_iter=120,max_leaf_nodes=12,min_samples_leaf=90,early_stopping=False,
                  max_bins=128,learning_rate=.06,l2_regularization=4.,random_state=91431)
    counts = fit.meta.groupby("env").turn.transform("size").to_numpy()
    weights = counts.astype(float)**-.5
    weights /= weights.mean()
    tree = HistGradientBoostingClassifier(**params).fit(fit.features,fit.meta.turn,sample_weight=weights)
    risk = [HistGradientBoostingRegressor(loss="quantile",quantile=.85,**params).fit(
            fit.features,fit.proxy[:,j],sample_weight=weights) for j in range(fit.proxy.shape[1])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fits = {kind:[cold_fit(fit,gate,stats,kind,seed,device) for seed in SEEDS] for kind in KINDS}
    pack = dict(tree=tree,risk=risk,stats=stats,fits=fits,features=list(fit.features),
                dims=(fit.seq.shape[-1],fit.features.shape[1],fit.movement.shape[-1]))
    scores = Scorer(pack)
    raw, _ = scores.outputs(gate.seq,gate.features,gate.meta.stage.to_numpy(),calibrated=False)
    scalers = {}
    for name,score in raw.items():
        x = logit(score.clip(.0001,.9999)).reshape(-1,1)
        lr = LogisticRegression(C=.5).fit(x,gate.meta.turn)
        scalers[name] = (float(max(.05,lr.coef_[0,0])),float(lr.intercept_[0]))
    pack["scalers"] = scalers
    return pack,cal


class Scorer:
    def __init__(self,pack):
        configure()
        self.pack = pack
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nets = {}
        for kind,entries in pack["fits"].items():
            nets = []
            for entry in entries:
                m = TurnNetwork(*pack["dims"]).to(self.device)
                m.load_state_dict(torch.load(OUT/entry["file"],map_location=self.device,weights_only=True))
                nets.append(m.eval())
            self.nets[kind] = nets

    def outputs(self,sequence,features,phase,calibrated=True):
        features = features[self.pack["features"]]
        tensors = as_tensors(sequence,features,phase,self.pack["stats"],self.device)
        out = {"tree":self.pack["tree"].predict_proba(features)[:,1]}
        for kind,models in self.nets.items():
            out[kind] = np.mean([predict_net(m,tensors) for m in models],axis=0)
        risk = np.exp(np.stack([m.predict(features) for m in self.pack["risk"]],axis=1))
        if calibrated:
            for key,(slope,intercept) in self.pack["scalers"].items():
                out[key] = expit(slope*logit(out[key].clip(.0001,.9999))+intercept)
            out["ensemble"] = (out["tree"]+out["wrist_aux"]+out["joint_aux"])/3
        return out,risk
