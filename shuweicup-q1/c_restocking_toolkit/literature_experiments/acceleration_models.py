"""Predict the 30-frame speed regime and an endpoint correction, using origin-visible inputs."""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from torch import nn
from torch.nn import functional as F

from acceleration_data import OFFSETS, RESULTS

SEEDS = (7341, 7342, 7343)
EPOCHS = 24


def linear_inputs(data, ids):
    # Five fixed temporal averages / samples of the same two-second motion stream.
    series = data.sensor_history(ids)
    blocks = [series[:, a:b].mean(1) for a,b in ((0,5),(5,10),(10,15),(15,20))]
    return np.column_stack([*blocks, series[:,-1], data.base[ids]]).astype(np.float32)


class LinearForecast:
    def fit(self, data):
        ids = data.train
        x = np.concatenate([linear_inputs(data,b) for b in np.array_split(ids,12)])
        labels = data.end_class[ids]
        residual = data.truth[ids] - data.base[ids]
        self.classifier = LogisticRegression(C=.012,max_iter=600).fit(x,labels)
        self.regressors = [Ridge(alpha=250.).fit(x[labels==i],residual[labels==i]) for i in range(3)]
        return self

    def predict(self, data, ids):
        probabilities, forecasts = [], []
        for batch in np.array_split(ids,int(np.ceil(len(ids)/768))):
            x = linear_inputs(data,batch)
            prob = self.classifier.predict_proba(x)
            corrections = np.stack([reg.predict(x) for reg in self.regressors],axis=1)
            probabilities.append(prob)
            forecasts.append(data.base[batch]+np.einsum("ni,nij->nj",prob,corrections))
        return np.vstack(probabilities),np.vstack(forecasts)


class AnticipationGRU(nn.Module):
    def __init__(self, n_features, visual=False):
        super().__init__()
        self.visual = visual
        self.encoder = nn.Sequential(nn.Linear(n_features,56),nn.LayerNorm(56),nn.SiLU())
        self.gru = nn.GRU(56,64,num_layers=2,dropout=.15,batch_first=True)
        self.current = nn.Sequential(nn.Linear(n_features+3,40),nn.LayerNorm(40),nn.SiLU())
        if visual:
            self.vision = nn.Sequential(
                nn.Conv2d(9,16,5,stride=2,padding=2),nn.SiLU(),
                nn.Conv2d(16,24,3,stride=2,padding=1),nn.SiLU(),
                nn.Conv2d(24,24,3,stride=2,padding=1),nn.SiLU(),
                nn.AdaptiveAvgPool2d((2,3)),nn.Flatten(),nn.Linear(144,32),nn.SiLU())
        self.shared = nn.Sequential(nn.Linear(168 + (32 if visual else 0),88),nn.LayerNorm(88),
                                    nn.SiLU(),nn.Dropout(.12))
        self.regime = nn.Linear(88,3)
        self.corrections = nn.Linear(88,9)
        nn.init.normal_(self.corrections.weight,std=.012)
        nn.init.zeros_(self.corrections.bias)

    def forward(self, sequence, centre, images=None):
        history,_ = self.gru(self.encoder(sequence))
        now = self.current(torch.cat((sequence[:,-1],centre),dim=-1))
        context = [history[:,-1],history.mean(1),now]
        if self.visual:
            context.append(self.vision(images))
        hidden = self.shared(torch.cat(context,dim=1))
        logits = self.regime(hidden)
        # All three endpoints are available from the same past. Inference only uses the learned probabilities.
        corrections = self.corrections(hidden).reshape(-1,3,3)
        ends = centre[:,None,:]+corrections
        forecast = (logits.softmax(1)[:,:,None]*ends).sum(1)
        return logits,ends,forecast


class Batches:
    def __init__(self, data, device):
        self.dev = device
        self.values = torch.as_tensor(data.sequence,device=device)
        self.starts = torch.as_tensor(data.start,device=device)
        self.offsets = torch.as_tensor(OFFSETS,device=device)
        self.centres = torch.as_tensor(data.base,device=device)
        self.images = torch.as_tensor(data.images,device=device) if data.images is not None else None
        self.image_ids = torch.as_tensor(data.image_index,device=device) if data.image_index is not None else None
        self.targets = torch.as_tensor(data.truth,device=device)
        self.regimes = torch.as_tensor(data.end_class,device=device)

    def inputs(self, ids, visual):
        timeline = torch.maximum(ids[:,None]-self.offsets,self.starts[ids,None])
        images = None
        if visual:
            if self.images is None:
                raise ValueError("The visual model requires synchronized images.")
            images = self.images[self.image_ids[ids].long()].permute(0,1,4,2,3).reshape(-1,9,48,64)
            images = images.float()/127.5-1.
        return self.values[timeline], self.centres[ids], images


def train_one(data, batches, name, visual, seed, output_dir=RESULTS):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AnticipationGRU(data.sequence.shape[1],visual).to(batches.dev)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0012,weight_decay=.003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS,eta_min=.00018)
    freq = np.bincount(data.end_class[data.train],minlength=3)/len(data.train)
    class_weights = torch.tensor(1/np.sqrt(freq),device=batches.dev,dtype=torch.float)
    # Roots are sampled only from the 16 training environments. Each run has the same effort.
    for epoch in range(EPOCHS):
        model.train()
        draws = rng.choice(data.train,9216,replace=False)
        losses = []
        for batch in np.array_split(draws,36):
            ids = torch.as_tensor(batch,device=batches.dev)
            y,cls = batches.targets[ids],batches.regimes[ids]
            logits,ends,forecast = model(*batches.inputs(ids,visual))
            chosen = ends[torch.arange(len(ids),device=batches.dev),cls]
            weights = class_weights[cls]
            loss_point = ((forecast-y).square().sum(-1)+.0064).sqrt().mean()
            loss_cond = (((chosen-y).square().sum(-1)+.0064).sqrt()*weights).sum()/weights.sum()
            loss = loss_point+.55*loss_cond+.55*F.cross_entropy(logits,cls)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),3.)
            optimizer.step()
            losses.append(loss.detach())
        schedule.step()
        if (epoch+1)%12==0:
            print(f"{name} {seed} epoch {epoch+1} loss {torch.stack(losses).mean().item():.4f}",flush=True)
    torch.save({"state_dict":model.state_dict(),"seed":seed,"visual":visual,"n_features":data.sequence.shape[1]},
               output_dir / f"{name}_{seed}.pt")
    return model.eval()


def predict_one(model,batches,ids):
    prob,point = [],[]
    model.eval()
    with torch.inference_mode():
        for batch in np.array_split(ids,int(np.ceil(len(ids)/640))):
            ix = torch.as_tensor(batch,device=batches.dev)
            logits,_,prediction = model(*batches.inputs(ix,model.visual))
            prob.append(logits.softmax(1).cpu().numpy())
            point.append(prediction.cpu().numpy())
    return np.vstack(prob),np.vstack(point)


def train_experiment():
    import json
    import joblib
    from acceleration_data import AccelerationData
    RESULTS.mkdir(parents=True,exist_ok=True)
    protocol = {
        "target": "future speed5 at origin+30, using the original training-tertile cutoffs",
        "population": "all slow origins with the original Q1 eligible target for evaluation",
        "history": "60 frames through origin, sampled every 3 frames, never across record or hand",
        "candidates": ["frozen_bayesian_gain","linear","motion_gru","video_motion_gru"],
        "fit": "16 original train environments only; fixed 24 epochs per GRU; average three prespecified seeds",
        "seeds": list(SEEDS),
        "selection": "4 validation environments only. Prefer lowest slow->fast endpoint error while requiring slow->slow and all slow-origin mean errors at most the incumbent. Keep incumbent on ties.",
        "blends": [0.25,0.5,1.0],
        "gates": ["all slow origins","above median validation predicted risk","upper quartile validation predicted risk"],
        "risk_operating_point": "maximum balanced accuracy on validation, then frozen",
        "test_use": "previously examined environments; retrospective comparison; no test tuning",
    }
    (RESULTS/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=AccelerationData(device)
    data.verify_baseline("validation")
    batches=Batches(data,device)
    np.savez(RESULTS/"preprocessing.npz",mean=data.feature_mean,sd=data.feature_sd,cuts=data.cuts,offsets=OFFSETS)
    print("Training origins",len(data.train),"history length",len(OFFSETS),"features",data.sequence.shape[1],flush=True)
    linear=LinearForecast().fit(data)
    # Save only the sklearn objects, so weights reload identically under the imported module.
    joblib.dump((linear.classifier,linear.regressors),RESULTS/"linear.joblib")
    print("Linear reference fitted.",flush=True)
    for name,visual in (("motion_gru",False),("video_motion_gru",True)):
        for seed in SEEDS:
            train_one(data,batches,name,visual,seed)
    print("Training finished. Test outcomes have not been used to choose any setting.",flush=True)


if __name__=="__main__":
    train_experiment()
