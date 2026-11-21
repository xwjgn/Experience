"""QSM-MoE under wrist-only masking. Train new experts and learn request timing from counterfactual replays."""
import json
from itertools import product

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qsm_data import ROOT,KIT,OUT,scoring_reference,reproducible_torch,write_json
from qsm_train import time_basis

H,L,DELAY=12,30,3
FIRST=(5,8,11,12)
SECOND=(17,19,22,25)
SEED=9461
KEY=["split","env","sample","hand","first_hidden_frame"]


def fresh_data():
    tracks,_=scoring_reference.load_tracks(KIT/"cache/canonical",ROOT/"Restocking_Problem_and_Data_EN")
    scoring_reference.qualify(tracks)
    envs=sorted({tr["environment_id"] for tr in tracks if tr["split"]=="train"})
    perm=np.random.default_rng(SEED).permutation(envs).tolist()
    sets=dict(fit=perm[:9],policy=perm[9:12],calibration=perm[12:])
    ts,zs,meta,ft,fz,natural=[],[],[],[],[],[]
    for tr in tracks:
        t,z=tr["t"],tr["points"]["wrist"]
        good=tr["qualified"] & np.isfinite(z).all(1)
        n=len(t)
        for quality in ("occlusion","near edge"):
            changes=np.diff(np.r_[False,tr["quality"]==quality,False].astype(int))
            for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
                next_good=np.flatnonzero(good[b:])
                recovery=(t[b+next_good[0]]-t[b-1]) if len(next_good) else np.nan
                natural.append(dict(split=tr["split"],sample=tr["sample"],hand=tr["hand"],quality=quality,
                             frames=int(b-a),duration_s=t[b-1]-t[a]+np.median(np.diff(t)),left_censored=(a==0),
                             recovery_s=recovery,right_censored=not len(next_good)))
        for first in range(H,n-L+1,60):
            ix=slice(first-H,first+L)
            if good[ix].all() and np.max(np.diff(t[ix]))<=.05:
                meta.append([tr["split"],tr["environment_id"],tr["sample"],tr["hand"],first])
                ts.append(t[ix]);zs.append(z[ix])
        if tr["environment_id"] in sets["fit"]:
            starts=np.arange(0,n-84+1,4)
            idx=starts[:,None]+np.arange(84)
            keep=good[idx].all(1)&(np.diff(t[idx],axis=1)<=.05).all(1)
            ix=idx[keep]
            ft.append(t[ix]-t[ix[:,0],None])
            fz.append(z[ix]-z[ix[:,0],None])
    pd.DataFrame(natural).to_csv(OUT/"q2_natural_flag_runs.csv",index=False)
    return pd.DataFrame(meta,columns=KEY),np.stack(ts),np.stack(zs),np.concatenate(ft).astype("float32"),np.concatenate(fz).astype("float32"),sets


class SparseMoE(nn.Module):
    """A damped velocity expert and three learned movement experts for the newest released history."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(8,40,batch_first=True)
        self.short=nn.Sequential(nn.Linear(4*8,32),nn.SiLU())
        self.combine=nn.Sequential(nn.Linear(72,64),nn.LayerNorm(64),nn.SiLU())
        self.stage=nn.Linear(64,3)
        self.gate=nn.Linear(64,1)
        self.experts=nn.Linear(64,36)
        self.width=nn.Linear(64,4)
        self.decay=nn.Parameter(torch.tensor(-1.))
        self.decay_weights=nn.Parameter(torch.zeros(3))

    def condition(self,t,z):
        gap=torch.diff(t,dim=1,prepend=t[:,:1])
        delta=torch.diff(z,dim=1,prepend=z[:,:1])
        velocity=delta/gap.clamp_min(.001)[:,:,None]
        feat=torch.cat(((z-z[:,-1:])/.16,velocity.clamp(-4.,4.)/.7,
                        (t-t[:,-1:])[:,:,None],gap[:,:,None]*5),2)
        _,h=self.encoder(feat)
        f=self.combine(torch.cat((h[-1],self.short(feat[:,-4:].reshape(len(t),-1))),1))
        stage=self.stage(f)
        g=self.gate(f).sigmoid()
        weights=torch.cat((1-g,g*stage.softmax(1)),1)
        vel=self.experts(f).reshape(-1,3,4,3)
        v0=(self.decay_weights.softmax(0)[None,:,None]*velocity[:,-3:]).sum(1)
        return v0,vel,weights,.25*F.softplus(self.width(f))+.005,stage

    def forecast(self,x,age):
        v0,vel,weights,rate,_=x
        tau=.06+F.softplus(self.decay)
        basis=time_basis(age)
        physical=(-tau*torch.expm1(-age/tau))[:,:,None]*v0[:,None]
        learned=torch.einsum("nhd,nkdc->nkhc",basis,vel)
        points=torch.cat((physical[:,None],learned),1)
        mean=(weights[:,:,None,None]*points).sum(1)
        var=(weights[:,:,None]*(points-mean[:,None]).square().sum(-1)).sum(1)
        sigma=torch.sqrt((.002+(basis*rate[:,None]).sum(-1))**2+.25*var)
        return mean,sigma,points


def fit_model(t,z):
    torch.manual_seed(SEED)
    rng=np.random.default_rng(SEED)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=SparseMoE().to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=.0016,weight_decay=.003)
    ft=torch.as_tensor(t,device=dev);fz=torch.as_tensor(z,device=dev)
    speed=np.linalg.norm(z[:,11]-z[:,6],axis=1)/np.maximum(t[:,11]-t[:,6],.001)
    threshold=float(np.quantile(speed,1/3))
    for epoch in range(23):
        for _ in range(30):
            n=192
            rows=torch.as_tensor(rng.integers(len(t),size=n),device=dev)[:,None]
            history=np.tile(np.arange(H),(n,1))
            count=rng.choice(3,n,p=[.4,.35,.25])
            first=rng.integers(4,16,n)
            second=first+rng.integers(5,17,n)
            for number in (1,2):
                take=count==number
                history[take,:H-number]=np.arange(number,H)
                history[take,H-number]=H-1+first[take]
                if number==2:
                    history[take,-1]=H-1+second[take]
            obs=torch.as_tensor(history,device=dev)
            future=obs[:,-1:]+torch.arange(1,L+1,device=dev)
            ht,hz=ft[rows,obs],fz[rows,obs]
            age=ft[rows,future]-ht[:,-1:]
            y=fz[rows,future]-hz[:,-1:]
            cond=net.condition(ht,hz)
            center,width,points=net.forecast(cond,age)
            err=((center-y).square().sum(2)+1e-7).sqrt()
            first_leg=y[:,4]; second_leg=y[:,9]-y[:,4]
            future_speed=second_leg.norm(dim=1)/(age[:,9]-age[:,4]).clamp_min(.001)
            turn=F.cosine_similarity(first_leg,second_leg)<.7071
            label=torch.where(future_speed<threshold,0,torch.where(turn,1,2))
            # An energy score supplies a proper joint-path objective for the four alternatives.
            flat=(points/np.sqrt(L)).flatten(2)
            truth=(y/np.sqrt(L)).flatten(1)
            w=cond[2]
            es=(w*(flat-truth[:,None]).norm(dim=2)).sum(1)-.5*(w[:,:,None]*w[:,None,:]*torch.cdist(flat,flat)).sum((1,2))
            loss=(err/(.025+.16*age)).mean()+.10*(err.detach()/width+width.log()).mean()+.13*F.cross_entropy(cond[-1],label)+2.*es.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),3.)
            opt.step()
    torch.save(net.state_dict(),OUT/"q2_sparse_moe_scratch.pt")
    return net.cpu().eval(),threshold


@torch.inference_mode()
def condition(net,t,z):
    tt=torch.from_numpy((t-t[:,-1:]).astype("float32"))
    zz=torch.from_numpy((z-z[:,-1:]).astype("float32"))
    v0,v,w,rate,_=net.condition(tt,zz)
    return tuple(a.numpy() for a in (v0,v,w,rate))


def policy_features(cond,t,z,innovation,frame):
    v0,v,w,rate=cond
    speed=np.linalg.norm(v0,axis=1)
    displacement=z[:,-1]-z[:,-4]
    trend=np.linalg.norm(displacement,axis=1)/np.maximum(t[:,-1]-t[:,-4],.001)
    diff=np.diff(z,axis=1)/np.maximum(np.diff(t,axis=1),.001)[:,:,None]
    jitter=np.linalg.norm(np.diff(diff[:,-4:],axis=1),axis=2).mean(1)
    dispersion=np.linalg.norm(v[:,:,0]-v0[:,None],axis=2).mean(1)
    return np.column_stack((speed,trend,jitter,w,rate,dispersion,innovation,t[:,-1]-t[:,-2],frame/30))


def simulate(net,t,z,calendar=None,policy=None):
    """Read a masked coordinate only at its delayed delivery. Policy uses the clock and released history."""
    n=len(t); ids=np.arange(n)
    ht=t[:,:H].copy();hz=z[:,:H].copy()
    cond=condition(net,ht,hz)
    tau=float(.06+F.softplus(net.decay.detach()).item())
    centers=np.empty((n,L,3)); scales=np.empty((n,L))
    request=np.zeros((n,L),bool);waiting=np.full(n,-1,int)
    count=np.zeros(n,int);innovation=np.zeros(n)
    initial=policy_features(cond,ht,hz,innovation,np.zeros(n))
    if calendar is None:
        a=np.asarray(FIRST)[policy[0].predict(initial).argmin(1)]
        b=np.full(n,31,int)
    else:
        a=np.full(n,calendar[0],int);b=np.full(n,calendar[1],int)
    after=np.zeros_like(initial)
    for j in range(L):
        arrived=ids[(waiting>=0)&(waiting+DELAY==j)]
        if len(arrived):
            src=waiting[arrived]
            sample=z[arrived,H+src]
            innovation[arrived]=np.linalg.norm(sample-centers[arrived,src],axis=1)
            ht[arrived]=np.column_stack((ht[arrived,1:],t[arrived,H+src]))
            hz[arrived]=np.concatenate((hz[arrived,1:],sample[:,None]),axis=1)
            upd=condition(net,ht[arrived],hz[arrived])
            for dest,values in zip(cond,upd):
                dest[arrived]=values
            first_return=arrived[count[arrived]==1]
            if len(first_return):
                f=policy_features(tuple(v[first_return] for v in cond),ht[first_return],hz[first_return],
                                  innovation[first_return],a[first_return])
                after[first_return]=f
                if calendar is None:
                    b[first_return]=np.asarray(SECOND)[policy[1].predict(f).argmin(1)]
            waiting[arrived]=-1
        age=t[:,H+j]-ht[:,-1]
        v0,v,w,rate=cond
        basis=time_basis(age)
        physics=(-tau*np.expm1(-age/tau))[:,None]*v0
        experts=np.concatenate((physics[:,None],np.einsum("nd,nkdc->nkc",basis,v)),1)
        delta=(w[:,:,None]*experts).sum(1)
        centers[:,j]=hz[:,-1]+delta
        var=(w*np.square(experts-delta[:,None]).sum(2)).sum(1)
        scales[:,j]=np.sqrt((.002+(basis*rate).sum(1))**2+.25*var)
        ask=((count==0)&(a==j+1))|((count==1)&(b==j+1))
        assert not (ask & (waiting>=0)).any()
        request[:,j]=ask;waiting[ask]=j;count+=ask
    expected=2 if calendar is None else sum(1<=j<=L-DELAY for j in calendar)
    assert (count==expected).all() and (waiting<0).all()
    return centers,scales,request,initial,after


def fit_policy(net,t,z):
    n=len(t)
    costs=np.empty((n,len(FIRST),len(SECOND)))
    states=[]
    for i,a in enumerate(FIRST):
        for j,b in enumerate(SECOND):
            y,_,_,first,after=simulate(net,t,z,(a,b))
            costs[:,i,j]=np.linalg.norm(y-z[:,H:],axis=2).mean(1)*100
            if j==0:
                states.append(after)
    def reg(fields,labels):
        m=make_pipeline(StandardScaler(),Ridge(alpha=50.))
        return m.fit(fields,labels-labels.mean(1,keepdims=True))
    # First choice averages second-action costs; the second regressor sees the actually returned observation.
    one=reg(first,costs.mean(2))
    two=reg(np.concatenate(states),np.concatenate([costs[:,i,:] for i in range(len(FIRST))]))
    return one,two


def rank95(x):
    rank=int(np.ceil((len(x)+1)*.95))
    if rank>len(x): raise ValueError("Too few calibration windows")
    return float(np.partition(x,rank-1)[rank-1])


def replay_check(net,t,z,policy):
    t,z=t[:48],z[:48].copy()
    ref=simulate(net,t,z,policy=policy)
    rng=np.random.default_rng(77)
    zh=z.copy()
    zh[:,H:]+=rng.normal(0,.7,size=z[:,H:].shape)*(~ref[2])[:,:,None]
    alt=simulate(net,t,zh,policy=policy)
    for x,y in zip(ref[:3],alt[:3]): np.testing.assert_array_equal(x,y)
    queried=ref[2].argmax(1);rows=np.arange(len(t))
    zh=z.copy();zh[rows,H+queried]+=rng.normal(0,.25,(len(t),3))
    alt=simulate(net,t,zh,policy=policy)
    locked=np.arange(L)[None]<queried[:,None]+DELAY
    for x,y in zip(ref[:3],alt[:3]): np.testing.assert_array_equal(x[locked],y[locked])


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    reproducible_torch()
    meta,t,z,ft,fz,sets=fresh_data()
    write_json(OUT/"q2_protocol.json",dict(source_weights_loaded=0,seed=SEED,environments=sets,history=H,masked=L,
         delay=DELAY,budget=2,cost=1,first_actions=FIRST,second_actions=SECOND,
         objective="Learn expected counterfactual complete-window Euclidean cost from training pilot environments; no test tuning",
         observation="clock, warm wrist and delivered wrist only; online stage/innovation drive decision",
         selection="Fixed calendar minimized validation environment-macro trajectory error. Active policy is independently frozen."))
    print("Fitting a fresh sparse-observation mixture",len(ft),"training clips",flush=True)
    net,cut=fit_model(ft,fz)
    ip=meta.env.isin(sets["policy"]).to_numpy()
    ic=meta.env.isin(sets["calibration"]).to_numpy()
    iv=meta.split.eq("validation").to_numpy()
    policy=fit_policy(net,t[ip],z[ip])
    joblib.dump(policy,OUT/"q2_counterfactual_request.joblib")
    replay_check(net,t[iv],z[iv],policy)
    trials=[]
    for a,b in product(FIRST,SECOND):
        y=simulate(net,t[iv],z[iv],(a,b))[0]
        err=np.linalg.norm(y-z[iv,H:],axis=2).mean(1)*100
        trials.append(dict(first=a,second=b,macro_cm=float(pd.Series(err).groupby(meta.env[iv].to_numpy()).mean().mean())))
    best=min(trials,key=lambda v:v["macro_cm"])
    pd.DataFrame(trials).to_csv(OUT/"q2_validation_calendars.csv",index=False)
    settings={"no_request":(0,0),"one_request":(11,0),"matched_11_19":(11,19),
              "new_fixed":(best["first"],best["second"]),"new_active":None}
    factors={}
    for k,cal in settings.items():
        y,width,*_=simulate(net,t[ic],z[ic],cal,policy)
        factors[k]=rank95((np.linalg.norm(y-z[ic,H:],axis=2)/width).max(1))
    write_json(OUT/"q2_locked.json",dict(fixed=best,calibration_windows=int(ic.sum()),policy_windows=int(ip.sum()),
        stage_speed_cut=cut,factors=factors,unrequested_perturbation_passed=True,before_delivery_perturbation_passed=True))
    records=[];summaries=[];horizons=[]
    for split in ("validation","test"):
        ix=meta.split.eq(split).to_numpy()
        for name,cal in settings.items():
            y,width,ask,_,_=simulate(net,t[ix],z[ix],cal,policy)
            err=np.linalg.norm(y-z[ix,H:],axis=2)
            band=factors[name]*width
            df=meta[ix].copy()
            df["method"]=name;df["error_cm"]=100*err.mean(1);df["endpoint_cm"]=100*err[:,-1]
            df["radius_cm"]=100*band.mean(1);df["path_covered"]=(err<=band).all(1)
            df["frame_coverage"]=(err<=band).mean(1);df["requests"]=ask.sum(1)
            df["within_10cm"]=(err<=.10).mean(1)
            df["waiting_s"]=((t[ix,H+DELAY:]-t[ix,H:-DELAY])*ask[:,:-DELAY]).sum(1)
            df["request_frames"]=["|".join(map(str,np.flatnonzero(a)+1)) for a in ask]
            records.append(df)
            for j in range(L):
                horizons.append(dict(split=split,method=name,frame=j+1,error_cm=100*err[:,j].mean(),
                                 within_10cm=(err[:,j]<=.10).mean(),mean_radius_cm=100*band[:,j].mean()))
            summaries.append(dict(split=split,method=name,n=len(df),env_macro_cm=df.groupby("env").error_cm.mean().mean(),
                      error_cm=df.error_cm.mean(),endpoint_cm=df.endpoint_cm.mean(),path_coverage=df.path_covered.mean(),
                      radius_cm=df.radius_cm.mean(),requests=df.requests.mean(),waiting_s=df.waiting_s.mean(),
                      within_10cm=df.within_10cm.mean()))
    logs=pd.concat(records,ignore_index=True)
    logs.to_csv(OUT/"q2_issued_windows.csv.gz",index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/"q2_compare.csv",index=False)
    pd.DataFrame(horizons).to_csv(OUT/"q2_horizons.csv",index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":
    run()
