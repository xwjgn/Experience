"""C-problem observations, author code adapters and a freshly fitted wrist state estimator."""
import json
import os
import sys
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scipy.spatial.transform import Rotation
from qsm_generator import fit_and_sample

KIT = Path(__file__).resolve().parents[1]
ROOT = KIT.parent
sys.path.insert(0, str(KIT))
import q1_state as scoring_reference

STEPS = np.arange(1,31)
OUT = ROOT / "分析报告" / "QSM-MoE融合方案对比"
SNAP = ROOT / "参考文献" / "论文与代码仓库" / "仓库快照"
CACHE = OUT / "cache"
SEED = 9317


def reproducible_torch():
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def author_path(filename, folder):
    archive = SNAP / filename
    with ZipFile(archive) as z:
        root = z.namelist()[0].split("/")[0]
    path = str(archive / root / folder)
    if path not in sys.path:
        sys.path.insert(0, path)


def author_transformer():
    author_path("03-LSTTN-68185aa8.zip", "models")
    from transformer import TransformerLayers
    return TransformerLayers


def diffusion_schedule():
    author_path("05-Uni-Hand-7f3edd59.zip", "unihand")
    from denoising_diffusion.gaussian_diffusion import GaussianDiffusion
    phase = (np.arange(17)/16+.008)/1.008 * np.pi/2
    alpha = np.maximum((np.cos(phase)/np.cos(phase[0]))**2,.005)
    betas = 1 - alpha[1:] / alpha[:-1]
    result = GaussianDiffusion(betas=betas, predict_xstart=False, rescale_learned_sigmas=False,
                               learn_sigmas=False, sigma_small=False, use_kl=False)
    np.testing.assert_allclose(result.alphas_cumprod, alpha[1:], atol=2e-7)
    return result


def motion_modes(t, z, speed_cut):
    o = np.arange(len(t))
    mid, first = np.maximum(0, o-5), np.maximum(0, o-10)
    a, b = z[mid]-z[first], z-z[mid]
    norm = np.linalg.norm(b, axis=1)
    speed = norm / np.maximum(t-t[mid], 1e-4)
    cos = (a*b).sum(1) / np.maximum(np.linalg.norm(a,axis=1)*norm, 1e-9)
    mode = np.where(speed < speed_cut, 0, np.where(cos < np.cos(np.pi/4), 1, 2))
    return mode, np.column_stack((np.minimum(speed, 4), np.clip(cos,-1,1)))


def quality_features(t, z, flag, seq):
    o = np.arange(len(t))
    prev = np.maximum(o-1, 0)
    dt = np.maximum(t-t[prev], 1e-4)
    vel = (z-z[prev]) / dt[:,None]
    # The raw geometry block passed here contains no quality flags.
    fields = [np.nan_to_num(seq), np.minimum(np.linalg.norm(vel,axis=1),8),
              np.minimum(np.linalg.norm(vel-vel[prev],axis=1),10), np.minimum(dt,.5)]
    for lag in (1,3,6,12):
        idx = o-lag
        fields.append(np.where(idx>=0, flag[np.maximum(idx,0)], 0))
    return np.column_stack(fields).astype("float32")


def filter_track(t, z, bad_prob, q, r, multiple):
    """Keep independent xyz covariances so a missing coordinate can skip its measurement update."""
    n = len(z)
    post = np.empty((n,3))
    velocity = np.empty_like(post)
    variance = np.empty_like(post)
    p, v = np.zeros(3), np.zeros(3)
    pp, pv, vv = np.ones(3), np.zeros(3), np.full(3,.25)
    seen = np.zeros(3, bool)
    for i in range(n):
        dt = max(t[i]-t[i-1], 0.) if i else 0.
        p = p + dt*v
        pp, pv, vv = (pp+2*dt*pv+dt*dt*vv+q*dt**4/4,
                      pv+dt*vv+q*dt**3/2, vv+q*dt*dt)
        obs = np.isfinite(z[i])
        initial = obs & ~seen
        p[initial] = z[i,initial]
        pp[initial], pv[initial] = r, 0.
        update = obs & seen
        rr = r * (1 + (multiple-1)*bad_prob[i])
        denom = pp[update] + rr
        kp, kv = pp[update]/denom, pv[update]/denom
        innov = z[i,update]-p[update]
        p[update] += kp*innov
        v[update] += kv*innov
        pp[update], pv[update], vv[update] = (
            rr*kp, rr*kv, vv[update]-kv*pv[update])
        seen |= obs
        post[i], velocity[i], variance[i] = p, v, pp
    return post, velocity, variance


def local(v, rot):
    return np.einsum("n...j,nji->n...i", v, rot)


def fit_physics(tracks, masks, seed):
    rng = np.random.default_rng(seed)
    ts, zs, bs, actual, cad, envs = [], [], [], [], [], []
    for tr, eligible in zip(tracks,masks):
        if tr["split"] != "train":
            continue
        options = np.flatnonzero(eligible.all(1) & (np.arange(len(eligible))>=40))
        if not len(options):
            continue
        o = rng.choice(options,min(64,len(options)),replace=False)
        idx = o[:,None]-np.arange(39,-1,-1)
        t,z=tr["t"],tr["points"]["wrist"]
        ts.append(t[idx]); zs.append(z[idx]); bs.append(tr["bad"][idx])
        actual.append(z[o[:,None]+np.array([10,20,30])])
        cad.append(np.median(np.diff(ts[-1][:,-6:],axis=1),axis=1))
        envs.extend([tr["environment_id"]]*len(o))
    t,z,flag,target,dt_med=map(np.concatenate,(ts,zs,bs,actual,cad))
    env=np.unique(envs,return_inverse=True)[1]
    weight=1/np.bincount(env)[env]
    rows=[]
    for q in (.04,.4):
        for r in (2e-6,2e-5):
            for m in (5.,25.):
                p,v=z[:,0].copy(),np.zeros_like(z[:,0])
                pp,pv,vv=np.full(len(z),r),np.zeros(len(z)),np.full(len(z),.25)
                for i in range(1,t.shape[1]):
                    dt=t[:,i]-t[:,i-1]
                    p+=dt[:,None]*v
                    a,b,c=pp+2*dt*pv+dt**2*vv+q*dt**4/4,pv+dt*vv+q*dt**3/2,vv+q*dt**2
                    rr=r*(1+(m-1)*flag[:,i])
                    k1,k2=a/(a+rr),b/(a+rr)
                    dz=z[:,i]-p
                    p+=k1[:,None]*dz; v+=k2[:,None]*dz
                    pp,pv,vv=rr*k1,rr*k2,c-k2*b
                for tau in (.18,.4,.8):
                    age=dt_med[:,None]*np.array([10,20,30])
                    pred=p[:,None]+v[:,None]*(-tau*np.expm1(-age/tau))[:,:,None]
                    err=np.linalg.norm(pred-target,axis=2).mean(1)
                    rows.append(dict(q=q,r=r,m=m,tau=tau,mean_cm=100*np.average(err,weights=weight)))
    pd.DataFrame(rows).to_csv(OUT/"q1_physics_training_grid.csv",index=False)
    best=min(rows,key=lambda row:row["mean_cm"])
    return {k:best[k] for k in ("q","r","m")},best["tau"]


def geometry(t,z,other,head,rot,fingers):
    o=np.arange(len(t))
    blocks=[local(z-head,rot),local(other-z,rot),local(fingers-z[:,None],rot).reshape(len(z),-1)]
    for lag in (1,4,12,32):
        j=np.maximum(o-lag,0)
        dt=np.maximum(t-t[j],1e-4)
        for pos in (z,other,head):
            blocks.append(local((pos-pos[j])/dt[:,None],rot))
        blocks.append(dt[:,None])
    return np.column_stack(blocks).astype(np.float32)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    (CACHE/"ready.json").unlink(missing_ok=True)
    reproducible_torch()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracks,frames=scoring_reference.load_tracks(KIT/"cache/canonical",ROOT/"Restocking_Problem_and_Data_EN")
    scoring_reference.qualify(tracks)
    environments=sorted({tr["environment_id"] for tr in tracks if tr["split"]=="train"})
    order=np.random.default_rng(SEED).permutation(environments).tolist()
    roles={"fit":order[:10],"gate":order[10:13],"calibration":order[13:]}
    groups={s:g.sort_values("frame_index") for s,g in frames.groupby("sample",sort=False)}
    by_key={(tr["sample"],tr["hand"]):tr for tr in tracks}
    # Raw head-camera thumbnails retain their recorded capture timestamps.
    images=[np.zeros((1,48,64,3),np.uint8)]
    video_index={}
    count=1
    for key,g in groups.items():
        with np.load(KIT/"cache/literature_features"/(key+".npz")) as f:
            stamps,rgb=f["t"],f["rgb"]
        t=g["t_s"].to_numpy()
        current=np.searchsorted(stamps,t,side="right")-1
        assert np.all(stamps[current[current>=0]]<=t[current>=0])
        idx=np.stack((current-1,current),1)
        video_index[key]=np.where(idx>=0,idx+count,0)
        images.append(rgb); count+=len(rgb)
    speed5=[]
    for tr in tracks:
        t,z=tr["t"],tr["points"]["wrist"]
        sp=np.r_[np.nan,np.linalg.norm(np.diff(z,axis=0),axis=1)/np.diff(t)]
        speed5.append(pd.Series(sp).rolling(5,min_periods=5).mean().to_numpy())
    cuts=np.quantile(np.concatenate([v[np.isfinite(v)] for tr,v in zip(tracks,speed5) if tr["split"]=="train"]),[1/3,2/3])
    mode_cut=float(np.quantile(np.concatenate([v[np.isfinite(v)] for tr,v in zip(tracks,speed5)
                                               if tr["environment_id"] in roles["fit"]]),1/3))
    features={key:[] for key in ("sequence","start","cadence","y","mask","ok","grid","mode","next_mode","state","video","recent")}
    qx,metadata,clocks,rotations=[],[],[],[]
    start=0
    for tr,sp in zip(tracks,speed5):
        s,hand=tr["sample"],tr["hand"]
        g=groups[s]; t,z=tr["t"],tr["points"]["wrist"]
        n=len(t); o=np.arange(n); flag=tr["bad"]
        other=by_key[(s,"right" if hand=="left" else "left")]
        head=g[["head_x","head_y","head_z"]].to_numpy()
        rot=Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].to_numpy()).as_matrix()
        rotations.append(rot)
        fingers=g[[f"{hand}_j{j:02}_{a}" for j in (4,5,8,9,12,17,20) for a in "xyz"]].to_numpy().reshape(n,7,3)
        geom=geometry(t,z,other["points"]["wrist"],head,rot,fingers)
        cut=min(51,n)
        np.testing.assert_allclose(geom[:cut],geometry(t[:cut],z[:cut],other["points"]["wrist"][:cut],head[:cut],rot[:cut],fingers[:cut]),atol=1e-5)
        qx.append(quality_features(t,z,flag,geom))
        look=o[:,None]+STEPS; safe=look.clip(max=n-1)
        mask=(look<n) & tr["qualified"][safe] & (o[:,None]>=10) & np.isfinite(sp[:,None])
        mode,state=motion_modes(t,z,mode_cut)
        future=np.full(n,-1,np.int8); future[:-10]=mode[10:]
        recent=np.convolve(flag,np.ones(10),"full")[:n]>0
        seq=np.column_stack((geom,flag,other["bad"],recent,np.full(n,hand=="right"),np.minimum(t-t[0],2)))
        cadence=pd.Series(np.r_[np.nan,np.diff(t)]).rolling(5,min_periods=1).median().fillna(1/30).to_numpy()
        values=dict(sequence=seq.astype("float32"),start=np.full(n,start),cadence=cadence.astype("float32"),
                    y=local(z[safe]-z[:,None],rot).astype("float32"),mask=mask,ok=~flag,grid=(o-10)%30==0,
                    mode=mode.astype("int8"),next_mode=future,state=state.astype("float32"),video=video_index[s],recent=recent)
        for k,v in values.items():
            features[k].append(v)
        target_bin=np.full(n,-1);target_bin[:-30]=np.digitize(sp[30:],cuts)
        metadata.append(pd.DataFrame(dict(sample=s,hand=hand,env=tr["environment_id"],split=tr["split"],origin=o,
                            slow=(o>=10)&(sp<cuts[0]),score=mask[:,-1],end_bin=target_bin)))
        if hand=="left":
            dt=np.diff(t)
            clocks.append(dict(sample=s,split=tr["split"],env=tr["environment_id"],frames=n,dt_median_ms=1000*np.median(dt),
                               dt_min_ms=1000*dt.min(),dt_max_ms=1000*dt.max(),h30_median_s=np.median(t[30:]-t[:-30])))
        start+=n
    fit_rows=[i for i,tr in enumerate(tracks) if tr["environment_id"] in roles["fit"]]
    kf,tau=fit_physics([tracks[i] for i in fit_rows],[features["mask"][i] for i in fit_rows],SEED)
    arr={k:np.concatenate(v) for k,v in features.items()}
    meta=pd.concat(metadata,ignore_index=True)
    train=meta.env.isin(roles["fit"]).to_numpy()
    xq=np.concatenate(qx); bad=~arr["ok"]
    quality=HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=12,l2_regularization=4,learning_rate=.08,
                                           early_stopping=False,random_state=SEED)
    quality.fit(xq[train],bad[train])
    prob=quality.predict_proba(xq)[:,1].astype("float32")
    joblib.dump(quality,OUT/"measurement_quality.joblib")
    quality_rows=[]
    for split in ("train","validation","test"):
        ix=meta.split.eq(split).to_numpy()
        quality_rows.append(dict(split=split,n=int(ix.sum()),flagged=int(bad[ix].sum()),prevalence=bad[ix].mean(),
                        AP=average_precision_score(bad[ix],prob[ix]),AUROC=roc_auc_score(bad[ix],prob[ix]),Brier=brier_score_loss(bad[ix],prob[ix])))
    pd.DataFrame(quality_rows).to_csv(OUT/"q1_quality.csv",index=False)
    physics=np.empty((len(meta),2,7),np.float32)
    for tr,rot,starts in zip(tracks,rotations,features["start"]):
        ix=np.arange(len(starts))+starts[0]
        t,z=tr["t"],tr["points"]["wrist"]
        for col,risk in enumerate((bad[ix].astype(float),np.maximum(bad[ix].astype(float),prob[ix]))):
            p,v,var=filter_track(t,z,risk,kf["q"],kf["r"],kf["m"])
            physics[ix,col]=np.column_stack((local(p-z,rot),local(v,rot),np.sqrt(var.mean(1))))
    z=np.array([[1.,0,0],[np.nan,1.,0],[np.nan,np.nan,np.nan],[1.,2.,1.]])
    p,v,s=filter_track(np.arange(4)/30,z,np.zeros(4),kf["q"],kf["r"],kf["m"])
    assert np.isfinite(p).all() and p[2,0]==1 and p[1,1]>0 and s[2,0]>s[0,0]
    seq=arr["sequence"]
    mu=seq[train].mean(0,dtype=float).astype("float32")
    sd=np.maximum(seq[train].std(0,dtype=float),.005).astype("float32")
    arr["sequence"]=np.clip((seq-mu)/sd,-8,8)
    print("Training new visual generator on official training environments",flush=True)
    vision,knots=fit_and_sample(arr["sequence"],arr["start"],arr.pop("video"),np.concatenate(images),arr["y"],arr["mask"],train,
                                OUT,diffusion_schedule(),dev,SEED)
    vm,vs=vision[train].mean(0),np.maximum(vision[train].std(0),.01)
    arr.update(vision=(vision-vm)/vs,knots=knots,physics=physics,prob=prob)
    for key,value in arr.items():
        np.save(CACHE/(key+".npy"),value,allow_pickle=False)
    meta.to_parquet(CACHE/"meta.parquet",index=False)
    pd.DataFrame(clocks).to_csv(OUT/"record_cadence.csv",index=False)
    np.savez(OUT/"feature_normalization.npz",motion_mean=mu,motion_sd=sd,vision_mean=vm,vision_sd=vs,
             evaluation_speed_cuts=cuts,mode_cut=mode_cut)
    write_json(CACHE/"ready.json",dict(rows=len(meta),records=len(clocks),tau=tau,quality_kalman=kf,seed=SEED,
                 version="qsm-independent-2",source_weights_loaded=0,environments=roles,generator_epochs=22))
    print("Independent QSM inputs ready",len(meta),flush=True)


if __name__=="__main__":
    prepare()
