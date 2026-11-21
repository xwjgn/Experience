from collections import defaultdict
import numpy as np
import pandas as pd
from study_paths import DATA,OUT,save
from release_reader import manifest,load_record,SIDES
from hand_features import H,encode,reference_geometry,speed_trace,stage,transform


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    catalog = manifest().sort_values(["dataset","sample"])
    train_envs = sorted(catalog[catalog.split=="train"].environment_id.unique())
    groups = np.random.default_rng(9317).permutation(train_envs).tolist()
    roles = dict(fit=groups[:10],gate=groups[10:13],cal=groups[13:])
    rows = []
    for m in catalog.itertuples(index=False):
        raw = load_record(m.dataset,m.sample)
        for side,hand in enumerate(SIDES):
            feats,geom = encode(raw,hand)
            t,z,_,_ = geom
            flagged = raw.flags[:,side]!=0
            bone = reference_geometry(raw.joints[:,side].astype(float))
            rows.append((m,hand,feats,geom,flagged,bone,speed_trace(t,z)))
    clean_train = np.concatenate([b[~bad] for m,_,_,_,bad,b,_ in rows if m.split=="train"])
    eligible_cut = float(np.quantile(clean_train,.995))
    speed_cut = float(np.quantile(np.concatenate([v[np.isfinite(v)] for m,_,_,_,_,_,v in rows if m.environment_id in roles["fit"]]),1/3))
    arrays = defaultdict(list)
    keys = []
    start = 0
    for m,hand,feats,(t,z,rot,mirror),flag,bone,sp in rows:
        n = len(t)
        o = np.arange(n)
        future = o[:,None]+np.arange(1,H+1)
        safe = future.clip(max=n-1)
        qualified = (~flag)&(bone<=eligible_cut)&feats["finite"]
        full = (future<n)&qualified[safe]&(o[:,None]>=10)&np.isfinite(sp[:,None])
        now = stage(t,z,speed_cut)
        target = np.full(n,-1,np.int8)
        target[:-10] = now[10:]
        vals = dict(**feats,start=np.full(n,start,np.int64),stage=target,now=now.astype(np.int8),flag=flag,eligible=full,
                    y=transform(z[safe]-z[:,None],rot,mirror).astype("float32"))
        for k,a in vals.items():
            arrays[k].append(a)
        keys.append(pd.DataFrame(dict(sample=m.sample,dataset=m.dataset,hand=hand,env=m.environment_id,split=m.split,origin=o)))
        start += n
    meta = pd.concat(keys,ignore_index=True)
    meta.to_parquet(DATA/"windows.parquet",index=False)
    catalog.to_csv(DATA/"records.csv",index=False)
    fit = meta.env.isin(roles["fit"]).to_numpy()
    scales = {}
    for name,parts in arrays.items():
        value = np.concatenate(parts)
        if name in ("nodes","global_x"):
            mu = value[fit].mean(axis=0,dtype=float).astype("float32")
            sd = value[fit].std(axis=0,dtype=float).astype("float32").clip(.004)
            scales[name+"_mean"],scales[name+"_sd"] = mu,sd
            value = ((value-mu)/sd).clip(-8,8)
        if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():
            raise ValueError("Nonfinite feature "+name)
        np.save(DATA/(name+".npy"),value,allow_pickle=False)
    np.savez(DATA/"fit_normalization.npz",**scales)
    save(OUT/"release_protocol.json",dict(roles=roles,records=len(catalog),environments=int(catalog.environment_id.nunique()),
          hand_frames=len(meta),speed_cut=speed_cut,qualification_cut=eligible_cut,
          inputs=["Raw_Data/EgoDemo.zip","Raw_Data/EgoPro.zip","Raw_Data/EgoStandard.zip","record_split.csv","source_segment_index.csv (record identifiers only)"],
          common_scoring="Every 30 frames from origin 10; all 30 reference wrist frames eligible. Whole-record bone medians are used exclusively to grade targets.",
          label="Reliability head estimates the released hand-quality flag from geometry and time. No clean unobserved wrist positions are provided.",
          prior_feature_caches_read=0,prior_checkpoints_read=0,kalman_fit_runs=0))
    print("Independent release decode:",len(catalog),"records,",len(meta),"hand-frames, speed threshold",speed_cut,flush=True)


if __name__=="__main__":
    prepare()
