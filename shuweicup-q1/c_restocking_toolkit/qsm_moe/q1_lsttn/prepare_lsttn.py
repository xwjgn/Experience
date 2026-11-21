from collections import defaultdict
import numpy as np
import pandas as pd

from paths_q1 import DATA, FRAMES, OUT, save_json
from wrist_observations import STEPS, bone_deviation, local, measured, mode, velocity_one, xyz


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    frames = pd.read_parquet(FRAMES)
    train_envs = sorted(frames.loc[frames.split.eq("train"),"environment_id"].unique())
    order = np.random.default_rng(9317).permutation(train_envs).tolist()
    roles = dict(fit=order[:10],gate=order[10:13],cal=order[13:])
    tracks = []
    for sample,g in frames.groupby("sample",sort=False):
        g = g.sort_values("frame_index")
        np.testing.assert_array_equal(g.frame_index,np.arange(len(g)))
        for hand in ("left","right"):
            raw,geometry = measured(g,hand)
            t,z,_,_ = geometry
            tracks.append((sample,hand,g.environment_id.iat[0],g.split.iat[0],raw,geometry,
                           bone_deviation(g,hand),velocity_one(t,z),g[hand+"_quality"].eq("ok").to_numpy()))
    speed_cut = float(np.quantile(np.concatenate([s[np.isfinite(s)] for _,_,e,_,_,_,_,s,_ in tracks if e in roles["fit"]]),1/3))
    # This common geometry convention qualifies retrospective scoring targets only.
    qual_cut = float(np.quantile(np.concatenate([b[q] for _,_,_,sp,_,_,b,_,q in tracks if sp=="train"]),.995))
    arrays,keys = defaultdict(list),[]
    start = 0
    for sample,hand,env,split,raw,(t,z,rot,mir),bone,speed,quality_ok in tracks:
        n = len(t)
        o = np.arange(n)
        future = o[:,None]+STEPS
        safe = future.clip(max=n-1)
        qualified = quality_ok & (bone<=qual_cut)
        good = (future<n)&qualified[safe]&(o[:,None]>=10)&np.isfinite(speed[:,None])
        current = mode(t,z,speed_cut)
        target = np.full(n,-1,np.int8)
        target[:-10] = current[10:]
        vals = dict(**raw, start=np.full(n,start,dtype=np.int64),mode=target,current_mode=current.astype(np.int8),
                    y=local(z[safe]-z[:,None],rot,mir).astype("float32"),valid=good)
        for name,x in vals.items():
            arrays[name].append(x)
        keys.append(pd.DataFrame(dict(split=split,env=env,sample=sample,hand=hand,origin=o)))
        start += n
    meta = pd.concat(keys,ignore_index=True)
    fit = meta.env.isin(roles["fit"]).to_numpy()
    for key,parts in arrays.items():
        value = np.concatenate(parts)
        if key=="x":
            mu,sd = value[fit].mean(0),value[fit].std(0).clip(min=.003)
            np.savez(DATA/"normalization.npz",mu=mu,sd=sd)
            value = np.clip((value-mu)/sd,-7,7).astype("float32")
        if np.issubdtype(value.dtype,np.floating) and not np.isfinite(value).all():
            raise ValueError("Nonfinite "+key)
        np.save(DATA/(key+".npy"),value,allow_pickle=False)
    meta.to_parquet(DATA/"keys.parquet",index=False)
    save_json(OUT/"data_protocol.json",dict(roles=roles,rows=len(meta),records=int(meta["sample"].nunique()),
          mode_cut=speed_cut,eligible_bone_cut=qual_cut,old_weights_loaded=0,
          geometry_reference="whole-record medians used only for common retrospective eligibility",
          features="raw causal wrist, fingers, opposite wrist, head, reported quality and timestamps; mirrored local head coordinates",
          split="10 train environments fit weights, 3 train select checkpoints and initialize calibration, 3 train fit outer radius; 4 validation, 4 test"))
    print("Prepared independent raw measurements",len(meta),flush=True)


if __name__=="__main__":
    prepare()
