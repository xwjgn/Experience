"""Causal wrist histories in the origin frame and aligned 30-step competition references."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
search_path=list(sys.path)
try:
    sys.path.insert(0,str(HERE.parent/"q1_origin_refinement"))
    from wrist_history import DATA,KEY,M4_DIR,OLD,ROOT,SIDES,load_record,observe,sequence
    from path_expert import frame_and_inputs,expert_points,fit_expert
finally:
    sys.path[:]=search_path
PREVIOUS=ROOT/"分析报告"/"第一问转向预警与分步半径改进"
OUT=ROOT/"分析报告"/"第一问贝叶斯增益与三维协方差"
LAGS=np.arange(15,-1,-1)
SEEDS=(62171,62189)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def isolate():
    def audit(event,args):
        if event!="open" or not isinstance(args[0],(str,bytes)):
            return
        p=Path(os.fsdecode(args[0]))
        model=p.suffix.lower() in {".pt",".pth",".ckpt",".pkl",".safetensors",".joblib"}
        write=(args[2] or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if model or write:
            p=p.resolve()
            if p.is_relative_to(ROOT) and not (p.is_relative_to(OUT) or (not model and p.is_relative_to(HERE))):
                raise RuntimeError(f"Access outside the independent Bayesian experiment: {p}")
    sys.addaudithook(audit)


@dataclass
class Inputs:
    # Positions and dynamic state are expressed in forward/lateral/normal axes fixed at issue time.
    history: np.ndarray
    delta: np.ndarray
    available: np.ndarray
    flag: np.ndarray
    context: np.ndarray
    cadence: np.ndarray
    axes: np.ndarray

    def take(self,idx):
        return Inputs(*(v[idx] for v in vars(self).values()))


def features_for(raw,m,hand,origins,seq,tab):
    axes,x,cad=frame_and_inputs(seq,tab)
    times=origins[:,None]-LAGS
    idx=times.clip(0)
    h=(m.wrist[idx]-m.wrist[origins,None])
    head_rotation=m.rotation[origins].as_matrix()
    local=np.einsum("nhj,nji->nhi",h,head_rotation)*m.mirror
    history=np.einsum("nhi,nij->nhj",local,axes)
    original=raw.joints[:,SIDES.index(hand),0]
    available=(times>=0) & np.isfinite(original[idx]).all(axis=2)
    flags=raw.flags[idx,SIDES.index(hand)]!=0
    dt=np.diff(m.t[idx],axis=1,prepend=m.t[idx[:,:1]])
    extra=tab[["w_recent_flags","noise_1","noise_3","noise_10","innovation_ratio_3","stage_age"]].to_numpy()
    return Inputs(history.astype("float32"),dt.astype("float32"),available,flags,np.column_stack((x,extra)).astype("float32"),
                  cad.astype("float32"),axes.astype("float32"))


@dataclass
class Windows:
    meta: pd.DataFrame
    features: pd.DataFrame
    seq: np.ndarray
    inputs: Inputs
    truth: np.ndarray

    def take(self,mask):
        ix=np.flatnonzero(mask) if np.asarray(mask).dtype==bool else mask
        return Windows(self.meta.iloc[ix].reset_index(drop=True),self.features.iloc[ix].reset_index(drop=True),
                       self.seq[ix],self.inputs.take(ix),self.truth[ix])


def dataset(split):
    catalog=pd.read_parquet(DATA/"windows.parquet")
    select=catalog[catalog.split==split]
    truth=np.load(DATA/"y.npy",mmap_mode="r")
    stage=np.load(DATA/"stage.npy",mmap_mode="r")
    eligible=np.load(DATA/"eligible.npy",mmap_mode="r")
    speed=read(M4_DIR/"release_protocol.json")["speed_cut"]
    stride=3 if split=="train" else 30
    parts=[]
    for (dataset_name,sample),record_rows in select.groupby(["dataset","sample"],sort=False):
        raw=load_record(dataset_name,sample)
        for hand,group in record_rows.groupby("hand",sort=False):
            valid=(group.origin>=10)&((group.origin-10)%stride==0)&eligible[group.index].all(1)
            meta=group.loc[valid].copy()
            if meta.empty:
                continue
            origins=meta.origin.to_numpy()
            m=observe(raw,hand,speed)
            seq=sequence(m,origins)
            tab=m.tab.iloc[origins]
            inp=features_for(raw,m,hand,origins,seq,tab)
            meta["row"]=meta.index
            meta["turn"]=stage[meta.index]==1
            meta["phase"]=m.phase[origins]
            parts.append((meta,tab,seq,inp,truth[meta.index]))
    a,b,c,d,e=zip(*parts)
    inputs=Inputs(*(np.concatenate([getattr(p,k) for p in d]) for k in vars(d[0])))
    return Windows(pd.concat(a,ignore_index=True),pd.concat(b,ignore_index=True),np.concatenate(c),inputs,np.concatenate(e))


def partition(all_train,role,grid=True):
    roles=read(M4_DIR/"release_protocol.json")["roles"]
    mask=all_train.meta.env.isin(roles[role])
    if grid:
        mask &= (all_train.meta.origin-10)%30==0
    return all_train.take(mask)
