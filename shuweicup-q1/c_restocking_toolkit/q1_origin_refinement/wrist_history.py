"""Causal palm motion and delayed extrapolation errors for the wrist-origin study."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
_search_path = list(sys.path)
try:
    sys.path.insert(0, str(HERE.parent/"q1_origin_study"))
    from origin_signals import DATA, M4_DIR, ROOT, carried, cosine, length, origin_features
    from release_reader import SIDES, load_record
    from hand_features import stage
finally:
    sys.path[:] = _search_path

OUT = ROOT/"分析报告"/"第一问转向预警与分步半径改进"
OLD = ROOT/"分析报告"/"第一问起点转向与收紧实测"
KEY = ["split","env","sample","hand","origin"]
BANDS = ((0,3),(3,10),(10,20),(20,30))
STEPS = np.arange(1,31)
OFFSETS = np.arange(22,-1,-2)
SEEDS = (91431,91437)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def protect_weights():
    def audit(event,args):
        if event != "open" or not isinstance(args[0],(str,bytes)):
            return
        path = Path(os.fsdecode(args[0]))
        flags = args[2] or 0
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            absolute = path.resolve()
            if absolute.is_relative_to(ROOT) and not (absolute.is_relative_to(OUT) or absolute.is_relative_to(HERE)):
                raise RuntimeError(f"Output outside this study: {absolute}")
        if path.suffix.lower() in {".pt",".pth",".ckpt",".safetensors",".joblib",".pkl"}:
            path = path.resolve()
            if path.is_relative_to(ROOT) and not path.is_relative_to(OUT):
                raise RuntimeError(f"Unexpected prior checkpoint {path}")
    sys.addaudithook(audit)


def unit(x):
    return x / np.maximum(length(x),1.e-7)[:,None]


def trailing(x, width, statistic):
    rolling = pd.Series(x).rolling(width,min_periods=1)
    if statistic == "median":
        return rolling.median().to_numpy()
    if statistic == "mean":
        return rolling.mean().to_numpy()
    return rolling.quantile(.9).to_numpy()


@dataclass
class Motion:
    tab: pd.DataFrame
    tokens: np.ndarray
    phase: np.ndarray
    quality: np.ndarray
    wrist: np.ndarray
    head: np.ndarray
    rotation: Rotation
    velocity: np.ndarray
    t: np.ndarray
    mirror: np.ndarray


def observe(raw, hand, speed_cut):
    side = SIDES.index(hand)
    t = raw.t
    n = len(t)
    i = np.arange(n)
    z, seen = carried(raw.joints[:,side,0],np.zeros(3))
    opposite,_ = carried(raw.joints[:,1-side,0],np.zeros(3))
    head,_ = carried(raw.head,np.zeros(3))
    quat = raw.quat.copy()
    quat[length(quat)<.001] = np.nan
    quat,_ = carried(quat,np.array([0.,0.,0.,1.]))
    rot = Rotation.from_quat(quat)
    mat = rot.as_matrix()
    mirror = np.array([-1. if side else 1.,1.,1.])
    def local(vector):
        return np.einsum("ij,ijk->ik",vector,mat)*mirror
    def measured(j):
        return carried(raw.joints[:,side,j],z[0])[0]
    def velocity(x, lag):
        old = np.maximum(i-lag,0)
        return np.clip((x-x[old])/np.maximum(t-t[old],.001)[:,None],-6.,6.)

    tab = origin_features(raw,hand)
    extras = {}
    phase = stage(t,z,speed_cut).astype(np.int64)
    change = np.r_[True,np.diff(phase)!=0]
    last = np.maximum.accumulate(np.where(change,i,0))
    extras["stage_age"] = np.minimum(t-t[last],2.)
    for state, name in enumerate(("dwell","turn","translate")):
        extras["current_"+name] = phase==state
        for w in (10,30):
            extras[f"past_{name}_{w}"] = trailing((phase==state).astype(float),w,"mean")

    forward = unit(measured(9)-z)
    normal = unit(np.cross(measured(5)-z,measured(17)-z))
    spread = unit(np.cross(forward,normal))
    palm = np.stack((forward,spread,normal),axis=2)
    for lag in (3,7,15):
        earlier = np.maximum(i-lag,0)
        axes = np.einsum("ij,ijk->ik",velocity(z,lag),palm)
        for j, name in enumerate(("forward","side","normal")):
            extras[f"palm_velocity_{lag}_{name}"] = axes[:,j]
        extras[f"palm_rotation_{lag}"] = np.arccos(np.clip(cosine(normal,normal[earlier]),-1.,1.))
        extras[f"palm_flex_{lag}"] = np.arccos(np.clip(cosine(forward,forward[earlier]),-1.,1.))

    good = seen & (raw.flags[:,side]==0)
    for lag in (1,3,10,20,30):
        old = np.maximum(i-lag,0)
        delta = (t-t[old])[:,None]
        predicted = z[old]+delta*velocity(z, max(3,min(lag,8)))[old]
        miss = length(predicted-z)
        # Update only with a measured endpoint and an observed preceding velocity interval.
        support = pd.Series(good).rolling(lag+max(3,min(lag,8))+1,min_periods=1).min().to_numpy(bool)
        measured_miss = np.where((i>=lag+max(3,min(lag,8))) & support,miss,np.nan)
        series = pd.Series(measured_miss)
        latest = series.ffill().fillna(0.).to_numpy()
        history = series.rolling(30,min_periods=1)
        typical = history.median().ffill().fillna(0.).to_numpy()
        upper = history.quantile(.9).ffill().fillna(0.).to_numpy()
        extras[f"innovation_{lag}"] = latest
        extras[f"noise_{lag}"] = upper
        extras[f"innovation_ratio_{lag}"] = np.clip(latest/(typical+.0008),0,10)
    v = velocity(z,5)
    old3 = np.maximum(i-3,0)
    ang = (rot[old3].inv()*rot).as_rotvec()/np.maximum(t-t[old3],.001)[:,None]
    tokens = np.column_stack([local(velocity(z,k)) for k in (2,5,12)] +
        [local(velocity(head,5)),ang.clip(-8,8),local(velocity(opposite,5)),local(z-head),
         local(forward),local(normal),local(measured(8)-measured(4)),
         extras["innovation_3"],extras["noise_10"],good, np.eye(3)[phase]])
    features = pd.concat([tab,pd.DataFrame(extras,dtype=np.float32)],axis=1)
    return Motion(features,tokens.astype(np.float32),phase,good,z,head,rot,v,t,mirror)


def sequence(motion, origins):
    idx = np.maximum(origins[:,None]-OFFSETS,0)
    age = motion.t[origins,None]-motion.t[idx]
    available = origins[:,None] >= OFFSETS
    return np.concatenate((motion.tokens[idx],age[:,:,None],available[:,:,None]),axis=2).astype(np.float32)


def training_targets(m, origins):
    future = origins[:,None]+STEPS
    elapsed = m.t[future]-m.t[origins,None]
    displacement = m.wrist[future]-m.wrist[origins,None]
    error = length(displacement-m.velocity[origins,None]*elapsed[:,:,None])
    # Learn how uncertain a past-velocity extrapolation is, in metres per second of lead time.
    proxy = np.column_stack([(error[:,lo:hi]/elapsed[:,lo:hi]).max(axis=1) for lo,hi in BANDS])
    aux = []
    axes = m.rotation[origins].as_matrix()
    def local(delta):
        return np.einsum("ni,nij->nj",delta,axes)*m.mirror
    for h in (5,10,20):
        aux.append(local(m.wrist[origins+h]-m.wrist[origins]))
    for h in (5,10,20):
        aux.append(local(m.head[origins+h]-m.head[origins]))
        aux.append((m.rotation[origins].inv()*m.rotation[origins+h]).as_rotvec())
    future_turn = m.phase[origins[:,None]+np.array([5,10,20])]==1
    return future_turn.astype(np.float32),np.column_stack(aux).astype(np.float32),np.log(proxy.clip(.001,8)).astype(np.float32)


@dataclass
class Cohort:
    meta: pd.DataFrame
    features: pd.DataFrame
    seq: np.ndarray
    events: np.ndarray
    movement: np.ndarray
    proxy: np.ndarray

    def take(self, rows):
        index = np.flatnonzero(rows) if np.asarray(rows).dtype==bool else rows
        return Cohort(self.meta.iloc[index].reset_index(drop=True),self.features.iloc[index].reset_index(drop=True),
                      self.seq[index],self.events[index],self.movement[index],self.proxy[index])


def load_split(split):
    proto = read(M4_DIR/"release_protocol.json")
    meta = pd.read_parquet(DATA/"windows.parquet")
    eligible = np.load(DATA/"eligible.npy",mmap_mode="r")
    expected = np.load(DATA/"stage.npy",mmap_mode="r")
    chunks = []
    step = 3 if split=="train" else 30
    for (dataset,sample),rows in meta[meta.split==split].groupby(["dataset","sample"],sort=False):
        raw = load_record(dataset,sample)
        for side,part in rows.groupby("hand",sort=False):
            possible = (part.origin>=10) & ((part.origin-10)%step==0) & eligible[part.index].all(1)
            rows_hand = part.loc[possible].copy()
            if rows_hand.empty:
                continue
            rows_hand["row"] = rows_hand.index
            ids = rows_hand.origin.to_numpy()
            m = observe(raw,side,proto["speed_cut"])
            y,movement,proxy = training_targets(m,ids)
            np.testing.assert_array_equal(y[:,1], expected[rows_hand.index]==1)
            rows_hand["stage"] = m.phase[ids]
            rows_hand["turn"] = y[:,1].astype(bool)
            chunks.append((rows_hand,m.tab.iloc[ids],sequence(m,ids),y,movement,proxy))
    columns = list(zip(*chunks))
    return Cohort(pd.concat(columns[0],ignore_index=True),pd.concat(columns[1],ignore_index=True),
                  *(np.concatenate(v) for v in columns[2:]))
