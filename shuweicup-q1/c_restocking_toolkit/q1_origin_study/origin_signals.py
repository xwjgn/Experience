"""Observable hand and head motion through the issue frame, with no learned state or future labels."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q1_transformer_lstm"))
from release_reader import SIDES, load_record
from study_paths import DATA, KEY, OUT as M4_DIR, ROOT

OUT = ROOT / "分析报告" / "第一问起点转向与收紧实测"
SUBSETS = {"wrist":("w",), "wrist_hand":("w","f"), "wrist_head":("w","h"),
           "wrist_hand_head":("w","f","h")}


def carried(x, default):
    x = np.asarray(x, float)
    valid = np.isfinite(x).all(1)
    last = np.maximum.accumulate(np.where(valid, np.arange(len(x)), -1))
    return np.where((last >= 0)[:, None], x[last.clip(min=0)], default), valid


def length(x):
    return np.linalg.norm(x, axis=-1)


def cosine(x, y):
    return np.clip(np.sum(x*y, axis=-1) / np.maximum(length(x)*length(y), 1.e-8), -1., 1.)


def origin_features(raw, hand):
    side = SIDES.index(hand)
    t, joints = raw.t, raw.joints
    i = np.arange(len(t))
    wrist, good = carried(joints[:, side, 0], np.zeros(3))
    other, _ = carried(joints[:, 1-side, 0], np.zeros(3))
    head, _ = carried(raw.head, np.zeros(3))
    quat = raw.quat.copy()
    quat[length(quat) < .001] = np.nan
    quat, _ = carried(quat, np.array([0.,0.,0.,1.]))
    orientation = Rotation.from_quat(quat)
    matrices = orientation.as_matrix()
    data = {}
    def add(group, name, values):
        if values.ndim == 1:
            data[group+"_"+name] = values
        else:
            for j in range(values.shape[1]):
                data[f"{group}_{name}_{j}"] = values[:,j]
    def vel(series, start, end):
        dt = np.maximum(t[end]-t[start], .001)
        return (series[end]-series[start]) / dt[:,None]
    gaps = ~good | (raw.flags[:,side] != 0)
    add("w", "flag", gaps)
    add("w", "recent_flags", pd.Series(gaps).rolling(10,min_periods=1).mean().to_numpy())
    path = np.r_[0., np.cumsum(length(np.diff(wrist, axis=0)))]
    tips = np.stack([carried(joints[:,side,j],np.zeros(3))[0] for j in (4,8,12,16,20)], axis=1)
    relative = np.einsum("ij,ijk->ik",wrist-head,matrices)
    relative[:,0] *= -1. if side else 1.
    add("h","wrist_head_position",relative)
    add("f","wrist_distance",length(other-wrist))
    spread = length(tips[:,0]-tips[:,1])
    opening = length(tips-wrist[:,None]).mean(1)
    add("f","thumb_index",spread)
    add("f","opening",opening)
    for lag in (3,5,10,20):
        earlier = np.maximum(i-lag,0)
        previous = np.maximum(i-2*lag,0)
        v, last = vel(wrist,earlier,i), vel(wrist,previous,earlier)
        speed, old_speed = length(v), length(last)
        add("w",f"speed_{lag}",speed)
        add("w",f"speed_change_{lag}",speed-old_speed)
        add("w",f"velocity_change_{lag}",length(v-last))
        add("w",f"heading_cos_{lag}",cosine(v,last))
        add("w",f"straightness_{lag}",length(wrist-wrist[earlier])/np.maximum(path-path[earlier],1.e-7))
        add("w",f"vertical_fraction_{lag}",v[:,2]/np.maximum(speed,.01))
        vother = vel(other,earlier,i)
        add("f",f"other_speed_{lag}",length(vother))
        add("f",f"both_v_cos_{lag}",cosine(v,vother))
        dt = np.maximum(t-t[earlier],.001)
        add("f",f"aperture_change_{lag}",(spread-spread[earlier])/dt)
        add("f",f"opening_change_{lag}",(opening-opening[earlier])/dt)
        add("f",f"other_distance_change_{lag}",(length(other-wrist)-length(other[earlier]-wrist[earlier]))/dt)
        vh = vel(head,earlier,i)
        last_head = vel(head,previous,earlier)
        omega = (orientation[earlier].inv()*orientation).as_rotvec()/dt[:,None]
        prev_dt = np.maximum(t[earlier]-t[previous],.001)
        last_omega = (orientation[previous].inv()*orientation[earlier]).as_rotvec()/prev_dt[:,None]
        add("h",f"head_speed_{lag}",length(vh))
        add("h",f"angular_speed_{lag}",length(omega))
        add("h",f"angular_change_{lag}",length(omega-last_omega))
        add("h",f"relative_speed_{lag}",length(v-vh))
        add("h",f"head_precedes_hand_{lag}",cosine(v,last_head))
        add("h",f"relative_vel_{lag}",(relative-relative[earlier])/dt[:,None])
    return pd.DataFrame(data,dtype=np.float32)


def columns(frame, name):
    return [c for c in frame if c.startswith(tuple(group+"_" for group in SUBSETS[name]))]


def cohort(split_names):
    meta = pd.read_parquet(DATA/"windows.parquet")
    meta = meta[meta.split.isin(split_names)]
    stages = np.load(DATA/"stage.npy",mmap_mode="r")
    now = np.load(DATA/"now.npy",mmap_mode="r")
    eligible = np.load(DATA/"eligible.npy",mmap_mode="r")
    chunks = []
    for (dataset,sample), part in meta.groupby(["dataset","sample"], sort=False):
        raw = load_record(dataset,sample)
        for hand, origins in part.groupby("hand",sort=False):
            stride = 3 if origins.split.iat[0] == "train" else 30
            idx = origins.index.to_numpy()
            select = (origins.origin.to_numpy() >= 10) & ((origins.origin.to_numpy()-10) % stride == 0)
            use = origins.iloc[np.flatnonzero(select & eligible[idx].all(axis=1))].copy()
            assert not len(use) or (use.origin.max()+30 < len(raw.t))
            feat = origin_features(raw,hand).iloc[use.origin.to_numpy()].copy()
            feat.index = use.index
            use["row"] = use.index
            use["turn"] = stages[use.index] == 1
            use["origin_stage"] = now[use.index]
            chunks.append(pd.concat([use,feat],axis=1))
    return pd.concat(chunks).sort_index()
