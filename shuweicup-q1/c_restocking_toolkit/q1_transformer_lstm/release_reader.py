"""Read the three competition packages directly. The forward API can stop at a recorded prefix."""
from dataclasses import dataclass
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from study_paths import RAW,TABLES

SIDES = ("left","right")
QUALITY = {"ok":0,"occlusion":1,"near edge":2}


def manifest():
    records = pd.read_csv(TABLES/"record_split.csv")
    ids = pd.read_csv(TABLES/"source_segment_index.csv")[["dataset","sample","episode_uuid"]].drop_duplicates()
    out = records.merge(ids,on=["dataset","episode_uuid"],how="left",validate="one_to_one")
    if out["sample"].isna().any() or out.groupby("environment_id").split.nunique().max()!=1:
        raise ValueError("The release manifest is incomplete or has overlapping environment splits.")
    return out


@dataclass
class Record:
    t: np.ndarray
    joints: np.ndarray       # frame, hand, anatomical joint, xyz
    head: np.ndarray
    quat: np.ndarray         # xyzw, head to world
    flags: np.ndarray        # frame, hand

    def prefix(self,n):
        return Record(self.t[:n],self.joints[:n],self.head[:n],self.quat[:n],self.flags[:n])


def quality_code(value):
    if value is None or not str(value).strip():
        return 0
    return QUALITY.get(str(value).lower().strip(),3)


def _demo(archive,base,limit):
    prefix = "observation.state."
    pos = [prefix+"hand_"+side+"_world" for side in SIDES]
    hp,hq = prefix+"head_world",prefix+"head_world_rotation"
    members = [n for n in archive.namelist() if n.startswith(base+"data/") and n.endswith(".parquet")]
    if len(members)!=1:
        raise ValueError("Unexpected number of raw episode files")
    with archive.open(members[0]) as stream:
        f = pq.ParquetFile(BytesIO(stream.read()))
        cols = ["timestamp","frame_index",*pos,hp,hq]
        batch = next(f.iter_batches(batch_size=min(limit or f.metadata.num_rows,f.metadata.num_rows),columns=cols))
        raw = batch.to_pydict()
    n = len(raw["frame_index"])
    np.testing.assert_array_equal(raw["frame_index"],np.arange(n))
    layouts = json.loads(archive.read(base+"meta/info.json"))["features"]
    names = [str(x).lower().removeprefix("q") for x in layouts[hq]["names"][-1]]
    quat = np.asarray(raw[hq],np.float32).reshape(n,4)[:,[names.index(c) for c in "xyzw"]]
    flags = np.zeros((n,2),np.uint8)
    report = json.loads(archive.read(base+"bad_frame_ratio.json"))
    for entry in report["hand_bad_frame_details"]:
        frame = int(entry["frame"])
        if 0<=frame<n:
            flags[frame] = [quality_code(entry.get(side+"_bad_reason")) for side in SIDES]
    return Record(np.asarray(raw["timestamp"],float),np.stack([np.asarray(raw[k],np.float32) for k in pos],1),
                  np.asarray(raw[hp],np.float32).reshape(n,3),quat,flags)


def _csv(archive,base,limit):
    def get(name,columns=None):
        with archive.open(base+"tables/"+name+".csv") as stream:
            return pd.read_csv(stream,usecols=columns,nrows=limit,float_precision="round_trip")
    joint_cols = [f"transforms[{j}].pos.{c}" for j in range(21) for c in "xyz"]
    clock = "_log_time_ns"
    t_col = "_offset_from_first_log_s"
    hands = [get("pose_"+s+"_hand",[clock,t_col,*joint_cols]) for s in SIDES]
    stamps = hands[0][clock].to_numpy()
    np.testing.assert_array_equal(stamps,hands[1][clock])
    n = len(stamps)
    positions = np.stack([x[joint_cols].to_numpy(np.float32).reshape(n,21,3) for x in hands],1)
    head_cols = [f"transforms[0].pos.{c}" for c in "xyz"]
    quat_cols = [f"transforms[0].quat.{c}" for c in "xyzw"]
    poses = get("pose_head",[clock,*head_cols,*quat_cols]).set_index(clock).reindex(stamps)
    quality = get("annotation_bad_frame_pose_hand").set_index(clock).reindex(stamps)
    flags = np.zeros((n,2),np.uint8)
    for i in np.flatnonzero(quality.is_bad.fillna(False).to_numpy(bool)):
        for c in quality.filter(regex=r"^problem_type\[").columns:
            value = quality[c].iat[i]
            if isinstance(value,str) and ":" in value:
                side,reason = value.split(":",1)
                if side.strip() in SIDES:
                    flags[i,SIDES.index(side.strip())] = quality_code(reason)
    return Record(hands[0][t_col].to_numpy(float),positions,poses[head_cols].to_numpy(np.float32),
                  poses[quat_cols].to_numpy(np.float32),flags)


def load_record(dataset,sample,limit=None):
    if limit is not None and limit<1:
        raise ValueError("A nonempty measured prefix is required")
    if dataset not in ("EgoDemo","EgoPro","EgoStandard") or "/" in sample or "\\" in sample:
        raise ValueError("Unknown released record")
    with ZipFile(RAW/(dataset+".zip")) as archive:
        fn = _demo if dataset=="EgoDemo" else _csv
        record = fn(archive,dataset+"/"+sample+"/",limit)
    if np.any(np.diff(record.t)<=0) or (limit is not None and len(record.t)!=limit):
        raise ValueError("Requested a missing frame or a non-increasing clock")
    return record
