"""Extract as-of head-left images and project the recorded hands into their capture frame."""
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from prepare_object_detector import ROOT as OBJECT_CACHE
from video_inputs import CACHE, KIT, ROOT, STRIDE as THUMB_STRIDE, ffmpeg_executable


STRIDE = 12
FRAME_ROOT = OBJECT_CACHE / "frames12"
BOX_ROOT = OBJECT_CACHE / "boxes12"
GOODS_ROOT = OBJECT_CACHE / "goods12"
HANDS = ("left", "right")
JOINTS = (0, 5, 9, 13, 17)
DIMENSIONS = ("cam_fx", "cam_fy", "cam_cx", "cam_cy", "cam_width", "cam_height")


def record_table():
    return pd.read_csv(KIT / "cache/canonical/records.csv")


def capture_times(sample):
    assert STRIDE % THUMB_STRIDE == 0
    with np.load(CACHE / (sample + ".npz")) as data:
        return data["t"][::STRIDE//THUMB_STRIDE].copy()


def extract():
    FRAME_ROOT.mkdir(parents=True, exist_ok=True)
    exe = ffmpeg_executable()
    table = record_table()
    for dataset, records in table.groupby("dataset", sort=False):
        with zipfile.ZipFile(ROOT/"Restocking_Problem_and_Data_EN/Raw_Data"/(dataset+".zip")) as archive:
            for row in records.itertuples(index=False):
                folder = FRAME_ROOT / row.sample
                times = capture_times(row.sample)
                if (folder/"complete.json").is_file() and len(list(folder.glob("*.jpg"))) == len(times):
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                prefix = f"{dataset}/{row.sample}/"
                if dataset == "EgoDemo":
                    member, = [f for f in archive.namelist()
                               if f.startswith(prefix+"videos/observation.images.head_left/") and f.endswith(".mp4")]
                else:
                    member = prefix+"videos/head_left.mp4"
                with tempfile.TemporaryDirectory(dir=FRAME_ROOT) as tmp:
                    video = Path(tmp)/"left.mp4"
                    with archive.open(member) as source, video.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "3",
                                    "-i", str(video), "-an", "-vf", f"select=not(mod(n\\,{STRIDE})),scale=640:-2:flags=area",
                                    "-vsync", "0", "-start_number", "0", "-q:v", "3", str(folder/"%05d.jpg")], check=True)
                if len(list(folder.glob("*.jpg"))) != len(times):
                    raise ValueError(f"Image/timestamp mismatch for {row.sample}")
                (folder/"complete.json").write_text(json.dumps(dict(n=len(times), stride=STRIDE)), encoding="utf8")
                print(dataset, row.sample, len(times), flush=True)


def pose_table():
    positions = [f"{hand}_j{j:02d}_{axis}" for hand in HANDS for j in JOINTS for axis in "xyz"]
    cols = ["sample", "frame_index", "t_s", "head_qx", "head_qy", "head_qz", "head_qw",
            "cam_x", "cam_y", "cam_z", "cam_qx", "cam_qy", "cam_qz",
            "cam_qw", *DIMENSIONS, *positions]
    return pd.read_parquet(KIT/"cache/canonical/frames.parquet", columns=cols)


def projected_hands(poses, image_times, hypothetical_displacements=None):
    t = poses.t_s.to_numpy()
    idx = np.searchsorted(t, image_times, side="right")-1
    safe = np.maximum(0, idx)
    valid_time = (idx >= 0) & (image_times-t[safe] <= .12)
    assert (t[safe[valid_time]] <= image_times[valid_time]).all()
    g = poses.iloc[safe]
    cam = g[["cam_x", "cam_y", "cam_z"]].to_numpy()
    quat = g[["cam_qx", "cam_qy", "cam_qz", "cam_qw"]].to_numpy()
    finite_q = np.isfinite(quat).all(1) & (np.linalg.norm(quat, axis=1) > 1e-7)
    quat = quat.copy()
    quat[~finite_q] = (0., 0., 0., 1.)
    rot = Rotation.from_quat(quat).as_matrix()
    if hypothetical_displacements is not None:
        head_q = g[["head_qx", "head_qy", "head_qz", "head_qw"]].to_numpy(copy=True)
        valid_head = np.isfinite(head_q).all(1) & (np.linalg.norm(head_q, axis=1)>1e-7)
        head_q[~valid_head] = (0.,0.,0.,1.)
        world_offsets = np.einsum("nij,kj->nki",Rotation.from_quat(head_q).as_matrix(),hypothetical_displacements)
        valid_time = valid_time & valid_head
    fx, fy, cx, cy, width, height = (g[c].to_numpy()[:, None] for c in DIMENSIONS)
    pixels = {}
    for hand in HANDS:
        points = g[[f"{hand}_j{j:02d}_{a}" for j in JOINTS for a in "xyz"]].to_numpy().reshape(-1, len(JOINTS), 3)
        if hypothetical_displacements is not None:
            points = points[:, :1]+world_offsets
        p = np.einsum("nji,nkj->nki", rot, points-cam[:, None])
        z = p[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack(((fx*p[:, :, 0]/z+cx)/width, (fy*p[:, :, 1]/z+cy)/height), axis=2)
        uv[(z <= 0) | ~valid_time[:, None] | ~finite_q[:, None]] = np.nan
        pixels[hand] = uv
    return pixels


if __name__ == "__main__":
    extract()
