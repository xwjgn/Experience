"""Decode the released left head camera; preserve capture times and original task text."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KIT = Path(__file__).resolve().parents[1]
ROOT = KIT.parent
CACHE = KIT / "cache" / "literature_features"
OUT = ROOT / "分析报告" / "三种文献方法实测"
STRIDE = 6
WIDTH, HEIGHT = 64, 48


def ffmpeg_executable():
    if os.getenv("FFMPEG_BINARY"):
        return os.environ["FFMPEG_BINARY"]
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    sys.path.insert(0, str(CACHE / "decoder"))
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_record(archive, row, exe):
    dataset, sample = row.dataset, row.sample
    prefix = f"{dataset}/{sample}/"
    if dataset == "EgoDemo":
        info = json.loads(archive.read(prefix + "meta/info.json"))
        ep = pq.read_table(io.BytesIO(archive.read(prefix + "meta/episodes/chunk-000/file-000.parquet"))).to_pylist()[0]
        member, = [n for n in archive.namelist() if n.startswith(prefix + "videos/observation.images.head_left/") and n.endswith(".mp4")]
        fps = info["features"]["observation.images.head_left"]["info"]["video.fps"]
        times = np.arange(row.frames, dtype=float) / fps - ep["videos/observation.images.head_left/from_timestamp"]
        assert abs(ep["videos/observation.images.head_left/from_timestamp"]) < 1e-8
        task = " ".join(ep.get("tasks", []))
    else:
        member = prefix + "videos/head_left.mp4"
        info = {r["field"]: r["value"] for r in csv.DictReader(io.StringIO(archive.read(prefix + "session_metadata.csv").decode("utf-8-sig")))}
        task = " ".join(info.get(k, "") for k in ("task_info.task_name", "task_info.task_description")).strip()
        v = pd.read_csv(archive.open(prefix + "videos/head_left_frames.csv"), usecols=["_log_time_ns"])
        first = pd.read_csv(archive.open(prefix + "tables/pose_left_hand.csv"), nrows=1, usecols=["_log_time_ns", "_offset_from_first_log_s"])
        times = (v["_log_time_ns"].to_numpy(np.int64) - first["_log_time_ns"].iat[0]) / 1e9 + first["_offset_from_first_log_s"].iat[0]
    assert np.all(np.diff(times) > 0)
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        video = Path(temp) / "camera.mp4"
        with archive.open(member) as src, video.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        cmd = [exe, "-nostdin", "-v", "error", "-threads", "4", "-i", str(video), "-an", "-vf",
               f"select=not(mod(n\\,{STRIDE})),scale={WIDTH}:{HEIGHT}:flags=area", "-vsync", "0",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        raw = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    images = np.frombuffer(raw, np.uint8).reshape(-1, HEIGHT, WIDTH, 3)
    stamps = times[::STRIDE]
    if len(images) != len(stamps):
        raise ValueError(f"{sample}: {len(images)} decoded thumbnails for {len(stamps)} clock samples")
    return images, stamps, task


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    exe = ffmpeg_executable()
    meta = pd.read_csv(KIT / "cache/canonical/records.csv")
    summary = []
    for dataset, subset in meta.groupby("dataset"):
        with zipfile.ZipFile(ROOT / "Restocking_Problem_and_Data_EN/Raw_Data" / (dataset + ".zip")) as archive:
            for row in subset.itertuples(index=False):
                path = CACHE / (row.sample + ".npz")
                if not path.exists():
                    images, stamps, task = read_record(archive, row, exe)
                    tmp = path.with_suffix(".working.npz")
                    np.savez_compressed(tmp, rgb=images, t=stamps, task=task)
                    tmp.replace(path)
                with np.load(path) as features:
                    summary.append(dict(sample=row.sample, split=row.split, source=dataset, frames=row.frames,
                                        video_samples=len(features["t"]), first_video_s=features["t"][0],
                                        last_video_s=features["t"][-1], task=str(features["task"])))
                print(f"{len(summary)}/{len(meta)} {dataset} {row.sample}", flush=True)
    pd.DataFrame(summary).to_csv(OUT / "video_alignment.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
