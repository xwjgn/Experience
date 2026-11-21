"""Shared causal inputs, physical target masks and aggregation for the first question."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from video_inputs import KIT, ROOT, CACHE, OUT
sys.path.insert(0, str(KIT))
import q1_improve as predictors
import q1_improve2 as reliability
import q1_round3 as intervals
from q1_state import POINTS

WAYPOINTS = (10, 20, 30)
SCALE = 0.20
SPLITS = ("train", "validation", "test")
PALM_JOINTS = POINTS["palm_center"]
PALM_TARGETS = ("palm_midpoint", "middle_midpoint", "area_center", "combined_center")


def palm_geometry(joints):
    """Landmarks in the order wrist, index, middle, ring and little MCP."""
    wrist = joints[:, 0]
    roots = joints[:, 1:] - wrist[:, None]
    root_mean = roots.mean(axis=1)
    midpoint = .5 * root_mean
    middle_midpoint = .5 * roots[:, 1]
    # The 3D fan has three triangles sharing the wrist. Twice the areas suffice.
    twice_area = np.linalg.norm(np.cross(roots[:, :-1], roots[:, 1:]), axis=2)
    total = twice_area.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(total) & (total > 0)):
        raise ValueError("Palm fan needs a finite, positive surface area.")
    area_center = ((roots[:, :-1] + roots[:, 1:]) * twice_area[:, :, None]).sum(axis=1) / (3 * total)
    # Average the five proposed locations, including the wrist (zero offset).
    combined = (.8 * root_mean + midpoint + middle_midpoint + area_center) / 5
    return {"palm_midpoint": wrist + midpoint, "middle_midpoint": wrist + middle_midpoint,
            "area_center": wrist + area_center, "combined_center": wrist + combined}


def rotation_matrices(g):
    q = g[["head_qx", "head_qy", "head_qz", "head_qw"]].to_numpy(float)
    return Rotation.from_quat(q).as_matrix()


def local(v, rot):
    return np.einsum("n...j,nji->n...i", v, rot)


def palm_history(points, rot, mode):
    """Current wrist-to-palm vector and its past changes in the current head axes."""
    if mode in ("zero", "multi_zero"):
        return np.zeros((len(rot), 12 if mode == "zero" else 36), dtype=np.float32)
    if mode == "measured":
        offset = points["palm_center"] - points["wrist"]
    elif mode == "multi":
        # These three vectors and the wrist reconstruct all five positions exactly:
        # the original five-point centre is W + 1.6 * (palm_midpoint - W).
        names = ("palm_midpoint", "middle_midpoint", "area_center")
        offset = np.stack([points[name] - points["wrist"] for name in names], axis=1)
    else:
        raise ValueError(f"Unknown palm context: {mode}")
    idx = np.arange(len(offset))
    values = [local(offset, rot).reshape(len(offset), -1)]
    for lag in (3, 9, 24):
        past = np.maximum(0, idx-lag)
        values.append(local(offset-offset[past], rot).reshape(len(offset), -1))
    return np.column_stack(values).astype(np.float32)


class MotionData:
    def __init__(self, visual=False, point="wrist", palm_context=None):
        self.point = point
        self.point_joints = PALM_JOINTS if point in PALM_TARGETS else POINTS[point]
        self.palm_context = palm_context
        canonical = KIT / "cache/canonical"
        self.tracks = reliability.load2(canonical, ROOT / "Restocking_Problem_and_Data_EN")
        trmap = {(r["sample"], r["hand"]): r for r in self.tracks}
        headcols = ["head_x", "head_y", "head_z", "head_qx", "head_qy", "head_qz", "head_qw"]
        extra_geometry = point in PALM_TARGETS or palm_context == "multi"
        handcols = {side: [f"{side}_j{j:02d}_{axis}" for j in PALM_JOINTS for axis in "xyz"]
                    for side in ("left", "right")} if extra_geometry else {}
        columns = ["sample", "frame_index", *headcols, *(col for cols in handcols.values() for col in cols)]
        frames = pd.read_parquet(canonical / "frames.parquet", columns=columns)
        groups = {s: g.sort_values("frame_index") for s, g in frames.groupby("sample", sort=False)}
        if extra_geometry:
            for tr in self.tracks:
                group = groups[tr["sample"]]
                joints = group[handcols[tr["hand"]]].to_numpy(float).reshape(-1, len(PALM_JOINTS), 3)
                tr["points"].update(palm_geometry(joints))
                rebuilt = tr["points"]["wrist"] + 1.6 * (tr["points"]["palm_midpoint"] - tr["points"]["wrist"])
                np.testing.assert_allclose(rebuilt, tr["points"]["palm_center"], atol=1e-12, rtol=0)
        self.parts, self.index = [], {}
        fields = {}
        self.images = None
        if visual:
            from sklearn.feature_extraction.text import HashingVectorizer
            texts = HashingVectorizer(n_features=64, alternate_sign=False, ngram_range=(1, 2))
            images = [np.zeros((1, 48, 64, 3), dtype=np.uint8)]
            vindex, count = {}, 1
            for sample, g in groups.items():
                with np.load(CACHE / (sample + ".npz")) as f:
                    rgb, timestamps, task = f["rgb"], f["t"], str(f["task"])
                t = trmap[(sample, "left")]["t"]
                cur = np.searchsorted(timestamps, t, side="right") - 1
                assert np.all(timestamps[cur[cur >= 0]] <= t[cur >= 0])
                prev = cur - 1
                vindex[sample] = (np.stack([np.where(cur >= 0, count + cur, 0),
                                             np.where(prev >= 0, count + prev, 0)], axis=1),
                                  texts.transform([task]).toarray()[0])
                images.append(rgb)
                count += len(rgb)
            self.images = np.concatenate(images)
        start = 0
        for tr in self.tracks:
            sample, hand = tr["sample"], tr["hand"]
            other = trmap[(sample, "left" if hand == "right" else "right")]
            g = groups[sample]
            z, opposite = tr["points"][point], other["points"][point]
            head = g[headcols[:3]].to_numpy(float)
            rot, time = rotation_matrices(g), tr["t"]
            n = len(z)
            idx = np.arange(n)
            lags = np.maximum(0, idx[:, None] - np.array([1, 3, 6, 10, 20, 30]))
            hist_idx = np.maximum(0, idx[:, None] - np.arange(10, -1, -1))
            history = local(z[hist_idx] - z[:, None], rot)
            steps = np.maximum(np.diff(time[hist_idx], axis=1), 1e-4)
            hands = np.full(n, float(hand == "right"))
            context = [local(z - head, rot), local(opposite - head, rot), hands[:, None]]
            for points in (z, opposite, head):
                context.append(local(points[:, None] - points[lags], rot).reshape(n, -1))
            heading = rot[:, :, 0]
            context.append(local(heading[:, None] - heading[lags], rot).reshape(n, -1))
            for b in (tr["bad"], other["bad"]):
                context.append(np.column_stack([b, b[lags]]))
            context.append(np.asarray(time[:, None] - time[lags]))
            if palm_context is not None:
                context.append(palm_history(tr["points"], rot, palm_context))
            tix = idx[:, None] + np.asarray(WAYPOINTS)
            safe = tix.clip(max=n - 1)
            qualified = (tix < n) & tr["qualified"][safe]
            target = local(z[safe] - z[:, None], rot) / SCALE
            head_target = local(head[safe] - head[:, None], rot) / SCALE
            cnt = np.r_[0, np.cumsum(tr["bad"])]
            clean = (idx >= 9) & (idx + 10 < n) & (cnt[idx + 1] == cnt[np.maximum(0, idx - 9)])
            flag_next = cnt[np.minimum(n, idx + 11)] > cnt[idx + 1]
            valid_origin = (idx >= predictors.MIN_ORIGIN) & np.isfinite(tr["speed5"])
            values = dict(x=np.column_stack(context).astype(np.float32),
                          hist=(history / SCALE).astype(np.float32), dt=steps.astype(np.float32),
                          hist_flags=tr["bad"][hist_idx].astype(np.float32),
                          y=target.astype(np.float32), y_head=head_target.astype(np.float32),
                          mask=qualified & valid_origin[:, None], head_mask=(tix < n) & valid_origin[:, None],
                          alert_mask=clean, alert_y=flag_next.astype(np.float32),
                          split=np.full(n, SPLITS.index(tr["split"]), np.uint8))
            if visual:
                vids, text = vindex[sample]
                values.update(video=vids.astype(np.int32), text=np.broadcast_to(text, (n, len(text))).astype(np.float32))
            for key, v in values.items():
                fields.setdefault(key, []).append(v)
            self.parts.append((tr, start, start + n))
            self.index[sample, hand] = slice(start, start + n)
            start += n
        self.a = {k: np.concatenate(v) for k, v in fields.items()}
        train = self.a["split"] == 0
        mean, sd = self.a["x"][train].mean(0), self.a["x"][train].std(0)
        self.a["x"] = np.clip((self.a["x"] - mean) / np.maximum(sd, 0.02), -9, 9)
        assert np.isfinite(self.a["x"]).all()

    def error_rows(self, local_prediction, scales=None):
        rows = []
        for tr, start, end in self.parts:
            if tr["split"] == "train":
                continue
            n = end - start
            o = np.arange(predictors.MIN_ORIGIN, n - 30)
            target = o + 30
            error = SCALE * np.linalg.norm(local_prediction[start + o] - self.a["y"][start + o, -1], axis=1)
            data = dict(sample=tr["sample"], hand=tr["hand"], environment_id=tr["environment_id"], split=tr["split"],
                        horizon=30, origin=o, target=target, err=error, qualified=tr["qualified"][target],
                        feedback_ok=~tr["bad"][target], speed=tr["speed5"][o], flag=tr["recent_flag"][o])
            if scales is not None:
                data["scale"] = scales[start + o]
            rows.append(pd.DataFrame(data))
        return pd.concat(rows, ignore_index=True)


def baseline(data):
    """Reproduce the published wrist-only gradient boosting comparator."""
    if data.point != "wrist":
        raise ValueError("The archived boosting comparator predicts the wrist.")
    reg = predictors.Learned("boosting").fit([tr for tr in data.tracks if tr["split"] == "train"], 30)
    rows = []
    for tr, _, _ in data.parts:
        if tr["split"] == "train":
            continue
        o = np.arange(predictors.MIN_ORIGIN, len(tr["t"]) - 30)
        target = o + 30
        err = np.linalg.norm(reg.predict(tr, o, 30) - tr["points"]["wrist"][target], axis=1)
        rows.append(pd.DataFrame(dict(sample=tr["sample"], hand=tr["hand"], environment_id=tr["environment_id"],
                     split=tr["split"], horizon=30, origin=o, target=target, err=err, qualified=tr["qualified"][target],
                     feedback_ok=~tr["bad"][target], speed=tr["speed5"][o], flag=tr["recent_flag"][o], scale=np.ones(len(o)))))
    return pd.concat(rows, ignore_index=True)


def point_tables(methods):
    by = pd.concat([x[x.qualified].assign(model=m) for m, x in methods.items()], ignore_index=True)
    env = by.groupby(["model", "split", "environment_id"]).err.agg(["size", "mean"]).reset_index()
    env["mean_cm"] = env.pop("mean") * 100
    pooled = by.groupby(["model", "split"]).err.agg(["size", "mean"]).reset_index()
    pooled["pooled_cm"] = pooled.pop("mean") * 100
    macro = env.groupby(["model", "split"]).mean_cm.mean().rename("macro_cm").reset_index()
    return pooled.merge(macro), env
