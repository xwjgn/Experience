"""Time-aligned causal inputs and the frozen 1-second wrist forecast for slow origins."""
import numpy as np
import pandas as pd
import torch

from bayesian_gain import GainFilter, SEED as GAIN_SEED, SAMPLES
from motion_data import KIT, ROOT, OUT as PRIOR_RESULTS, SCALE, MotionData, local, palm_history, rotation_matrices

RESULTS = ROOT / "分析报告" / "慢到快预测实验"
WINDOW = 60
STRIDE = 3
OFFSETS = np.arange(WINDOW, -1, -STRIDE)
J = (4, 5, 8, 17, 20)
STATE = ("slow", "middle", "fast")


def history_features(t, rot, head, wrist, other, fingers, bad, other_bad, right, cuts, anatomical_wrist=None):
    """Each row uses its own observations and a backward-looking window. Coordinates are head-local."""
    idx = np.arange(len(t))
    first = np.maximum(0, idx - 1)
    dt = np.maximum(t - t[first], 1e-4)
    dist = np.linalg.norm(wrist - wrist[first], axis=1)
    speed = dist / dt
    speed5 = pd.Series(speed).rolling(5, min_periods=1).mean().to_numpy()
    flags = np.asarray(bad, np.float32)
    opposite_flags = np.asarray(other_bad, np.float32)
    finger_origin = wrist if anatomical_wrist is None else anatomical_wrist
    normal = np.cross(fingers[:, 1] - finger_origin, fingers[:, 3] - finger_origin)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-7)
    # Thumb/index tip distance and index/little tip distance are direct hand-shape cues.
    opening = np.column_stack((np.linalg.norm(fingers[:, 0] - fingers[:, 2], axis=1),
                               np.linalg.norm(fingers[:, 2] - fingers[:, 4], axis=1)))
    values = [local(wrist-head, rot), local(other-head, rot), local(normal, rot),
              local(fingers-finger_origin[:, None], rot).reshape(len(t), -1), opening]
    for lag in (3, 9, 24):
        past = np.maximum(0, idx-lag)
        elapsed = np.maximum(t-t[past], 1e-4)[:, None]
        for pos in (wrist, other, head):
            values.append(local((pos-pos[past])/elapsed, rot))
        values.append(local((rot[:, :, 0]-rot[past, :, 0])/elapsed, rot))
        values.append((opening-opening[past])/elapsed)
        values.append(local((normal-normal[past])/elapsed, rot))
    # Do not represent short recordings as if their prehistory had been observed.
    age = t-t[0]
    last_fast = np.maximum.accumulate(np.where(speed5 >= cuts[1], idx, 0))
    last_non_slow = np.maximum.accumulate(np.where(speed5 >= cuts[0], idx, 0))
    meta = np.column_stack((flags, opposite_flags, speed5,
                            pd.Series(speed).rolling(15,min_periods=1).mean().to_numpy(),
                            pd.Series(speed).rolling(45,min_periods=1).mean().to_numpy(),
                            np.minimum(t-t[last_fast],3), np.minimum(t-t[last_non_slow],3),
                            np.minimum(age,3), np.full(len(t),float(right)), np.minimum(dt,.2)))
    values.append(meta)
    result = np.column_stack(values).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite causal motion inputs")
    return result


def incumbent_centres(data, eligible_train, device, model=None):
    """Reproduce the prior model's exact frozen mean. Confidence estimates are unused here."""
    keys = ("x", "hist", "dt", "hist_flags")
    a = {k: torch.as_tensor(data.a[k], device=device) for k in keys}
    if model is None:
        if data.point != "wrist" or data.palm_context is not None:
            raise ValueError("This representative point or input set requires its fitted gain filter.")
        info = torch.load(PRIOR_RESULTS/"bayesian_gain.pt", map_location=device, weights_only=True)
        assert info["seed"] == GAIN_SEED
        model = GainFilter(a["x"].shape[1]).to(device)
        model.load_state_dict(info["state"])
    model.eval()
    centre = np.zeros((len(data.a["x"]),3), np.float32)
    # Preserve the validation/test batching and MC draws of the published 17.52-cm model.
    for indices, seed in ((np.flatnonzero(data.a["split"] != 0), GAIN_SEED+1),
                          (eligible_train, GAIN_SEED+3)):
        torch.manual_seed(seed)
        with torch.no_grad():
            for batch in np.array_split(indices,int(np.ceil(len(indices)/1024))):
                ids = torch.as_tensor(batch,device=device)
                draws = [model(*(a[k][ids] for k in keys))[0] for _ in range(SAMPLES)]
                centre[batch] = torch.stack(draws).mean(0).cpu().numpy()
    return centre


class AccelerationData:
    def __init__(self, device, motion=None, gain_model=None):
        data = MotionData(visual=True) if motion is None else motion
        self.point, self.point_joints, self.palm_context = data.point, data.point_joints, data.palm_context
        self.cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])]
                                  for tr in data.tracks if tr["split"]=="train"]), [1/3,2/3])
        trmap = {(tr["sample"],tr["hand"]):tr for tr in data.tracks}
        cols = ["sample","frame_index","head_x","head_y","head_z","head_qx","head_qy","head_qz","head_qw"]
        cols += [f"{side}_j{j:02}_{axis}" for side in ("left","right") for j in J for axis in "xyz"]
        f = pd.read_parquet(KIT/"cache/canonical/frames.parquet", columns=cols)
        frames = {s:g.sort_values("frame_index") for s,g in f.groupby("sample",sort=False)}
        features, metadata, leftmost = [], [], []
        for tr, start, end in data.parts:
            sample, hand = tr["sample"], tr["hand"]
            opposite = trmap[sample,"right" if hand=="left" else "left"]
            g = frames[sample]
            finger_cols = [f"{hand}_j{j:02}_{axis}" for j in J for axis in "xyz"]
            obs = [tr["t"], rotation_matrices(g), g[["head_x","head_y","head_z"]].to_numpy(),
                   tr["points"][self.point], opposite["points"][self.point], g[finger_cols].to_numpy().reshape(-1,len(J),3),
                   tr["bad"], opposite["bad"]]
            anatomical_wrist = tr["points"]["wrist"]
            feat = history_features(*obs,hand=="right",self.cuts,anatomical_wrist)
            for cut in (11,min(95,len(g)-1)):
                prefix = history_features(*(x[:cut+1] for x in obs),hand=="right",self.cuts,anatomical_wrist[:cut+1])
                np.testing.assert_allclose(prefix,feat[:cut+1],atol=1e-6,rtol=1e-6)
            if data.palm_context is not None:
                feat = np.column_stack((feat, palm_history(tr["points"], obs[1], data.palm_context)))
            features.append(feat)
            leftmost.append(np.full(end-start,start,np.int64))
            future = np.full(len(g),-1, int)
            future[:-30] = np.digitize(tr["speed5"][30:],self.cuts)
            origin = np.arange(len(g))
            slow = (origin >= 10) & (tr["speed5"] < self.cuts[0])
            # This mask is post-hoc scoring eligibility only; gate and features use origin speed.
            score = data.a["mask"][start:end,-1]
            metadata.append(pd.DataFrame(dict(sample=sample,hand=hand,env=tr["environment_id"],split=tr["split"],
                            origin=origin,slow=slow,score=score,end_bin=future)))
        self.meta = pd.concat(metadata,ignore_index=True)
        self.start = np.concatenate(leftmost)
        self.meta.index.name = "row"
        self.train = self.meta.index[self.meta.split.eq("train") & self.meta.slow & self.meta.score].to_numpy(copy=True)
        self.base = incumbent_centres(data,self.train,device,gain_model)
        self.truth = data.a["y"][:,-1]
        self.sequence = np.concatenate(features)
        fit = self.sequence[data.a["split"]==0]
        self.feature_mean = fit.mean(0,dtype=np.float64).astype(np.float32)
        self.feature_sd = np.maximum(fit.std(0,dtype=np.float64),.01).astype(np.float32)
        self.sequence = np.clip((self.sequence-self.feature_mean)/self.feature_sd,-8,8)
        self.images = data.images
        # Current, approximately 0.5-s-old and 1-s-old head camera frames, each chosen as-of
        # the corresponding trajectory timestamp. MotionData verifies the capture-time condition.
        flat = np.arange(len(self.start))
        at = np.maximum(flat[:,None]-np.array([0,15,30]),self.start[:,None])
        self.image_index = data.a["video"][at,0] if data.images is not None else None
        self.end_class = self.meta.end_bin.to_numpy(copy=True)
        self.eval_ids = {s:self.meta.index[self.meta.split.eq(s) & self.meta.slow].to_numpy(copy=True)
                         for s in ("validation","test")}
        assert len(self.train)==28949 and int((self.end_class[self.train]==2).sum())==5892

    def score_ids(self, split):
        return self.meta.index[self.meta.split.eq(split) & self.meta.slow & self.meta.score].to_numpy()

    def sensor_history(self, ids):
        index = np.maximum(ids[:,None]-OFFSETS,self.start[ids,None])
        return self.sequence[index]

    def verify_baseline(self, split):
        if self.point != "wrist" or self.palm_context is not None:
            raise ValueError("The archived gain-error table describes the original wrist inputs and target.")
        old = pd.read_csv(PRIOR_RESULTS/"gain_error_by_speed.csv")
        ids = self.score_ids(split)
        err = np.linalg.norm(self.base[ids]-self.truth[ids],axis=1)*SCALE*100
        for cls in range(3):
            group = "slow->"+STATE[cls]
            expected = old[(old.split==split)&(old.group_type=="speed_origin_to_target")&(old.group==group)].iloc[0]
            selected = err[self.end_class[ids]==cls]
            assert len(selected) == expected.n
            np.testing.assert_allclose(selected.mean(), expected.mean_error_cm, atol=1e-4, rtol=0)
