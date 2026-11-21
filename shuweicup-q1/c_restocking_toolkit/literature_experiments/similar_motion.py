"""Retrieve same-hand motion histories from distinct training records."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from motion_data import KIT, PALM_TARGETS, SCALE, local, palm_geometry, rotation_matrices


LAGS_S = np.array([0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
BLOCK_SIZES = (6, 21, 21, 21, 42, 1)
SCALE_FLOORS = (0.03, 0.015, 0.015, 0.015, 0.04, 0.15)


def describe(t, wrist, other, head, rotation):
    # Use the same head axes for the entire history and its future displacement.
    assert np.all(np.diff(t) > 0)
    past = np.maximum(0, np.searchsorted(t, t[:, None] - LAGS_S, side="right") - 1)
    pose = local(np.stack((wrist - head, other - head), axis=1), rotation)
    blocks = [pose.reshape(len(t), 6)]
    for track in (wrist, other, head):
        motion = local(track[:, None] - track[past], rotation)
        blocks.append(motion.reshape(len(t), -1))
    axes = rotation[:, :, (0, 2)].transpose(0, 2, 1)
    turns = local(axes[:, None] - axes[past], rotation)
    blocks.append(turns.reshape(len(t), -1))
    # An early origin has a short real history even when its wrist barely moves.
    blocks.append(np.minimum(t - t[0], LAGS_S[-1])[:, None])
    return np.column_stack(blocks).astype(np.float32)


def history_descriptors(data):
    joints = getattr(data, "point_joints", (0,))
    hand_cols = {hand: [f"{hand}_j{j:02d}_{axis}" for j in joints for axis in "xyz"]
                 for hand in ("left", "right")}
    head_cols = [f"head_{axis}" for axis in "xyz"]
    columns = ["sample", "frame_index", "t_s", *head_cols,
               "head_qx", "head_qy", "head_qz", "head_qw", *hand_cols["left"], *hand_cols["right"]]
    frames = pd.read_parquet(KIT / "cache/canonical/frames.parquet", columns=columns)
    by_track = data.meta.groupby(["sample", "hand"], sort=False).indices
    result = np.empty((len(data.meta), sum(BLOCK_SIZES)), dtype=np.float32)
    written = np.zeros(len(result), dtype=bool)
    for sample, group in frames.groupby("sample", sort=False):
        group = group.sort_values("frame_index")
        t = group.t_s.to_numpy(dtype=float)
        head = group[head_cols].to_numpy()
        rotation = rotation_matrices(group)
        wrists = {}
        for hand, cols in hand_cols.items():
            positions = group[cols].to_numpy(dtype=float if len(joints)>1 else None).reshape(len(group),len(joints),3)
            point = getattr(data, "point", "wrist")
            wrists[hand] = palm_geometry(positions)[point] if point in PALM_TARGETS else positions.mean(axis=1)
        for hand, opposite in (("left", "right"), ("right", "left")):
            ids = by_track[sample, hand]
            np.testing.assert_array_equal(group.frame_index, data.meta.origin.to_numpy()[ids])
            inputs = (t, wrists[hand], wrists[opposite], head, rotation)
            result[ids] = describe(*inputs)
            written[ids] = True
            stop = min(len(ids), 95)
            np.testing.assert_allclose(describe(*(x[:stop] for x in inputs)), result[ids[:stop]], atol=1e-6)
            if data.meta.split.iat[ids[0]] == "train":
                origins = np.arange(0, len(t)-30, 31)
                future = local(wrists[hand][origins+30] - wrists[hand][origins], rotation[origins]) / SCALE
                np.testing.assert_allclose(future, data.truth[ids[origins]], atol=1e-5)
    assert written.all() and np.isfinite(result).all()
    # Normalize coordinates using only the eligible training library. Give each
    # physical block equal total weight, with a smaller contribution from duration.
    lower, centre, upper = np.quantile(result[data.train], [0.1, 0.5, 0.9], axis=0)
    scale = np.maximum((upper-lower)/2.563, np.repeat(SCALE_FLOORS, BLOCK_SIZES))
    block_weight = np.repeat(np.sqrt(np.array([1, 1, 1, 1, 1, .25])/BLOCK_SIZES), BLOCK_SIZES)
    return (np.clip((result-centre)/scale, -5, 5)*block_weight).astype(np.float32)


class MotionLibrary:
    def __init__(self, data, device):
        self.device = device
        self.features = torch.as_tensor(history_descriptors(data), device=device)
        bank_ids = data.train
        bank = data.meta.iloc[bank_ids]
        assert bank.split.eq("train").all() and bank.score.all() and bank.slow.all()
        self.groups = bank.groupby(["sample", "hand"], sort=True).indices
        self.bank_ids = bank_ids
        self.hands = data.meta.hand.to_numpy()
        self.names = data.meta["sample"].to_numpy()
        self.training_names = set(bank["sample"])
        self.training_environments = set(bank.env)
        self.outcomes = data.truth[bank_ids].copy()
        lengths = np.linalg.norm(self.outcomes, axis=1)
        self.quiet, self.far = np.quantile(lengths, [1/3, 2/3])

    def nearest(self, query, k, eligible=None):
        # Indexes returned to the caller refer to rows in the training library.
        assert self.training_names.isdisjoint(self.names[query])
        if eligible is not None:
            eligible = np.asarray(eligible, dtype=bool)
            if eligible.shape != (len(self.bank_ids),):
                raise ValueError("The candidate mask must index training-library rows.")
        nearest = np.empty((len(query), k), dtype=np.int64)
        for hand in ("left", "right"):
            positions = np.flatnonzero(self.hands[query] == hand)
            records = [idx for (_, side), idx in self.groups.items() if side == hand]
            if eligible is not None:
                records = [idx[eligible[idx]] for idx in records]
                records = [idx for idx in records if len(idx)]
            assert len(records) >= k
            for start in range(0, len(positions), 512):
                take = positions[start:start+512]
                x = self.features[torch.as_tensor(query[take], device=self.device)]
                distances, matches = [], []
                for record in records:
                    ids = torch.as_tensor(self.bank_ids[record], device=self.device)
                    dist, col = torch.cdist(x, self.features[ids]).min(dim=1)
                    distances.append(dist)
                    matches.append(torch.as_tensor(record, device=self.device)[col])
                distance = torch.stack(distances, dim=1)
                order = torch.argsort(distance, dim=1, stable=True)[:, :k]
                nearest[take] = torch.stack(matches, dim=1).gather(1, order).cpu().numpy()
        donors = self.bank_ids[nearest]
        assert (self.hands[donors] == self.hands[query, None]).all()
        names = np.sort(self.names[donors], axis=1)
        assert (names[:, 1:] != names[:, :-1]).all()
        return nearest

    def shuffled(self, seed):
        rng = np.random.default_rng(seed)
        order = np.arange(len(self.outcomes))
        for record in self.groups.values():
            order[record] = rng.permutation(record)
        return self.outcomes[order]


@dataclass(frozen=True)
class Policy:
    k: int
    quiet_consensus: bool
    fraction: float


def consensus(neighbors, outcomes, baseline, quiet_cut, far_cut, use_quiet):
    """Return a proposed displacement and the vote margin over a 50% majority."""
    vectors = outcomes[neighbors]
    lengths = np.linalg.norm(vectors, axis=2)
    units = vectors / np.maximum(lengths[:, :, None], 1e-8)
    far = lengths >= far_cut
    heading = (units*far[:, :, None]).sum(axis=1)
    heading /= np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-8)
    aligned = far & (np.einsum("nkd,nd->nk", units, heading) >= .5)
    far_votes = aligned.mean(axis=1)
    near = lengths <= quiet_cut
    near_votes = near.mean(axis=1)
    extend = far_votes > .5
    shorten = (near_votes > .5) & use_quiet
    assert not (extend & shorten).any()
    target = baseline.copy()
    if extend.any():
        agree = aligned[extend]
        direction = (units[extend]*agree[:, :, None]).sum(axis=1)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-8)
        length = np.nanmedian(np.where(agree, lengths[extend], np.nan), axis=1)
        target[extend] = direction*length[:, None]
    if shorten.any():
        length = np.nanmedian(np.where(near[shorten], lengths[shorten], np.nan), axis=1)
        base_length = np.linalg.norm(baseline[shorten], axis=1)
        target[shorten] *= np.minimum(1, length/np.maximum(base_length, 1e-8))[:, None]
    strength = np.maximum(0, 2*np.where(shorten, near_votes, far_votes)-1)
    return target, strength, extend, shorten


def correct(baseline, target, strength, fraction):
    return baseline + (fraction*strength[:, None])*(target-baseline)
