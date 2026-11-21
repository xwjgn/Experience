from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from stable_config import KIT, REPLICA, WORK
from config import KEY, read_json
from features import wrist, xyz
from intervals import adjust_width, band, risk_fields, sequences
from predict import filled
from score import phase_calls
from settings import array
from origin_signals import PastErrors, raw_signals, issue_features, stable_truth


@lru_cache(maxsize=2)
def track(sample):
    columns = ["frame_index","t_s","head_x","head_y","head_z","left_quality","right_quality"]
    columns.extend(f"{hand}_j00_{axis}" for hand in ("left","right") for axis in "xyz")
    g = pd.read_parquet(KIT/"cache"/"canonical"/"frames.parquet",columns=columns,filters=[("sample","==",sample)])
    return g.sort_values("frame_index").reset_index(drop=True)


def measurements(g, hand):
    head = xyz(g,"head")
    raw = wrist(g,hand)
    valid = g[hand+"_quality"].eq("ok").to_numpy() & np.isfinite(raw).all(1)
    return g.t_s.to_numpy(), filled(raw,head[0]), head, valid


@dataclass
class Windows:
    meta: pd.DataFrame
    x: pd.DataFrame
    radius: np.ndarray
    error: np.ndarray
    stable: np.ndarray
    turn: np.ndarray
    ids: np.ndarray

    @property
    def ratio(self):
        return np.max(self.error/self.radius,axis=1)

    def take(self,mask):
        return Windows(self.meta.loc[mask].reset_index(drop=True), self.x.loc[mask].reset_index(drop=True),
                       self.radius[mask],self.error[mask],self.stable[mask],self.turn[mask],self.ids[mask])


class FixedReference:
    def __init__(self,d):
        self.d = d
        self.plan = read_json(REPLICA/"frozen_before_test.json")["interval"]
        self.center = np.load(WORK/"corrected_center.npy",mmap_mode="r")
        self.prior = np.load(WORK/"reference_center.npy",mmap_mode="r")
        self.scale = np.load(WORK/"reference_scale.npy",mmap_mode="r")
        self.risk_model,self.mid = joblib.load(WORK/"path_risk_scale.joblib")
        with np.load(WORK/(self.plan["shape"]+"_feedback_priors.npz")) as fp:
            self.reference,self.typical = fp["reference"],float(fp["typical"])
        self.next_mode = array("next_mode")

    def radius(self, rows, error, valid):
        d, plan = self.d, self.plan
        center = self.center[rows]
        fields = risk_fields(d.phase[rows], d.state[rows], d.q[rows], d.v[rows], center, center-self.prior[rows])
        raw = adjust_width(plan["shape"],self.scale[rows],fields,self.risk_model,self.mid)
        if plan["scheme"]=="static":
            return raw*plan["factors"]["static"]
        parts = sequences(d.meta.iloc[rows].reset_index(drop=True), phase_calls(d,rows),error,raw,valid,self.reference,self.typical)
        return band(parts,plan["scheme"],plan["factors"])

    def windows(self, allowed, stride=30):
        d = self.d
        logs, features, radii, errors, stabilities, indices = [], [], [], [], [], []
        for (sample,hand),group in d.meta.loc[allowed].groupby(["sample","hand"],sort=False):
            rows = group.index.to_numpy()
            t,z,head,valid = measurements(track(sample),hand)
            np.testing.assert_array_equal(group.origin,np.arange(len(t)))
            np.testing.assert_array_equal(valid,array("ok")[rows])
            full_error = np.linalg.norm(self.center[rows]-d.y[rows],axis=2)
            radius = self.radius(rows,full_error,valid)
            # Schedule earlier issues and process observed outcomes even if an earlier window
            # was ineligible for retrospective scoring. Each independent wrist starts empty.
            state = PastErrors()
            times = np.arange(10,len(rows),stride)
            selected = times[d.mask[rows[times]].all(1)]
            if not len(selected):
                continue
            receipts = iter(range(40,len(rows),30))
            next_receipt = next(receipts,None)
            memory = []
            for now in selected:
                while next_receipt is not None and next_receipt<=now:
                    old = next_receipt-30
                    state.receive(next_receipt,full_error[old],radius[old],valid[old+1:next_receipt+1])
                    next_receipt = next(receipts,None)
                memory.append(state.values(now))
            keep = rows[selected]
            raw = raw_signals(t,z,head,valid,selected)
            features.append(issue_features(raw, d.phase[keep], d.q[keep], self.center[keep], radius[selected], memory))
            indices.append(keep)
            logs.append(group.iloc[selected][KEY].reset_index(drop=True))
            radii.append(radius[selected])
            errors.append(full_error[selected])
            stabilities.append(stable_truth(t,selected,d.y[keep]))
        ids = np.concatenate(indices)
        if stride==30:
            np.testing.assert_array_equal(np.sort(ids),np.flatnonzero(allowed & d.grid))
        return Windows(pd.concat(logs,ignore_index=True), pd.concat(features,ignore_index=True),
                       np.concatenate(radii),np.concatenate(errors),np.concatenate(stabilities),
                       self.next_mode[ids]==1, ids)
