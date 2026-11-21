"""Issue a QSM wrist forecast and apply the frozen stability rule to its 30 radii."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from stable_config import OUT
from config import digest, read_json
from features import rows as read_prefix
from forecast import Forecaster as QSMForecaster
from origin_signals import PastErrors, issue_features, raw_signals
from phase_rule import apply_rule
from predict import history
from stable_data import measurements
from stability_router import Policy


class StableWristForecaster:
    def __init__(self, policy="phase_only"):
        lock = read_json(OUT/"frozen_before_test.json")
        self.name = policy
        self.policy = Policy(**lock["policies"][policy])
        self.simple = None
        if policy=="phase_only":
            exported = read_json(OUT/"phase_rule.json")
            if exported["derived_from_sha256"]!=digest(OUT/"frozen_before_test.json"):
                raise ValueError("Exported thresholds do not match the frozen policy")
            if exported["simple"]:
                self.simple = exported
        self.router = None if self.simple else joblib.load(OUT/"origin_router.joblib")
        self.qsm = QSMForecaster()
        self.records = {}

    @lru_cache(maxsize=256)
    def reference(self, sample, hand, origin):
        return self.qsm.issue(sample,hand,origin)

    def observed_history(self,sample,hand,origin,z,valid):
        key = sample,hand
        last,state = self.records.get(key,(10,PastErrors()))
        chronological = origin>=last
        if not chronological:
            last,state = 10,PastErrors()
        for arrived in range(last+30,origin+1,30):
            old = arrived-30
            previous = self.reference(sample,hand,old)
            observed = slice(old+1,arrived+1)
            error = np.linalg.norm(z[observed]-previous["center_world"],axis=1)
            state.receive(arrived,error,np.asarray(previous["radius_m"]),valid[observed])
            last = arrived
        if chronological:
            self.records[key] = last,state
        return state.values(origin),last if last>=40 else None

    def issue(self,sample,hand,origin):
        reference = self.reference(sample,hand,int(origin))
        g = read_prefix(sample,origin)
        t,z,head,valid = measurements(g,hand)
        past,matured = self.observed_history(sample,hand,origin,z,valid)
        raw = raw_signals(t,z,head,valid,np.array([origin]))
        phase = reference["calibrated_probabilities"]
        radius = np.asarray(reference["radius_m"])[None]
        details = {}
        if self.simple:
            x = pd.DataFrame({**raw,**past,"turn_probability":phase["turn"]},index=[0],dtype="float32")
            new,reduced,padded = apply_rule(self.simple,x,radius)
            details["maximum_turn_probability"] = self.simple["maximum_turn_probability"]
        else:
            b = self.qsm.base
            _,_,_,quality,_,_,_,_,_ = history(g,hand,b.quality,b.config,b.cut)
            probs = np.array([[phase[k] for k in ("dwell","turn","translate")]])
            center = np.array(reference["center_world"])-z[-1]
            x = issue_features(raw,probs,quality[-1:],center[None],radius,[past])
            scores = self.router.scores(x)
            new,reduced,padded = self.policy.issue(self.router,x,radius,scores)
            details = dict(stability_score=float(scores["stable"][0]),turn_score=float(scores["turn"][0]))
        decision = "tighten" if reduced[0] else ("protect" if padded[0] else "reference")
        return {**reference,"radius_m":new[0].tolist(),"policy":self.name,"band_decision":decision,
                "mean_radius_fraction":float(new.mean()/radius.mean()),"last_matured_frame":matured,
                "previous_observed_path_ratio":float(past["last_path_ratio"]),
                "reference_radius_m":radius[0].tolist(),**details}


def forecasting_model(predictor="qsm",policy=None,calibration=None):
    if predictor=="qsm" and calibration is None:
        return StableWristForecaster(policy or "phase_only")
    if predictor=="lsttn" and policy is None:
        standalone = str(Path(__file__).resolve().parent.parent/"q1_lsttn")
        if standalone not in sys.path:
            sys.path.insert(0,standalone)
        from issue_lsttn import WristForecaster
        return WristForecaster(calibration or "cafht_cptc")
    raise ValueError("Use a QSM policy with --predictor qsm or an LSTTN calibration with --predictor lsttn")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("hand",choices=("left","right"))
    parser.add_argument("origin",type=int)
    parser.add_argument("--predictor",choices=("qsm","lsttn"),default="qsm")
    parser.add_argument("--policy",choices=("conservative","efficient","recognition","phase_only","motion_only"))
    parser.add_argument("--calibration",choices=("static_joint","cafht","cafht_cptc","cafht_cptc_buffer","cafht_cptc_pooled_target"))
    args = parser.parse_args()
    try:
        forecaster = forecasting_model(args.predictor,args.policy,args.calibration)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(forecaster.issue(args.sample,args.hand,args.origin),ensure_ascii=False,allow_nan=False))
