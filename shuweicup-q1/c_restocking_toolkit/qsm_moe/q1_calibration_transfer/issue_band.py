"""Issue a new 30-step band from raw recorded observations through the requested frame."""
import argparse
from copy import copy
from functools import lru_cache
import json

import numpy as np
import torch

from transfer_config import OUT
from config import read_json
from forecast import Forecaster as PointForecaster
from features import rows, wrist, xyz
from predict import filled, history
from calibration import RecordFeedback, freeze_endpoint_feedback, load_priors
from error_scale import ErrorScale, Features, SEEDS, predict_models


class BandForecaster:
    def __init__(self, decision="matched"):
        lock = read_json(OUT/"frozen_before_test.json")
        if decision in ("selected","matched"):
            self.recipe = lock[decision]["recipe"]
        elif decision=="fixed_feedback_control":
            self.recipe = freeze_endpoint_feedback(lock["selected"]["recipe"])
        elif decision in ("gru_endpoint_95","gru_static_95"):
            self.recipe = lock["controls"][decision]
        else:
            raise ValueError("Unknown interval decision")
        self.decision = decision
        self.point = PointForecaster()
        self.raw_width = copy(self.point)
        self.raw_width.plan = dict(shape="kinematic",scheme="static",factors={"static":1.})
        self.priors = load_priors()
        self.models = []
        if self.recipe.get("shape")=="gru":
            base = self.point.base
            for seed in SEEDS:
                model = ErrorScale(len(base.mu)).to(base.device)
                model.load_state_dict(torch.load(OUT/f"error_scale_{seed}.pt",map_location=base.device,weights_only=True))
                self.models.append(model.eval())

    @lru_cache(maxsize=256)
    def _issue_qsm(self, sample, hand, origin):
        return self.point.issue(sample,hand,int(origin))

    def _scale(self, sample, hand, origin, result):
        plan = self.point.plan
        if plan["shape"]=="kinematic" and plan["scheme"]=="static":
            return np.asarray(result["radius_m"])/plan["factors"]["static"]
        return np.asarray(self.raw_width.issue(sample,hand,int(origin))["radius_m"])

    def issue(self, sample, hand, origin):
        recipe = self.recipe
        issued = self._issue_qsm(sample,hand,origin)
        if recipe["kind"]=="incumbent":
            return {**issued,"band_decision":self.decision}
        g = rows(sample,origin)
        raw = wrist(g,hand)
        z = filled(raw,xyz(g,"head")[0])
        good = g[hand+"_quality"].eq("ok").to_numpy() & np.isfinite(raw).all(1)
        matured = np.arange(10,origin-30+1,30) if recipe["scheme"]!="static" else np.empty(0,int)
        issue_times = np.r_[matured,origin]
        shape = recipe["shape"]
        prior = self.priors[shape]
        state = RecordFeedback(prior["reference"],prior["typical"])
        if shape=="gru":
            b = self.point.base
            k,_,_,_,_,cad,_,_,_ = history(g,hand,b.quality,b.config,b.cut)
            x = np.clip((k-b.mu)/b.sd,-6,6)
            data = Features(x,np.zeros(len(k),int),cad,b.device)
            scales = predict_models(self.models,data,issue_times)
        else:
            scales = [self._scale(sample,hand,int(t),self._issue_qsm(sample,hand,int(t))) for t in issue_times]
        for j,t in enumerate(matured):
            past = self._issue_qsm(sample,hand,int(t))
            obs = slice(t+1,t+31)
            distance = np.linalg.norm(z[obs]-np.asarray(past["center_world"]),axis=1)
            state.observe(distance,scales[j],good[obs])
        radius = state.radii(scales[-1])[recipe["scheme"]]*recipe["factor"]
        if recipe["weighted"]:
            radius = radius*prior["weights"]
        if recipe["floor"]:
            radius = np.maximum(radius,recipe["floor"]*np.asarray(issued["radius_m"]))
        return {**issued,"radius_m":radius.tolist(),"interval_scheme":recipe["scheme"],
                "band_decision":self.decision,"last_feedback_end":int(matured[-1]+30) if len(matured) else None}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("hand",choices=["left","right"])
    parser.add_argument("origin",type=int)
    parser.add_argument("--decision",default="matched",choices=["selected","matched","gru_endpoint_95","gru_static_95","fixed_feedback_control"])
    args = parser.parse_args()
    print(json.dumps(BandForecaster(args.decision).issue(args.sample,args.hand,args.origin),
                     ensure_ascii=False,allow_nan=False))
