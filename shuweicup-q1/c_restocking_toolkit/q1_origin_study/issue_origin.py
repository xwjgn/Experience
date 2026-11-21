"""Issue the frozen turn alarm and radius adjustment for a single raw prefix, without loading a trajectory model."""
import argparse
import json

import joblib
import numpy as np

from origin_signals import OUT, columns, origin_features, load_record
from release_reader import manifest
from evaluate_origin import route, tail_features


class OriginDecision:
    def __init__(self):
        self.lock = json.loads((OUT/"frozen_before_replay.json").read_text(encoding="utf8"))
        saved = joblib.load(OUT/"independent_origin_models.joblib")
        self.turn, self.tail = saved["turn"], saved["tail"]

    def issue(self, record, hand, radius_m=None):
        if len(record.t) < 11 or np.any(np.diff(record.t) <= 0):
            raise ValueError("Provide at least eleven strictly time-ordered raw frames through the issue time.")
        feat = origin_features(record,hand).iloc[[-1]].copy()
        scores = {k:float(model.predict_proba(feat[columns(feat,k)])[0,1]) for k,model in self.turn.items()}
        name = "wrist_hand_head"
        clean = (feat.w_recent_flags == 0).to_numpy()
        def factor_for(key,score):
            allowed = route(np.array([score]),clean,self.lock["routing"][key]["cut"])[0]
            return self.lock["shrink_factor"] if allowed else 1.
        factor = factor_for("low_turn", scores[name])
        output = dict(hand=hand,origin=len(record.t)-1,turn_score=scores[name],turn_scores=scores,
                      turn_alert=scores[name]>=self.lock["turn_thresholds"][name],
                      recent_wrist_observations_clean=bool(clean[0]),low_turn_radius_factor=factor)
        if radius_m is not None:
            radius = np.asarray(radius_m, dtype=float)
            if radius.shape != (30,) or not np.isfinite(radius).all() or not (radius > 0).all():
                raise ValueError("Supply thirty positive, finite M4 radii in metres.")
            feat["wrist_hand_head"] = scores[name]
            feat["radius_cm"] = 100*radius.mean()
            tail_score = float(self.tail.predict(tail_features(feat))[0])
            tail_factor = factor_for("tail",tail_score)
            output.update(low_turn_radius_m=(radius*factor).tolist(),tail_radius_factor=tail_factor,
                          tail_radius_m=(radius*tail_factor).tolist())
        return output


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample")
    parser.add_argument("hand",choices=("left","right"))
    parser.add_argument("origin",type=int)
    parser.add_argument("--radius-json",help="Optional JSON list of M4's 30 already-issued radii, in metres.")
    args = parser.parse_args()
    catalog = manifest().set_index("sample")
    prefix = load_record(catalog.at[args.sample,"dataset"],args.sample,args.origin+1)
    radii = json.loads(args.radius_json) if args.radius_json else None
    print(json.dumps(OriginDecision().issue(prefix,args.hand,radii),ensure_ascii=False,allow_nan=False))
