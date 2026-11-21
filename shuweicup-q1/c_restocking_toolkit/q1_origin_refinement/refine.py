"""Live wrist alert and forecast refinement using a raw prefix and an optional already issued M4 forecast."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from wrist_history import M4_DIR,OUT,SIDES,load_record,observe,protect_weights,read,sequence
from release_reader import manifest
from event_models import Scorer
from horizon_budget import issue_radii
from alarm_rules import alarms_on_record,combine
from path_expert import expert_points
from freeze_choices import fingerprint


class WristRefinement:
    def __init__(self):
        self.origin=read(OUT/"frozen_before_test.json")
        self.point=read(OUT/"point_locked_before_test.json")
        assert self.point["origin_sha256"]==fingerprint(OUT/"frozen_before_test.json")
        for name,sig in {**self.origin["weights"],"fresh_path_expert.joblib":self.point["expert_sha256"]}.items():
            if fingerprint(OUT/name)!=sig:
                raise ValueError("A frozen model has changed: "+name)
        self.scorer=Scorer(joblib.load(OUT/"fresh_tree_and_scales.joblib"))
        self.calibration=joblib.load(OUT/"validation_horizon_calibration.joblib")
        self.expert=joblib.load(OUT/"fresh_path_expert.joblib")
        self.cut=read(M4_DIR/"release_protocol.json")["speed_cut"]

    def issue(self,prefix,hand,m4=None,previous_factor=None):
        end=len(prefix.t)-1
        if end<10 or hand not in SIDES or np.any(np.diff(prefix.t)<=0):
            raise ValueError("A hand and at least eleven increasing timestamped frames are required.")
        m=observe(prefix,hand,self.cut)
        grid=np.arange(10,end+1,30)
        if grid[-1]!=end:
            grid=np.r_[grid,end]
        seq=sequence(m,grid)
        features=m.tab.iloc[grid]
        scores,proxy=self.scorer.outputs(seq,features,m.phase[grid])
        chosen=scores[self.origin["detector"]]
        cfg=self.origin["online_alarm"]
        key=(cfg["fpr"],cfg["memory"])
        cuts=alarms_on_record(m,grid,chosen,self.origin["negative_prior"],[key])
        cut=float(cuts[f"cut_{key[0]}_{key[1]}"][-1])
        local=expert_points(self.expert,seq[-1:],features.iloc[-1:])[0]
        world=m.wrist[-1]+np.einsum("hi,ji->hj",local*m.mirror,m.rotation[-1].as_matrix())
        output=dict(origin=end,hand=hand,last_observed_frame=end,turn_score=float(chosen[-1]),
                    turn_threshold=cut,turn_alert=bool(chosen[-1]>=cut),independent_center_world=world.tolist())
        if m4 is None:
            return output
        if int(m4["origin"])!=end or m4["hand"]!=hand or int(m4.get("last_read_frame",end))!=end:
            raise ValueError("The M4 issue must match the hand and the final prefix frame.")
        base=np.asarray(m4["center_world"],float)
        radius=np.asarray(m4["radius_m"],float)
        if base.shape!=(30,3) or not np.isfinite(base).all() or radius.shape!=(30,):
            raise ValueError("Supply the issued M4 3D centre and radius at all thirty steps.")
        factor=None
        if previous_factor is not None:
            if not 0<float(previous_factor)<=1:
                raise ValueError("The previous radius factor must be in (0, 1].")
            factor=[float(previous_factor)]
        radii=issue_radii(features.iloc[-1:],chosen[-1:],proxy[-1:],radius[None],self.calibration,self.origin["radii"],factor)
        mix=self.origin["optional_mixture"]
        m4_score=float(m4["phase_probabilities"]["turn"])
        if not 0<=m4_score<=1:
            raise ValueError("Invalid M4 stage score.")
        combined=float(combine(np.array([m4_score]),chosen[-1:],mix["anchors"],mix["share"])[0])
        output["fused_turn_score"]=combined
        output["fused_turn_alert"]=bool(combined>=mix["threshold"])
        output["m4_center_radii_m"]={k:v[0].tolist() for k,v in radii.items()}
        output["mixed_center_world"]=(base+self.point["share"]*(world-base)).tolist()
        output["mixed_center_radii_m"]={k:(radii[k][0]*s).tolist() for k,s in self.point["radius_multipliers"].items()}
        if factor is not None:
            output["mixed_center_radii_m"]["same_window_budget"]=radii["same_window_budget"][0].tolist()
        return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample")
    p.add_argument("hand",choices=SIDES)
    p.add_argument("origin",type=int)
    p.add_argument("--m4-json",type=Path,help="File with the existing forecast output at this exact origin")
    p.add_argument("--previous-factor",type=float,help="Previously issued per-window width multiplier, e.g. 0.8 or 1")
    args=p.parse_args()
    protect_weights()
    records=manifest().set_index("sample")
    prefix=load_record(records.at[args.sample,"dataset"],args.sample,args.origin+1)
    base=read(args.m4_json) if args.m4_json else None
    if base and base.get("sample",args.sample)!=args.sample:
        p.error("M4 sample differs from the release record.")
    result=WristRefinement().issue(prefix,args.hand,base,args.previous_factor)
    print(json.dumps(dict(sample=args.sample,**result),ensure_ascii=False,allow_nan=False))
