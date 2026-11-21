"""Forecast from a raw wrist prefix; optional inputs are issued trajectories, never reference-model weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from gain_records import M4_DIR,OUT,SIDES,features_for,isolate,load_record,observe,read,sequence
from train_gain import ModelBank
from gain_calibration import geometric,redistribute,third_mix
from release_reader import manifest


class BayesianWrist:
    def __init__(self):
        self.bank=ModelBank(("bayesian",))
        self.settings=read(OUT/"frozen_before_test.json")["models"]["bayesian"]
        self.speed_cut=read(M4_DIR/"release_protocol.json")["speed_cut"]

    def issue(self,prefix,hand,reference=None,reference_width="third"):
        o=len(prefix.t)-1
        if hand not in SIDES or o<10 or np.any(np.diff(prefix.t)<=0):
            raise ValueError("Supply a hand and at least eleven increasing timestamps.")
        if reference_width not in ("third","fourth"):
            raise ValueError("Reference width must identify row three or four.")
        m=observe(prefix,hand,self.speed_cut)
        ix=np.array([o])
        seq=sequence(m,ix)
        inp=features_for(prefix,m,hand,ix,seq,m.tab.iloc[ix])
        pred=self.bank.issue(inp)
        g=geometric(pred["cov"])
        local_to_world=m.rotation[-1].as_matrix()@np.diag(m.mirror)
        heading_to_world=local_to_world@inp.axes[0]
        def world_points(x):
            return m.wrist[-1]+x@local_to_world.T
        def world_cov(c):
            return local_to_world@c@local_to_world.T
        a=heading_to_world
        t6=np.block([[a,np.zeros((3,3))],[np.zeros((3,3)),a]])
        stand=self.settings["standalone"]
        cov=world_cov(pred["cov"][0])
        mu=world_points(pred["mean"][0])
        gain=[a@np.diag(v)@a.T for v in pred["gain"][0]]
        output=dict(hand=hand,origin=o,last_read_frame=o,coordinate_frame="record world",
            position_m=(m.wrist[-1]+a@pred["state"][0,:3]).tolist(),velocity_m_s=(a@pred["state"][0,3:]).tolist(),
            position_velocity_covariance=(t6@pred["state_cov"][0]@t6.T).tolist(),
            last_effective_gain_world=np.concatenate(gain,axis=0).tolist(),
            gain_axes="last effective history update in the issue coordinate frame; position and velocity blocks",
            center_world=mu.tolist(),covariance_m2=cov.tolist(),parameter_covariance_m2=world_cov(pred["weight_cov"][0]).tolist(),
            sphere_radius_m=(stand["sphere_q"]*g["radial"][0]).tolist(),ellipsoid_mahalanobis_radius=stand["ellipsoid_q"],
            posterior_draws_per_seed=16,seeds=2)
        if reference is None:
            return output
        if reference.get("origin")!=o or reference.get("hand")!=hand or reference.get("last_observed_frame",o)!=o:
            raise ValueError("The supplied issue must have the same hand and origin.")
        key="same_window_budget" if reference_width=="third" else "compact"
        if "mixed_center_world" in reference:
            ref_world=np.asarray(reference["mixed_center_world"],float)
            r0=np.asarray(reference["mixed_center_radii_m"][key],float)
        else:
            ref_world=np.asarray(reference["reference_center_world"],float)
            r0=np.asarray(reference["reference_radius_m"],float)
        if ref_world.shape!=(30,3) or r0.shape!=(30,) or not np.isfinite(ref_world).all() or not np.isfinite(r0).all() or (r0<=0).any():
            raise ValueError("Supply thirty finite 3-D reference points and positive sphere radii in metres.")
        spec=self.settings["hybrid"]
        local=(ref_world-m.wrist[-1])@local_to_world
        joint=third_mix(local[None],pred["mean"],g,spec["share"],spec["mode"])
        radius=redistribute(r0[None],g["radial"],spec["power"])[0]
        output["reference_width"]=reference_width
        output["hybrid_center_world"]=(ref_world if spec["share"]==0 else world_points(joint[0])).tolist()
        output["hybrid_radius_m"]=radius.tolist()
        if reference_width=="third":
            output["hybrid_guarded_radius_m"]=(spec["q_guarded"]*radius).tolist()
        return output


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample")
    p.add_argument("hand",choices=SIDES)
    p.add_argument("origin",type=int)
    p.add_argument("--reference-json",type=Path)
    p.add_argument("--reference-width",choices=("third","fourth"),default="third")
    args=p.parse_args()
    isolate()
    m=manifest().set_index("sample")
    raw=load_record(m.at[args.sample,"dataset"],args.sample,args.origin+1)
    ref=read(args.reference_json) if args.reference_json else None
    if ref and ref.get("sample",args.sample)!=args.sample:
        p.error("Reference sample differs from the released record.")
    ans=BayesianWrist().issue(raw,args.hand,ref,args.reference_width)
    print(json.dumps(dict(sample=args.sample,**ans),ensure_ascii=False,allow_nan=False))
