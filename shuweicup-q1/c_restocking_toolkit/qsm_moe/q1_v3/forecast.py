"""Forecast from a recorded prefix with only this run's learned parameters."""
import argparse
import json

import joblib
import numpy as np
import torch

from config import OUT,WORK,read_json
from predict import WristForecaster,history
from features import rows,camera_data,video_features
from motion import base_path,ridge_prediction,visual_shift
from phase import raw_predict,calibrated,decide
from distribution import noise_scale,draw,gate_features,predict_block
from qsm_train import time_basis
from settings import MODES
from intervals import Feedback,PARTS,adjust_width,risk_fields,band
from turn_models import TurnExperts,STEPS


class Forecaster:
    def __init__(self):
        self.lock=read_json(OUT/"frozen_before_test.json")
        self.base=WristForecaster()
        self.plan=self.lock["interval"]
        self.point=self.lock["point"]["selected"]
        self.experts=TurnExperts((self.point["family"],))
        self.risk_model,self.midpoint=joblib.load(WORK/"path_risk_scale.joblib")
        with np.load(WORK/(self.plan["shape"]+"_feedback_priors.npz")) as p:
            self.reference=p["reference"]
            self.typical=float(p["typical"])

    @torch.inference_mode()
    def issue(self,sample,hand,origin):
        if hand not in ("left","right") or origin<10:
            raise ValueError("A wrist forecast needs a left/right hand and at least eleven frames.")
        g=rows(sample,origin)
        if not np.array_equal(g.frame_index.to_numpy(),np.arange(origin+1)):
            raise ValueError("The requested record does not contain that full prefix.")
        b=self.base
        k,v,phys,q,state,cad,rot,z,bad=history(g,hand,b.quality,b.config,b.cut)
        adaptive=any(x!="static" for x in PARTS[self.plan["scheme"]])
        ids=np.unique(np.r_[np.arange(10,origin+1,30),origin]) if adaptive else np.array([origin])
        k,v,phys,q,state,cad=(a[ids] for a in (k,v,phys,q,state,cad))
        raw=raw_predict(b.phase_model,k)
        p=calibrated(raw,b.phase_cal).astype("float32")
        phase=decide(raw,p,b.lock["phase"]["rule"])
        x=np.clip((k-b.mu)/b.sd,-6,6)
        physics=base_path(phys,cad,b.config["tau"])
        motion=ridge_prediction(v,p,state,q,cad,physics,b.rscale,b.rcoef)
        age=cad[:,None]*STEPS
        args=(b.tensor(x),b.tensor(v/b.vscale[None,:,None]),b.tensor(p),b.tensor(time_basis(age)))
        motion+=b.lock["motion"]["neural_weight"]*(sum(net(*args) for net in b.nets)/len(b.nets)).cpu().numpy()
        if b.lock["motion"]["visual_weight"]>0:
            stamps,pairs,uv=camera_data(sample,g)
            video=video_features(g,stamps,pairs,hand,uv)[ids]
            motion+=b.lock["motion"]["visual_weight"]*visual_shift(video,v,b.vm,b.vs,cad,b.vsd,b.vcoef)
        residual=draw(b.gen,b.tensor(np.column_stack((x,p))),b.tensor(noise_scale(cad,state)),b.noise,b.schedule,b.alpha)
        ref,samples,w,base_w=predict_block(motion,residual,gate_features(p,q,state,phys),p,state,q,cad,b.lock,b.width)
        family=self.point["family"]
        if family=="none":
            delta=np.zeros_like(ref)
        else:
            candidates=self.experts.corrections([family],ref,v,x,p,state,cad,np.full(len(ids),hand=="right"),
                                                np.full(len(ids),sample),phase==1)
            delta=self.point["strength"]*candidates[family]
        center=ref+delta
        fields=risk_fields(p,state,q,v,center,delta)
        shape=adjust_width(self.plan["shape"],base_w,fields,self.risk_model,self.midpoint)
        feedback=Feedback(self.reference,self.typical)
        if adaptive:
            for i,o in enumerate(ids):
                if o+30>origin:
                    continue
                actual=(z[o+1:o+31]-z[o])@rot[o]
                errors=np.linalg.norm(actual-center[i],axis=1)
                feedback.observe(phase[i],errors,shape[i],~bad[o+1:o+31])
        ep,whole=feedback.multipliers(phase[-1])
        parts={"static":shape[-1],"endpoint":shape[-1]*ep,"completed":shape[-1]*whole}
        radius=band(parts,self.plan["scheme"],self.plan["factors"])
        # Translate every distribution support by the same conditional correction as the point readout.
        supports=np.concatenate((motion[-1,None],samples[-1]),axis=0)+delta[-1,None]
        return dict(sample=sample,hand=hand,origin=int(origin),last_read_frame=int(g.frame_index.max()),
                    timestamp_s=float(g.t_s.iat[-1]),future_time_s=(g.t_s.iat[-1]+age[-1]).tolist(),
                    calibrated_probabilities=dict(zip(MODES,p[-1].tolist())),phase_decision=MODES[phase[-1]],
                    coordinate_frame="record world, metres",center_world=(z[-1]+center[-1]@rot[-1].T).tolist(),
                    radius_m=radius.tolist(),supports_world=(z[-1]+supports@rot[-1].T).tolist(),
                    support_weights=[float(1-w[-1])]+[float(w[-1]/8)]*8,
                    turn_correction=family,interval_scheme=self.plan["scheme"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("hand",choices=["left","right"])
    parser.add_argument("origin",type=int)
    options=parser.parse_args()
    print(json.dumps(Forecaster().issue(options.sample,options.hand,options.origin),ensure_ascii=False,allow_nan=False))
