"""Issue a fixed 30-frame forecast from a raw wrist prefix, in the record's world frame."""
import argparse
from types import SimpleNamespace
import json
import numpy as np
import torch

from paths_q1 import DATA,OUT,STAGES,digest,read_json
from wrist_observations import measured,read_frames
from lsttn_model import RHO,SequenceInputs,deterministic
from train_lsttn import checkpoint,load_nets,predict
from freeze_lsttn import readout
from path_calibration import CalibrationState,load_bank


class WristForecaster:
    def __init__(self,calibration="cafht_cptc"):
        deterministic()
        lock = read_json(OUT/"frozen_before_test.json")
        self.spec = lock["experiments"][lock["primary"]]
        self.band = calibration
        self.recipe = self.spec["intervals"][calibration]["recipe"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in self.spec["seeds"]:
            if digest(checkpoint(self.spec["variant"],seed))!=self.spec["checkpoint_sha256"][str(seed)]:
                raise ValueError("Model parameters changed after the study was frozen")
        self.nets = load_nets(self.spec["variant"],self.spec["seeds"],lock["dims"],self.device)
        profile = self.recipe["profile"]
        prior = OUT/(lock["primary"]+"_"+profile+"_prior.npz")
        if digest(prior)!=self.spec["prior_sha256"][profile]:
            raise ValueError("Calibration reference changed")
        self.shape,self.banks = load_bank(prior)
        with np.load(DATA/"normalization.npz") as normalization:
            self.mu,self.sd = normalization["mu"],normalization["sd"]

    def issue_prefix(self,g,hand):
        if hand not in ("left","right") or len(g)<11 or not np.array_equal(g.frame_index,np.arange(len(g))):
            raise ValueError("Expected a chronological raw left/right wrist prefix starting at frame zero with at least eleven frames")
        o = len(g)-1
        data,(t,z,rot,mirror) = measured(g,hand)
        data["x"] = np.clip((data["x"]-self.mu)/self.sd,-7,7).astype("float32")
        data["start"] = np.zeros(len(g),dtype=np.int64)
        seq = SequenceInputs(data,self.device)
        past = np.arange(10,o-30+1,30) if self.recipe["scheme"]!="static_joint" else np.empty(0,int)
        ids = np.r_[past,o]
        p,w,means,sigma = predict(self.nets,seq,ids,self.spec["temperature"])
        point,radial = readout(SimpleNamespace(prob=p,weight=w,means=means,sigma=sigma),self.spec["point"])
        state = CalibrationState(self.recipe,self.shape,self.banks)
        last_valid_receipt = None
        for i,start in enumerate(past):
            provisional = state.provisional(radial[i],p[i])
            steps = slice(start+1,start+31)
            truth = np.einsum("hj,ji->hi",z[steps]-z[start],rot[start])*mirror
            valid = data["ok"][steps]
            state.receive(np.linalg.norm(truth-point[i],axis=-1),radial[i],p[i],provisional,valid)
            if valid.all():
                last_valid_receipt = int(start+30)
        radius = state.provisional(radial[-1],p[-1])*self.recipe["outer_factor"]
        def world(x):
            return z[-1]+np.einsum("...i,ji->...j",x*mirror,rot[-1])
        probabilities = dict(zip(STAGES,p[-1].tolist()))
        alert = bool(p[-1,1]>=self.spec["alert_threshold"])
        decision = "turn" if alert else ("dwell" if p[-1,0]>p[-1,2] else "translate")
        return dict(hand=hand,origin=o,last_read_frame=o,last_valid_path_receipt=last_valid_receipt,
            timestamp_s=float(t[-1]),future_time_s=(t[-1]+data["cad"][-1]*np.arange(1,31)).tolist(),
            coordinate_frame="record world, metres",center_world=world(point[-1]).tolist(),radius_m=radius.tolist(),
            predictor="lsttn",radial_rms_m=radial[-1].tolist(),calibration=self.band,interval_scheme=self.band,
            stage_probabilities=probabilities,calibrated_probabilities=probabilities,phase_decision=decision,
            turn_alert=alert,mixture_weights=w[-1].tolist(),
            component_stage=list(STAGES)*len(self.nets),component_means_world=world(means[-1]).tolist(),
            component_axis_sigma_m=sigma[-1].tolist(),standardized_residual_ar1=RHO)

    def issue(self,sample,hand,origin):
        if origin<10:
            raise ValueError("The requested issue frame must be at least 10")
        g = read_frames(sample,origin)
        if len(g)!=origin+1:
            raise ValueError("Record does not contain the specified prefix")
        return dict(sample=sample,**self.issue_prefix(g,hand))


def sample_paths(issued,n,seed):
    if n<0:
        raise ValueError("Requested a negative sample count")
    rng = np.random.default_rng(seed)
    weights = np.asarray(issued["mixture_weights"],float)
    comp = rng.choice(len(weights),size=n,p=weights/weights.sum())
    means = np.asarray(issued["component_means_world"])[comp]
    sigmas = np.asarray(issued["component_axis_sigma_m"])[comp]
    z = rng.normal(size=means.shape)
    rho = issued["standardized_residual_ar1"]
    for h in range(1,z.shape[1]):
        z[:,h] = rho*z[:,h-1]+np.sqrt(1-rho**2)*z[:,h]
    return means+sigmas[:,:,None]*z


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("hand",choices=("left","right"))
    parser.add_argument("origin",type=int)
    parser.add_argument("--calibration",default="cafht_cptc",choices=("static_joint","cafht","cafht_cptc","cafht_cptc_buffer","cafht_cptc_pooled_target"))
    parser.add_argument("--draws",type=int,default=0)
    parser.add_argument("--seed",type=int,default=61717)
    args = parser.parse_args()
    result = WristForecaster(args.calibration).issue(args.sample,args.hand,args.origin)
    if args.draws:
        result["draws_world"] = sample_paths(result,args.draws,args.seed).tolist()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
