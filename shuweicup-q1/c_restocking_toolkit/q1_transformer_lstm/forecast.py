"""Online API for the standalone Transformer-enhanced LSTM, reading official observations through the issue frame."""
import argparse
import json
import numpy as np
from scipy.special import expit
import torch

from study_paths import DATA,MODES,OUT,digest,read
from release_reader import load_record,manifest,SIDES
from hand_features import encode
from recurrent_network import Histories,deterministic
from fit_wrist import center_and_scale,issue_components,load_own,weights_file
from trajectory_bands import BandMemory


class Forecaster:
    def __init__(self,model="M2",band="joint_stage"):
        deterministic()
        frozen=read(OUT/"frozen_before_test.json")
        self.requested_model=model
        self.name="M3" if model=="M4" else model
        self.band=band
        self.spec=frozen["models"][self.name]
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in self.spec["seeds"]:
            if digest(weights_file(self.name,seed))!=self.spec["params"][str(seed)]:
                raise ValueError("Model file does not match the frozen study")
        self.nets=load_own(self.name,self.spec["seeds"],frozen["dims"],frozen["flag_rate"],self.device)
        cfg=self.spec["bands"][band]
        self.recipe=cfg["recipe"]
        path=OUT/f"{self.name}_{self.recipe['family']}_reference.npz"
        if digest(path)!=cfg["reference_sha256"]:
            raise ValueError("Interval file does not match the frozen study")
        with np.load(path) as f:
            self.reference={k:f[k] for k in f}
        with np.load(DATA/"fit_normalization.npz") as f:
            self.stats={k:f[k] for k in f}
        self.records=manifest().set_index("sample")

    def issue_prefix(self,record,hand):
        if hand not in SIDES or len(record.t)<11:
            raise ValueError("A wrist forecast needs a left/right hand and at least eleven recorded frames")
        if np.any(np.diff(record.t)<=0):
            raise ValueError("Input timestamps must increase")
        o=len(record.t)-1
        data,(t,z,rot,mirror)=encode(record,hand)
        for k in ("nodes","global_x"):
            data[k]=((data[k]-self.stats[k+"_mean"])/self.stats[k+"_sd"]).clip(-8,8).astype("float32")
        data["start"]=np.zeros(len(t),np.int64)
        history=Histories(data,self.device)
        matured=np.arange(10,o-30+1,30) if self.recipe["family"]!="fixed" else np.empty(0,int)
        ids=np.r_[matured,o]
        p,w,mus,sd,flag=issue_components(self.nets,history,ids,self.spec["temperature"])
        point,scale=center_and_scale(w,mus,sd,self.spec["readout"])
        state=BandMemory(self.reference,self.recipe)
        last_valid=None
        for j,start in enumerate(matured):
            prior=state.issue(scale[j],p[j])
            observed=slice(start+1,start+31)
            truth=np.einsum("hj,ji->hi",z[observed]-z[start],rot[start])*mirror
            good=data["good"][observed]
            state.receive(np.linalg.norm(truth-point[j],axis=1),scale[j],p[j],prior,good)
            if good.all():
                last_valid=int(start+30)
        radius=state.issue(scale[-1],p[-1])*self.recipe["outer"]
        qc=self.spec["quality"]
        risk=float(expit(qc["slope"]*flag[-1]+qc["intercept"]))
        def world(x):
            return z[-1]+np.einsum("...i,ji->...j",x*mirror,rot[-1])
        return dict(model=self.requested_model,point_model=self.name,calibration=self.band,hand=hand,origin=o,last_read_frame=o,
           last_complete_observation=last_valid,timestamp_s=float(t[-1]),future_time_s=(t[-1]+data["cad"][-1]*np.arange(1,31)).tolist(),
           coordinate_frame="record world, metres",center_world=world(point[-1]).tolist(),radius_m=radius.tolist(),
           observation_flag_probability=risk,quality_alert=bool(risk>=qc["threshold"]),
           phase_probabilities=dict(zip(MODES,p[-1].tolist())),turn_alert=bool(p[-1,1]>=self.spec["turn_cut"]),
           component_weights=w[-1].tolist(),component_means_world=world(mus[-1]).tolist(),
           component_axis_sigma_m=sd[-1].tolist(),component_residual_ar=self.spec["ar"],component_stages=list(MODES)*len(self.nets))

    def issue(self,sample,hand,origin):
        if origin<10 or sample not in self.records.index:
            raise ValueError("Unknown record or an issue frame earlier than 10")
        record=load_record(self.records.at[sample,"dataset"],sample,origin+1)
        return dict(sample=sample,**self.issue_prefix(record,hand))


def sample_paths(issued,n,seed):
    if n<0:
        raise ValueError("The number of draws must be nonnegative")
    rng=np.random.default_rng(seed)
    w=np.asarray(issued["component_weights"],float)
    k=rng.choice(len(w),n,p=w/w.sum())
    means=np.asarray(issued["component_means_world"])[k]
    sd=np.asarray(issued["component_axis_sigma_m"])[k]
    rho=np.asarray(issued["component_residual_ar"])[k,None]
    noise=rng.normal(size=means.shape)
    for h in range(1,30):
        noise[:,h]=rho*noise[:,h-1]+np.sqrt(1-rho*rho)*noise[:,h]
    return means+sd[:,:,None]*noise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("hand",choices=SIDES)
    parser.add_argument("origin",type=int)
    parser.add_argument("--model",choices=("M1","M2","M3","M4","M2_uniform"),default="M2")
    parser.add_argument("--band",choices=("joint_stage","joint_stage_env","cafht","fixed"),default="joint_stage")
    parser.add_argument("--draws",type=int,default=0)
    parser.add_argument("--seed",type=int,default=78109)
    args=parser.parse_args()
    answer=Forecaster(args.model,args.band).issue(args.sample,args.hand,args.origin)
    if args.draws:
        answer["sampled_paths_world"]=sample_paths(answer,args.draws,args.seed).tolist()
    print(json.dumps(answer,ensure_ascii=False,allow_nan=False))
