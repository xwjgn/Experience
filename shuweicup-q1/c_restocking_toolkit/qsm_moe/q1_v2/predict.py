"""Issue the locked v2 forecast for one wrist. Inputs stop at the requested canonical frame."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from settings import OUT, WORK, KIT, MODES
from features import rows, xyz, wrist, kinematics, camera_data, video_features
from motion import VelocityResidual, base_path, ridge_prediction, visual_shift
from phase import raw_predict, calibrated, decide
from distribution import ResidualDiffusion, noise_scale, draw, gate_features, predict_block
from qsm_data import geometry, quality_features, filter_track, local, motion_modes, diffusion_schedule
from qsm_train import time_basis


def filled(values,initial):
    if np.isfinite(values).all():
        return values
    copy=values.copy()
    last=np.asarray(initial).copy()
    for i,row in enumerate(values):
        last=np.where(np.isfinite(row),row,last)
        copy[i]=last
    return copy


def history(g,hand,quality_model,config,cut):
    t=g.t_s.to_numpy()
    head=xyz(g,"head")
    rot=Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].to_numpy()).as_matrix()
    other="right" if hand=="left" else "left"
    raw=wrist(g,hand)
    z=filled(raw,head[0])
    opp=filled(wrist(g,other),head[0])
    finger_ids=(4,5,8,9,12,17,20)
    fingers=np.stack([filled(xyz(g,f"{hand}_j{j:02}"),z[0]) for j in finger_ids],1)
    bad=g[hand+"_quality"].ne("ok").to_numpy() | ~np.isfinite(raw).all(1)
    geom=geometry(t,z,opp,head,rot,fingers)
    prob=quality_model.predict_proba(quality_features(t,z,bad,geom))[:,1].astype("float32")
    k=config["quality_kalman"]
    post,vel,var=filter_track(t,raw,np.maximum(prob,bad),k["q"],k["r"],k["m"])
    physics=np.column_stack((local(post-z,rot),local(vel,rot),np.sqrt(var.mean(1)))).astype("float32")
    if not np.isfinite(raw).all() or not np.isfinite(wrist(g,other)).all() or not np.isfinite(
            g[[f"{hand}_j{j:02}_{a}" for j in (4,8,12,20) for a in "xyz"]]).all().all():
        g=g.copy()
        for name,points in [(hand+"_j00",z),(other+"_j00",opp),
                            *((f"{hand}_j{j:02}",fingers[:,i]) for i,j in enumerate(finger_ids))]:
            g[[name+"_"+a for a in "xyz"]]=points
    kine,v=kinematics(g,hand,physics,prob,cut)
    _,state=motion_modes(t,z,cut)
    cadence=pd.Series(np.r_[np.nan,np.diff(t)]).rolling(5,min_periods=1).median().fillna(1/30).to_numpy(dtype="float32")
    return kine,v,physics,prob,state.astype("float32"),cadence,rot,z,bad


class WristForecaster:
    def __init__(self):
        self.lock=json.loads((OUT/"frozen_before_test.json").read_text(encoding="utf8"))
        self.config=json.loads((WORK/"cache"/"ready.json").read_text(encoding="utf8"))
        if self.lock["previous_checkpoints_loaded"]!=0:
            raise ValueError("Unexpected source weights")
        self.quality=joblib.load(WORK/"measurement_quality.joblib")
        self.phase_model,self.phase_cal=joblib.load(WORK/(self.lock["phase"]["model"]+"_phase.joblib"))
        with np.load(WORK/"feature_normalization.npz") as f:
            self.cut=float(f["mode_cut"])
        with np.load(WORK/"v2_normalization.npz") as f:
            self.mu,self.sd=f["mu"],f["sd"]
            self.vm,self.vs=f["video_mu"],f["video_sd"]
        with np.load(WORK/"motion_ridges.npz") as f:
            self.rscale=f["scale"]
            self.rcoef=f["c"+str(self.lock["motion"]["ridge_index"])]
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vscale=np.load(WORK/"vector_scale.npy")
        self.nets=[]
        for seed in self.lock["motion"]["seeds"]:
            net=VelocityResidual(len(self.mu),len(self.vscale)).to(self.device)
            net.load_state_dict(torch.load(WORK/f"velocity_{seed}.pt",map_location=self.device,weights_only=True))
            self.nets.append(net.eval())
        self.gen=ResidualDiffusion(len(self.mu)+3).to(self.device)
        self.gen.load_state_dict(torch.load(WORK/"residual_diffusion.pt",map_location=self.device,weights_only=True))
        self.gen.eval()
        self.schedule=diffusion_schedule()
        self.alpha=self.tensor(np.r_[1.,self.schedule.alphas_cumprod])
        self.noise=self.tensor(np.load(WORK/"residual_noise.npy"))
        self.width=np.load(WORK/"width_parameters.npy")
        self.reference=np.load(WORK/"feedback_reference.npy")
        if self.lock["motion"]["visual_weight"]>0:
            with np.load(WORK/"visual_adapter.npz") as f:
                self.vsd,self.vcoef=f["sd"],f["coef"]

    def tensor(self,a):
        return torch.as_tensor(np.ascontiguousarray(a),dtype=torch.float32,device=self.device)

    @torch.inference_mode()
    def issue(self,sample,hand,origin):
        if hand not in ("left","right") or origin<10:
            raise ValueError("An issue needs eleven wrist frames and a left or right hand.")
        g=rows(sample,origin)
        if not np.array_equal(g.frame_index.to_numpy(),np.arange(origin+1)):
            raise ValueError("Requested wrist prefix is incomplete or outside the record.")
        k,v,phys,q,state,cad,rot,z,bad=history(g,hand,self.quality,self.config,self.cut)
        level=self.lock["band"]["scheme"]
        ids=np.unique(np.r_[np.arange(10,origin+1,30),origin]) if level=="feedback" else np.array([origin])
        k,v,phys,q,state,cad=(value[ids] for value in (k,v,phys,q,state,cad))
        raw=raw_predict(self.phase_model,k)
        p=calibrated(raw,self.phase_cal).astype("float32")
        call=decide(raw,p,self.lock["phase"]["rule"])
        x=np.clip((k-self.mu)/self.sd,-6,6)
        base=base_path(phys,cad,self.config["tau"])
        motion=ridge_prediction(v,p,state,q,cad,base,self.rscale,self.rcoef)
        age=cad[:,None]*np.arange(1,31,dtype="float32")
        args=(self.tensor(x),self.tensor(v/self.vscale[None,:,None]),self.tensor(p),self.tensor(time_basis(age)))
        delta=sum(net(*args) for net in self.nets)/len(self.nets)
        motion+=self.lock["motion"]["neural_weight"]*delta.cpu().numpy()
        vw=self.lock["motion"]["visual_weight"]
        if vw>0:
            stamps,pairs,uv=camera_data(sample,g)
            video=video_features(g,stamps,pairs,hand,uv)[ids]
            motion+=vw*visual_shift(video,v,self.vm,self.vs,cad,self.vsd,self.vcoef)
        res=draw(self.gen,self.tensor(np.column_stack((x,p))),self.tensor(noise_scale(cad,state)),
                 self.noise,self.schedule,self.alpha)
        center,gen,w,width=predict_block(motion,res,gate_features(p,q,state,phys),p,state,q,cad,self.lock,self.width)
        radius=width[-1]*self.lock["band"]["factor"]
        if level=="feedback":
            risk=np.full(3,.05)
            for j,o in enumerate(ids):
                if o+30>origin or bad[o+30]:
                    continue
                observed=(z[o+30]-z[o])@rot[o]
                err=np.linalg.norm(observed-center[j,-1])
                s=call[j]
                factor=self.reference[int(np.ceil((len(self.reference)-1)*(1-risk[s]))),-1]
                risk[s]=np.clip(risk[s]+.04*(.05-float(err>width[j,-1]*factor)),.01,.2)
            radius*=self.reference[int(np.ceil((len(self.reference)-1)*(1-risk[call[-1]])))]
        sup=np.concatenate((motion[-1,None],gen[-1]),0)
        return dict(sample=sample,hand=hand,origin=int(origin),time_s=float(g.t_s.iat[-1]),
                    cutoff_frame=int(g.frame_index.max()),future_time_s=(g.t_s.iat[-1]+age[-1]).tolist(),
                    probabilities=dict(zip(MODES,p[-1].tolist())),phase_decision=MODES[call[-1]],
                    highest_calibrated_phase=MODES[p[-1].argmax()],quality_risk=float(q[-1]),
                    visual_weight=vw,readout=self.lock["point"],coordinate_frame="record world, metres",
                    center_world=(z[-1]+center[-1]@rot[-1].T).tolist(),radius_m=radius.tolist(),
                    supports_world=(z[-1]+sup@rot[-1].T).tolist(),
                    support_weights=[float(1-w[-1])]+[float(w[-1]/8)]*8)


if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("sample");args.add_argument("hand",choices=["left","right"]);args.add_argument("origin",type=int)
    cfg=args.parse_args()
    print(json.dumps(WristForecaster().issue(cfg.sample,cfg.hand,cfg.origin),ensure_ascii=False,allow_nan=False))
