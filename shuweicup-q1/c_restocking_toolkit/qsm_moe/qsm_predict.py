"""Issue one QSM-MoE forecast from a canonical prefix and this experiment's own fitted files."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from qsm_data import KIT,OUT,STEPS,geometry,motion_modes,quality_features,filter_track,local,diffusion_schedule
from qsm_generator import WristDiffusion,draw_knots
from qsm_train import ResidualPath,time_basis
from qsm_q1 import gate_value,interpolate


def causal_fill(z,initial):
    z=np.asarray(z).copy()
    last=np.asarray(initial).copy()
    for i,row in enumerate(z):
        mask=np.isfinite(row)
        last=np.where(mask,row,last)
        z[i]=last
    return z


@torch.inference_mode()
def issue_prefix(sample,hand,origin):
    if origin<10 or hand not in ("left","right"):
        raise ValueError("An issue needs at least eleven frames and a left or right hand.")
    other="left" if hand=="right" else "right"
    joints=(4,5,8,9,12,17,20)
    cols=["frame_index","t_s","head_x","head_y","head_z","head_qx","head_qy","head_qz","head_qw",hand+"_quality",other+"_quality"]
    cols += [f"{h}_j{j:02}_{a}" for h,js in ((hand,(0,*joints)),(other,(0,))) for j in js for a in "xyz"]
    g=pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=cols,
                      filters=[("sample","==",sample),("frame_index","<=",origin)]).sort_values("frame_index")
    if not np.array_equal(g.frame_index.to_numpy(),np.arange(origin+1)):
        raise ValueError(f"Record {sample} has no continuous prefix through frame {origin}.")
    t=g.t_s.to_numpy()
    head=g[["head_x","head_y","head_z"]].to_numpy()
    rot=Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].to_numpy()).as_matrix()
    raw=g[[f"{hand}_j00_{a}" for a in "xyz"]].to_numpy()
    raw_other=g[[f"{other}_j00_{a}" for a in "xyz"]].to_numpy()
    z=causal_fill(raw,head[0])
    opp=causal_fill(raw_other,head[0])
    fingers=g[[f"{hand}_j{j:02}_{a}" for j in joints for a in "xyz"]].to_numpy(copy=True).reshape(-1,len(joints),3)
    for j in range(len(joints)):
        fingers[:,j]=causal_fill(fingers[:,j],z[0])
    bad=g[hand+"_quality"].ne("ok").to_numpy() | ~np.isfinite(raw).all(1)
    other_bad=g[other+"_quality"].ne("ok").to_numpy() | ~np.isfinite(raw_other).all(1)
    recent=np.convolve(bad,np.ones(10),"full")[:len(z)]>0
    geom=geometry(t,z,opp,head,rot,fingers)
    seq=np.column_stack((geom,bad,other_bad,recent,np.full(len(z),hand=="right"),np.minimum(t-t[0],2))).astype("float32")
    spec=json.loads((OUT/"q1_protocol.json").read_text(encoding="utf8"))
    assert spec["version"]=="qsm-independent-2"
    lock=json.loads((OUT/"q1_locked.json").read_text(encoding="utf8"))
    with np.load(OUT/"feature_normalization.npz") as params:
        seq=np.clip((seq-params["motion_mean"])/params["motion_sd"],-8,8)
        _,state=motion_modes(t,z,float(params["mode_cut"]))
        vm,vs=params["vision_mean"],params["vision_sd"]
    qx=quality_features(t,z,bad,geom)
    prob=joblib.load(OUT/"measurement_quality.joblib").predict_proba(qx)[:,1].astype("float32")
    params=spec["quality_kalman"]
    post,vel,var=filter_track(t,raw,np.maximum(prob,bad),params["q"],params["r"],params["m"])
    p=np.column_stack((local(post-z,rot),local(vel,rot),np.sqrt(var.mean(1)))).astype("float32")
    cadence=pd.Series(np.r_[np.nan,np.diff(t)]).rolling(5,min_periods=1).median().fillna(1/30).to_numpy(dtype="float32")
    # Only previous grid endpoints have matured. No other issue needs to be evaluated to reconstruct the feedback.
    ids=np.unique(np.r_[np.arange(10,origin+1,30),origin])
    dt=cadence[ids,None]*STEPS.astype("float32")
    phys=p[ids]
    base=phys[:,None,:3]+phys[:,None,3:6]*(-spec["tau"]*np.expm1(-dt/spec["tau"]))[:,:,None]
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def tensor(v): return torch.as_tensor(v,device=dev,dtype=torch.float32)
    with np.load(KIT/"cache/literature_features"/(sample+".npz")) as f:
        stamps=f["t"]
        last=np.searchsorted(stamps,t[-1],side="right")
        rgb=np.concatenate((np.zeros((1,48,64,3),np.uint8),f["rgb"][:last]))
    cur=np.searchsorted(stamps[:last],t[ids],side="right")-1
    idx=np.stack((cur-1,cur),1)
    assert np.all(stamps[cur[cur>=0]]<=t[ids][cur>=0])
    rgb=tensor(rgb[np.where(idx>=0,idx+1,0)]).permute(0,1,4,2,3).reshape(-1,6,48,64)/127.5-1.
    gen=WristDiffusion(seq.shape[1]).to(dev)
    gen.load_state_dict(torch.load(OUT/"qsm_visual_diffusion_scratch.pt",map_location=dev,weights_only=True));gen.eval()
    vis,c,mean=gen.condition(tensor(seq[np.maximum(ids[:,None]-[18,12,8,5,3,1,0],0)]),rgb)
    schedule=diffusion_schedule()
    knots=draw_knots(gen,c,mean,tensor(np.load(OUT/"q1_sampler_noise.npy")),schedule,
                     tensor(np.r_[1.,schedule.alphas_cumprod])).cpu().numpy()
    b=interpolate(knots)
    net=ResidualPath(seq.shape[1],"B4").to(dev)
    net.load_state_dict(torch.load(OUT/"B4_scratch.pt",map_location=dev,weights_only=True));net.eval()
    a,scale,stage,change=net(tensor(seq[np.maximum(ids[:,None]-np.arange(8,-1,-1),0)]),
                  tensor(seq[np.maximum(ids[:,None]-np.arange(60,-1,-5),0)]),tensor(phys),tensor(state[ids]),
                  tensor(prob[ids]),(vis-tensor(vm))/tensor(vs),tensor(time_basis(dt)),tensor(base))
    a,scale,pi,change=(a.cpu().numpy(),scale.cpu().numpy(),stage.softmax(1).cpu().numpy(),change.sigmoid().cpu().numpy())
    phase=pi.argmax(1)
    x=np.column_stack((prob[ids],bad[ids],recent[ids],pi[:,0],pi[:,1],change,state[ids,0],state[ids,1],np.log(1e-5+phys[:,6]),scale[:,-1]))
    weight=gate_value(x,lock["gate"])
    supports=np.concatenate((a[:,None],b),1)
    weights=np.column_stack((1-weight,np.broadcast_to(weight[:,None]/8,(len(ids),8))))
    flat=supports.reshape(len(ids),9,-1)
    distances=np.linalg.norm(flat[:,:,None]-flat[:,None,:],axis=3)/np.sqrt(30)
    choice=(distances*weights[:,None]).sum(2).argmin(1)
    center=supports[np.arange(len(ids)),choice]
    spread=(weights[:,:,None]*np.square(supports-center[:,None]).sum(3)).sum(1)
    scale=np.sqrt(scale**2+.25*spread)
    ref=np.load(OUT/"q1_feedback_reference.npy")
    risk=np.full(3,.05)
    for i,o in enumerate(ids):
        if o+30>origin or bad[o+30]:
            continue
        actual=(z[o+30]-z[o])@rot[o]
        err=np.linalg.norm(actual-center[i,-1])
        s=phase[i]
        q=ref[int(np.ceil((len(ref)-1)*(1-risk[s]))),-1]
        risk[s]=np.clip(risk[s]+.04*(.05-float(err>q*scale[i,-1])),.01,.20)
    radius=scale[-1]*ref[int(np.ceil((len(ref)-1)*(1-risk[phase[-1]])))]*lock["feedback_factor"]
    world=z[-1]+supports[-1]@rot[-1].T
    return dict(sample=sample,hand=hand,origin=int(origin),timestamp_s=float(t[-1]),
                coordinate_frame="record world, metres",last_wrist_world=z[-1].tolist(),future_time_s=(t[-1]+dt[-1]).tolist(),
                stage_probabilities=dict(zip(("dwell","turn","translate"),pi[-1].tolist())),
                quality_risk=float(prob[-1]),phase_change_risk=float(change[-1]),expert_weights=weights[-1].tolist(),
                selected_support=int(choice[-1]),supports_world=world.tolist(),center_world=world[choice[-1]].tolist(),
                radius_m=radius.tolist())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("sample");parser.add_argument("hand",choices=["left","right"]);parser.add_argument("origin",type=int)
    cfg=parser.parse_args()
    print(json.dumps(issue_prefix(cfg.sample,cfg.hand,cfg.origin),ensure_ascii=False,allow_nan=False))
