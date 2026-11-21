"""Causal multi-scale wrist kinematics and wrist-local temporal image changes."""
import sys
import json

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.spatial.transform import Rotation

from settings import WORK, KIT, OUT, array
from qsm_data import local, motion_modes
sys.path.insert(0,str(KIT/"literature_experiments"))
from object_frames import projected_hands

KNOTS = np.array([1,3,6,10,15,22,30])
JOINTS = (0,4,5,8,9,12,13,17,20)
COLS = ["frame_index","t_s","head_x","head_y","head_z","head_qx","head_qy","head_qz","head_qw",
        "cam_x","cam_y","cam_z","cam_qx","cam_qy","cam_qz","cam_qw","cam_fx","cam_fy","cam_cx","cam_cy","cam_width","cam_height",
        "left_quality","right_quality"]+[f"{side}_j{j:02}_{a}" for side in ("left","right") for j in JOINTS for a in "xyz"]


def rows(sample, origin=None):
    filters=[("sample","==",sample)]
    if origin is not None:
        filters.append(("frame_index","<=",int(origin)))
    return pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=COLS,filters=filters).sort_values("frame_index")


def xyz(g, key):
    return g[[key+"_"+a for a in "xyz"]].to_numpy(dtype=float)


def wrist(g, hand):
    return xyz(g,hand+"_j00")


def norm(x):
    return np.linalg.norm(x,axis=-1)


def clipvec(x,limit):
    return x*np.minimum(1.,limit/np.maximum(norm(x),1e-8))[...,None]


def cosine(a,b):
    return np.clip((a*b).sum(-1)/np.maximum(norm(a)*norm(b),1e-6),-1,1)


def image_pairs(rgb):
    gray=np.einsum("nhwc,c->nhw",rgb.astype("float32")/255.,np.array([.299,.587,.114],np.float32))
    old=np.concatenate((gray[:1],gray[:-1]))
    gy,gx=np.gradient((gray+old)*.5,axis=(1,2))
    change=gray-old
    # Local Lucas–Kanade flow. Window operators act on one image pair at a time.
    def box(x): return uniform_filter(x,size=(1,7,7),mode="nearest")
    xx,yy,xy=box(gx*gx)+.001,box(gy*gy)+.001,box(gx*gy)
    xt,yt=box(gx*change),box(gy*change)
    determinant=np.maximum(xx*yy-xy*xy,1e-8)
    fx=np.clip((xy*yt-yy*xt)/determinant,-6,6)
    fy=np.clip((xy*xt-xx*yt)/determinant,-6,6)
    diff=box(np.abs(change))
    texture=box(gx*gx+gy*gy)
    global_xy=np.stack((np.median(fx,axis=(1,2)),np.median(fy,axis=(1,2))),1)
    return fx,fy,diff,texture,global_xy


def video_features(g, stamps, pair, hand, uv):
    t=g.t_s.to_numpy()
    result=np.zeros((len(g),10),np.float32)
    if not len(stamps):
        return result
    fx,fy,diff,texture,global_xy=pair
    uv=uv[hand][:,0]
    finite=np.isfinite(uv).all(1)
    fov=finite & (uv>=.03).all(1) & (uv<=.97).all(1)
    px=(np.nan_to_num(uv,nan=0.,posinf=0.,neginf=0.).clip(0,.9999)*[fx.shape[2],fx.shape[1]]).astype(int)
    ids=np.arange(len(stamps))
    u,v=px.T
    fxp,fyp=fx[ids,v,u],fy[ids,v,u]
    pixels=np.column_stack((fxp-global_xy[:,0],fyp-global_xy[:,1],fxp,fyp,diff[ids,v,u],
                            texture[ids,v,u],diff.mean((1,2)),np.linalg.norm(global_xy,axis=1)))
    at=np.searchsorted(stamps,t,side="right")-1
    i=at.clip(min=0)
    age=t-stamps[i]
    pair_dt=stamps-np.r_[stamps[0],stamps[:-1]]
    good=(at>=1)&fov[i]&(age>=0)&(age<=.30)&(pair_dt[i]>0)&(pair_dt[i]<=.32)
    result[good,:8]=pixels[i[good]]
    result[good,8]=age[good]
    result[good,9]=1.
    return result


def kinematics(g,hand,physics,quality,mode_cut):
    t=g.t_s.to_numpy()
    i=np.arange(len(t))
    side=-1. if hand=="right" else 1.
    mirror=np.array([1.,side,1.])
    z=wrist(g,hand)
    opposite=wrist(g,"right" if hand=="left" else "left")
    head=xyz(g,"head")
    rot=Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].to_numpy()).as_matrix()
    tips=np.stack([xyz(g,f"{hand}_j{j:02}") for j in (4,8,12,20)],1)
    hand_rel=local(z-head,rot)
    palm=local(tips.mean(1)-z,rot)
    mode,motion=motion_modes(t,z,mode_cut)
    features=[hand_rel*mirror,local(opposite-z,rot)*mirror,palm*mirror,quality,
              norm(tips[:,0]-tips[:,1]), norm(tips[:,1]-tips[:,3]),
              np.minimum(t-t[0],3.), np.log1p(physics[:,6]*1000), motion]
    vectors=[physics[:,3:6],physics[:,:3]*10]
    def delta(pos,a,b):
        return (pos[a]-pos[b])/np.maximum(t[a]-t[b],.001)[:,None]
    for lag in (2,5,10,20,40):
        m=np.maximum(i-lag,0); l=np.maximum(i-2*lag,0)
        w=clipvec(delta(z,i,m),3.5); old=clipvec(delta(z,m,l),3.5)
        h=clipvec(delta(head,i,m),2.5); o=clipvec(delta(opposite,i,m),3.5)
        speed,last=norm(w),norm(old)
        features.extend([speed,last,cosine(w,old),norm(w-old),speed-last,norm(h),norm(o),
                         cosine(w,h),cosine(w,o),norm(w-h),cosine(w-h,old-delta(head,m,l))])
        if lag in (2,5,10,20):
            features.append(local(w,rot)*mirror)
        vectors.append(local(w,rot))
        if lag in (5,20):
            vectors.extend((local(h,rot),local(o,rot)))
            co_rotate=np.einsum("nij,nj->ni",rot[m],hand_rel)
            spin=(z-head-co_rotate)/np.maximum(t-t[m],.001)[:,None]
            vectors.append(local(clipvec(spin,3.),rot))
    for window in (10,30,60):
        mean=pd.DataFrame(hand_rel).rolling(window,min_periods=1).mean().to_numpy()
        vectors.append((mean-hand_rel)*3)
        prev=np.maximum(i-window,0)
        lengths=np.r_[0.,np.cumsum(norm(np.diff(z,axis=0)))]
        features.extend([norm(z-z[prev])/np.maximum(lengths-lengths[prev],1e-5),
                         pd.Series(mode==1).rolling(window,min_periods=1).mean().to_numpy(),
                         pd.Series(mode==0).rolling(window,min_periods=1).mean().to_numpy()])
    last_turn=np.maximum.accumulate(np.where(mode==1,i,-1))
    since=np.where(last_turn>=0,t-t[last_turn.clip(min=0)],3.)
    features.extend([np.minimum(since,3.),*(mode==j for j in range(3))])
    # Spatial vector coefficients are shared between axes and between hands.
    vectors.extend([hand_rel,local(opposite-z,rot),palm,local(np.tile([0.,0.,1.],(len(i),1)),rot)])
    for axis in np.eye(3)*mirror:
        vectors.append(np.broadcast_to(axis,(len(i),3)))
    scalar=np.column_stack(features).astype("float32")
    vec=np.stack(vectors,1).astype("float32")
    if not np.isfinite(scalar).all() or not np.isfinite(vec).all():
        raise ValueError("Non-finite visible kinematics")
    return scalar,vec


def camera_data(sample,g):
    with np.load(KIT/"cache/literature_features"/(sample+".npz")) as f:
        all_t=f["t"]
        count=np.searchsorted(all_t,g.t_s.iat[-1],side="right")
        stamps,images=all_t[:count],f["rgb"][:count]
    if count==0:
        return stamps,None,None
    return stamps,image_pairs(images),projected_hands(g,stamps)


def make_features():
    meta=pd.read_parquet(WORK/"cache"/"meta.parquet")
    physics=array("physics")
    q=array("prob")
    with np.load(WORK/"feature_normalization.npz") as f:
        cut=float(f["mode_cut"])
    blocks={}
    for sample, group in meta.groupby("sample",sort=False):
        g=rows(sample)
        stamps,pair,uv=camera_data(sample,g)
        for hand,sub in group.groupby("hand",sort=False):
            idx=sub.index.to_numpy()
            np.testing.assert_array_equal(sub.origin.to_numpy(),g.frame_index.to_numpy())
            scalars,vectors=kinematics(g,hand,physics[idx,1],q[idx],cut)
            visual=video_features(g,stamps,pair,hand,uv)
            if not blocks:
                for k,a in (("kine",scalars),("vectors",vectors),("local_video",visual)):
                    blocks[k]=np.empty((len(meta),*a.shape[1:]),np.float32)
            for k,a in (("kine",scalars),("vectors",vectors),("local_video",visual)):
                blocks[k][idx]=a
    for key,values in blocks.items():
        np.save(WORK/(key+".npy"),values)
    spec=json.loads((WORK/"cache"/"ready.json").read_text(encoding="utf8"))
    fit=meta.env.isin(spec["environments"]["fit"]).to_numpy()
    x=blocks["kine"]
    mu=x[fit].mean(0,dtype=float).astype("float32")
    sd=np.maximum(x[fit].std(0,dtype=float),.02).astype("float32")
    video=blocks["local_video"]
    valid=fit & (video[:,-1]>0)
    np.savez(WORK/"v2_normalization.npz",mu=mu,sd=sd,video_mu=video[valid,:-1].mean(0),
             video_sd=np.maximum(video[valid,:-1].std(0),.002))
    print("New inputs",[(k,v.shape) for k,v in blocks.items()],"usable projected image pairs",int(video[:,-1].sum()),flush=True)


if __name__=="__main__":
    make_features()
