"""Raw measurements in local head axes, including timestamps and validity. No state filter is used."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from paths_q1 import FRAMES

JOINTS = (0,4,8,12,16,20)
BONELINKS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20))
STEPS = np.arange(1,31)
LENGTH = 96


def read_frames(sample,origin):
    columns = ["frame_index","t_s","head_x","head_y","head_z","head_qx","head_qy","head_qz","head_qw",
               "left_quality","right_quality"]+[f"{h}_j{j:02}_{a}" for h in ("left","right") for j in JOINTS for a in "xyz"]
    data = pd.read_parquet(FRAMES,columns=columns,filters=[("sample","==",sample),("frame_index","<=",int(origin))])
    return data.sort_values("frame_index").reset_index(drop=True)


def xyz(g,prefix):
    return g[[prefix+"_"+a for a in "xyz"]].to_numpy(np.float64)


def length(x):
    return np.linalg.norm(x,axis=-1)


def filled(x,initial):
    valid = np.isfinite(x).all(-1)
    if valid.all():
        return x,valid
    old = np.maximum.accumulate(np.where(valid,np.arange(len(x)),-1))
    return np.where((old>=0)[:,None],x[old.clip(min=0)],initial),valid


def unitcap(x,limit):
    return x*np.minimum(1.,limit/np.maximum(length(x),1e-7))[...,None]


def local(v,rotation,mirror):
    return np.einsum("n...j,nji->n...i",v,rotation)*mirror


def measured(g,hand):
    t = g.t_s.to_numpy(dtype=np.float64)
    if len(t)==0 or np.any(np.diff(t)<=0):
        raise ValueError("Expected a nonempty, strictly increasing wrist clock")
    h = xyz(g,"head")
    rot = Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].to_numpy()).as_matrix()
    m = np.array([1.,-1. if hand=="right" else 1.,1.])
    other = "right" if hand=="left" else "left"
    z,finite = filled(xyz(g,hand+"_j00"),h[0])
    opp,_ = filled(xyz(g,other+"_j00"),h[0])
    js,okj = zip(*(filled(xyz(g,f"{hand}_j{j:02}"),z[0]) for j in JOINTS))
    js,okj = np.stack(js,axis=1),np.stack(okj,axis=1)
    flag = g[hand+"_quality"].astype(str).to_numpy()
    ok = (flag=="ok") & finite
    o = np.arange(len(t))
    lag = lambda k: np.maximum(o-k,0)
    def velocity(x,a,b):
        dt = (t[a]-t[b]).clip(min=.001)
        return unitcap((x[a]-x[b])/dt.reshape((-1,)+(1,)*(x.ndim-1)),4.)
    relative = local(z-h,rot,m)
    node_pos = local(js-z[:,None],rot,m)
    node_pos[:,0] = relative
    node_vel = local(velocity(js,o,lag(3)),rot,m)
    nodes = np.concatenate([node_pos*4,node_vel],axis=2)
    observed = np.maximum.accumulate(np.where(ok,o,-1))
    gap = np.where(observed<0,3.,t-t[observed.clip(min=0)]).clip(0,3)
    quality = np.column_stack([ok,flag=="occlusion",flag=="near edge",finite,
                 g[other+"_quality"].eq("ok"),gap,pd.Series(ok).rolling(10,min_periods=1).mean(),okj.mean(1)])
    head_v = local(velocity(h,o,lag(5)),rot,m)
    opposite_v = local(velocity(opp,o,lag(5)),rot,m)
    dt = np.r_[1/30,np.diff(t)]
    cad = pd.Series(np.r_[np.nan,np.diff(t)]).rolling(5,min_periods=1).median().fillna(1/30).to_numpy()
    features = [relative,local(opp-z,rot,m),node_pos[:,1:].mean(1),head_v,opposite_v,
                np.clip(dt*30,0,5),np.minimum(t-t[0],3.)]
    vectors = []
    for k in (2,5,10,20,40):
        j,prev = lag(k),lag(2*k)
        v = local(velocity(z,o,j),rot,m)
        old = local(velocity(z,j,prev),rot,m)
        s,sp = length(v),length(old)
        cs = np.clip((v*old).sum(1)/np.maximum(s*sp,1e-8),-1,1)
        features.extend([v,old,s,sp,cs,length(v-old)])
        vectors.extend([v,old])
    for k in (10,30,60):
        past = pd.DataFrame(z).rolling(k,min_periods=1).mean().to_numpy()
        vectors.append(local(past-z,rot,m))
    vectors.extend([head_v,opposite_v,local(velocity(h,o,lag(20)),rot,m),relative,local(opp-z,rot,m),node_pos[:,1:].mean(1)])
    return dict(x=np.column_stack(features).astype("float32"),nodes=nodes.astype("float32"),node_ok=okj,quality=quality.astype("float32"),
                vec=np.stack(vectors,axis=1).astype("float32"),time=(t-t[0]).astype("float32"),
                cad=cad.astype("float32"),ok=ok), (t,z,rot,m)


def velocity_one(t,z):
    return pd.Series(np.r_[np.nan,length(np.diff(z,axis=0))/np.diff(t)]).rolling(5,min_periods=5).mean().to_numpy()


def mode(t,z,cut):
    o = np.arange(len(z))
    mid,old = np.maximum(o-5,0),np.maximum(o-10,0)
    a,b=z[mid]-z[old],z-z[mid]
    speed = length(b)/np.maximum(t-t[mid],.0001)
    cs = (a*b).sum(1)/np.maximum(length(a)*length(b),1e-9)
    return np.where(speed<cut,0,np.where(cs<np.cos(np.pi/4),1,2))


def bone_deviation(g,hand):
    joints = np.stack([xyz(g,f"{hand}_j{j:02}") for j in range(21)],axis=1)
    bones = np.stack([length(joints[:,a]-joints[:,b]) for a,b in BONELINKS],axis=1)
    return np.max(np.abs(bones/np.median(bones,axis=0)-1.),axis=1)
