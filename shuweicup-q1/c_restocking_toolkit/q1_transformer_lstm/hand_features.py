"""Geometric inputs, displayed quality flags and evaluation-only references."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from release_reader import SIDES

H = 30
HISTORY = 80
FINGERS = np.array([[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]])
BONES = np.stack((FINGERS[:,:-1],FINGERS[:,1:]),-1).reshape(-1,2)


def norm(x):
    return np.linalg.norm(x,axis=-1)


def carry(x,initial):
    x = np.asarray(x,dtype=float)
    valid = np.isfinite(x).all(-1)
    if valid.all():
        return x,valid
    ix = np.maximum.accumulate(np.where(valid,np.arange(len(x)),-1))
    return np.where((ix>=0)[:,None],x[ix.clip(min=0)],initial),valid


def transform(v,rot,mirror):
    return np.einsum("i...j,ijk->i...k",v,rot)*mirror


def speed_trace(t,z):
    return pd.Series(np.r_[np.nan,norm(np.diff(z,axis=0))/np.diff(t)]).rolling(5,min_periods=5).mean().to_numpy()


def stage(t,z,speed_cut):
    i = np.arange(len(z))
    m,b = (i-5).clip(0),(i-10).clip(0)
    v,prev = z-z[m],z[m]-z[b]
    cos = (prev*v).sum(1)/np.maximum(norm(prev)*norm(v),1e-9)
    return np.where(norm(v)/np.maximum(t-t[m],1e-4)<speed_cut,0,np.where(cos<np.cos(np.pi/4),1,2))


def reference_geometry(joints):
    size = norm(joints[:,BONES[:,0]]-joints[:,BONES[:,1]])
    return np.max(np.abs(size/np.maximum(np.median(size,axis=0),1e-9)-1),axis=1)


def encode(record,hand):
    s = SIDES.index(hand)
    t = record.t
    i = np.arange(len(t))
    h,_ = carry(record.head,np.zeros(3))
    quat,_ = carry(record.quat,np.array([0.,0.,0.,1.]))
    rot = Rotation.from_quat(quat).as_matrix()
    flip = np.array([-1. if s else 1.,1.,1.])
    points,masks = zip(*(carry(record.joints[:,s,j],h[0]) for j in range(21)))
    joints = np.stack(points,1)
    marks = np.stack(masks,1)
    z = joints[:,0]
    other,_ = carry(record.joints[:,1-s,0],h[0])
    def local(v):
        return transform(v,rot,flip)
    def velocity(x,lo,hi):
        dt = np.maximum(t[hi]-t[lo],.001)
        v = (x[hi]-x[lo])/dt.reshape((-1,)+(1,)*(x.ndim-1))
        return v*np.minimum(1.,4./np.maximum(norm(v),1e-7))[...,None]
    roots,tips = FINGERS[:,1],FINGERS[:,-1]
    selected = np.r_[0,tips]
    tip = joints[:,selected]
    relative = local(tip-z[:,None])
    relative[:,0] = local(z-h)
    proximal = local(joints[:,np.r_[0,roots]]-z[:,None])
    proximal[:,0] = local(other-z)
    lengths = norm(np.diff(joints[:,FINGERS],axis=2))
    recent = pd.DataFrame(lengths.reshape(len(t),-1)).rolling(25,min_periods=1).median().to_numpy().reshape(lengths.shape)
    irregular = ((lengths-recent)/np.maximum(recent,.007)).clip(-2,2)
    geometry = np.concatenate([lengths*8,irregular],axis=2)
    geometry = np.concatenate([geometry.mean(1,keepdims=True),geometry],axis=1)
    jmask = np.column_stack([marks[:,0],marks[:,FINGERS[:,1:]].all(2)])
    nodes = np.concatenate((relative,proximal,local(velocity(tip,(i-3).clip(0),i)),
                            local(velocity(tip,(i-9).clip(0),i)),geometry,jmask[:,:,None]),axis=2)
    high = []
    common = [local(z-h),local(other-z),relative[:,1:].mean(1),np.minimum(t-t[0],3.)]
    for lag in (2,6,12,24):
        earlier,old = (i-lag).clip(0),(i-2*lag).clip(0)
        v,prior = local(velocity(z,earlier,i)),local(velocity(z,old,earlier))
        vlen,plen = norm(v),norm(prior)
        common.extend([v,prior,vlen,plen,norm(v-prior),(v*prior).sum(1)/np.maximum(vlen*plen,1e-6)])
        high.extend((v,prior))
    hv,ov = local(velocity(h,(i-8).clip(0),i)),local(velocity(other,(i-8).clip(0),i))
    common.extend((hv,ov))
    for win in (12,36):
        mean = pd.DataFrame(z).rolling(win,min_periods=1).mean().to_numpy()
        high.append(local(mean-z)*3)
    high.extend([hv,ov,local(z-h),relative[:,1:].mean(1)])
    flag,opp = record.flags[:,s],record.flags[:,1-s]
    good = (flag==0) & marks[:,0]
    last = np.maximum.accumulate(np.where(good,i,-1))
    since = np.where(last>=0,t-t[last.clip(min=0)],3.).clip(0,3)
    qinfo = np.column_stack([flag==1,flag==2,flag==3,good,opp==0,since,
                 pd.Series(good).rolling(8,min_periods=1).mean(),jmask.mean(1)])
    dt = np.r_[1/30,np.diff(t)]
    common.append((dt*30).clip(0,5))
    cad = pd.Series(np.r_[np.nan,np.diff(t)]).rolling(5,min_periods=1).median().fillna(1/30).to_numpy()
    return dict(nodes=nodes.astype("float32"),global_x=np.column_stack(common).astype("float32"),
                node_mask=jmask,qinfo=qinfo.astype("float32"),basis=np.stack(high,1).astype("float32"),
                finite=marks[:,0],good=good,time=(t-t[0]).astype("float32"),cad=cad.astype("float32")), (t,z,rot,flip)
