"""Causal corrections for uncertain changes in wrist direction. Turn labels are training targets only."""
import numpy as np
from scipy.spatial import cKDTree

from config import WORK
from features import KNOTS
from motion import INTERP

STEPS=np.arange(1,31,dtype=np.float32)
FAMILIES=("brake","body_brake","broad_regression","turn_regression","hard_alert",
          "turn_analog","regression_analog","brake_regression")
STRENGTHS=(.25,.5,.75,1.)


def unit(v,default=None):
    length=np.linalg.norm(v,axis=-1,keepdims=True)
    result=v/np.maximum(length,1e-8)
    return np.where(length>.015,result,default) if default is not None else result


def limit(v,radius):
    return v*np.minimum(1.,np.asarray(radius)/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8))


def rotation_features(v,cad):
    # Five- and ten-frame displacements share the frozen head axes at the issue time.
    # The previous five-frame average is estimated from them under the trailing cadence.
    current=v[:,3]-.5*v[:,4]
    earlier=2*v[:,7]-v[:,3]-.5*v[:,4]
    cross=np.cross(earlier,current)
    cosine=(earlier*current).sum(1)
    angle=np.arctan2(np.linalg.norm(cross,axis=1),cosine)
    trust=np.minimum(1.,np.minimum(np.linalg.norm(current,axis=1),np.linalg.norm(earlier,axis=1))/.16)
    omega=np.clip(angle/np.maximum(5*cad,.01),0,7)*trust
    lateral=np.cross(unit(cross),current)
    return current,lateral,omega


def base_vectors(v,step,cad,center_k):
    age=np.maximum(cad*step,.002)
    cur,lat,omega=rotation_features(v,cad)
    elapsed=-.36*np.expm1(-age/.36)
    angle=omega*elapsed
    arc=(cur*(elapsed*np.sinc(angle/np.pi))[:,None]+
         lat*(elapsed*(1-np.cos(angle))/np.maximum(angle,1e-7))[:,None])
    head=v[:,4]*(-.65*np.expm1(-age/.65))[:,None]
    shape=[center_k/age[:,None],(head+arc-center_k)/age[:,None],
           (head-center_k)/age[:,None],lat,cur]
    return np.concatenate((np.stack(shape,axis=1),v[:,[0,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22]]),axis=1)


def design(v,p,state,cad,center_k,step):
    vectors=base_vectors(v,step,cad,center_k)
    speed,cos=state.T
    context=np.column_stack((np.ones(len(v)),p[:,0]-.35,p[:,1]-.15,
                 speed/(speed+.28)-.5,.5*(1-cos)-.25))
    return (context[:,:,None,None]*vectors[:,None]).reshape(len(v),-1,3).astype(np.float32)


def fit_regressions(d,center):
    import torch
    ids=np.flatnonzero(d.fit & d.mask.all(1) & (d.meta.origin.to_numpy()%3==1))
    turns=np.load(WORK/"cache"/"next_mode.npy",mmap_mode="r")[ids]==1
    _,inv,counts=np.unique(d.env[ids],return_inverse=True,return_counts=True)
    unit_env=1/counts[inv]
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results={}
    for family in ("broad","turn"):
        use=np.ones(len(ids),bool) if family=="broad" else turns
        ix=ids[use]
        base_weight=unit_env[use]*(1+5*turns[use] if family=="broad" else 1.)
        weights=torch.tensor(base_weight,dtype=torch.float32,device=dev)
        scales=[];coefs=[]
        for h in KNOTS:
            x=design(d.v[ix],d.phase[ix],d.state[ix],d.cad[ix],center[ix,h-1],h)
            sd=np.maximum(np.sqrt(np.mean(x*x,axis=(0,2))),.04)
            z=torch.tensor((x/sd[None,:,None]).transpose(0,2,1).copy(),device=dev)
            age=d.cad[ix,None]*h
            y=torch.tensor((d.y[ix,h-1]-center[ix,h-1])/age,dtype=torch.float32,device=dev)
            flat=z.flatten(0,1)
            ident=torch.eye(z.shape[-1],device=dev)
            w=weights/weights.sum()
            for _ in range(5):
                weighted=(z*w[:,None,None]).flatten(0,1)
                coef=torch.linalg.solve(weighted.T@flat+.055*ident,weighted.T@y.flatten())
                distance=(z@coef-y).norm(dim=1)
                w=weights/distance.clamp_min(.065)
                w/=w.sum()
            scales.append(sd); coefs.append(coef.cpu().numpy())
        results[family+"_scale"]=np.stack(scales)
        results[family+"_coef"]=np.stack(coefs)
    np.savez(WORK/"turn_regressions.npz",**results)
    return results


def regress(v,p,state,cad,center,params,family):
    knots=[]
    for j,h in enumerate(KNOTS):
        x=design(v,p,state,cad,center[:,h-1],h)
        coef=params[family+"_coef"][j]/params[family+"_scale"][j]
        vel=np.einsum("nkd,k->nd",x,coef,optimize=True)
        knots.append(limit(vel,.32)*(cad*h)[:,None])
    return (INTERP@np.stack(knots,axis=1)).astype(np.float32)


def embedding(x):
    # The standardized input is already mirrored for right wrists. Weight pose, local
    # direction changes and longer motion history separately in the retrieval distance.
    return np.concatenate((.70*x[:,:9],x[:,14:16],.6*x[:,16:30],.85*x[:,30:44],
                           .60*x[:,44:58],.45*x[:,58:72],.55*x[:,-12:]),axis=1).clip(-3,3).astype(np.float32)


def frame(v,right):
    mirror=np.column_stack((np.ones(len(v)),np.where(right,-1.,1.),np.ones(len(v)))).astype(np.float32)
    tangent=unit((.65*v[:,3]+.35*v[:,0]-.35*v[:,4])*mirror,np.array([1.,0.,0.],np.float32))
    lateral=np.cross(np.broadcast_to([0.,0.,1.],tangent.shape),tangent)
    fallback=np.cross(np.broadcast_to([0.,1.,0.],tangent.shape),tangent)
    lateral=unit(lateral,unit(fallback))
    axes=np.stack((tangent,lateral,np.cross(tangent,lateral)),2).astype(np.float32)
    return axes,mirror


def build_library(d,center):
    labels=np.load(WORK/"cache"/"next_mode.npy",mmap_mode="r")
    eligible=d.fit & d.mask.all(1) & (labels==1)
    take=[]
    for rows in d.meta.groupby(["sample","hand"],sort=False).indices.values():
        options=rows[eligible[rows]]
        last=-100
        for row in options:
            if row-last>=7:
                take.append(row)
                last=row
    ids=np.array(take)
    axes,mir=frame(d.v[ids],d.meta.hand.to_numpy()[ids]=="right")
    err=d.y[ids][:,KNOTS-1]-center[ids][:,KNOTS-1]
    local=(err*mir[:,None])@axes
    library=dict(x=embedding(d.x[ids]),residual=local.astype(np.float32),
           sample=d.meta["sample"].to_numpy()[ids].astype(str),origin=d.meta.origin.to_numpy()[ids],
           hand=d.meta.hand.to_numpy()[ids].astype(str))
    np.savez_compressed(WORK/"turn_analog_library.npz",**library)
    return library


class TurnExperts:
    def __init__(self,families=FAMILIES):
        names=set(families)
        if names & {"broad_regression","turn_regression","hard_alert","regression_analog","brake_regression"}:
            with np.load(WORK/"turn_regressions.npz") as p:
                self.reg={k:p[k] for k in p.files}
        if names & {"turn_analog","regression_analog"}:
            with np.load(WORK/"turn_analog_library.npz") as lib:
                self.lib={k:lib[k] for k in lib.files}
            self.tree=cKDTree(self.lib["x"])

    def analog(self,x,v,right,sample):
        ds,inds=self.tree.query(embedding(x),k=min(64,len(self.lib["x"])),workers=1)
        axes,mir=frame(v,right)
        chosen=np.zeros((len(x),len(KNOTS),3),np.float32)
        for i,(d,neighbors) in enumerate(zip(ds,inds)):
            ids=[]; distances=[]; counts={}
            for dist,j in zip(d,neighbors):
                s=self.lib["sample"][j]
                if s==sample[i] or counts.get(s,0)>=2:
                    continue
                ids.append(j);distances.append(dist)
                counts[s]=counts.get(s,0)+1
                if len(ids)==14: break
            if not ids: continue
            distances=np.array(distances)
            weight=1/np.maximum(distances,.7)**2
            residual=self.lib["residual"][ids]
            mid=np.average(residual,axis=0,weights=weight)
            for _ in range(4):
                distance=np.sqrt(np.square(residual-mid).sum(2).mean(1)+.01**2)
                mid=np.average(residual,axis=0,weights=weight/distance)
            chosen[i]=mid
        return (INTERP@((chosen@axes.transpose(0,2,1))*mir[:,None])).astype(np.float32)

    def corrections(self,names,center,v,x,p,state,cad,right,sample,alert):
        names=set(names)
        risk=np.clip((p[:,1]-.025)/.25,.08,1.)*(1-.25*p[:,0])
        soft=risk[:,None,None]
        age=cad[:,None]*STEPS
        ans={}
        if names & {"brake","body_brake","brake_regression"}:
            softness=1-np.exp(-age/.12)
            head=v[:,None,4]*(-.55*np.expm1(-age/.55))[:,:,None]
            ans["brake"]=-.55*soft*softness[:,:,None]*center
            ans["body_brake"]=-.55*soft*softness[:,:,None]*(center-head)
        if "broad_regression" in names:
            ans["broad_regression"]=regress(v,p,state,cad,center,self.reg,"broad")
        if names & {"turn_regression","hard_alert","regression_analog","brake_regression"}:
            expert=regress(v,p,state,cad,center,self.reg,"turn")
            ans["turn_regression"]=soft*expert
            ans["hard_alert"]=alert[:,None,None]*expert
        if names & {"turn_analog","regression_analog"}:
            retrieved=self.analog(x,v,right,sample)
            ans["turn_analog"]=soft*limit(retrieved,(.015+.19*age)[:,:,None])
        if "brake_regression" in names:
            ans["brake_regression"]=.5*(ans["brake"]+ans["turn_regression"])
        if "regression_analog" in names:
            ans["regression_analog"]=.5*(ans["turn_regression"]+ans["turn_analog"])
        return {name:ans[name] for name in names}
