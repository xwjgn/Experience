"""Point readout, fixed-width comparisons, and simultaneous calibration of balls and ellipsoids."""
import math
import numpy as np

SHARES=(0.,.10,.25,.40,.60,.80,1.)
POWERS=(0.,.20,.40,.65,1.)


def geometric(cov):
    cov=np.asarray(cov,dtype=np.float64)
    cov=(cov+cov.swapaxes(-1,-2))*.5
    values,vectors=np.linalg.eigh(cov)
    if not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError("Non-positive or nonfinite 3-D covariance")
    return dict(values=values,vectors=vectors,radial=np.sqrt(values.sum(-1)),
                volume=np.exp(np.log(values).mean(-1)*.5),max_axis=np.sqrt(values[...,-1]))


def distance_axes(error,geom):
    coords=np.einsum("...ji,...j->...i",geom["vectors"],error)
    return np.sqrt(np.sum(coords**2/geom["values"],axis=-1))


def nll_values(error,geom):
    d2=distance_axes(error,geom)**2
    trace=geom["values"].sum(-1)
    full=.5*(d2+np.log(geom["values"]).sum(-1)+3*np.log(2*np.pi))
    iso=.5*(3*np.square(error).sum(-1)/trace+3*np.log(trace/3)+3*np.log(2*np.pi))
    return full,iso


def quantile(x,level=.95):
    vals=np.asarray(x,dtype=float)
    rank=math.ceil((len(vals)+1)*level)-1
    if rank>=len(vals) or rank<0:
        raise ValueError("Too few complete windows for requested conformal level.")
    return float(np.partition(vals,rank)[rank])


def path_metrics(center,truth,radius,turn,env):
    error=np.linalg.norm(center-truth,axis=2)
    hit=(error<=radius).all(1)
    return dict(n=len(hit),turns=int(turn.sum()),ade_cm=float(100*error.mean()),turn_ade_cm=float(100*error[turn].mean()),
                fde_cm=float(100*error[:,-1].mean()),covered=int(hit.sum()),coverage=float(hit.mean()),
                turn_covered=int(hit[turn].sum()),turn_coverage=float(hit[turn].mean()),radius_cm=float(100*radius.mean()),
                least_env=float(min(hit[env==e].mean() for e in np.unique(env))))


def third_mix(base,bayes,covgeom,share,mode):
    if share==0.:
        return base.copy()
    if share==1. or mode=="scalar":
        return base+share*(bayes-base)
    spectrum=covgeom["values"]
    relative=spectrum/spectrum.mean(-1,keepdims=True)
    # Reduce the new expert's influence in directions in which that expert is less certain.
    # This decision rule does not assume independence of the two forecasters.
    weight=np.clip(share/(share+(1-share)*relative),.45*share,min(1.,1.75*share))
    vec=covgeom["vectors"]
    diff=np.einsum("...ji,...j->...i",vec,bayes-base)
    return base+np.einsum("...ji,...i->...j",vec,weight*diff)


def redistribute(reference,scale,power):
    if power==0.:
        return reference.copy()
    rel=np.power(scale/reference,power)
    adjust=rel/(np.sum(reference*rel,axis=1)/reference.sum(1))[:,None]-1
    damping=np.minimum(1.,np.minimum(.25/np.maximum(-adjust.min(1),1.e-9),1.5/np.maximum(adjust.max(1),1.e-9)))
    return reference*(1+damping[:,None]*adjust)


def choose(base,r0,bayes,geom,truth,turn,env):
    original=np.linalg.norm(base-truth,axis=2)
    def point_value(err):
        ade=100*err.mean(1)
        return float(.45*ade[turn].mean()+.35*np.mean([ade[env==e].mean() for e in np.unique(env)])+.20*ade.mean())
    trials=[]
    for mode in ("scalar","directional"):
        for share in SHARES:
            if mode=="directional" and share in (0.,1.):
                continue
            mu=third_mix(base,bayes,geom,share,mode)
            err=np.linalg.norm(mu-truth,axis=2)
            admissible=err.mean()<=original.mean()+1.e-9 and err[turn].mean()<=original[turn].mean()+1.e-9
            trials.append(dict(mode=mode,share=share,objective=point_value(err),admissible=bool(admissible),
                               ade_cm=float(100*err.mean()),turn_ade_cm=float(100*err[turn].mean())))
    choice=min((p for p in trials if p["admissible"]),key=lambda p:(p["objective"],p["share"]))
    center=third_mix(base,bayes,geom,choice["share"],choice["mode"])
    error=np.linalg.norm(center-truth,axis=2)
    old_hit=(original<=r0).all(1)
    bands=[]
    for g in POWERS:
        r=redistribute(r0,geom["radial"],g)
        score=(error/r).max(1)
        hit=score<=1
        bands.append(dict(power=g,turn_lost=int((old_hit & turn & ~hit).sum()),
                          deficit=float(max(0.,.95-hit.mean(),.95-hit[turn].mean())),
                          covered=int(hit.sum()),turn_covered=int(hit[turn].sum()),
                          stress_misses=int(sum((score>q).sum() for q in (.90,1.,1.08))),
                          tail=float(np.mean(np.maximum(score-.8,0)))))
    band=min(bands,key=lambda b:(b["turn_lost"],b["deficit"],-b["covered"],b["stress_misses"],b["tail"],b["power"]))
    return dict(mode=choice["mode"],share=choice["share"],power=band["power"]),trials,bands


def view_geometry(g,select):
    return {k:a[select] for k,a in g.items()}


def hybrid_freeze(base,r0,pred,truth,turn,env):
    geom=geometric(pred["cov"])
    oof_center=np.empty_like(base)
    oof_radius=np.empty_like(r0)
    folds={}
    for e in np.unique(env):
        hold=env==e
        spec,_,_=choose(base[~hold],r0[~hold],pred["mean"][~hold],view_geometry(geom,~hold),
                        truth[~hold],turn[~hold],env[~hold])
        oof_center[hold]=third_mix(base[hold],pred["mean"][hold],view_geometry(geom,hold),spec["share"],spec["mode"])
        oof_radius[hold]=redistribute(r0[hold],geom["radial"][hold],spec["power"])
        folds[e]=spec
    chosen,points,bands=choose(base,r0,pred["mean"],geom,truth,turn,env)
    err=truth-oof_center
    ratio=(np.linalg.norm(err,axis=2)/oof_radius).max(1)
    orig_hit=(np.linalg.norm(base-truth,axis=2)<=r0).all(1)
    covered_turn=turn & orig_hit
    q=quantile(ratio)
    guarded=max(q,float(ratio[covered_turn].max()),*(quantile(ratio[env==e]) for e in np.unique(env)))
    mahal=distance_axes(err,geom).max(1)
    return dict(**chosen,folds=folds,q_pooled=q,q_guarded=guarded,ellipsoid_q=quantile(mahal),
                oof=path_metrics(oof_center,truth,oof_radius,turn,env)),points,bands


def standalone_freeze(gate,cal,gtruth,ctruth):
    gg,cc=geometric(gate["cov"]),geometric(cal["cov"])
    gerr=np.linalg.norm(gate["mean"]-gtruth,axis=2)
    cerr=cal["mean"]-ctruth
    flat=np.maximum(np.quantile(gerr,.75,axis=0),.0005)
    return dict(sphere_q=quantile((np.linalg.norm(cerr,axis=2)/cc["radial"]).max(1)),
                ellipsoid_q=quantile(distance_axes(cerr,cc).max(1)),flat_shape=flat.tolist(),
                flat_q=quantile((np.linalg.norm(cerr,axis=2)/flat).max(1)),
                gate_windows=len(gg["radial"]),calibration_windows=len(cerr))
