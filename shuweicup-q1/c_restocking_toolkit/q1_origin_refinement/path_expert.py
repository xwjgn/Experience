"""A fresh wrist trajectory expert in a frame aligned with the current wrist velocity."""
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wrist_history import DATA,M4_DIR,OUT,STEPS,load_split,protect_weights,read,save
from event_models import Scorer
from freeze_choices import fingerprint,reference
from horizon_budget import fit_context,issue_radii

KNOTS = (1,3,6,10,15,20,25,30)
SHARES = (0.,.15,.30,.50,.75,1.)


def frame_and_inputs(seq,tab):
    last = seq[:,-1]
    n=len(last)
    heading=last[:,3:6].copy()
    for fallback in (last[:,6:9],last[:,21:24],np.broadcast_to([1.,0.,0.],(n,3))):
        small=np.linalg.norm(heading,axis=1)<.045
        heading[small]=fallback[small]
    heading/=np.linalg.norm(heading,axis=1)[:,None]
    up=np.tile([0.,0.,1.],(n,1))
    up[np.abs(heading[:,2])>.87]=[0,1,0]
    side=np.cross(up,heading)
    side/=np.linalg.norm(side,axis=1)[:,None]
    basis=np.stack((heading,side,np.cross(heading,side)),axis=2)
    vectors=[]
    for point in (last,seq[:,-4],seq.mean(1)):
        # These triplets are polar vectors in the origin head axes, except the angular/palm normal triplets omitted here.
        vec=point[:,:30].reshape(n,10,3)[:,[0,1,2,3,5,6,7,9]]
        vectors.append(np.einsum("nqj,njk->nqk",vec,basis).reshape(n,-1))
    extra = [c for c in tab if c.startswith(("w_speed","w_heading","w_straight","current_","past_","h_angular_speed",
             "innovation_","noise_","palm_rotation_","f_opening","f_wrist_distance"))]
    x=np.column_stack([*vectors,tab[extra]]).astype("float32")
    cadence=np.clip(seq[:,-2,-2]/2,.015,.075)
    return basis,x,cadence


def fit_expert(fit,gate):
    basis,x,dt=frame_and_inputs(fit.seq,fit.features)
    raw=np.load(DATA/"y.npy",mmap_mode="r")
    truth=raw[fit.meta.row.to_numpy()][:,np.array(KNOTS)-1]
    target=np.einsum("nhj,njk->nhk",truth,basis)/(dt[:,None,None]*np.array(KNOTS)[None,:,None])
    y=target.reshape(len(x),-1).clip(-3,3)
    count=fit.meta.groupby("env").turn.transform("size").to_numpy(float)
    weight=count**-.5
    weight/=weight.mean()
    linear=make_pipeline(StandardScaler(),Ridge(alpha=240.))
    linear.fit(x,y,ridge__sample_weight=weight)
    trees=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=22,max_features=.70,max_depth=17,
                              n_jobs=3,random_state=91717)
    trees.fit(x,y-linear.predict(x),sample_weight=weight)
    pack=dict(linear=linear,trees=trees,residual_fraction=1.)
    gate_truth=raw[gate.meta.row.to_numpy()]
    trials=[]
    for fraction in (0.,.5,1.):
        pack["residual_fraction"]=fraction
        errors=np.linalg.norm(expert_points(pack,gate.seq,gate.features)-gate_truth,axis=2)
        trials.append((point_cost(errors,gate.meta),fraction))
    pack["residual_fraction"]=min(trials)[1]
    return pack


def expert_points(pack,seq,tab):
    basis,x,dt=frame_and_inputs(seq,tab)
    speed=pack["linear"].predict(x)+pack["residual_fraction"]*pack["trees"].predict(x)
    knots=speed.reshape(len(x),len(KNOTS),3)*(dt[:,None,None]*np.array(KNOTS)[None,:,None])
    knots=np.einsum("nhk,njk->nhj",knots,basis)
    xp=np.array((0,*KNOTS))
    interpolation=np.stack([np.interp(STEPS,xp,column) for column in np.eye(len(xp))],axis=1)
    return np.einsum("hq,nqj->nhj",interpolation[:,1:],knots)


def point_cost(error,meta):
    ade=100*error.mean(1)
    env=meta.env.to_numpy()
    macro=np.mean([ade[env==e].mean() for e in np.unique(env)])
    return float(.45*ade[meta.turn.to_numpy()].mean()+.35*macro+.20*ade.mean())


def issued_centers(meta):
    with np.load(M4_DIR/"M3_issued.npz") as f:
        import pandas as pd
        ix=pd.Index(f["row"]).get_indexer(meta.row)
        if (ix<0).any():
            raise ValueError("Missing M4 issue record")
        return f["center"][ix]


def choose_share(old,new,truth,meta):
    base=np.linalg.norm(old-truth,axis=2)
    turns=meta.turn.to_numpy()
    choices=[]
    for a in SHARES:
        e=np.linalg.norm(old+a*(new-old)-truth,axis=2)
        if e.mean()<=base.mean()+1.e-10 and e[turns].mean()<=base[turns].mean()+1.e-10:
            choices.append((point_cost(e,meta),a))
    return min(choices)[1]


def freeze_path():
    proto=read(M4_DIR/"release_protocol.json")
    train=load_split("train")
    fit=train.take(train.meta.env.isin(proto["roles"]["fit"]))
    gate=train.take(train.meta.env.isin(proto["roles"]["gate"]) & ((train.meta.origin-10)%30==0))
    pack=fit_expert(fit,gate)
    joblib.dump(pack,OUT/"fresh_path_expert.joblib",compress=3)
    val=load_split("validation")
    old=issued_centers(val.meta)
    new=expert_points(pack,val.seq,val.features)
    truth=np.load(DATA/"y.npy",mmap_mode="r")[val.meta.row.to_numpy()]
    locked=read(OUT/"frozen_before_test.json")
    scorer=Scorer(joblib.load(OUT/"fresh_tree_and_scales.joblib"))
    predictions,proxy=scorer.outputs(val.seq,val.features,val.meta.stage.to_numpy())
    w,_=reference(val,predictions[locked["detector"]],proxy)
    base_hit=(w.error<=w.radius).all(1)
    envs=val.meta.env.to_numpy()
    turn=val.meta.turn.to_numpy()
    shares={}
    error_oof=np.empty_like(w.error)
    ratios={key:np.zeros(len(val.meta)) for key in ("compact","reliable")}
    for env in np.unique(envs):
        hold=envs==env
        shares[env]=choose_share(old[~hold],new[~hold],truth[~hold],val.meta.loc[~hold])
        mixed=old[hold]+shares[env]*(new[hold]-old[hold])
        error_oof[hold]=np.linalg.norm(mixed-truth[hold],axis=2)
        fold=w.take(hold)
        cal=fit_context(w.take(~hold))
        radii=issue_radii(fold.features,fold.score,fold.proxy,fold.radius,cal,locked["radii"])
        for key in ratios:
            ratios[key][hold]=np.max(error_oof[hold]/radii[key],axis=1)
    # Centre shifts change which windows are covered. Recalibration uses out-of-environment residuals.
    factors={}
    for name,values in ratios.items():
        factors[name]=float(max(1.,values[base_hit&turn].max(),np.quantile(values,.96,method="higher"),
                              *(np.quantile(values[envs==env],.95,method="higher") for env in np.unique(envs))))
    share=choose_share(old,new,truth,val.meta)
    vals=dict(m4_cm=float(100*w.error.mean()),m4_turn_cm=float(100*w.error[turn].mean()),
              oof_cm=float(100*error_oof.mean()),oof_turn_cm=float(100*error_oof[turn].mean()))
    save(OUT/"point_locked_before_test.json",dict(share=share,fold_shares=shares,radius_multipliers=factors,
         expert_sha256=fingerprint(OUT/"fresh_path_expert.joblib"),origin_sha256=fingerprint(OUT/"frozen_before_test.json"),
         calibration="Held-out validation environments, maximum previously covered turn score and 95% environment quantiles.",
         validation=vals))
    print("New wrist displacement expert",share,vals,factors,flush=True)


if __name__=="__main__":
    protect_weights()
    freeze_path()
