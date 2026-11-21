"""Whole-path calibration with reference, endpoint-feedback and completed-trajectory scales."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import OUT, WORK, BASE, SEED, read_json, save
from motion import Inputs
from phase import decide
from settings import array
from scipy.stats import norm

LEVELS=(.95,.97,.985,.99)
SCHEMES=("static","endpoint","completed","static_endpoint","static_completed","envelope")
PARTS={"static":("static",),"endpoint":("endpoint",),"completed":("completed",),
       "static_endpoint":("static","endpoint"),"static_completed":("static","completed"),
       "envelope":("static","endpoint","completed")}


def risk_fields(phase,state,q,v,center,delta):
    speed,cos=state.T
    return np.column_stack((phase[:,:2],np.log1p(4*speed),cos,np.sqrt(q.clip(0,1)),
             np.linalg.norm(v[:,4],axis=1),np.linalg.norm(v[:,3]-v[:,7],axis=1),
             np.linalg.norm(v[:,2]-v[:,3],axis=1),np.linalg.norm(center[:,9],axis=1),
             np.linalg.norm(center[:,-1],axis=1),np.linalg.norm(delta[:,9],axis=1),
             np.linalg.norm(delta[:,-1],axis=1))).astype(np.float32)


def fit_risk_model(x,err,base,env):
    target=np.log(np.maximum((err/base).max(1),1e-4))
    _,idx,n=np.unique(env,return_inverse=True,return_counts=True)
    w=1/n[idx];w/=w.mean()
    # Fit an upper conditional score on official gate-training environments only.
    model=HistGradientBoostingRegressor(loss="quantile",quantile=.75,learning_rate=.055,max_iter=70,
                   max_leaf_nodes=5,min_samples_leaf=100,l2_regularization=5,early_stopping=False,random_state=SEED)
    model.fit(x,target,sample_weight=w)
    midpoint=float(np.median(model.predict(x)))
    joblib.dump((model,midpoint),WORK/"path_risk_scale.joblib")
    return model,midpoint


def adjust_width(shape,base,fields,model,midpoint):
    if shape=="kinematic":
        factor=np.exp(.65*(model.predict(fields)-midpoint)).clip(.72,1.65)
        return base*factor[:,None]
    return base.copy()


class Feedback:
    """Independent state per record and wrist, using a completed prediction only after its last frame."""
    def __init__(self,reference,typical):
        self.reference=reference
        self.typical=typical
        self.alpha=np.full(3,.05)
        self.fast=0.
        self.slow=0.
        self.stage=np.zeros(3)

    def rank(self,phase):
        return np.ceil((len(self.reference)-1)*(1-self.alpha[phase])).astype(int)

    def observe(self,phase,error,scale,visible):
        if visible[-1]:
            q=self.reference[self.rank(phase),-1]
            miss=float(error[-1]>q*scale[-1])
            self.alpha[phase]=np.clip(self.alpha[phase]+.04*(.05-miss),.01,.20)
        if np.all(visible):
            score=float(np.max(error/scale))
            change=np.clip(np.log(max(score/self.typical,.05)),-1.6,1.6)
            self.fast=.72*self.fast+.28*change
            self.slow=.90*self.slow+.10*change
            self.stage[phase]=.75*self.stage[phase]+.25*change

    def multipliers(self,phase):
        end=self.reference[self.rank(phase)]
        level=.78*max(self.fast,self.slow)+.22*self.stage[phase]
        whole=np.exp(level).clip(.80,2.6)
        return end,whole


def sequences(meta,phase,err,shape,valid_obs,reference,typical):
    n=len(meta)
    endpoint=np.empty_like(shape)
    completed=np.empty_like(shape)
    origin=meta.origin.to_numpy()
    for rows in meta.groupby(["sample","hand"],sort=False).indices.values():
        if not np.array_equal(origin[rows],np.arange(len(rows))):
            raise ValueError("Feedback needs every recorded tick, including unscored origins")
        slots=rows[(origin[rows]>=10)&((origin[rows]-10)%30==0)]
        state=Feedback(reference,typical)
        first=rows[0]
        schedule=np.r_[first,slots] if len(slots) and slots[0]!=first else slots
        if not len(schedule):
            schedule=np.array([first])
        for j,now in enumerate(schedule):
            old=now-30
            if old>=first+10:
                assert origin[now]-origin[old]==30
                state.observe(phase[old],err[old],shape[old],valid_obs[old+1:now+1])
            stop=schedule[j+1] if j+1<len(schedule) else rows[-1]+1
            sl=slice(now,stop)
            ep,whole=state.multipliers(phase[sl])
            endpoint[sl]=shape[sl]*ep
            completed[sl]=shape[sl]*whole[:,None]
    return dict(static=shape,endpoint=endpoint,completed=completed)


def wilson_lower(count,n):
    # Descriptive validation margin. Windows from one environment are correlated.
    z=norm.ppf(.90)
    p=count/n
    return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def order_stat(scores,level):
    n=len(scores)
    k=min(n,int(np.ceil((n+1)*level)))
    return float(np.partition(scores,k-1)[k-1])


def quantile_factors(scales,err,env,level,rule):
    output={}
    for name,shape in scales.items():
        score=(err/shape).max(1)
        values=[order_stat(score,level)]
        if rule=="max_env":
            values.extend(order_stat(score[env==e],level) for e in sorted(set(env)))
        output[name]=max(values)
    return output


def band(scales,scheme,factors):
    pieces=PARTS[scheme]
    value=scales[pieces[0]]*factors[pieces[0]]
    for key in pieces[1:]:
        value=np.maximum(value,scales[key]*factors[key])
    return value


def evaluate_candidate(err,radius,env,turn):
    hit=(err<=radius).all(1)
    lower=wilson_lower(int(hit.sum()),len(hit))
    per_env=pd.Series(hit).groupby(env).mean()
    group=float(hit[turn].mean())
    gap=max(0.,.95-lower,.95-per_env.min(),.95-group)
    return dict(n=len(hit),covered=int(hit.sum()),coverage=float(hit.mean()),lower90=lower,
                minimum_environment=float(per_env.min()),turns=int(turn.sum()),turn_coverage=group,
                radius_cm=float(radius.mean()*100),shortfall=float(gap),eligible=gap<=1e-12)


def select():
    d=Inputs()
    locked=read_json(BASE/"phase_selection.json")
    with np.load(WORK/(locked["model"]+"_phase.npz")) as fp:
        mode=decide(fp["raw"],fp["p"],locked["rule"])
    used=np.flatnonzero(d.gate|d.cal|d.val)
    if d.test[used].any(): raise AssertionError("Test environment in interval selection")
    meta=d.meta.iloc[used].reset_index(drop=True); env=d.env[used]
    point=np.load(WORK/"corrected_center.npy",mmap_mode="r")[used]
    prior=np.load(WORK/"reference_center.npy",mmap_mode="r")[used]
    base=np.load(WORK/"reference_scale.npy",mmap_mode="r")[used]
    err=np.linalg.norm(point-d.y[used],axis=2)
    x=risk_fields(d.phase[used],d.state[used],d.q[used],d.v[used],point,point-prior)
    trained=d.gate[used]&d.mask[used].all(1)&(meta.origin.to_numpy()%5==0)
    model,midpoint=fit_risk_model(x[trained],err[trained],base[trained],env[trained])
    gf=d.gate[used]&d.grid[used];cf=d.cal[used]&d.grid[used];vf=d.val[used]&d.grid[used]
    truth_turn=array("next_mode")[used][vf]==1
    all_rows=[]
    for form in ("reference","kinematic"):
        shape=adjust_width(form,base,x,model,midpoint)
        scores=err[gf]/shape[gf]
        reference=np.sort(scores,axis=0)
        typical=float(np.median(scores.max(1)))
        np.savez(WORK/(form+"_feedback_priors.npz"),reference=reference,typical=typical)
        scales=sequences(meta,mode[used],err,shape,array("ok")[used],reference,typical)
        cal={k:v[cf] for k,v in scales.items()}
        val={k:v[vf] for k,v in scales.items()}
        for level in LEVELS:
            for rule in ("pooled","max_env"):
                factors=quantile_factors(cal,err[cf],env[cf],level,rule)
                for scheme in SCHEMES:
                    rad=band(val,scheme,factors)
                    row=evaluate_candidate(err[vf],rad,env[vf],truth_turn)
                    all_rows.append(dict(shape=form,scheme=scheme,level=level,pooling=rule,factors=factors,**row))
    chosen=min(all_rows,key=lambda r:(r["shortfall"],r["radius_cm"],len(PARTS[r["scheme"]])))
    pd.DataFrame([{**r,"factors":json.dumps(r["factors"],sort_keys=True)} for r in all_rows]).to_csv(OUT/"coverage_validation.csv",index=False)
    save("frozen_before_test.json",dict(point=read_json(OUT/"point_lock.json"),interval=chosen,
        target=.95,selection_partitions=["train","validation"],old_checkpoints_loaded=0,roles=d.roles,
        gate_grid=int(gf.sum()),calibration_grid=int(cf.sum()),validation_grid=int(vf.sum()),
        reference_interval=read_json(BASE/"frozen_before_test.json")["band"],
        calibration_environment_counts=pd.Series(env[cf]).value_counts().to_dict(),
        small_group_quantile="rank min(n, ceil((n+1)*q)); small calibration groups can use their observed maximum; no new-environment coverage guarantee"))
    print("Locked interval:",chosen,flush=True)


if __name__=="__main__":
    select()
