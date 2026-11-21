"""Validation-locked comparison of anticipating acceleration and forecasting its 3D endpoint."""
import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from acceleration_data import AccelerationData, RESULTS, SCALE, STATE
from acceleration_models import AnticipationGRU, Batches, LinearForecast, SEEDS, predict_one

METHODS = ("linear","motion_gru","video_motion_gru")
BLENDS = (.25,.5,1.)
RISK_QUANTILES = (0.,.5,.75)


@dataclass(frozen=True)
class Policy:
    model: str
    fraction: float
    threshold: float


BASE = Policy("incumbent",0.,-1.)


def read_models(data,device):
    linear=LinearForecast()
    linear.classifier,linear.regressors=joblib.load(RESULTS/"linear.joblib")
    nominals={}
    for name in METHODS[1:]:
        nominals[name]=[]
        for seed in SEEDS:
            weights=torch.load(RESULTS/f"{name}_{seed}.pt",map_location=device,weights_only=True)
            net=AnticipationGRU(data.sequence.shape[1],weights["visual"]).to(device)
            net.load_state_dict(weights["state_dict"])
            nominals[name].append(net.eval())
    return linear,nominals


def predict_split(data,batches,models,split):
    # Prediction is issued for every slow origin. Reference quality/ending state
    # are consulted only after the predictions have been produced.
    all_slow=data.eval_ids[split]
    mask=data.meta.score.to_numpy()[all_slow]
    ids=all_slow[mask]
    linear,gru=models
    all_predictions={"linear":linear.predict(data,all_slow)}
    members={}
    for name,nets in gru.items():
        pairs=[predict_one(net,batches,all_slow) for net in nets]
        all_predictions[name]=tuple(np.mean([p[k] for p in pairs],axis=0) for k in (0,1))
        for seed,(p,m) in zip(SEEDS,pairs):
            members[f"{name}_{seed}"]=(p[mask],m[mask])
    pred={name:(p[mask],m[mask]) for name,(p,m) in all_predictions.items()}
    for probs,point in pred.values():
        np.testing.assert_allclose(probs.sum(1),1,atol=2e-6)
        assert np.isfinite(point).all()
    return ids,pred,members


def apply(policy,base,pred):
    if policy.model=="incumbent":
        return base.copy()
    prob,candidate=pred[policy.model]
    weight=policy.fraction*(prob[:,2]>=policy.threshold)
    return base+weight[:,None]*(candidate-base)


def error_cm(point,truth):
    return np.linalg.norm(point-truth,axis=1)*SCALE*100


def make_policy(data,ids,pred):
    base,truth,labels=data.base[ids],data.truth[ids],data.end_class[ids]
    old=error_cm(base,truth)
    def effect(policy):
        err=error_cm(apply(policy,base,pred),truth)
        return dict(slow_to_slow_cm=float(err[labels==0].mean()),slow_to_fast_cm=float(err[labels==2].mean()),
                    all_slow_cm=float(err.mean()))
    baseline=effect(BASE)
    grid=[]
    protected={}
    def row(policy):
        scores=effect(policy)
        feasible=(scores["slow_to_slow_cm"]<=baseline["slow_to_slow_cm"]+1e-7 and
                  scores["all_slow_cm"]<=baseline["all_slow_cm"]+1e-7)
        return {**asdict(policy),**scores,"passes_validation_guards":bool(feasible)}
    grid.append(row(BASE))
    best=BASE
    best_score=baseline["slow_to_fast_cm"]
    for name in METHODS:
        best_here=BASE
        cutoff_best=baseline["slow_to_fast_cm"]
        for quantile in RISK_QUANTILES:
            threshold=float(np.quantile(pred[name][0][:,2],quantile)) if quantile else -1.
            for fraction in BLENDS:
                policy=Policy(name,fraction,threshold)
                r=row(policy)
                grid.append(r)
                if r["passes_validation_guards"] and r["slow_to_fast_cm"]<cutoff_best-1e-7:
                    best_here,cutoff_best=policy,r["slow_to_fast_cm"]
        protected[name]=best_here
        if cutoff_best<best_score-1e-7:
            best,best_score=best_here,cutoff_best
    assert np.isclose(old.mean(),baseline["all_slow_cm"])
    return protected,best,pd.DataFrame(grid)


def risk_metrics(actual,score,cut):
    pred=score>=cut
    tp=int(np.sum(actual&pred))
    fp=int(np.sum(~actual&pred))
    positives=int(actual.sum())
    negatives=len(actual)-positives
    recall=tp/positives if positives else np.nan
    specificity=1-fp/negatives if negatives else np.nan
    return dict(n=len(actual),positives=positives,prevalence=actual.mean(),
        auc=roc_auc_score(actual,score) if positives and negatives else np.nan,
        ap=average_precision_score(actual,score) if positives else np.nan,
        brier=brier_score_loss(actual,score),decision_threshold=cut,predicted_fast=int(pred.sum()),
        true_positives=tp,false_positives=fp,recall=recall,precision=tp/(tp+fp) if tp+fp else np.nan,
        specificity=specificity,balanced_accuracy=(recall+specificity)/2)


def risk_tables(data,ids,pred,split,cuts):
    actual=data.end_class[ids]==2
    env=data.meta.env.to_numpy()[ids]
    scores={name:p[:,2] for name,(p,_) in pred.items()}
    scores["constant_train_rate"]=np.full(len(ids),np.mean(data.end_class[data.train]==2))
    rows=[]
    for name,score in scores.items():
        for e in ["all",*np.unique(env)]:
            select=np.ones(len(ids),bool) if e=="all" else env==e
            rows.append(dict(split=split,model=name,environment=e,**risk_metrics(actual[select],score[select],cuts[name])))
    df=pd.DataFrame(rows)
    macro=df[df.environment.ne("all")].groupby("model").auc.mean()
    df["env_macro_auc"]=df.model.map(macro)
    return df


def risk_threshold(y,p):
    fpr,tpr,cuts=roc_curve(y,p,drop_intermediate=False)
    index=int(np.argmax(tpr-fpr))
    return float(cuts[index]) if np.isfinite(cuts[index]) else 1.01


def position_tables(data,split,ids,pred,protected,best):
    base=data.base[ids]
    points={"incumbent":base,"hold":np.zeros_like(base)}
    for name,(_,point) in pred.items():
        points[name+"_full"]=point
        points[name+"_protected"]=apply(protected[name],base,pred)
    points["validation_selected"]=apply(best,base,pred)
    group=data.end_class[ids]
    env=data.meta.env.to_numpy()[ids]
    all_ids=data.meta.index[data.meta.split.eq(split)&data.meta.score].to_numpy()
    slots=np.searchsorted(all_ids,ids)
    assert np.array_equal(all_ids[slots],ids)
    original_all=error_cm(data.base[all_ids],data.truth[all_ids])
    target=data.truth[ids]
    old=error_cm(base,target)
    records=[]
    comparison=[]
    def summarize(name,group_name,err,ref,envs):
        env_gain=[]
        env_mean=[]
        for e in np.unique(envs):
            sub=envs==e
            avg=float(err[sub].mean())
            delta=float((err[sub]-ref[sub]).mean())
            comparison.append(dict(split=split,model=name,group=group_name,environment=e,n=int(sub.sum()),
                                   pooled_cm=avg,delta_vs_incumbent_cm=delta))
            env_gain.append(delta)
            env_mean.append(avg)
        records.append(dict(split=split,model=name,group=group_name,n=len(err),pooled_cm=float(err.mean()),
                            macro_cm=float(np.mean(env_mean)),p90_cm=float(np.quantile(err,.9)),
                            delta_vs_incumbent_cm=float((err-ref).mean()),environments_improved=int(np.sum(np.array(env_gain)<-1e-5)),
                            environments=len(env_mean)))
    for name,point in points.items():
        err=error_cm(point,target)
        for label,select in [("all_slow",np.ones(len(ids),bool)),
            *[("slow_to_"+STATE[c],group==c) for c in range(3)]]:
            summarize(name,label,err[select],old[select],env[select])
        # Non-slow origins keep the frozen incumbent for all proposed specialist policies.
        all_err=original_all.astype(float).copy()
        all_err[slots]=err
        summarize(name,"all_origins",all_err,original_all,data.meta.env.to_numpy()[all_ids])
    scorelog=data.meta.loc[ids,["sample","hand","env","split","origin","end_bin"]].copy()
    scorelog["incumbent_error_cm"]=old
    scorelog["actual_displacement_cm"]=error_cm(np.zeros_like(target),target)
    for name,(score,point) in pred.items():
        scorelog[name+"_p_fast"]=score[:,2]
        scorelog[name+"_full_cm"]=error_cm(point,target)
        scorelog[name+"_protected_cm"]=error_cm(points[name+"_protected"],target)
    scorelog["validation_selected_cm"]=error_cm(points["validation_selected"],target)
    return pd.DataFrame(records),pd.DataFrame(comparison),scorelog


def run():
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=AccelerationData(dev)
    d.verify_baseline("validation")
    saved_pre=np.load(RESULTS/"preprocessing.npz")
    for key,value in (("mean",d.feature_mean),("sd",d.feature_sd),("cuts",d.cuts)):
        np.testing.assert_allclose(value,saved_pre[key],atol=1e-6)
    models=read_models(d,dev)
    batches=Batches(d,dev)
    ids,v,vmembers=predict_split(d,batches,models,"validation")
    protected,best,grid=make_policy(d,ids,v)
    grid.to_csv(RESULTS/"validation_policy_grid.csv",index=False)
    y=d.end_class[ids]==2
    cuts={name:risk_threshold(y,p[:,2]) for name,(p,_) in v.items()}
    cuts["constant_train_rate"]=1.01
    val_risk=risk_tables(d,ids,v,"validation",cuts)
    ranked=val_risk[val_risk.environment.eq("all") & val_risk.model.isin(METHODS)].sort_values("env_macro_auc",ascending=False)
    risk_best=ranked.model.iloc[0]
    lock={"point_policy":asdict(best),"protected_by_method":{m:asdict(p) for m,p in protected.items()},
          "risk_thresholds":cuts,"risk_model_selected_by_validation_macro_auc":risk_best,
          "slow_upper_mps":float(d.cuts[0]),"fast_lower_mps":float(d.cuts[1])}
    (RESULTS/"validation_decision.json").write_text(json.dumps(lock,indent=2,allow_nan=False),encoding="utf-8")
    print("Validation-locked choices",json.dumps(lock,allow_nan=False),flush=True)
    # First assessment of the new test scores is below the persisted validation decision.
    d.verify_baseline("test")
    test_ids,t,tmembers=predict_split(d,batches,models,"test")
    risk=[val_risk,risk_tables(d,test_ids,t,"test",cuts)]
    group_rows,environment_rows,logs,seed_rows=[],[],[],[]
    for split,idx,pairs,members in (("validation",ids,v,vmembers),("test",test_ids,t,tmembers)):
        pos,env,detail=position_tables(d,split,idx,pairs,protected,best)
        group_rows.append(pos)
        environment_rows.append(env)
        logs.append(detail)
        for name,(p,m) in members.items():
            err=error_cm(m,d.truth[idx])
            label=d.end_class[idx]
            seed_rows.append(dict(split=split,model=name,auc=roc_auc_score(label==2,p[:,2]),all_slow_cm=err.mean(),
                                  slow_to_slow_cm=err[label==0].mean(),slow_to_fast_cm=err[label==2].mean()))
    risks=pd.concat(risk,ignore_index=True)
    positions=pd.concat(group_rows,ignore_index=True)
    risks.to_csv(RESULTS/"fast_classification.csv",index=False)
    positions.to_csv(RESULTS/"endpoint_errors.csv",index=False)
    pd.concat(environment_rows,ignore_index=True).to_csv(RESULTS/"endpoint_by_environment.csv",index=False)
    pd.concat(logs).to_csv(RESULTS/"origin_predictions.csv.gz",compression="gzip",index=False)
    pd.DataFrame(seed_rows).to_csv(RESULTS/"seed_sensitivity.csv",index=False)
    print(risks.query("environment=='all'")[["split","model","prevalence","auc","env_macro_auc","ap","recall","precision"]].round(3).to_string(index=False))
    print(positions[positions.model.isin(["incumbent","hold","linear_full","motion_gru_full","video_motion_gru_full","validation_selected"])]
          .pivot(index=["split","model"],columns="group",values="pooled_cm").round(3).to_string())


if __name__=="__main__":
    run()
