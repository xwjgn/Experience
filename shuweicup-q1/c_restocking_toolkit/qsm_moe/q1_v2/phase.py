"""Independently supervised movement and turn prediction, probability calibration, and decision selection."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from settings import OUT, WORK, PREVIOUS, SEED, KEY, MODES, array, write_json

BUDGETS=(.08,.12,.16,.20,.24,.28,.32)


def weights(env, label=None):
    _,offset,count=np.unique(np.asarray(env),return_inverse=True,return_counts=True)
    w=1/np.sqrt(count[offset].astype(float))
    if label is not None:
        freq=np.bincount(label,weights=w)
        w/=freq[label]
    return w/w.mean()


def learner():
    return HistGradientBoostingClassifier(max_iter=115,max_leaf_nodes=13,min_samples_leaf=100,
            learning_rate=.055,l2_regularization=6.,early_stopping=False,random_state=SEED)


def raw_predict(model,x):
    if "tree" in model:
        return model["tree"].predict_proba(x)
    moving=model["moving"].predict_proba(x)[:,1]
    turn=model["conditional"].predict_proba(x)[:,1]
    return np.column_stack((1-moving,moving*turn,moving*(1-turn)))


def calibrated(raw,platt):
    log=np.log(np.maximum(raw,1e-7))
    return platt.predict_proba(log-log[:,-1:])


def decide(raw,p,rule):
    if rule["type"]=="calibrated_argmax":
        return p.argmax(1)
    if rule["type"]=="raw_argmax":
        return raw.argmax(1)
    other=np.where(p[:,0]>p[:,2],0,2)
    return np.where(p[:,1]>=rule["threshold"],1,other)


def scores(y,p,c,env):
    t=(y==1); a=(c==1)
    tp=int((t&a).sum()); fp=int((~t&a).sum()); fn=int((t&~a).sum())
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    env_f1=[]
    for e in np.unique(env):
        use=(env==e)
        env_f1.append(f1_score(t[use],a[use],zero_division=0))
    return dict(n=len(y),turns=int(t.sum()),alarms=int(a.sum()),tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
                turn_f1=2*precision*recall/max(precision+recall,1e-12),env_f1=float(np.mean(env_f1)),
                overall_accuracy=float((c==y).mean()),macro_f1=f1_score(y,c,labels=[0,1,2],average="macro",zero_division=0),
                AP=average_precision_score(t,p),AUROC=roc_auc_score(t,p) if np.unique(t).size==2 else np.nan,
                brier=float(np.mean((p-t)**2)),call_rate=float(a.mean()))


def rule_grid(raw,p,gate,allow_raw):
    rules=[{"type":"calibrated_argmax"}]
    if allow_raw:
        rules.append({"type":"raw_argmax"})
    for budget in BUDGETS:
        rules.append(dict(type="threshold",training_alert_fraction=budget,threshold=float(np.quantile(p[gate,1],1-budget))))
    return rules


def evaluate_rules(model,raw,p,y,env,validation,rules):
    result=[]
    for rule in rules:
        c=decide(raw[validation],p[validation],rule)
        metrics=scores(y[validation],p[validation,1],c,env[validation])
        feasible=metrics["precision"]>=.25 and metrics["call_rate"]<=.35 and metrics["tp"]>0
        result.append(dict(model=model,rule=rule,feasible=feasible,**metrics))
    return result


def best(rows):
    return max(rows,key=lambda r:(r["feasible"],r["env_f1"],r["turn_f1"],r["precision"],-r["call_rate"]))


def fit_stage():
    x=np.load(WORK/"kine.npy",mmap_mode="r")
    meta=pd.read_parquet(WORK/"cache"/"meta.parquet")
    env=meta.env.to_numpy()
    spec=json.loads((WORK/"cache"/"ready.json").read_text(encoding="utf8"))["environments"]
    mask=array("mask")
    y=array("next_mode").astype(int)
    fit=np.isin(env,spec["fit"]) & mask[:,9] & mask[:,-1]
    density=meta.origin.to_numpy()%4==2
    cal=np.isin(env,spec["gate"]) & mask[:,9] & mask[:,-1] & density
    grid=array("grid") & mask.all(1)
    gate=grid & np.isin(env,spec["gate"])
    validation=grid & meta.split.eq("validation").to_numpy()
    protocols=[]
    for name in ("plain_tree","balanced_tree","hurdle"):
        if name=="hurdle":
            moving=y[fit]!=0
            sel=fit & (y!=0)
            model={
                "moving":learner().fit(x[fit],moving,sample_weight=weights(env[fit],moving.astype(int))),
                "conditional":learner().fit(x[sel],y[sel]==1,sample_weight=weights(env[sel],(y[sel]==1).astype(int)))
            }
        else:
            w=weights(env[fit],y[fit] if name=="balanced_tree" else None)
            model={"tree":learner().fit(x[fit],y[fit],sample_weight=w)}
        raw=raw_predict(model,x)
        z=np.log(np.maximum(raw,1e-7));z-=z[:,-1:].copy()
        platt=LogisticRegression(C=.8,max_iter=300,random_state=SEED).fit(z[cal],y[cal],sample_weight=weights(env[cal]))
        p=calibrated(raw,platt)
        rules=rule_grid(raw,p,gate,True)
        protocols.extend(evaluate_rules(name,raw,p,y,env,validation,rules))
        np.savez(WORK/(name+"_phase.npz"),raw=raw.astype("float32"),p=p.astype("float32"))
        joblib.dump((model,platt),WORK/(name+"_phase.joblib"))
    choice=best(protocols)
    with np.load(PREVIOUS/"cache"/"B4_issues.npz") as f:
        ids=f["ids"]; prev=f["stage"]
    old=evaluate_rules("v1_threshold_only",prev,prev,y[ids],env[ids],validation[ids],rule_grid(prev,prev,gate[ids],False))
    old_choice=best(old)
    write_json(OUT/"phase_selection.json",dict(model=choice["model"],rule=choice["rule"],feasible=bool(choice["feasible"]),
               old_rule=old_choice["rule"],gate_budgets=BUDGETS,precision_floor=.25,max_alarm_share=.35,
               validation_envs=sorted(set(env[validation])),objective="validation environment-mean turn F1 under precision and alarm caps"))
    pd.DataFrame([{**r,"rule":json.dumps(r["rule"])} for r in protocols+old]).to_csv(OUT/"phase_validation_candidates.csv",index=False)
    # Historical uncertainty of the ten-frame proxy is inspected using the same label definition only in scoring.
    print("Locked turn head",choice["model"],choice["rule"],"validation TP/FP",choice["tp"],choice["fp"],flush=True)


if __name__=="__main__":
    fit_stage()
