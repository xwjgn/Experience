"""Measure advance turn recognition separately from the decision to narrow a fixed M4 forecast."""
import json
import os
import sys
from pathlib import Path
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from origin_signals import DATA, KEY, M4_DIR, OUT, ROOT, SUBSETS, cohort, columns, load_record, origin_features

FRACTIONS = (0., .1, .2, .3, .5, .75, 1.)
SHRINK = .80
ALERT_FRACTION = .25


def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False), encoding="utf8")


def turn_models(train, roles):
    fit = train[train.env.isin(roles["fit"])]
    gate = train[train.env.isin(roles["gate"]) & ((train.origin-10) % 30 == 0)]
    counts = fit.groupby("env").turn.transform("size").to_numpy()
    weights = counts.astype(float)**-.5
    models, thresholds = {}, {}
    for name in SUBSETS:
        cols = columns(fit,name)
        clf = HistGradientBoostingClassifier(learning_rate=.055,max_iter=100,max_leaf_nodes=12,
                min_samples_leaf=80,l2_regularization=4.,early_stopping=False,random_state=36511)
        clf.fit(fit[cols],fit.turn,sample_weight=weights/weights.mean())
        models[name] = clf
        thresholds[name] = float(np.quantile(clf.predict_proba(gate[cols])[:,1],1-ALERT_FRACTION))
    return models, thresholds, len(fit), len(gate)


def attach_reference(frame, models):
    # Only previously issued centres and radii enter the interval experiment. Original data provide the truth.
    y = np.load(DATA/"y.npy", mmap_mode="r")
    with np.load(M4_DIR/"M3_issued.npz") as issued:
        ix = pd.Index(issued["row"]).get_indexer(frame.row)
        if (ix < 0).any() or frame.row.duplicated().any():
            raise ValueError("M4 and the new detectors must use the same split, hand and origin")
        center, radius, phases = (issued[k][ix] for k in ("center","radius","phase"))
    if not (radius > 0).all():
        raise ValueError("Radii must be positive")
    error = np.linalg.norm(y[frame.row.to_numpy()] - center, axis=2)
    table = frame.copy()
    table["ratio"] = np.max(error/radius,axis=1)
    table["radius_cm"] = radius.mean(axis=1)*100
    table["ade_cm"] = error.mean(axis=1)*100
    table["m4_score"] = phases[:,1]
    for name, model in models.items():
        table[name] = model.predict_proba(frame[columns(frame,name)])[:,1]
    return table


def recognition(table, thresholds):
    labels = table.turn.to_numpy()
    original = table.m4_score.to_numpy() >= thresholds["m4_score"]
    budget = int(original.sum())
    rows = []
    for name in ("m4_score", *SUBSETS):
        scores = table[name].to_numpy()
        order = np.argsort(-scores,kind="stable")
        paired = np.zeros(len(table),bool)
        paired[order[:budget]] = True
        fixed = scores >= thresholds[name]
        for rule, alarm in (("online_frozen",fixed),("same_alarm_count_ranking",paired)):
            hit = int((labels & alarm).sum())
            overlap = int((alarm & labels & original).sum())
            rows.append(dict(split=table.split.iat[0],model=name,rule=rule,windows=len(table),turns=int(labels.sum()),
                n_alert=int(alarm.sum()),true_alert=hit,false_alert=int(alarm.sum()-hit),recall=hit/max(1,labels.sum()),
                precision=hit/max(1,alarm.sum()),average_precision=average_precision_score(labels,scores),
                shared_hits_with_m4=overlap,newly_identified=int((alarm & labels & ~original).sum()),
                previously_identified_now_missed=int((~alarm & labels & original).sum())))
    return rows


def tail_features(df):
    # Restrict the calibration regression to seven quantities already available at the issue frame.
    return np.column_stack((df.wrist_hand_head, np.log1p(df.radius_cm), np.log1p(df.w_speed_5),
            np.log1p(df.w_velocity_change_5), df.w_heading_cos_10, np.log1p(df.h_angular_speed_5),
            df.w_recent_flags))


def fit_tail(df):
    model = make_pipeline(StandardScaler(),QuantileRegressor(quantile=.85,alpha=.035,solver="highs"))
    # Predict the upper tail of the largest normalised path error under the original centre.
    return model.fit(tail_features(df),np.log(df.ratio.clip(.015,8.)))


def threshold(train_score, clean, fraction):
    if fraction == 0:
        return None
    if fraction == 1:
        return "all_clean"
    return float(np.quantile(np.asarray(train_score)[clean],fraction,method="lower"))


def route(score,clean,cut):
    if cut is None:
        return np.zeros(len(clean),bool)
    if cut == "all_clean":
        return np.asarray(clean,bool)
    return (np.asarray(score)<=cut) & clean


def performance(df, factor, routed):
    ratio,turn = df.ratio.to_numpy(),df.turn.to_numpy()
    factor = np.broadcast_to(factor,ratio.shape)
    hit = ratio <= factor
    base_hit = ratio <= 1.
    return dict(n=len(df),turns=int(turn.sum()),routed=int(routed.sum()),routed_turns=int((turn&routed).sum()),
        covered=int(hit.sum()),coverage=float(hit.mean()),turn_covered=int(hit[turn].sum()),turn_coverage=float(hit[turn].mean()),
        additional_turn_misses=int((base_hit & ~hit & turn).sum()),additional_misses=int((base_hit & ~hit).sum()),
        radius_cm=float(np.mean(factor*df.radius_cm)),ade_cm=float(df.ade_cm.mean()),turn_ade_cm=float(df.loc[turn,"ade_cm"].mean()))


def freeze_routing(val):
    clean = val.w_recent_flags.to_numpy() == 0
    families = ("low_turn","tail")
    routed = {(family,f):np.zeros(len(val),bool) for family in families for f in FRACTIONS}
    for env in val.env.unique():
        hold = val.env.to_numpy() == env
        a,b = val.loc[~hold],val.loc[hold]
        fitted = fit_tail(a)
        scores = {"tail":(fitted.predict(tail_features(a)),fitted.predict(tail_features(b))),
                  "low_turn":(a.wrist_hand_head.to_numpy(),b.wrist_hand_head.to_numpy())}
        for family,(train_score,test_score) in scores.items():
            for f in FRACTIONS:
                cut = threshold(train_score,clean[~hold],f)
                routed[family,f][hold] = route(test_score,clean[hold],cut)
    trials = []
    for (family,f),mask in routed.items():
        factor = np.where(mask,SHRINK,1.)
        metrics = performance(val,factor,mask)
        # Both coverage targets and preserving previously covered turns are specified before testing.
        feasible = metrics["coverage"] >= .95 and metrics["turn_coverage"] >= .95 and metrics["additional_turn_misses"] == 0
        trials.append(dict(family=family,fraction=f,feasible=bool(feasible),**metrics))
    tail = fit_tail(val)
    policies = {}
    for family in families:
        candidates = [r for r in trials if r["family"] == family and r["feasible"]]
        if candidates:
            pick = min(candidates,key=lambda row:(row["radius_cm"],row["fraction"]))
        else:
            pick = next(r for r in trials if r["family"]==family and r["fraction"]==0)
        score = tail.predict(tail_features(val)) if family=="tail" else val.wrist_hand_head.to_numpy()
        policies[family] = dict(fraction=pick["fraction"],cut=threshold(score,clean,pick["fraction"]),
                   uniform_validation_factor=float(pick["radius_cm"]/val.radius_cm.mean()),validation=pick)
    return tail, policies, trials


def score_routes(df, tail, policies):
    clean = df.w_recent_flags.to_numpy() == 0
    none = np.zeros(len(df),bool)
    rows = [dict(policy="m4",**performance(df,1.,none))]
    for family,rule in policies.items():
        score = tail.predict(tail_features(df)) if family=="tail" else df.wrist_hand_head.to_numpy()
        use = route(score,clean,rule["cut"])
        factor = np.where(use,SHRINK,1.)
        actual = performance(df,factor,use)
        rows.append(dict(policy=family,**actual))
        df[family+"_routed"] = use
        df[family+"_hit"] = df.ratio.to_numpy()<=factor
        uniform = rule["uniform_validation_factor"]
        rows.append(dict(policy=family+"_uniform_frozen",**performance(df,uniform,np.full(len(df),uniform<1.))))
        # Matching widths on this test cohort is descriptive, independent of the future labels.
        exact = actual["radius_cm"]/df.radius_cm.mean()
        rows.append(dict(policy=family+"_same_width_diagnostic",**performance(df,exact,np.full(len(df),exact<1.))))
    return rows


def origin_breakdown(df,thresholds):
    low_quality = df.w_recent_flags.to_numpy() > 0
    original = df.m4_score >= thresholds["m4_score"]
    rows = []
    for label,mask in (("all",df.turn),("recognized",df.turn&original),("missed",df.turn&~original)):
        for phase,name in enumerate(("dwell","turn","translate")):
            take = (mask & (df.origin_stage==phase)).to_numpy()
            rows.append(dict(group=label,origin_stage=name,windows=int(take.sum()),
                             past_10_flag_or_missing=int((take&low_quality).sum())))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    def reject_prior_weights(event,args):
        if event == "open" and isinstance(args[0], (str,bytes)):
            path = Path(os.fsdecode(args[0]))
            if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".joblib", ".pkl"}:
                path = path.resolve()
                if path.is_relative_to(ROOT) and not path.is_relative_to(OUT):
                    raise RuntimeError(f"A previous study's weights were requested: {path}")
    sys.addaudithook(reject_prior_weights)
    roles = json.loads((M4_DIR/"release_protocol.json").read_text(encoding="utf8"))["roles"]
    train = cohort(["train"])
    models, thresholds, n_fit, n_gate = turn_models(train,roles)
    thresholds["m4_score"] = json.loads((M4_DIR/"frozen_before_test.json").read_text(encoding="utf8"))["models"]["M3"]["turn_cut"]
    val = attach_reference(cohort(["validation"]),models)
    assert len(val)==401 and val.turn.sum()==75
    tail,rules,trials = freeze_routing(val)
    pd.DataFrame(trials).to_csv(OUT/"validation_routing.csv",index=False)
    save("frozen_before_replay.json",dict(roles=roles,fit_rows=n_fit,threshold_rows=n_gate,seed=36511,
            validation_envs=sorted(val.env.unique()),validation_windows=len(val),turn_alert_train_fraction=ALERT_FRACTION,
            shrink_factor=SHRINK,turn_thresholds=thresholds,routing=rules,sklearn=version("scikit-learn"),
            original_checkpoint_reads=0,source="Official poses, original benchmark labels and the already-issued M4 forecasts."))
    joblib.dump({"turn":models,"tail":tail},OUT/"independent_origin_models.joblib",compress=3)
    print("Frozen event thresholds and leave-environment-out radius rules",flush=True)
    test = attach_reference(cohort(["test"]),models)
    assert len(test)==760 and test.turn.sum()==99
    assert (test.m4_score>=thresholds["m4_score"]).sum()==181
    assert ((test.ratio<=1)&test.turn).sum()==99 and (test.ratio<=1).sum()==751
    recognition_rows = recognition(val,thresholds)+recognition(test,thresholds)
    ranks = pd.DataFrame(recognition_rows)
    ranks.to_csv(OUT/"event_recognition.csv",index=False)
    radius = pd.DataFrame(score_routes(test,tail,rules))
    radius.to_csv(OUT/"intervals.csv",index=False)
    breakdown = pd.DataFrame(origin_breakdown(test,thresholds))
    breakdown.to_csv(OUT/"origin_phase_and_quality.csv",index=False)
    env_rows = []
    for env,group in test.groupby("env",sort=False):
        base = group.ratio <= 1.
        for family in ("m4","low_turn","tail"):
            hit = base if family=="m4" else group[family+"_hit"]
            env_rows.append(dict(env=env,policy=family,n=len(group),covered=int(hit.sum()),turns=int(group.turn.sum()),
                turn_covered=int((hit & group.turn).sum()),radius_cm=float(group.radius_cm.mean() if family=="m4" else
                np.mean(group.radius_cm*np.where(group[family+"_routed"],SHRINK,1.)))))
    pd.DataFrame(env_rows).to_csv(OUT/"environment_coverage.csv",index=False)
    fields = KEY+["turn","origin_stage","ratio","radius_cm","ade_cm","m4_score",*SUBSETS,
                   "w_recent_flags","tail_routed","tail_hit","low_turn_routed","low_turn_hit"]
    test[fields].to_csv(OUT/"paired_origins.csv",index=False)
    for dataset, group in test.groupby("dataset"):
        r = group.iloc[0]
        limit = int(r.origin)+1
        full = load_record(dataset,r["sample"],limit+5)
        for side in ("left","right"):
            a = origin_features(full.prefix(limit),side)
            b = origin_features(full,side).iloc[:limit]
            np.testing.assert_allclose(a,b,rtol=0,atol=1.e-7)
    print(ranks[(ranks.split=="test") & (ranks.rule=="same_alarm_count_ranking")][["model","true_alert","false_alert","average_precision"]].round(3).to_string(index=False))
    print(radius[["policy","radius_cm","covered","turn_covered","routed"]].round(2).to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
    from replay_check import check
    check()
    from study_report import report
    report()
