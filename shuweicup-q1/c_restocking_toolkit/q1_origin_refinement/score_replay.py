"""Replay locked rules. Historical model artifacts are accessed only as already-issued outputs."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from wrist_history import BANDS,DATA,KEY,OLD,OUT,load_split,protect_weights,read,save
from event_models import Scorer
from horizon_budget import Windows,context,issue_radii,path_metrics,rankings
from alarm_rules import alert_metrics,combine,stream
from freeze_choices import fingerprint,reference
from path_expert import expert_points,issued_centers


def top_k(scores,n):
    chosen=np.zeros(len(scores),bool)
    chosen[np.argsort(-scores,kind="stable")[:n]]=True
    return chosen


def score():
    origin=read(OUT/"frozen_before_test.json")
    point=read(OUT/"point_locked_before_test.json")
    for name,signature in origin["weights"].items():
        if fingerprint(OUT/name)!=signature:
            raise ValueError("Changed artifact "+name)
    assert point["origin_sha256"]==fingerprint(OUT/"frozen_before_test.json")
    assert point["expert_sha256"]==fingerprint(OUT/"fresh_path_expert.joblib")
    scorer=Scorer(joblib.load(OUT/"fresh_tree_and_scales.joblib"))
    calib=joblib.load(OUT/"validation_horizon_calibration.joblib")
    data=load_split("test")
    scores,proxy=scorer.outputs(data.seq,data.features,data.meta.stage.to_numpy())
    chosen=origin["detector"]
    w,m4=reference(data,scores[chosen],proxy)
    prev=pd.read_csv(OLD/"paired_origins.csv")
    prior=data.meta[KEY].merge(prev,on=KEY,validate="one_to_one")
    assert len(data.meta)==len(prior)==len(prev)==760
    turn=data.meta.turn.to_numpy()
    np.testing.assert_array_equal(turn,prior.turn)
    assert turn.sum()==99
    old_factor=np.where(prior.low_turn_routed,.8,1.)
    old_radius=w.radius*old_factor[:,None]
    old_hit=(w.error<=old_radius).all(1)
    m4_hit=(w.error<=w.radius).all(1)
    np.testing.assert_array_equal(old_hit,prior.low_turn_hit)
    failures=m4_hit & ~old_hit
    assert failures.sum()==14 and not (failures & turn).any()
    rad=dict(m4=w.radius,previous_low_turn=old_radius,
             **issue_radii(w.features,w.score,w.proxy,w.radius,calib,origin["radii"],old_factor))
    np.testing.assert_allclose(rad["same_window_budget"].mean(1),old_radius.mean(1),rtol=0,atol=1.e-10)
    streams=stream(scorer,"test",chosen,origin["negative_prior"])
    events=w.meta[KEY].merge(streams,on=KEY,validate="one_to_one")
    np.testing.assert_allclose(events.stream_score,w.score,rtol=0,atol=1.e-6)
    alarm_cfg=origin["online_alarm"]
    cuts=events[f"cut_{alarm_cfg['fpr']}_{alarm_cfg['memory']}"].to_numpy()
    baseline_turn=m4>=origin["m4_threshold"]
    k=int(baseline_turn.sum())
    assert k==181
    mix_cfg=origin["optional_mixture"]
    mixed_score=combine(m4,w.score,mix_cfg["anchors"],mix_cfg["share"])
    alternatives={"m4":m4,"previous_wrist":prior.wrist,"previous_full":prior.wrist_hand_head,**scores,"late_fusion":mixed_score}
    prev_cuts=read(OLD/"frozen_before_replay.json")["turn_thresholds"]
    deployed={"m4":baseline_turn,"previous_wrist":np.asarray(prior.wrist)>=prev_cuts["wrist"],
              "previous_full":np.asarray(prior.wrist_hand_head)>=prev_cuts["wrist_hand_head"],
              chosen:w.score>=cuts,"late_fusion":mixed_score>=mix_cfg["threshold"]}
    alarm_rows=[]
    for name,values in alternatives.items():
        for rule,alert in [("same_alarm_count",top_k(values,k))]+([("online",deployed[name])] if name in deployed else []):
            alarm_rows.append(dict(model=name,rule=rule,average_precision=average_precision_score(turn,values),
                new_vs_m4=int((alert & turn & ~baseline_turn).sum()),lost_vs_m4=int((~alert&turn&baseline_turn).sum()),
                **alert_metrics(turn,alert)))
    old_center=issued_centers(w.meta)
    fresh=expert_points(joblib.load(OUT/"fresh_path_expert.joblib"),data.seq,data.features)
    truth=np.load(DATA/"y.npy",mmap_mode="r")[w.meta.row.to_numpy()]
    mixed=old_center+point["share"]*(fresh-old_center)
    new_error=np.linalg.norm(mixed-truth,axis=2)
    path_rows=[]
    for name,points in (("m4",old_center),("fresh_expert",fresh),("frozen_mix",mixed)):
        loss=np.linalg.norm(points-truth,axis=2)
        path_rows.append(dict(model=name,n=len(w.meta),turns=int(turn.sum()),ade_cm=float(100*loss.mean()),
               turn_ade_cm=float(100*loss[turn].mean()),fde_cm=float(100*loss[:,-1].mean()),
               turn_fde_cm=float(100*loss[turn,-1].mean())))
    mixed_windows=Windows(w.meta,w.features,w.score,w.proxy,w.radius,new_error)
    all_policies=[(name,w,arr) for name,arr in rad.items()]
    # Cross the already frozen centre decision with both old-budget shapes. No parameter is refit here.
    all_policies.extend((("mixed_center_previous_budget",mixed_windows,old_radius),
                         ("mixed_center_same_window_budget",mixed_windows,rad["same_window_budget"])))
    for name in ("compact","reliable"):
        all_policies.append(("mixed_center_"+name,mixed_windows,rad[name]*point["radius_multipliers"][name]))
    # These width-matched controls use only cohort widths and are retrospective.
    for name in ("same_window_budget","compact"):
        all_policies.append((name+"_uniform_matched",w,w.radius*rad[name].mean()/w.radius.mean()))
    detail=w.meta[KEY+["dataset","row","stage","turn"]].copy()
    detail["previous_routed"]=prior.low_turn_routed
    detail["turn_score"]=w.score
    detail["online_threshold"]=cuts
    detail["online_alert"]=deployed[chosen]
    detail["m4_turn_alert"]=baseline_turn
    detail["fused_alert"]=deployed["late_fusion"]
    detail["wrist_speed_m_s"]=w.features.w_speed_5
    detail["previous_additional_miss"]=failures
    risk,_=context(calib,w.features,w.score,w.proxy,w.radius,calib["risk_reference"],calib["turn_reference"])
    difficulty=np.max(risk["proxy"],axis=1)
    detail["physical_risk_percentile"]=rankings(calib["risk_reference"],difficulty)
    rows,environments,steps=[],[],[]
    for name,window,radius in all_policies:
        hit=(window.error<=radius).all(1)
        stats=path_metrics(window,radius)
        rows.append(dict(model=name,**stats,old14_saved=int((failures & hit).sum()),
                   other_previously_missed_now_hit=int((~old_hit & ~failures & hit).sum()),
                   newly_missed_vs_previous=int((old_hit & ~hit).sum()),
                   windows_with_smaller_budget_than_m4=int((radius.mean(1)<w.radius.mean(1)-1.e-8).sum())))
        detail[name+"_hit"]=hit
        detail[name+"_radius_cm"]=100*radius.mean(1)
        for h in range(30):
            steps.append(dict(model=name,step=h+1,radius_cm=100*radius[:,h].mean(),error_cm=100*window.error[:,h].mean(),
                              turn_error_cm=100*window.error[turn,h].mean(),covered=int((window.error[:,h]<=radius[:,h]).sum()),
                              turn_covered=int((window.error[turn,h]<=radius[turn,h]).sum())))
        for env in w.meta.env.unique():
            sub=(w.meta.env==env).to_numpy()
            environments.append(dict(model=name,env=env,**path_metrics(window.take(sub),radius[sub])))
    previous_ratio=w.error/old_radius
    detail["old_first_outside_step"]=np.where(old_hit,0,(previous_ratio>1).argmax(1)+1)
    detail["old_worst_step"]=previous_ratio.argmax(1)+1
    detail["old_peak_error_ratio"]=previous_ratio.max(1)
    detail["past_30_q90_10step_innovation_cm"]=100*w.features.noise_10
    heading=data.seq[:,-1,3:6].astype(float)
    heading/=np.maximum(np.linalg.norm(heading,axis=1)[:,None],1.e-7)
    peak=previous_ratio.argmax(1)
    difference=(old_center-truth)[np.arange(len(peak)),peak]
    axial=np.sum(difference*heading,axis=1)
    lateral=np.linalg.norm(difference-axial[:,None]*heading,axis=1)
    detail["peak_axial_error_cm"]=100*axial
    detail["peak_lateral_error_cm"]=100*lateral
    detail["m4_ade_cm"]=100*w.error.mean(1)
    detail["new_ade_cm"]=100*new_error.mean(1)
    detail.to_csv(OUT/"760个起点逐窗回放.csv",index=False,encoding="utf-8-sig")
    detail[turn].to_csv(OUT/"99个真实转向逐窗对照.csv",index=False,encoding="utf-8-sig")
    detail[failures].to_csv(OUT/"14个新增漏失逐窗复核.csv",index=False,encoding="utf-8-sig")
    diag=[]
    for group,mask in (("old14",failures),("other_previously_routed_and_covered",prior.low_turn_routed.to_numpy()&old_hit)):
        tab=detail.loc[mask]
        diag.append(dict(group=group,n=len(tab),stage_dwell=int((tab.stage==0).sum()),stage_turn=int((tab.stage==1).sum()),
             stage_translate=int((tab.stage==2).sum()),median_wrist_speed=float(tab.wrist_speed_m_s.median()),
             mainly_overshot=int((tab.peak_axial_error_cm>tab.peak_lateral_error_cm).sum()),
             mainly_undershot=int((tab.peak_axial_error_cm < -tab.peak_lateral_error_cm).sum()),
             mainly_lateral=int((tab.peak_axial_error_cm.abs()<=tab.peak_lateral_error_cm).sum()),
             median_innovation_upper_cm=float(tab.past_30_q90_10step_innovation_cm.median()),
             median_physical_risk_percentile=float(tab.physical_risk_percentile.median()),
             **{f"worst_{lo+1}_{hi}":int(tab.old_worst_step.between(lo+1,hi).sum()) for lo,hi in BANDS},
             **{f"first_{lo+1}_{hi}":int(tab.old_first_outside_step.between(lo+1,hi).sum()) for lo,hi in BANDS}))
    pd.DataFrame(diag).to_csv(OUT/"miss_diagnosis.csv",index=False)
    pd.DataFrame(path_rows).to_csv(OUT/"point_errors.csv",index=False)
    pd.DataFrame(alarm_rows).to_csv(OUT/"turn_alerts.csv",index=False)
    pd.DataFrame(environments).to_csv(OUT/"by_environment.csv",index=False)
    pd.DataFrame(steps).to_csv(OUT/"by_horizon.csv",index=False)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"intervals.csv",index=False)
    save(OUT/"replay_audit.json",dict(rows=len(w.meta),true_turns=int(turn.sum()),old14=int(failures.sum()),
            event_lock=fingerprint(OUT/"frozen_before_test.json"),point_lock=fingerprint(OUT/"point_locked_before_test.json"),
            equality_max_m=float(np.max(np.abs(rad["same_window_budget"].mean(1)-old_radius.mean(1)))),
            checked_keys=KEY,prior_checkpoint_reads=0))
    print(result[["model","covered","turn_covered","radius_cm","ade_cm","turn_ade_cm","old14_saved","newly_missed_vs_previous"]].round(3).to_string(index=False),flush=True)
    print(pd.DataFrame(alarm_rows)[["model","rule","tp","fp","average_precision"]].round(3).to_string(index=False),flush=True)
    print(pd.DataFrame(diag).round(2).to_string(index=False),flush=True)


if __name__=="__main__":
    protect_weights()
    score()
