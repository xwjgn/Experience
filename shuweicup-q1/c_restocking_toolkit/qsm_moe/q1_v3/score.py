"""Read the frozen decisions first, then score and pair all four methods at exactly the same origins."""
import numpy as np
import pandas as pd

from config import BASE,WORK,OUT,V2,ORIGINAL,KEY,digest,joined,macro,read_json,save
from motion import Inputs
from phase import decide
from settings import array
from intervals import risk_fields,adjust_width,sequences,band
import joblib


def phase_calls(d,ix):
    pc=read_json(BASE/"phase_selection.json")
    with np.load(WORK/(pc["model"]+"_phase.npz")) as f:
        return decide(f["raw"][ix],f["p"][ix],pc["rule"])


def rate_rows(log,all_points):
    rows=[]
    for split,df in log.groupby("split",sort=False):
        turn=df.turn.to_numpy()
        for key in ("original","v1","v2","v3"):
            col=df[key+"_path_cm"]
            hit=df[key+"_hit"].to_numpy(bool)
            r=df[key+"_radius_cm"]
            p=all_points[(all_points["split"]==split)&(all_points["method"]==key)].iloc[0]
            rows.append(dict(
                    method=key,split=split,windows=len(df),turns=int(turn.sum()),turn_ade_cm=float(col[turn].mean()),
                    turn_env_macro_cm=macro(col[turn],df.env.to_numpy()[turn]),nonturn_ade_cm=col[~turn].mean(),
                    covered=int(hit.sum()),coverage=hit.mean(),turn_covered=int(hit[turn].sum()),turn_coverage=hit[turn].mean(),
                    mean_radius_cm=r.mean(),turn_radius_cm=r[turn].mean(),
                    overall_ade_cm=p.path_macro_cm,overall_fde_cm=p.endpoint_macro_cm))
    return pd.DataFrame(rows)


def score():
    frozen=OUT/"frozen_before_test.json"
    stamp=digest(frozen)
    lock=read_json(frozen)
    d=Inputs()
    ix=np.flatnonzero(d.val|d.test)
    meta=d.meta.iloc[ix].reset_index(drop=True)
    env=d.env[ix]; grid=d.grid[ix]; mask=d.mask[ix]
    turn=array("next_mode")[ix]==1
    current=array("mode")[ix]==1
    mode=phase_calls(d,ix)
    reference=np.load(WORK/"reference_center.npy",mmap_mode="r")[ix]
    pred=np.load(WORK/"corrected_center.npy",mmap_mode="r")[ix]
    width=np.load(WORK/"reference_scale.npy",mmap_mode="r")[ix]
    err=np.linalg.norm(pred-d.y[ix],axis=2)
    olderr=np.linalg.norm(reference-d.y[ix],axis=2)
    plan=lock["interval"]
    fields=risk_fields(d.phase[ix],d.state[ix],d.q[ix],d.v[ix],pred,pred-reference)
    model,midpoint=joblib.load(WORK/"path_risk_scale.joblib")
    shape=adjust_width(plan["shape"],width,fields,model,midpoint)
    with np.load(WORK/(plan["shape"]+"_feedback_priors.npz")) as f:
        ref=f["reference"]; typical=float(f["typical"])
    components=sequences(meta,mode,err,shape,array("ok")[ix],ref,typical)
    radius=band(components,plan["scheme"],plan["factors"])
    hit=(err<=radius).all(1)
    # Old-architecture width, applied to the newly corrected center as a paired calibration ablation.
    previous_band=read_json(BASE/"frozen_before_test.json")["band"]
    if previous_band["scheme"]!="static":
        raise ValueError("The archived v2 comparator used a static radius")
    old_radius=width*previous_band["factor"]
    counter=(err<=old_radius).all(1)
    pt=[];horizon=[]
    for split in ("validation","test"):
        part=meta.split.eq(split).to_numpy()
        end=part&mask[:,-1]; complete=part&mask.all(1)
        pt.append(dict(split=split,method="v3",n=int(end.sum()),path_n=int(complete.sum()),
             endpoint_macro_cm=macro(100*err[end,-1],env[end]),path_macro_cm=macro(100*err[complete].mean(1),env[complete])))
        group=part&grid&turn
        for j in range(30):
            horizon.append(dict(split=split,frame=j+1,n=int(group.sum()),v2_error_cm=100*olderr[group,j].mean(),
                        v3_error_cm=100*err[group,j].mean(),new_coverage=(err[group,j]<=radius[group,j]).mean(),
                        mean_new_radius_cm=100*radius[group,j].mean()))
    points=pd.concat((pd.read_csv(V2/"three_way_metrics.csv"),pd.DataFrame(pt)),ignore_index=True)
    new=meta.loc[grid,KEY].copy()
    for name,values in dict(actual_turn=turn,already_turning=current,issue_alert=mode==1,
            v3_path_cm=100*err.mean(1),v3_fde_cm=100*err[:,-1],v3_hit=hit,v3_radius_cm=100*radius.mean(1),
            unchanged_band_on_new_center=counter,retrained_v2_cm=100*olderr.mean(1)).items():
        new[name]=values[grid]
    old=pd.read_csv(V2/"paired_windows.csv.gz",usecols=KEY+["turn","current_phase","v2_turn","v2_path_cm","v2_coverage","v2_radius_cm",
            "v1_path_cm","original_path_cm","joint_adaptive_covered","adaptive_covered","adaptive_radius_cm"])
    old=old.rename(columns={"v2_coverage":"v2_hit","joint_adaptive_covered":"original_hit","adaptive_covered":"v1_hit",
                            "adaptive_radius_cm":"v1_radius_cm"})
    joined_grid=joined(new,old)
    original=pd.read_csv(ORIGINAL/"nonoverlap_windows.csv.gz").query("method=='anchored_damped'")
    joined_grid=joined(joined_grid,original[KEY+["joint_adaptive_radius_cm"]].rename(columns={"joint_adaptive_radius_cm":"original_radius_cm"}))
    np.testing.assert_array_equal(joined_grid.actual_turn,joined_grid.turn)
    np.testing.assert_array_equal(joined_grid.issue_alert,joined_grid.v2_turn)
    np.testing.assert_allclose(joined_grid.retrained_v2_cm,joined_grid.v2_path_cm,atol=8e-5)
    assert len(joined_grid)==1161 and len(joined_grid.query("split=='test' and turn"))==99
    metrics=rate_rows(joined_grid,points)
    tr=[]
    paired=[]
    group_rows=[]
    for split,df in joined_grid.groupby("split",sort=False):
        use=df.turn
        for key in ("original","v1","v2"):
            diff=df.loc[use,"v3_path_cm"]-df.loc[use,key+"_path_cm"]
            both=df[["v3_hit",key+"_hit"]].astype(bool)
            paired.append(dict(split=split,comparison=f"v3 minus {key}",n_turn=len(diff),mean_delta_cm=diff.mean(),
                median_delta_cm=diff.median(),turn_improved=int((diff<0).sum()),turn_worse=int((diff>0).sum()),
                coverage_gained=int((both.v3_hit&~both[key+"_hit"]).sum()),
                coverage_lost=int((~both.v3_hit&both[key+"_hit"]).sum())))
        for label,sel in (("all_turns",df.turn),("onset",df.turn & ~df.already_turning),
                         ("ongoing",df.turn & df.already_turning),("alerted_turn",df.turn & df.issue_alert),
                         ("unalerted_turn",df.turn & ~df.issue_alert),("nonturn",~df.turn)):
            for key in ("original","v1","v2","v3"):
                group_rows.append(dict(split=split,group=label,method=key,n=int(sel.sum()),
                    ade_cm=df.loc[sel,key+"_path_cm"].mean(),coverage=df.loc[sel,key+"_hit"].mean(),
                    radius_cm=df.loc[sel,key+"_radius_cm"].mean()))
        for e,one in df.groupby("env"):
            sub=one[one.turn]
            row=dict(split=split,env=e,n=len(one),turns=len(sub))
            for key in ("original","v1","v2","v3"):
                row[key+"_turn_ade_cm"]=sub[key+"_path_cm"].mean()
                row[key+"_coverage"]=one[key+"_hit"].mean()
                row[key+"_turn_covered"]=int(sub[key+"_hit"].sum())
                row[key+"_radius_cm"]=one[key+"_radius_cm"].mean()
            tr.append(row)
    for frame,name in ((metrics,"four_way_metrics.csv"),(pd.DataFrame(tr),"by_environment.csv"),
            (pd.DataFrame(paired),"paired_differences.csv"),(pd.DataFrame(group_rows),"subgroups.csv"),
            (pd.DataFrame(horizon),"turn_by_horizon.csv"),(joined_grid,"paired_windows.csv.gz")):
        frame.to_csv(OUT/name,index=False)
    t=joined_grid.query("split=='test' and turn")
    cases=t[KEY+["already_turning","issue_alert","original_path_cm","v1_path_cm","v2_path_cm","v3_path_cm","original_hit","v2_hit","v3_hit",
                 "v2_radius_cm","v3_radius_cm"]].copy()
    cases["delta_cm"]=cases.v3_path_cm-cases.v2_path_cm
    cases.hand=cases.hand.map({"left":"左手","right":"右手"})
    cases=cases.rename(columns=dict(split="划分",env="环境",sample="记录",hand="手",origin="起点帧_从0计数",
        already_turning="起点已经转向",issue_alert="起点已发转向预警",original_path_cm="原方案轨迹误差_厘米",
        v1_path_cm="QSM_v1轨迹误差_厘米",v2_path_cm="QSM_v2轨迹误差_厘米",v3_path_cm="QSM_v3轨迹误差_厘米",
        original_hit="原方案整段覆盖",v2_hit="QSM_v2整段覆盖",v3_hit="QSM_v3整段覆盖",v2_radius_cm="QSM_v2平均半径_厘米",
        v3_radius_cm="QSM_v3平均半径_厘米",delta_cm="v3减v2轨迹误差_厘米"))
    cases.to_csv(OUT/"99个转向窗口逐窗对照.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(OUT/"issued_grid.npz",row=ix[grid],center=pred[grid],prior_center=reference[grid],radius=radius[grid],
        truth=d.y[ix][grid],prior_radius=old_radius[grid])
    assert digest(frozen)==stamp
    save("score_audit.json",dict(frozen_sha256=stamp,number_paired=len(joined_grid),test_turns=len(t),
         preexisting_checkpoint_loads=0,matched_reference_max_error_cm=float(np.abs(joined_grid.retrained_v2_cm-joined_grid.v2_path_cm).max())))
    print(metrics[["method","split","turn_ade_cm","covered","windows","coverage","turn_covered","mean_radius_cm","overall_ade_cm","overall_fde_cm"]].round(4).to_string(index=False),flush=True)


if __name__=="__main__":
    score()
