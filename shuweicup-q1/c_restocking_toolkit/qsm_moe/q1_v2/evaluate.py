"""Score the locked Q1 revision once and join it to both archived comparators on exact wrist origins."""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from settings import WORK, OUT, PREVIOUS, ORIGINAL, KEY, MODES, array, equal_env, write_json
from motion import Inputs
from phase import decide,scores
from distribution import gate_features, predict_block
from qsm_q1 import geometry,feedback_scales


def paired(left,right,keys):
    result=left.merge(right,on=keys,how="inner",validate="one_to_one",sort=False)
    if len(result)!=len(left) or len(result)!=len(right):
        raise ValueError("Wrist cohorts differ between methods")
    return result


def safe_scores(y,p,c,env):
    if len(set(y==1))==2:
        return scores(y,p,c,env)
    return dict(n=len(y),turns=int((y==1).sum()),alarms=int((c==1).sum()),tp=int(((y==1)&(c==1)).sum()),
                fp=int(((y!=1)&(c==1)).sum()))


def score_frozen():
    lock_file=OUT/"frozen_before_test.json"
    lock_hash=hashlib.sha256(lock_file.read_bytes()).hexdigest()
    lock=json.loads(lock_file.read_text(encoding="utf8"))
    d=Inputs()
    ids=np.flatnonzero(d.test | d.val)
    m=d.meta.iloc[ids].reset_index(drop=True)
    env=m.env.to_numpy()
    test=m.split.eq("test").to_numpy()
    val=~test
    mask=d.mask[ids]
    full=mask.all(1); grid=d.grid[ids]
    y=d.y[ids]
    point=np.load(WORK/"motion_center.npy",mmap_mode="r")[ids]
    res=np.load(WORK/"residual_knots.npy",mmap_mode="r")[ids]
    x=gate_features(d.phase[ids],d.q[ids],d.state[ids],d.phys[ids])
    coeff=np.load(WORK/"width_parameters.npy")
    center=[]; scales=[]; energy=[]; g=[]
    for b in np.array_split(np.arange(len(ids)),int(np.ceil(len(ids)/512))):
        pred,candidates,w,width=predict_block(point[b],res[b],x[b],d.phase[ids[b]],d.state[ids[b]],d.q[ids[b]],d.cad[ids[b]],lock,coeff)
        _,_,co=geometry(point[b],candidates,y[b])
        center.append(pred);scales.append(width);g.append(w)
        energy.append((co[:,0]+co[:,1]*w+co[:,2]*w*w)*100)
    center=np.concatenate(center); base=np.concatenate(scales); weights=np.concatenate(g)
    energy=np.concatenate(energy)
    err=np.linalg.norm(center-y,axis=2)
    with np.load(WORK/(lock["phase"]["model"]+"_phase.npz")) as f:
        raw,p=f["raw"][ids],f["p"][ids]
    stage=decide(raw,p,lock["phase"]["rule"])
    ref=np.load(WORK/"feedback_reference.npy")
    base_adapt=feedback_scales(m,stage,err,base,array("ok")[ids],ref)
    first=m.groupby(["sample","hand"],sort=False).indices
    segment=next(v for v in first.values() if len(v)>180)
    origin=m.origin.to_numpy()[segment]
    corrupt=err[segment].copy()
    corrupt[origin[:,None]+np.arange(1,31)>103]=400.
    checked=feedback_scales(m.iloc[segment].reset_index(drop=True),stage[segment],corrupt,base[segment],array("ok")[ids[segment]],ref)
    np.testing.assert_array_equal(checked[origin<=103],base_adapt[segment][origin<=103])
    selected_band=(base if lock["band"]["scheme"]=="static" else base_adapt)*lock["band"]["factor"]
    hit=(err<=selected_band).all(1)
    band_checks=[]
    for candidate in pd.read_csv(OUT/"band_validation.csv").itertuples():
        radii=(base if candidate.scheme=="static" else base_adapt)*candidate.factor
        for split,part in (("validation",val),("test",test)):
            use=grid & part
            band_checks.append(dict(split=split,scheme=candidate.scheme,quantile=candidate.quantile,
                locked=(candidate.scheme==lock["band"]["scheme"] and candidate.quantile==lock["band"]["quantile"]),
                n=int(use.sum()),coverage=(err[use]<=radii[use]).all(1).mean(),radius_cm=100*radii[use].mean()))
    labels=array("next_mode")[ids]
    current=array("mode")[ids]
    old_np=np.load(PREVIOUS/"cache"/"B4_issues.npz")
    oi=np.searchsorted(old_np["ids"],ids)
    np.testing.assert_array_equal(old_np["ids"][oi],ids)
    oldp=old_np["stage"][oi]
    old_rule=decide(oldp,oldp,lock["phase"]["old_rule"])
    stage_rows=[];confusion=[];environments=[];risk_metrics=[]
    phase_methods={
        "persistence":(np.eye(3)[current],current),
        "v1_argmax":(oldp,oldp.argmax(1)),
        "v1_threshold_only":(oldp,old_rule),
        "v2_raw_argmax":(raw,raw.argmax(1)),
        "v2_calibrated_argmax":(p,p.argmax(1)),
        "v2_decision":(p,stage)}
    summary=[]
    def lowerr(v,ix): return equal_env(v[ix]*100,env[ix])
    for split,sel in (("validation",val),("test",test)):
        ix=sel & grid; endpoints=sel & mask[:,-1]; complete=sel & full
        summary.append(dict(method="v2",split=split,n=int(endpoints.sum()),path_n=int(complete.sum()),windows=int(ix.sum()),
                 endpoint_macro_cm=lowerr(err[:,-1],endpoints),path_macro_cm=lowerr(err.mean(1),complete),
                 spaced_path_cm=lowerr(err.mean(1),ix),energy_cm=equal_env(energy[ix],env[ix]),
                 coverage=hit[ix].mean(),radius_cm=100*selected_band[ix].mean()))
        for name,(pr,decision) in phase_methods.items():
            stage_rows.append(dict(method=name,split=split,**scores(labels[ix],pr[ix,1],decision[ix],env[ix])))
            mat=confusion_matrix(labels[ix],decision[ix],labels=[0,1,2])
            for a in range(3):
                for b in range(3):
                    confusion.append(dict(method=name,split=split,actual=MODES[a],predicted=MODES[b],n=mat[a,b]))
            for e in sorted(set(env[ix])):
                take=ix&(env==e)
                environments.append(dict(method=name,split=split,env=e,**safe_scores(labels[take],pr[take,1],decision[take],env[take])))
        for group,sub in (("turn",labels==1),("dwell",labels==0),("translate",labels==2),
                          ("onset_turn",(labels==1)&(current!=1)),("ongoing_turn",(labels==1)&(current==1)),
                          ("recent_flag",array("recent")[ids])):
            use=ix&sub
            if use.any():
                risk_metrics.append(dict(split=split,group=group,n=int(use.sum()),coverage=hit[use].mean(),
                        radius_cm=selected_band[use].mean()*100,path_cm=err[use].mean()*100,
                        predicted_turns=int((stage[use]==1).sum())))
    bands0=pd.read_csv(ORIGINAL/"band_coverage.csv").query("method=='anchored_damped' and band=='joint_adaptive' and sampling=='grid' and scope=='all' and env=='all'")
    bands1=pd.read_csv(PREVIOUS/"q1_bands.csv").query("band=='adaptive' and group=='all'")
    energy1=pd.read_csv(PREVIOUS/"q1_energy.csv").query("method=='quality_stage_gate'").set_index("split")
    for name,root,method,bands in (("original",ORIGINAL,"anchored_damped",bands0),("v1",PREVIOUS,"B5",bands1)):
        points=pd.read_csv(root/("point_errors.csv" if name=="original" else "q1_points.csv"))
        for row in points[points.method.eq(method)].itertuples():
            br=bands[bands.split.eq(row.split)].iloc[0]
            summary.append(dict(method=name,split=row.split,n=int(row.n),path_n=int(row.path_n),windows=int(row.spaced_n),
                    endpoint_macro_cm=row.endpoint_macro_cm,path_macro_cm=row.path_macro_cm,spaced_path_cm=row.spaced_path_cm,
                    energy_cm=energy1.at[row.split,"energy_score_cm"] if name=="v1" else np.nan,coverage=br.path_coverage,
                    radius_cm=br["mean_radius_cm" if name=="original" else "radius_cm"]))
    endpoints=m.loc[mask[:,-1],KEY].copy()
    endpoints["v2_fde_cm"]=err[mask[:,-1],-1]*100
    prior=pd.read_csv(PREVIOUS/"q1_paired_endpoints.csv.gz",usecols=KEY+["anchored_damped","B5"]).rename(
         columns={"anchored_damped":"original_fde_cm","B5":"v1_fde_cm"})
    endpoints=paired(endpoints,prior,KEY)
    windows=m.loc[grid,KEY].copy()
    for col,a in dict(turn=(labels==1),target_phase=labels,current_phase=current,v2_turn=(stage==1),
              v1_turn=(oldp.argmax(1)==1),v2_phase=stage,v2_turn_prob=p[:,1],v1_turn_prob=oldp[:,1],
              v2_calibrated_argmax=p.argmax(1),v2_raw_argmax=raw.argmax(1),v1_threshold_turn=(old_rule==1),
              v2_path_cm=100*err.mean(1),v2_fde_cm=100*err[:,-1],v2_max_error_cm=100*err.max(1),v2_coverage=hit,v2_radius_cm=100*selected_band.mean(1),
              v2_energy_cm=energy,diffusion_weight=weights).items():
        windows[col]=a[grid]
    old=pd.read_csv(PREVIOUS/"q1_paired_windows.csv.gz",usecols=KEY+["B3_path_cm","B4_path_cm","B5_path_cm","path_mean_error_cm",
                      "stage","actual_mode_at_10","joint_adaptive_covered","adaptive_covered","adaptive_radius_cm","max_error_cm","B5_max_error_cm"])
    windows=paired(windows,old.rename(columns={"B5_path_cm":"v1_path_cm","path_mean_error_cm":"original_path_cm"}),KEY)
    np.testing.assert_array_equal(windows.target_phase,windows.actual_mode_at_10)
    assert (windows.query("split=='test'").turn.sum(),len(windows),len(endpoints))==(99,1161,36466)
    common=[]
    for split,sub in windows.groupby("split"):
        for radius in (10,20,25,30):
            common.append(dict(split=split,fixed_radius_cm=radius,n=len(sub),original=(sub.max_error_cm<=radius).mean(),
                               v1=(sub.B5_max_error_cm<=radius).mean(),v2=(sub.v2_max_error_cm<=radius).mean()))
    by_env=[]
    for (split,e),sub in windows.groupby(["split","env"]):
        ep=endpoints[endpoints.split.eq(split)&endpoints.env.eq(e)]
        turn=sub[sub.turn]
        row=dict(split=split,env=e,n=len(sub),turns=len(turn),turn_tp=int(turn.v2_turn.sum()),
                 turn_fp=int(((~sub.turn)&sub.v2_turn).sum()))
        for name in ("original","v1","v2"):
            row[name+"_spaced_ade"]=sub[name+"_path_cm"].mean()
            row[name+"_fde"]=ep[name+"_fde_cm"].mean()
            row[name+"_turn_ade"]=turn[name+"_path_cm"].mean()
        by_env.append(row)
    # Verify the original and v1 window metrics can be recomputed from the same joined cohorts.
    metrics=pd.DataFrame(summary)
    for (method,sp),sub in metrics.groupby(["method","split"]):
        subw=windows[windows.split.eq(sp)]
        sube=endpoints[endpoints.split.eq(sp)]
        np.testing.assert_allclose(equal_env(subw[method+"_path_cm"],subw.env),sub.spaced_path_cm.iat[0],atol=2e-5)
        np.testing.assert_allclose(equal_env(sube[method+"_fde_cm"],sube.env),sub.endpoint_macro_cm.iat[0],atol=2e-5)
    vid=np.load(WORK/"visual_shift.npy",mmap_mode="r")[ids]
    motion0=np.load(WORK/"motion_only.npy",mmap_mode="r")[ids]
    vr=[]
    for sp,sel in (("validation",val),("test",test)):
        end=sel&mask[:,-1]; complete=sel&full
        for w in (0.,.25,.5,1.):
            er=np.linalg.norm(motion0+w*vid-y,axis=2)
            vr.append(dict(split=sp,alpha=w,selected=(w==lock["motion"]["visual_weight"]),
                           ade_cm=lowerr(er.mean(1),complete),fde_cm=lowerr(er[:,-1],end)))
    for f,name in ((metrics,"three_way_metrics.csv"),(pd.DataFrame(stage_rows),"turn_metrics.csv"),
             (pd.DataFrame(confusion),"confusion_matrices.csv"),(pd.DataFrame(environments),"turn_by_environment.csv"),
             (pd.DataFrame(by_env),"paired_by_environment.csv"),(pd.DataFrame(risk_metrics),"v2_subgroups.csv"),
             (pd.DataFrame(vr),"local_visual_ablation.csv"),(pd.DataFrame(common),"common_radius.csv"),
             (pd.DataFrame(band_checks),"interval_sensitivity.csv"),(endpoints,"paired_endpoints.csv.gz"),(windows,"paired_windows.csv.gz")):
        f.to_csv(OUT/name,index=False)
    np.savez_compressed(OUT/"issued_grid.npz",ids=ids[grid],center=center[grid],motion=point[grid],residual_knots=res[grid],
              truth=y[grid],radius=selected_band[grid],p=p[grid],raw_phase=raw[grid],called=stage[grid],diffusion_weight=weights[grid])
    assert hashlib.sha256(lock_file.read_bytes()).hexdigest()==lock_hash
    write_json(OUT/"evaluation_audit.json",dict(frozen_sha256=lock_hash,old_parameters_loaded=0,pairing_keys=KEY,
         matched_endpoints=len(endpoints),matched_windows=len(windows),test_true_turns=99,future_feedback_invariance=True))
    print(metrics.round(4).to_string(index=False),flush=True)
    print(pd.DataFrame(stage_rows).query("split=='test'")[["method","tp","fp","precision","recall","turn_f1","AP"]].round(3).to_string(index=False))


if __name__=="__main__":
    score_frozen()
