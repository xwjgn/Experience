"""Score frozen predictions. Only this module reads historical scalar results for comparison."""
import numpy as np
import pandas as pd
from scipy.special import chndtr,expit,logsumexp
from sklearn.metrics import average_precision_score,roc_auc_score

from study_paths import DATA,KEY,OUT,ROOT,digest,read,save
from recurrent_network import deterministic
from fit_wrist import Study,center_and_scale,issue_components,load_own,matrix,weights_file
from lock_candidates import get_outputs,stage_alarm
from trajectory_bands import play


def join(a,b):
    out=a.merge(b,on=KEY,how="inner",sort=False,validate="one_to_one")
    if len(out)!=len(a) or len(out)!=len(b):
        raise ValueError("Previously published origins differ from the official-data evaluation grid")
    return out


def marginal95(w,mu,sd,center):
    distance=np.linalg.norm(mu-center[:,None],axis=3)
    noncentral=np.square(distance/sd).astype(float)
    high=(distance+5*sd).max(1).astype(float)
    low=np.zeros_like(high)
    for _ in range(22):
        r=(high+low)*.5
        mass=(w[:,:,None]*chndtr(np.square(r[:,None]/sd),3.,noncentral)).sum(1)
        high=np.where(mass>=.95,r,high)
        low=np.where(mass<.95,r,low)
    return high


def distribution_metrics(out,ar):
    keep=out.complete
    y,mu,sd,w=out.truth[keep],out.means[keep],out.sd[keep],out.weight[keep]
    z=(y[:,None]-mu)/sd[:,:,:,None]
    rho=np.asarray(ar)[None,:,None,None]
    innovation=(z[:,:,1:]-rho*z[:,:,:-1])/np.sqrt(1-rho*rho)
    ll=-.5*(np.square(z[:,:,0]).sum(2)+np.square(innovation).sum((2,3)))-3*np.log(sd).sum(2)
    ll-=45*np.log(2*np.pi)+1.5*29*np.log(1-np.square(np.asarray(ar)))[None]
    joint=-logsumexp(np.log(w.clip(1e-12))+ll,axis=1)/30
    marginal_ll=-1.5*np.log(2*np.pi)-3*np.log(sd)-.5*np.square(z).sum(3)
    marginal=-logsumexp(np.log(w.clip(1.e-12))[:,:,None]+marginal_ll,axis=1).mean(1)
    rng=np.random.default_rng(78109)
    pieces=[]
    for i in range(0,len(y),64):
        take=slice(i,i+64)
        ws=w[take]
        n=len(ws); samples=[]
        for _ in range(2):
            u=rng.random((n,40))
            chosen=np.minimum((u[:,:,None]>ws.cumsum(1)[:,None]).sum(2),ws.shape[1]-1)
            noise=rng.standard_normal((n,40,30,3))
            corr=np.asarray(ar)[chosen,None]
            for j in range(1,30):
                noise[:,:,j]=corr*noise[:,:,j-1]+np.sqrt(1-corr*corr)*noise[:,:,j]
            samples.append(mu[take][np.arange(n)[:,None],chosen]+sd[take][np.arange(n)[:,None],chosen][:,:,:,None]*noise)
        def distance(diff):
            return np.sqrt(np.square(diff).sum(3).mean(2)).mean(1)
        pieces.append(distance(samples[0]-y[take,None])-.5*distance(samples[0]-samples[1]))
    es=100*np.concatenate(pieces)
    return dict(path_nll_per_step=float(joint.mean()),marginal_nll=float(marginal.mean()),energy_score_cm=float(es.mean()),
                turn_energy_score_cm=float(es[out.stage[keep]==1].mean()))


def evaluate_quality(d,nets,spec,role):
    ids=d.rows(role,stride=1,eligible=False,start=0)
    logits,=issue_components(nets,d.features,ids,quality_only=True)
    cfg=spec["quality"]
    risk=expit(cfg["slope"]*logits+cfg["intercept"])
    bad=d.flag[ids]
    rows=[]
    for group,select in [("all",np.ones(len(ids),bool)),*[(e,d.meta.env.to_numpy()[ids]==e) for e in np.unique(d.meta.env.to_numpy()[ids])]]:
        label,b=bad[select],risk[select]
        alert=b>=cfg["threshold"]
        tp=int(np.sum(alert&label))
        rows.append(dict(environment=group,n=len(b),positive=int(label.sum()),prevalence=label.mean(),
            average_precision=average_precision_score(label,b) if label.any() else None,
            auc=roc_auc_score(label,b) if len(np.unique(label))==2 else None,brier=np.square(b-label).mean(),
            base_rate_brier=np.square(d.flag_rate-label).mean(),tp=tp,fp=int((alert&~label).sum()),
            recall=tp/max(1,label.sum()),precision=tp/max(1,alert.sum())))
    return rows


def row_metrics(err,radius,turn,env):
    ade=err.mean(1)*100
    good=(err<=radius).all(1)
    return dict(n=len(ade),turns=int(turn.sum()),ade_cm=ade.mean(),turn_ade_cm=ade[turn].mean() if turn.any() else None,
            endpoint_cm=100*err[:,-1].mean(),covered=int(good.sum()),coverage=good.mean(),
            turn_covered=int(good[turn].sum()),turn_coverage=good[turn].mean() if turn.any() else None,
            frame_coverage=(err<=radius).mean(),radius_cm=100*radius.mean(),turn_radius_cm=100*radius[turn].mean() if turn.any() else None,
            min_env=min(good[env==s].mean() for s in np.unique(env)))


def natural_flagged(d,nets,spec,role):
    ids=d.rows(role,stride=1,eligible=False,start=10)
    marked=(matrix("qinfo")[ids,6]<1.) & d.eligible[ids].any(1)
    ids=ids[marked]
    if not len(ids):
        return []
    _,w,mu,sd,_=issue_components(nets,d.features,ids,spec["temperature"])
    center,_=center_and_scale(w,mu,sd,spec["readout"])
    error=100*np.linalg.norm(center-d.y[ids],axis=2)
    rows=[]
    for h in (1,3,10,30):
        for label,mask in (("recent_8",np.ones(len(ids),bool)),("at_origin",d.flag[ids])):
            use=mask&d.eligible[ids,h-1]
            rows.append(dict(subset=label,horizon=h,n=int(use.sum()),error_cm=error[use,h-1].mean() if use.any() else None))
    return rows


def score():
    lockpath=OUT/"frozen_before_test.json"
    stamp=digest(lockpath); lock=read(lockpath)
    deterministic()
    d=Study()
    # Prior studies are result-only controls. No derived feature array, posterior or parameter is loaded.
    previous=ROOT/"分析报告"/"第一问质量感知LSTTN与阶段概率预测"
    controls=pd.read_csv(previous/"comparison.csv")
    kalman=pd.read_csv(previous/"current_kalman_anchor.csv").set_index("split")
    historic=pd.read_csv(previous/"paired_windows.csv.gz")
    historic=historic[historic.calibration=="cafht_cptc"][KEY+["turn","ade_cm","hit","radius_cm","qsm_ade_cm","qsm_hit","qsm_radius_cm","original_ade_cm","original_hit","original_radius_cm"]].rename(columns={v:"lsttn_"+v for v in ("ade_cm","hit","radius_cm")})
    results,qs,ph,ps,grouped,paired,occluded,horizons=[],[],[],[],[],[],[],[]
    clock=matrix("time")
    stored={"M2":[],"M3":[]}
    for split in ("validation","test"):
        kh=kalman.loc[split]
        results.append(dict(split=split,model="M0",band="archived_quality_kalman_damped",n=int(kh.windows),
                        turns=int(kh.turns),ade_cm=kh.ade,turn_ade_cm=kh.turn))
        for key,band in (("original","archived"),("qsm_low_turn","archived"),("quality_stage","cafht_cptc")):
            m=controls[(controls.model==key)&(controls.calibration==band)&(controls.split==split)].iloc[0]
            results.append(dict(split=split,model="published_"+key,band=band,n=m.windows,turns=m.turns,ade_cm=m.ade_cm,
                 turn_ade_cm=m.turn_ade_cm,covered=m.covered,coverage=m.coverage,turn_covered=m.turn_covered,turn_coverage=m.turn_coverage,radius_cm=m.mean_radius_cm,min_env=m.min_environment))
        old=historic[historic.split==split]
        for kind,spec in lock["models"].items():
            if any(digest(weights_file(kind,seed))!=spec["params"][str(seed)] for seed in spec["seeds"]):
                raise ValueError("Candidate weights changed after locking")
            nets=load_own(kind,spec["seeds"],lock["dims"],lock["flag_rate"],d.device)
            a=get_outputs(d,split,nets,spec["temperature"])
            center,scale=a.point(spec["readout"])
            err=np.linalg.norm(center-a.truth,axis=2)
            use=a.complete
            turn=a.stage[use]==1
            env=a.meta.env.to_numpy()[use]
            aligned=join(a.meta.loc[use,KEY],old)
            np.testing.assert_array_equal(turn,aligned.turn)
            raw=marginal95(a.weight[use],a.means[use],a.sd[use],center[use])
            bands={"raw_step95":raw}
            for name,cfg in spec["bands"].items():
                rec=cfg["recipe"]
                path=OUT/f"{kind}_{rec['family']}_reference.npz"
                if digest(path)!=cfg["reference_sha256"]:
                    raise ValueError("Calibration prior changed")
                with np.load(path) as pf:
                    base=play(a.meta,err,scale,a.p,a.seen,{k:pf[k] for k in pf.files},rec)
                bands[name]=base[use]*rec["outer"]
            main=bands["joint_stage"]
            for horizon in (1,3,10,30):
                target=horizon-1
                elapsed=clock[a.ids[use]+horizon]-clock[a.ids[use]]
                distances=err[use,target]*100
                inside=err[use,target]<=main[:,target]
                horizons.append(dict(split=split,model=kind,horizon=horizon,n=len(turn),turns=int(turn.sum()),
                    mean_error_cm=distances.mean(),turn_mean_error_cm=distances[turn].mean(),
                    covered=int(inside.sum()),turn_covered=int(inside[turn].sum()),mean_radius_cm=100*main[:,target].mean(),
                    min_seconds=float(elapsed.min()),median_seconds=float(np.median(elapsed)),max_seconds=float(elapsed.max())))
            ratio=aligned.qsm_radius_cm.mean()/(100*main.mean())
            bands["equal_QSM_width_diagnostic"]=main*ratio
            for band,radius in bands.items():
                results.append(dict(split=split,model=kind,band=band,**row_metrics(err[use],radius,turn,env)))
                if band in ("joint_stage","joint_stage_env"):
                    for e in np.unique(env):
                        sub=env==e
                        grouped.append(dict(model=kind,split=split,band=band,env=e,**row_metrics(err[use][sub],radius[sub],turn[sub],env[sub])))
            table=aligned.copy()
            table["model"]=kind
            table["ade_cm"]=100*err[use].mean(1)
            table["hit"]=(err[use]<=main).all(1)
            table["radius_cm"]=100*main.mean(1)
            table["turn_probability"]=a.p[use,1]
            table["flag_probability"]=expit(spec["quality"]["slope"]*a.qlogit[use]+spec["quality"]["intercept"])
            table["actual_flag"]=d.flag[a.ids[use]]
            table["turn_alert"]=stage_alarm(a.p[use],spec["turn_cut"])
            paired.append(table)
            for rule,alert in (("gate_threshold",table.turn_alert.to_numpy()),("argmax",a.p[use].argmax(1)==1)):
                tp=int((alert&turn).sum())
                ph.append(dict(split=split,model=kind,rule=rule,turns=int(turn.sum()),tp=tp,fp=int((alert&~turn).sum()),
                             recall=tp/max(1,turn.sum()),precision=tp/max(1,alert.sum())))
            qs.extend(dict(model=kind,split=split,**v) for v in evaluate_quality(d,nets,spec,split))
            ps.append(dict(model=kind,split=split,**distribution_metrics(a,spec["ar"])))
            if kind in ("M1","M2","M3"):
                occluded.extend(dict(model=kind,split=split,**r) for r in natural_flagged(d,nets,spec,split))
            if kind in stored:
                stored[kind].append(dict(row=a.ids[use],center=center[use],scale=scale[use],radius=main,
                    guarded=bands["joint_stage_env"],phase=a.p[use],weights=a.weight[use],means=a.means[use],sd=a.sd[use],qlogit=a.qlogit[use]))
    assert digest(lockpath)==stamp
    results=pd.DataFrame(results)
    results.to_csv(OUT/"comparison.csv",index=False)
    pd.DataFrame(qs).to_csv(OUT/"quality_all_handframes.csv",index=False)
    pd.DataFrame(ph).to_csv(OUT/"stage_scores.csv",index=False)
    pd.DataFrame(ps).to_csv(OUT/"distribution_scores.csv",index=False)
    pd.DataFrame(occluded).to_csv(OUT/"natural_bad_observations.csv",index=False)
    pd.DataFrame(grouped).to_csv(OUT/"by_environment.csv",index=False)
    pd.DataFrame(horizons).to_csv(OUT/"by_horizon.csv",index=False)
    log=pd.concat(paired,ignore_index=True)
    log.to_csv(OUT/"paired_issues.csv.gz",index=False)
    turnlog=historic[(historic.split=="test")&historic.turn].copy()
    for kind in lock["models"]:
        cols=["ade_cm","hit","radius_cm","turn_alert"]
        sub=log[(log.split=="test")&log.turn&(log.model==kind)][KEY+cols].rename(columns={c:kind+"_"+c for c in cols})
        turnlog=join(turnlog,sub)
    turnlog.to_csv(OUT/"99个真实转向逐窗对照.csv",index=False,encoding="utf-8-sig")
    for name,items in stored.items():
        np.savez_compressed(OUT/(name+"_issued.npz"),**{k:np.concatenate([v[k] for v in items]) for k in items[0]})
    save(OUT/"scoring_audit.json",dict(frozen_sha256=stamp,test_turns=len(turnlog),prior_weights_loaded=0,M0_refitted=False,
         historical_results_used=["current_kalman_anchor.csv","comparison.csv","paired_windows.csv.gz"],
         selected_rows_checked_by=KEY,kalman_interval="No matching archived 30-step radius for the standalone damped Kalman; left empty."))
    print(results[(results.split=="test")&results.band.isin(["joint_stage","joint_stage_env","archived_quality_kalman_damped","archived","cafht_cptc"])][["model","band","ade_cm","turn_ade_cm","covered","turn_covered","radius_cm"]].round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    score()
