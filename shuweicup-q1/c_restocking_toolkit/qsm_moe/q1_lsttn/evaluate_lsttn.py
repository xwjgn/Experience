"""Held-out evaluation. Historical predictions are read only in this scoring entry point."""
import numpy as np
import pandas as pd
from scipy.special import chndtr
import torch

from paths_q1 import DATA,KEY,OUT,PREVIOUS,ROOT,digest,read_json,save_json
from lsttn_model import RHO,deterministic,joint_log_density
from train_lsttn import Study,checkpoint,load_nets
from freeze_lsttn import panel,readout
from path_calibration import load_bank,replay


def exact_marginal(p,center):
    # Each spherical marginal of a Gaussian about the published center is noncentral chi-square.
    dist = np.linalg.norm(p.means-center[:,None],axis=-1)
    nc = np.square(dist/p.sigma).astype(float)
    high = (dist+5*p.sigma).max(1).astype(float)
    low = np.zeros_like(high)
    for _ in range(21):
        mid = (high+low)*.5
        cdf = (p.weight[:,:,None]*chndtr(np.square(mid[:,None]/p.sigma),3.,nc)).sum(1)
        low = np.where(cdf<.95,mid,low)
        high = np.where(cdf>=.95,mid,high)
    return high


def probability_scores(p):
    use = p.full
    y,mu,sd,w = p.y[use],p.means[use],p.sigma[use],p.weight[use]
    ll = joint_log_density(torch.tensor(y),torch.tensor(mu),torch.tensor(sd))
    nll = (-torch.logsumexp(torch.tensor(w).clamp_min(1e-12).log()+ll,dim=1)/30).numpy()
    if not np.isfinite(nll).all():
        raise ValueError("Nonfinite probability score")
    rng = np.random.default_rng(62419)
    energies = []
    for start in range(0,len(y),64):
        take = slice(start,start+64)
        weight = w[take]
        count = len(weight)
        samples = []
        for _ in range(2):
            uniform = rng.random((count,48))
            idx = (uniform[:,:,None]>weight.cumsum(1)[:,None]).sum(-1).clip(max=w.shape[1]-1)
            noise = rng.standard_normal((count,48,30,3),dtype=np.float32)
            for h in range(1,30):
                noise[:,:,h] = RHO*noise[:,:,h-1]+np.sqrt(1-RHO**2)*noise[:,:,h]
            samples.append(mu[take][np.arange(count)[:,None],idx]+sd[take][np.arange(count)[:,None],idx][:,:,:,None]*noise)
        def distance(diff):
            return np.sqrt(np.square(diff).sum(-1).mean(-1)).mean(1)
        energies.append(100*(distance(samples[0]-y[take,None])-.5*distance(samples[0]-samples[1])))
    return nll,np.concatenate(energies)


def summarize(frame):
    t = frame.turn.to_numpy(bool)
    return dict(windows=len(frame),turns=int(t.sum()),ade_cm=frame.ade_cm.mean(),turn_ade_cm=frame.loc[t,"ade_cm"].mean(),
                env_macro_ade_cm=frame.groupby("env").ade_cm.mean().mean(),
                covered=int(frame.hit.sum()),coverage=frame.hit.mean(),turn_covered=int(frame.loc[t,"hit"].sum()),
                turn_coverage=frame.loc[t,"hit"].mean(),mean_radius_cm=frame.radius_cm.mean(),turn_radius_cm=frame.loc[t,"radius_cm"].mean(),
                min_environment=frame.groupby("env").hit.mean().min())


def merge_exact(a,b):
    joined = a.merge(b,on=KEY,how="inner",sort=False,validate="one_to_one")
    if len(joined)!=len(a) or len(joined)!=len(b):
        raise ValueError("Inconsistent evaluation origins")
    return joined


def score():
    frozen = OUT/"frozen_before_test.json"
    stamp = digest(frozen)
    lock = read_json(frozen)
    deterministic()
    d = Study()
    old = pd.read_csv(PREVIOUS/"paired_windows.csv.gz")
    old = old[old.model=="phase_only"][KEY+["turn","ade_cm","radius_cm","hit","v3_ade_cm","v3_hit","v3_radius_cm",
                                            "original_ade_cm","original_hit","original_radius_cm"]].copy()
    old = old.rename(columns={n:"qsm_"+n for n in ("ade_cm","radius_cm","hit")})
    old_issue = ROOT/"分析报告"/"QSM-MoE第一问转向与覆盖优化"/"issued_grid.npz"
    old_meta = pd.read_parquet(PREVIOUS/"qsm_retrained"/"backbone"/"fresh"/"cache"/"meta.parquet")
    split_logs,phase_rows,prob_rows,horizons,detail,groups = [],[],[],[],[],[]
    issued = []
    for split in ("validation","test"):
        history = old[old.split==split]
        for name,columns in (("original","original"),("qsm_v3","v3"),("qsm_low_turn","qsm")):
            tmp = history[KEY+["turn"]].assign(ade_cm=history[columns+"_ade_cm"],radius_cm=history[columns+"_radius_cm"],hit=history[columns+"_hit"])
            split_logs.append(dict(model=name,calibration="archived",split=split,**summarize(tmp)))
            for env,part in tmp.groupby("env"):
                groups.append(dict(model=name,calibration="archived",split=split,group="environment",value=env,**summarize(part)))
        for name,settings in lock["experiments"].items():
            seeds = settings["seeds"]
            for seed in seeds:
                if digest(checkpoint(settings["variant"],seed))!=settings["checkpoint_sha256"][str(seed)]:
                    raise ValueError("Checkpoint changed after selection")
            nets = load_nets(settings["variant"],seeds,d.dims,d.device)
            p = panel(d,split,nets,settings["temperature"])
            center,radial = readout(p,settings["point"])
            error = np.linalg.norm(p.y-center,axis=-1)
            use = p.full
            take = p.ids[use]
            compared = merge_exact(p.meta.loc[use,KEY],history)
            target = p.mode[use]
            np.testing.assert_array_equal(target==1,compared.turn.to_numpy())
            turn = target==1
            pr = p.prob[use]
            action = pr[:,1]>=settings["alert_threshold"]
            hard = pr.argmax(1)
            for rule,alert in (("frozen_alarm",action),("argmax",hard==1)):
                tp = int((alert&turn).sum()); fp = int((alert&~turn).sum())
                phase_rows.append(dict(model=name,split=split,rule=rule,actual=int(turn.sum()),tp=tp,fp=fp,
                    recall=tp/max(1,turn.sum()),precision=tp/max(1,tp+fp),f1=2*tp/max(1,alert.sum()+turn.sum()),
                    stage_cross_entropy=-np.log(pr[np.arange(len(target)),target].clip(1e-9)).mean(),
                    multiclass_brier=np.square(pr-np.eye(3)[target]).sum(1).mean()))
            nll,es = probability_scores(p)
            prob_rows.append(dict(model=name,split=split,windows=len(es),path_nll_per_3d_step=nll.mean(),
                                  energy_score_cm=es.mean(),turn_energy_score_cm=es[turn].mean(),ar_rho=RHO,samples=48))
            all_bands = {}
            if name=="quality_stage":
                all_bands["marginal_95"] = exact_marginal(p,center)[use]
            for band,choice in settings["intervals"].items():
                if name!="quality_stage" and band!="cafht_cptc":
                    continue
                recipe = choice["recipe"]
                prior_path = OUT/(name+"_"+recipe["profile"]+"_prior.npz")
                if digest(prior_path)!=settings["prior_sha256"][recipe["profile"]]:
                    raise ValueError("Calibration prior changed")
                shape,banks = load_bank(prior_path)
                base = replay(p.meta,error,radial,p.prob,p.visible,recipe,shape,banks)
                all_bands[band] = base[use]*recipe["outer_factor"]
            for band,radius in all_bands.items():
                hits = (error[use]<=radius).all(1)
                table = compared.copy()
                table["ade_cm"] = error[use].mean(1)*100
                table["radius_cm"],table["hit"] = radius.mean(1)*100,hits
                split_logs.append(dict(model=name,calibration=band,split=split,frame_coverage=(error[use]<=radius).mean(),
                                       fde_cm=100*error[use,-1].mean(),**summarize(table)))
                if name=="quality_stage":
                    table["model"],table["calibration"] = name,band
                    table["turn_probability"] = pr[:,1]
                    table["issued_turn_alert"] = action
                    table["newly_lost"] = table.qsm_hit & ~table.hit
                    table["newly_covered"] = ~table.qsm_hit & table.hit
                    table["stage_changed"] = np.load(DATA/"current_mode.npy",mmap_mode="r")[take]!=target
                    table["recent_flag"] = np.load(DATA/"quality.npy",mmap_mode="r")[take,6]<1.
                    detail.append(table)
                    for tag,part in table.groupby("env"):
                        groups.append(dict(model=name,calibration=band,split=split,group="environment",value=tag,**summarize(part)))
                    for col in ("stage_changed","recent_flag"):
                        for tag,part in table.groupby(col):
                            groups.append(dict(model=name,calibration=band,split=split,group=col,value=str(tag),**summarize(part)))
                    for h in (1,3,10,20,30):
                        horizons.append(dict(split=split,calibration=band,step=h,ade_cm=100*error[use,h-1].mean(),
                            turn_error_cm=100*error[use,h-1][turn].mean(),coverage=(error[use,h-1]<=radius[:,h-1]).mean(),
                            average_radius_cm=100*radius[:,h-1].mean()))
            if name=="quality_stage":
                sign = np.where(p.meta.hand.to_numpy()[use]=="right",-1.,1.)
                truth = p.y[use].copy()
                truth[:,:,1] *= sign[:,None]
                with np.load(old_issue) as cache:
                    raw_keys = old_meta.iloc[cache["row"]][KEY].copy()
                    raw_keys["row_in_archive"] = np.arange(len(raw_keys))
                    reorder = merge_exact(compared[KEY],raw_keys[raw_keys.split==split])["row_in_archive"].to_numpy()
                    np.testing.assert_allclose(truth,cache["truth"][reorder],atol=2e-6,rtol=0)
                issued.append(dict(row=take,center=center[use],radial=radial[use],radius=all_bands["cafht_cptc"],
                    guard=all_bands["cafht_cptc_buffer"],weight=p.weight[use],means=p.means[use],sigma=p.sigma[use],phase=pr))
    result = pd.DataFrame(split_logs)
    result.to_csv(OUT/"comparison.csv",index=False)
    pd.DataFrame(groups).to_csv(OUT/"by_environment_and_quality.csv",index=False)
    pd.DataFrame(phase_rows).to_csv(OUT/"stage_metrics.csv",index=False)
    pd.DataFrame(prob_rows).to_csv(OUT/"probability_metrics.csv",index=False)
    pd.DataFrame(horizons).to_csv(OUT/"by_horizon.csv",index=False)
    logs = pd.concat(detail,ignore_index=True)
    logs.to_csv(OUT/"paired_windows.csv.gz",index=False)
    logs[(logs.split=="test") & logs.turn & (logs.calibration=="cafht_cptc")].to_csv(OUT/"99个转向窗口对照.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(OUT/"issued_distribution.npz",**{k:np.concatenate([v[k] for v in issued]) for k in issued[0]})
    assert digest(frozen)==stamp
    save_json(OUT/"scoring_audit.json",dict(frozen_sha256=stamp,matching_raw_reference_tolerance_m=2e-6,
      old_checkpoints_loaded=0,finite_dimensional_distribution="one latent component per full path, isotropic horizon sigmas, Gaussian AR(1) normalized residual",
      coordinate_frame_of_npz="displacement in head axes at origin, right-hand lateral component mirrored; metres"))
    print(result.loc[result.split=="test",["model","calibration","turn_ade_cm","covered","turn_covered","mean_radius_cm","ade_cm"]].round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    score()
