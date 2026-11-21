"""Score the frozen Bayesian branch with paired spherical coverage and separate ellipsoidal coverage."""
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gain_records import KEY,OUT,PREVIOUS,dataset,digest,isolate,read,save
from comparison_reference import table_reference,verify_test_reference
from train_gain import ModelBank
from freeze_gain import variants
from gain_calibration import distance_axes,geometric,nll_values,path_metrics,redistribute,third_mix


def score():
    lock=read(OUT/"frozen_before_test.json")
    stamp=digest(OUT/"frozen_before_test.json")
    for p,sig in lock["hashes"].items():
        if digest(OUT/p)!=sig:
            raise ValueError("Frozen artifact has changed: "+p)
    w=dataset("test")
    truth,turn,env=w.truth,w.meta.turn.to_numpy(),w.meta.env.to_numpy()
    assert len(w.meta)==760 and turn.sum()==99
    ref,r0=table_reference(w,joblib.load(OUT/"cold_comparison_only.joblib"))
    published=verify_test_reference(w,ref,r0)
    # The fourth row has the same centre and the same within-window horizon profile,
    # with a different (previously issued) scalar width for each window.
    p4=read(PREVIOUS/"frozen_before_test.json")["radii"]["policies"]["compact"]
    assert p4["shape"]=="equal" and p4["timing"]=="flat"
    r4=r0*(published.mixed_center_compact_radius_cm.to_numpy()/(100*r0.mean(1)))[:,None]
    np.testing.assert_array_equal((np.linalg.norm(ref-w.truth,axis=2)<=r4).all(1),published.mixed_center_compact_hit)
    models=variants(ModelBank(),w)
    rows,env_rows,ellipsoids,diagnostics=[],[],[],[]
    paired=w.meta[KEY+["dataset","row","turn"]].copy()
    old_hit=(np.linalg.norm(truth-ref,axis=2)<=r0).all(1)
    assert old_hit.sum()==747 and old_hit[turn].sum()==99
    records={}
    def compare(name,rule,mu,radius):
        e=np.linalg.norm(truth-mu,axis=2)
        hit=(e<=radius).all(1)
        row=dict(model=name,rule=rule,**path_metrics(mu,truth,radius,turn,env),
                 gained_vs_third=int((hit & ~old_hit).sum()),lost_vs_third=int((~hit & old_hit).sum()))
        rows.append(row)
        if rule in ("matched","standalone_sphere","hybrid_guarded","archived","transfer_fourth_width"):
            for label in np.unique(env):
                use=env==label
                env_rows.append(dict(model=name,rule=rule,environment=label,**path_metrics(
                    mu[use],truth[use],radius[use],turn[use],env[use])))
        return hit
    compare("third","matched",ref,r0)
    compare("fourth","archived",ref,r4)
    paired["third_ade_cm"]=100*np.linalg.norm(truth-ref,axis=2).mean(1)
    paired["third_hit"]=old_hit
    paired["third_radius_cm"]=100*r0.mean(1)
    for name,pred in models.items():
        cfg=lock["models"][name]
        stand,hybrid=cfg["standalone"],cfg["hybrid"]
        geom=geometric(pred["cov"])
        center=third_mix(ref,pred["mean"],geom,hybrid["share"],hybrid["mode"])
        band=redistribute(r0,geom["radial"],hybrid["power"])
        np.testing.assert_allclose(band.mean(1),r0.mean(1),rtol=0,atol=1.e-10)
        hit=compare(name,"matched",center,band)
        compare(name,"center_only",center,r0)
        compare(name,"covariance_only",ref,band)
        compare(name,"fixed_quarter_center",third_mix(ref,pred["mean"],geom,.25,"scalar"),r0)
        compare(name,"fixed_half_covariance_profile",ref,redistribute(r0,geom["radial"],.50))
        bayes_power=lock["models"]["bayesian"]["hybrid"]["power"]
        compare(name,"same_profile_power_cov_only",ref,redistribute(r0,geom["radial"],bayes_power))
        compare(name,"hybrid_pooled",center,band*hybrid["q_pooled"])
        compare(name,"hybrid_guarded",center,band*hybrid["q_guarded"])
        compare(name,"standalone_sphere",pred["mean"],geom["radial"]*stand["sphere_q"])
        flat=np.broadcast_to(np.array(stand["flat_shape"])*stand["flat_q"],r0.shape)
        compare(name,"standalone_horizon_only",pred["mean"],flat)
        for who,mu,q in (("standalone",pred["mean"],stand["ellipsoid_q"]),("hybrid",center,hybrid["ellipsoid_q"])):
            complete=(distance_axes(truth-mu,geom)<=q).all(1)
            ellipsoids.append(dict(model=name,center=who,q=q,n=len(turn),covered=int(complete.sum()),coverage=complete.mean(),
                    turn_covered=int(complete[turn].sum()),turns=int(turn.sum()),turn_coverage=complete[turn].mean(),
                    volume_equivalent_cm=float(100*q*geom["volume"].mean()),
                    largest_semiaxis_cm=float(100*q*geom["max_axis"].mean())))
        e=truth-pred["mean"]
        full,iso=nll_values(e,geom)
        trace=lambda v: np.trace(v,axis1=-2,axis2=-1)
        total=trace(pred["cov"]).clip(1.e-10)
        p30=spearmanr(geom["radial"][:,-1],np.linalg.norm(e[:,-1],axis=1)).statistic
        diagnostics.append(dict(model=name,gaussian_nll_3d_per_step=full.mean(),same_trace_isotropic_nll=iso.mean(),
              step30_radial_error_spearman=float(p30),learned_weight_trace_share=float((trace(pred["weight_cov"])/total).mean()),
              between_seed_trace_share=float((trace(pred["seed_cov"])/total).mean()),
              min_forecast_eigenvalue=float(geom["values"].min()),min_state_eigenvalue=float(np.linalg.eigvalsh(pred["state_cov"].astype(float)).min()),
              mean_position_gain=float(pred["gain"][:,0].mean()),mean_position_gain_posterior_sd=float(pred["gain_weight_sd"][:,0].mean())))
        if name=="bayesian":
            # Mechanism checks at the identical frozen Bayesian exponent and third-row centre.
            no_spread=geometric(pred["cov"]-pred["weight_cov"])
            compare(name,"without_gain_sampling_variance",ref,redistribute(r0,no_spread["radial"],bayes_power))
            compare(name,"gate_time_profile_cov_only",ref,redistribute(r0,np.array(stand["flat_shape"])[None],bayes_power))
            r4c=redistribute(r4,geom["radial"],hybrid["power"])
            np.testing.assert_allclose(r4c.mean(1),r4.mean(1),rtol=0,atol=1.e-10)
            paired["fourth_hit"]=published.mixed_center_compact_hit
            paired["fourth_covariance_hit"]=compare(name,"transfer_fourth_width",center,r4c)
            paired["bayes_only_ade_cm"]=100*np.linalg.norm(truth-pred["mean"],axis=2).mean(1)
            paired["hybrid_ade_cm"]=100*np.linalg.norm(truth-center,axis=2).mean(1)
            paired["hybrid_hit"]=hit
            paired["hybrid_radius_cm"]=100*band.mean(1)
            records=dict(row=w.meta.row.to_numpy(),third_center=ref,third_radius=r0,mean=pred["mean"],cov=pred["cov"],
                 conditional=pred["conditional"],weight_cov=pred["weight_cov"],seed_cov=pred["seed_cov"],
                 state=pred["state"],state_cov=pred["state_cov"],gain=pred["gain"],gain_weight_sd=pred["gain_weight_sd"],
                 hybrid_center=center,hybrid_radius=band,fourth_radius=r4,fourth_cov_radius=r4c)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"sphere_comparison.csv",index=False)
    pd.DataFrame(env_rows).to_csv(OUT/"environment_spheres.csv",index=False)
    pd.DataFrame(ellipsoids).to_csv(OUT/"ellipsoid_coverage.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"covariance_diagnostics.csv",index=False)
    paired.to_csv(OUT/"760个起点配对.csv",index=False,encoding="utf-8-sig")
    paired[turn].to_csv(OUT/"99个转向配对.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(OUT/"bayesian_issued.npz",**records)
    assert digest(OUT/"frozen_before_test.json")==stamp
    save(OUT/"test_audit.json",dict(frozen=stamp,windows=len(w.meta),test_environments=sorted(np.unique(env)),
         reference_row_verified_by_key=True,previous_checkpoint_reads=0,
         fixed_width_max_difference_m=float(np.max(np.abs(records["hybrid_radius"].mean(1)-r0.mean(1))))))
    keep=result.rule.isin(["matched","standalone_sphere","hybrid_guarded"])
    print(result.loc[keep,["model","rule","ade_cm","turn_ade_cm","covered","turn_covered","radius_cm"]].round(3).to_string(index=False),flush=True)


if __name__=="__main__":
    isolate()
    score()
