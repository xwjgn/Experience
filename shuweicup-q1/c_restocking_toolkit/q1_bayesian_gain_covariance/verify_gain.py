"""Replay individual raw prefixes, verify covariance geometry, and test missing wrist measurements."""
from dataclasses import replace

import numpy as np
import pandas as pd

from gain_records import OUT,SIDES,features_for,isolate,load_record,observe,save,sequence
from forecast_bayesian import BayesianWrist


def verify():
    table=pd.read_csv(OUT/"760个起点配对.csv")
    changed=(table.third_hit!=table.hybrid_hit)|(table.fourth_hit!=table.fourth_covariance_hit)
    selected=pd.concat((table[changed],table.groupby(["env","hand"],sort=False).head(1),
                        table.groupby(["dataset","hand"],sort=False).head(1),
                        table[table.turn].groupby("env",sort=False).head(1))).drop_duplicates("row")
    sys=BayesianWrist()
    max_pos=max_cov=max_radius=0.
    with np.load(OUT/"bayesian_issued.npz") as f:
        all_cov=f["cov"]
        np.testing.assert_allclose(all_cov,f["conditional"]+f["weight_cov"]+f["seed_cov"],rtol=2.e-5,atol=1.e-7)
        np.testing.assert_allclose(all_cov,all_cov.swapaxes(-2,-1),rtol=0,atol=1.e-7)
        eig=float(np.linalg.eigvalsh(all_cov.astype(float)).min())
        state_eig=float(np.linalg.eigvalsh(f["state_cov"].astype(float)).min())
        assert eig>0 and state_eig>0 and np.mean(f["gain_weight_sd"])>0
        take=pd.Index(f["row"]).get_indexer(selected.row)
        stored={k:f[k][take] for k in ("third_center","third_radius","mean","cov","hybrid_center","hybrid_radius","fourth_radius","fourth_cov_radius")}
    for j,r in enumerate(selected.itertuples()):
        n=r.origin+1
        prefix=load_record(r.dataset,r.sample,n)
        full=load_record(r.dataset,r.sample,n+30)
        motion=observe(prefix,r.hand,sys.speed_cut)
        rot=motion.rotation[-1].as_matrix()@np.diag(motion.mirror)
        def to_world(relative):
            return motion.wrist[-1]+relative@rot.T
        third=dict(hand=r.hand,origin=r.origin,reference_center_world=to_world(stored["third_center"][j]),
                   reference_radius_m=stored["third_radius"][j])
        ans=sys.issue(prefix,r.hand,third)
        assert ans["last_read_frame"]==r.origin
        mean=to_world(stored["mean"][j])
        cworld=rot@stored["cov"][j]@rot.T
        max_pos=max(max_pos,float(np.max(np.abs(mean-ans["center_world"]))),
                    float(np.max(np.abs(to_world(stored["hybrid_center"][j])-ans["hybrid_center_world"]))))
        max_cov=max(max_cov,float(np.max(np.abs(cworld-ans["covariance_m2"]))))
        max_radius=max(max_radius,float(np.max(np.abs(stored["hybrid_radius"][j]-ans["hybrid_radius_m"]))))
        true=full.joints[n:,SIDES.index(r.hand),0]
        error=np.linalg.norm(np.asarray(ans["hybrid_center_world"])-true,axis=1)
        assert bool((error<=ans["hybrid_radius_m"]).all()) == r.hybrid_hit
        fourth=dict(third,reference_radius_m=stored["fourth_radius"][j])
        got=sys.issue(prefix,r.hand,fourth,"fourth")
        np.testing.assert_allclose(got["hybrid_radius_m"],stored["fourth_cov_radius"][j],atol=3.e-6,rtol=0)
        assert bool((error<=got["hybrid_radius_m"]).all())==r.fourth_covariance_hit
        # None of the forward inputs can change when only later observations are changed.
        corrupted=full.joints.copy()
        corrupted[n:]+=5.
        perturbed=replace(full,joints=corrupted)
        extended=observe(perturbed,r.hand,sys.speed_cut)
        ids=np.array([r.origin])
        def at(raw,m):
            seq=sequence(m,ids)
            return features_for(raw,m,r.hand,ids,seq,m.tab.iloc[ids])
        a,b=at(prefix,motion),at(perturbed,extended)
        for key in vars(a):
            np.testing.assert_allclose(getattr(a,key),getattr(b,key),atol=1.e-7,rtol=0)
    assert max_pos<3.e-6 and max_radius<5.e-6 and max_cov<2.e-6
    sample=selected.iloc[0]
    raw=load_record(sample.dataset,sample["sample"],max(51,int(sample.origin)+1))
    points=raw.joints.copy()
    points[-16:,SIDES.index(sample.hand),0]=np.nan
    missing=sys.issue(replace(raw,joints=points),sample.hand)
    np.testing.assert_allclose(missing["last_effective_gain_world"],0,atol=0,rtol=0)
    assert np.isfinite(missing["center_world"]).all() and np.all(np.linalg.eigvalsh(missing["covariance_m2"])>0)
    save(OUT/"raw_prefix_checks.json",dict(prefixes=len(selected),changed_windows=int(changed.sum()),
         environments=selected.env.nunique(),datasets=selected.dataset.nunique(),hands=sorted(selected.hand.unique()),
         max_center_error_m=max_pos,max_covariance_difference_m2=max_cov,max_radius_difference_m=max_radius,
         forecast_min_eigenvalue=eig,state_min_eigenvalue=state_eig,skips_missing_measurements=True,
         future_perturbation_test=True,coverage_matches_online=True))
    print("Raw prefix and covariance checks passed for",len(selected),"origins and every changed outcome.",flush=True)


if __name__=="__main__":
    isolate()
    verify()
