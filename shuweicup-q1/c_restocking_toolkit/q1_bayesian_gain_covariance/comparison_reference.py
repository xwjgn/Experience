"""Recover the third-row forecast from archived M4 issues and cold control fits. No old checkpoints."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from gain_records import KEY,M4_DIR,OLD,OUT,PREVIOUS,dataset,expert_points,fit_expert,isolate,partition,read,save


def fit_reference(training=None):
    OUT.mkdir(parents=True,exist_ok=True)
    if training is None:
        training=dataset("train")
    fit,gate=partition(training,"fit",False),partition(training,"gate")
    expert=fit_expert(fit,gate)
    cols=[c for c in fit.features if c.startswith(("w_","f_","h_"))]
    count=fit.meta.groupby("env").turn.transform("size").to_numpy(float)
    weights=count**-.5
    tree=HistGradientBoostingClassifier(learning_rate=.055,max_iter=100,max_leaf_nodes=12,min_samples_leaf=80,
                l2_regularization=4.,early_stopping=False,random_state=36511)
    tree.fit(fit.features[cols],fit.meta.turn,sample_weight=weights/weights.mean())
    old_policy=read(OLD/"frozen_before_replay.json")
    selection=read(PREVIOUS/"frozen_before_test.json")["radii"]["shape"]
    if selection!=["short_guard",1.]:
        raise ValueError("The reference row no longer has the short-horizon profile")
    pack=dict(expert=expert,budget_tree=tree,cols=cols,share=read(PREVIOUS/"point_locked_before_test.json")["share"],
              cut=old_policy["routing"]["low_turn"]["cut"],shrink=old_policy["shrink_factor"])
    joblib.dump(pack,OUT/"cold_comparison_only.joblib",compress=3)
    save(OUT/"reference_provenance.json",dict(fit_windows=len(fit.meta),gate_windows=len(gate.meta),
         shared=["released measurements","scoring grid","archived M4 forecast outputs","published reference rules"],
         comparison_models_refit_from_zero=["previous wrist displacement expert","previous low-turn radius gate"],
         old_checkpoints_opened=0))
    return pack


def table_reference(windows,pack):
    with np.load(M4_DIR/"M3_issued.npz") as f:
        ix=pd.Index(f["row"]).get_indexer(windows.meta.row)
        if (ix<0).any() or windows.meta.duplicated(KEY).any():
            raise ValueError("Archived M4 issue not available for every common origin")
        m4,r0=f["center"][ix],f["radius"][ix]
    new=expert_points(pack["expert"],windows.seq,windows.features)
    center=m4+pack["share"]*(new-m4)
    risk=pack["budget_tree"].predict_proba(windows.features[pack["cols"]])[:,1]
    reduce=(risk<=pack["cut"])&(windows.features.w_recent_flags.to_numpy()==0)
    a=np.exp(.72*np.exp(-np.arange(30)/5))
    z=a[None]/(np.sum(a*r0,axis=1)/r0.sum(axis=1))[:,None]-1
    damp=np.minimum(1.,np.minimum(.22/np.maximum(-z.min(1),1.e-8),1.70/np.maximum(z.max(1),1.e-8)))
    radius=r0*(1+z*damp[:,None])*np.where(reduce,pack["shrink"],1.)[:,None]
    return center,radius


def verify_test_reference(windows,center,radius):
    archive=pd.read_csv(PREVIOUS/"760个起点逐窗回放.csv")
    published=windows.meta[KEY].merge(archive,on=KEY,validate="one_to_one")
    if len(published)!=len(archive) or len(published)!=len(windows.meta):
        raise ValueError("Reference key mismatch")
    errors=np.linalg.norm(center-windows.truth,axis=2)
    np.testing.assert_allclose(100*errors.mean(1),published.new_ade_cm,rtol=0,atol=3.e-5)
    np.testing.assert_allclose(100*radius.mean(1),published.mixed_center_same_window_budget_radius_cm,rtol=0,atol=1.e-7)
    np.testing.assert_array_equal((errors<=radius).all(1),published.mixed_center_same_window_budget_hit)
    np.testing.assert_array_equal(windows.meta.turn,published.turn)
    return published


if __name__=="__main__":
    isolate()
    fit_reference()
