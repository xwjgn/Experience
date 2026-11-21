"""Choose centre readouts and radius rules without opening the test windows."""
import joblib
import numpy as np
import pandas as pd

from gain_records import OUT,SEEDS,dataset,digest,isolate,partition,read,save
from comparison_reference import table_reference
from neural_gain import KINDS
from train_gain import ModelBank,seed_file
from gain_calibration import hybrid_freeze,standalone_freeze


def variants(bank,win):
    preds={k:bank.issue(win.inputs,k) for k in KINDS}
    preds["posterior_mean_weights"]=bank.issue(win.inputs,"bayesian",mean_only=True)
    return preds


def freeze(training=None,reference_fit=None):
    if training is None:
        training=dataset("train")
    gate=partition(training,"gate")
    cal=partition(training,"cal")
    val=dataset("validation")
    if reference_fit is None:
        reference_fit=joblib.load(OUT/"cold_comparison_only.joblib")
    base,r0=table_reference(val,reference_fit)
    bank=ModelBank()
    gpred,cpred,vpred=(variants(bank,win) for win in (gate,cal,val))
    turn,env=val.meta.turn.to_numpy(),val.meta.env.to_numpy()
    models={}
    pts,bands=[],[]
    for name,pred in vpred.items():
        standalone=standalone_freeze(gpred[name],cpred[name],gate.truth,cal.truth)
        mix,points,intervals=hybrid_freeze(base,r0,pred,val.truth,turn,env)
        models[name]=dict(standalone=standalone,hybrid=mix)
        pts.extend(dict(model=name,**r) for r in points)
        bands.extend(dict(model=name,**r) for r in intervals)
        print(name,"frozen share, covariance profile",mix["mode"],mix["share"],mix["power"],
              "held-environment validation",mix["oof"],flush=True)
    pd.DataFrame(pts).to_csv(OUT/"validation_center_choices.csv",index=False)
    pd.DataFrame(bands).to_csv(OUT/"validation_radius_choices.csv",index=False)
    hashes={p.name:digest(p) for p in [OUT/"cold_comparison_only.joblib",OUT/"training_normalization.npz",
               *[seed_file(k,s) for k in KINDS for s in SEEDS]]}
    save(OUT/"frozen_before_test.json",dict(models=models,primary="bayesian",hashes=hashes,
        fit_windows=read(OUT/"fresh_fit.json")["fit"],gate_windows=len(gate.meta),calibration_windows=len(cal.meta),
        validation_windows=len(val.meta),validation_turns=int(turn.sum()),validation_environments=sorted(val.meta.env.unique()),
        comparison="Primary third-row comparison holds each issued window's mean radius fixed. Extra spherical and ellipsoidal calibration uses held-environment validation residuals for the hybrid, independent training calibration environments for standalones.",
        trial="Both centre-only and covariance-only controls and a fixed 25% new-forecast control are scored after freezing.",
        preexisting_checkpoint_reads=0))
    np.savez_compressed(OUT/"validation_issue_reference.npz",row=val.meta.row.to_numpy(),reference_center=base,reference_radius=r0)


if __name__=="__main__":
    isolate()
    freeze()
