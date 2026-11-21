"""Additional descriptive controls. No scaling in this file is used by the deployed forecaster."""
import numpy as np
import pandas as pd

from paths_q1 import DATA,KEY,OUT,PREVIOUS,read_json
from evaluate_lsttn import merge_exact,summarize
from train_lsttn import point_stats


def diagnose():
    fixed = pd.read_csv(OUT/"paired_windows.csv.gz")
    fixed = fixed[fixed.calibration=="cafht_cptc"]
    keys = pd.read_parquet(DATA/"keys.parquet")
    y = np.load(DATA/"y.npy",mmap_mode="r")
    v = fixed[fixed.split=="validation"]
    validation_factor = v.qsm_radius_cm.mean()/v.radius_cm.mean()
    # Read the archived state estimates for the *comparator* only, after the neural model is frozen.
    base = PREVIOUS/"qsm_retrained"/"backbone"/"fresh"/"cache"
    previous_keys = pd.read_parquet(base/"meta.parquet")[KEY].copy()
    previous_keys["archived_row"] = np.arange(len(previous_keys))
    posterior = np.load(base/"physics.npy",mmap_mode="r")
    cadence = np.load(base/"cadence.npy",mmap_mode="r")
    tau = read_json(base/"ready.json")["tau"]
    rows,kalman = [],[]
    with np.load(OUT/"issued_distribution.npz") as f:
        for split in ("validation","test"):
            select = keys.split.to_numpy()[f["row"]]==split
            ids = f["row"][select]
            sub = merge_exact(keys.iloc[ids][KEY],fixed[fixed.split==split])
            truth = y[ids]
            error = np.linalg.norm(f["center"][select]-truth,axis=-1)
            radius = f["radius"][select]
            equal = float(sub.qsm_radius_cm.mean()/(100*radius.mean()))
            for label,k in (("frozen_validation_mean",validation_factor),("equal_split_mean_diagnostic",equal)):
                new = sub.copy()
                adjusted = radius*k
                new["hit"] = (error<=adjusted).all(1)
                new["radius_cm"] = 100*adjusted.mean(1)
                rows.append(dict(split=split,diagnostic=label,factor=k,**summarize(new)))
            prev = previous_keys[previous_keys.split==split].merge(sub[KEY],on=KEY,validate="one_to_one")
            ordered = merge_exact(sub[KEY],prev)["archived_row"].to_numpy()
            state = posterior[ordered,1]
            age = cadence[ordered,None]*np.arange(1,31,dtype=np.float32)
            physical = state[:,None,:3]+state[:,None,3:6]*(-tau*np.expm1(-age/tau))[:,:,None]
            sign = np.where(sub.hand.to_numpy()=="right",-1.,1.)
            physical[:,:,1] *= sign[:,None]
            stats = point_stats(truth,physical,sub.turn.to_numpy().astype(int),sub.env.to_numpy())
            kalman.append(dict(split=split,tau_s=tau,windows=len(ids),turns=int(sub.turn.sum()),**stats))
    pd.DataFrame(rows).to_csv(OUT/"matched_radius_diagnostics.csv",index=False)
    pd.DataFrame(kalman).to_csv(OUT/"current_kalman_anchor.csv",index=False)


if __name__=="__main__":
    diagnose()
