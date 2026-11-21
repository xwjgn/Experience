"""Run only the independent training part of the study, using official training environments."""
import joblib
import numpy as np

from wrist_history import M4_DIR,OUT,load_split,protect_weights,read,save
from event_models import Scorer,train_models


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    roles = read(M4_DIR/"release_protocol.json")["roles"]
    data = load_split("train")
    pack,cal = train_models(data,roles)
    joblib.dump(pack,OUT/"fresh_tree_and_scales.joblib",compress=3)
    score = Scorer(pack)
    outputs,_ = score.outputs(cal.seq,cal.features,cal.meta.stage.to_numpy())
    np.savez_compressed(OUT/"training_env_scores.npz",**outputs,turn=cal.meta.turn.to_numpy(),
                        stage=cal.meta.stage.to_numpy(),env=cal.meta.env.to_numpy(str))
    save(OUT/"fit_protocol.json",dict(roles=roles,fit_windows=int(data.meta.env.isin(roles["fit"]).sum()),
             gate_windows=int((data.meta.env.isin(roles["gate"]) & ((data.meta.origin-10)%30==0)).sum()),
             calibration_windows=len(cal.meta),runs=pack["fits"],input_features=pack["features"],prior_weights_loaded=0))
    print("Fresh detectors and physical uncertainty estimates saved",flush=True)


if __name__=="__main__":
    protect_weights()
    fit()
