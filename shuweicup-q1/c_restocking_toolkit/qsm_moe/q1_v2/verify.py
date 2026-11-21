"""Verify prefix-only deployment against the independent batch run and fingerprint the frozen artifacts."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from settings import ROOT,OUT,WORK,PREVIOUS,KIT,KEY,MODES,write_json
from predict import WristForecaster


def sha(file):
    h=hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<19),b""):
            h.update(block)
    return h.hexdigest()


def audit():
    old=json.loads((PREVIOUS/"verification.json").read_text(encoding="utf8"))
    archive={}
    for file,expected in old["source_and_parameter_sha256"].items():
        path=ROOT/file
        if path.suffix in (".pt",".joblib") or (path.suffix==".py" and path.parent.name=="qsm_moe"):
            digest=sha(path)
            if digest!=expected:
                raise ValueError(f"Archived source or parameter hash differs for {path.name}")
            archive[path.name]=digest
    logs=pd.read_csv(OUT/"paired_windows.csv.gz")
    capture=np.load(OUT/"issued_grid.npz")
    batchmeta=pd.read_parquet(WORK/"cache"/"meta.parquet").iloc[capture["ids"]][KEY].reset_index(drop=True)
    pd.testing.assert_frame_equal(logs[KEY],batchmeta,check_dtype=False)
    choices=[]
    for split in ("validation","test"):
        for true,call in ((True,True),(True,False),(False,True),(False,False)):
            g=logs.index[logs.split.eq(split)&logs.turn.eq(true)&logs.v2_turn.eq(call)]
            if len(g):
                choices.append(int(g[len(g)//2]))
    model=WristForecaster()
    results=[]
    for i in choices:
        v=logs.iloc[i]
        sample,hand,o=v["sample"],v.hand,int(v.origin)
        result=model.issue(sample,hand,o)
        cols=[f"{hand}_j00_{a}" for a in "xyz"]+["head_qx","head_qy","head_qz","head_qw"]
        row=pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=cols,
                            filters=[("sample","==",sample),("frame_index","==",o)]).iloc[0].to_numpy()
        rot=Rotation.from_quat(row[3:]).as_matrix()
        center=(result["center_world"]-row[:3])@rot
        probs=np.array([result["probabilities"][k] for k in MODES])
        delta=float(np.abs(center-capture["center"][i]).max())
        radius=float(np.abs(result["radius_m"]-capture["radius"][i]).max())
        dp=float(np.abs(probs-capture["p"][i]).max())
        assert result["cutoff_frame"]==o and MODES.index(result["phase_decision"])==capture["called"][i]
        np.testing.assert_allclose(center,capture["center"][i],atol=2e-4,rtol=2e-5)
        np.testing.assert_allclose(result["radius_m"],capture["radius"][i],atol=2e-4)
        np.testing.assert_allclose(probs,capture["p"][i],atol=3e-6)
        np.testing.assert_allclose(sum(result["support_weights"]),1.,atol=1e-6)
        results.append(dict(sample=sample,hand=hand,origin=o,split=v.split,actual_turn=bool(v.turn),
                    alert=bool(v.v2_turn),max_xyz_difference_m=delta,max_radius_difference_m=radius,max_phase_difference=dp))
    np.testing.assert_array_equal(capture["called"],pd.Series(logs.v2_phase).to_numpy())
    lock=OUT/"frozen_before_test.json"
    audit0=json.loads((OUT/"evaluation_audit.json").read_text(encoding="utf8"))
    assert audit0["frozen_sha256"]==sha(lock)
    learned=[WORK/"measurement_quality.joblib", WORK/"motion_ridges.npz", WORK/"visual_adapter.npz",
            *sorted(WORK.glob("velocity_*.pt")), WORK/"residual_diffusion.pt",WORK/"width_parameters.npy",
            WORK/"feedback_reference.npy",WORK/(model.lock["phase"]["model"]+"_phase.joblib")]
    write_json(OUT/"verification.json",dict(**audit0,prefixes=results,archive_files_verified=archive,
          own_parameter_sha256={p.name:sha(p) for p in learned},frozen_decisions=model.lock,
          audit_note="Issue prefixes read timestamps and coordinates through the named origin. No test record enters parameter fitting or selection."))
    print("Prefix forecasts verified",len(results),"maximum coordinate difference (m)",max(v["max_xyz_difference_m"] for v in results),flush=True)


if __name__=="__main__":
    audit()
