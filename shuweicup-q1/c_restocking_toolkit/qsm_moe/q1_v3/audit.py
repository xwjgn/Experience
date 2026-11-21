"""Independent prefix replay, feedback cutoff checks and comparison-cohort verification."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from config import OUT,V2,V1,ROOT,KIT,KEY,WORK,read_json,digest,save
from forecast import Forecaster
from intervals import sequences


def audit_feedback():
    # A completed path is available only at origin+30. Change all future evidence after the cutoff.
    n=151
    meta=pd.DataFrame(dict(sample="audit_record",hand="left",origin=np.arange(n),index=np.arange(n)))
    rng=np.random.default_rng(579)
    ref=np.sort(np.exp(rng.normal(size=(130,30))),axis=0)
    phase=rng.integers(0,3,n)
    scale=np.exp(rng.normal(size=(n,30))*.2)
    err=scale*np.exp(rng.normal(size=(n,30)))
    ok=np.ones(n,bool); ok[55]=False
    cutoff=83
    before=sequences(meta,phase,err,scale,ok,ref,4.)
    altered=err.copy()
    altered[np.arange(n)[:,None]+np.arange(1,31)>cutoff]=9000.
    flags=ok.copy();flags[cutoff+1:]=False
    after=sequences(meta,phase,altered,scale,flags,ref,4.)
    for k in before:
        np.testing.assert_array_equal(before[k][:cutoff+1],after[k][:cutoff+1])
    # A corrupt, visibly incomplete prior window cannot modify the completed-path update.
    changed=err.copy(); changed[40]=1000.
    partial=sequences(meta,phase,changed,scale,ok,ref,4.)
    np.testing.assert_array_equal(partial["completed"][70:100],before["completed"][70:100])


def audit():
    audit_feedback()
    logs=pd.read_csv(OUT/"paired_windows.csv.gz")
    raw=np.load(OUT/"issued_grid.npz")
    score=read_json(OUT/"score_audit.json")
    assert digest(OUT/"frozen_before_test.json")==score["frozen_sha256"]
    meta=pd.read_parquet(WORK/"cache"/"meta.parquet").iloc[raw["row"]][KEY].reset_index(drop=True)
    pd.testing.assert_frame_equal(meta,logs[KEY],check_dtype=False)
    comparator=pd.read_csv(V2/"99个转向窗口.csv",encoding="utf-8-sig")
    prev=comparator[["划分","环境","记录","手","起点帧_从0计数"]].copy()
    prev.columns=KEY
    prev.hand=prev.hand.map({"左手":"left","右手":"right"})
    turns=logs[logs.split.eq("test")&logs.turn][KEY].reset_index(drop=True)
    pd.testing.assert_frame_equal(turns,prev,check_dtype=False)
    solver=Forecaster()
    examples=[]
    for split in ("validation","test"):
        for a,c in ((True,True),(True,False),(False,True),(False,False)):
            idx=logs.index[logs.split.eq(split)&logs.turn.eq(a)&logs.issue_alert.eq(c)]
            if not len(idx):
                continue
            i=int(idx[len(idx)//2])
            v=logs.iloc[i];sample=v["sample"];hand=v.hand;o=int(v.origin)
            live=solver.issue(sample,hand,o)
            cols=[f"{hand}_j00_{axis}" for axis in "xyz"]+["head_qx","head_qy","head_qz","head_qw"]
            row=pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=cols,
                filters=[("sample","==",sample),("frame_index","==",o)]).iloc[0].to_numpy()
            local=(live["center_world"]-row[:3])@Rotation.from_quat(row[3:]).as_matrix()
            dx=np.abs(local-raw["center"][i]).max(); dr=np.abs(live["radius_m"]-raw["radius"][i]).max()
            np.testing.assert_allclose(local,raw["center"][i],atol=2e-4)
            np.testing.assert_allclose(live["radius_m"],raw["radius"][i],atol=2e-4)
            np.testing.assert_allclose(sum(live["support_weights"]),1.,atol=1e-6)
            assert live["last_read_frame"]==o and (live["phase_decision"]=="turn")==bool(v.issue_alert)
            examples.append(dict(sample=sample,hand=hand,origin=o,split=split,max_xyz_delta_m=float(dx),max_radius_delta_m=float(dr)))
    # Verify old parameter bytes against manifests without deserializing them.
    legacy={}
    for tag,directory in (("v1",V1),("v2",V2)):
        doc=read_json(directory/"verification.json")
        if tag=="v1":
            files={ROOT/f:h for f,h in doc["source_and_parameter_sha256"].items() if f.endswith((".pt",".joblib"))}
        else:
            files={directory/"fresh"/f:h for f,h in doc["own_parameter_sha256"].items()}
        for path,expected in files.items():
            if digest(path)!=expected:
                raise ValueError("Historical model file changed: "+path.name)
        legacy[tag]=len(files)
    save("verification.json",dict(**score,prefix_checks=examples,identical_test_turn_keys=len(prev),
          old_parameter_hashes_verified=legacy,future_perturbation_passed=True,
          partial_path_feedback_skipped=True,positions_in_metres=True,
          case_sources="official training environments fit, separate validation selects both rules; cached test labels first scored after freeze"))
    print("Verified",len(examples),"independent prefixes and",len(prev),"identical turning-window keys",flush=True)


if __name__=="__main__":
    audit()
