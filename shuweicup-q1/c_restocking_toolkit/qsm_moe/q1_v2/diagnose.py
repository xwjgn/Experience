"""Describe the frozen v1 predictions, including the ninety-nine turns; no saved model is loaded."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from settings import PREVIOUS, OUT, KIT, KEY, MODES, equal_env, make_dirs


def diagnose():
    make_dirs()
    cache = PREVIOUS / "cache"
    meta = pd.read_parquet(cache / "meta.parquet")
    spec = json.loads((cache / "ready.json").read_text(encoding="utf8"))
    roles = meta.split.to_numpy().astype(object)
    for name, envs in spec["environments"].items():
        roles[meta.env.isin(envs)] = name
    masks = np.load(cache / "mask.npy", mmap_mode="r")
    next_phase = np.load(cache / "next_mode.npy")
    current = np.load(cache / "mode.npy")
    grid = np.load(cache / "grid.npy") & masks.all(1)
    train = (roles == "fit") & masks[:, 9] & masks[:, -1]
    stages, confusion, visual, samples, timing = [], [], [], [], []
    predictions = {}
    ids = None
    for model in ("B2", "B3", "B4"):
        with np.load(cache / (model + "_issues.npz")) as f:
            if ids is None:
                ids = f["ids"]
            np.testing.assert_array_equal(ids, f["ids"])
            predictions[model] = (f["stage"], f["path"])
    truth = np.load(cache / "y.npy", mmap_mode="r")[ids]
    flags = ["fit", "gate", "calibration", "validation", "test"]
    for split in flags:
        take = (roles[ids] == split) & grid[ids]
        y = next_phase[ids[take]]
        env = meta.env.to_numpy()[ids[take]]
        for model, (probs, _) in predictions.items():
            p = probs[take]
            label = p.argmax(1)
            is_turn = y == 1
            turn = p[:, 1]
            positives = turn[is_turn]
            row = dict(role=split, model=model, n=len(y), turns=int(is_turn.sum()), called=int((label==1).sum()),
                       tp=int(((label==1)&is_turn).sum()), prevalence=is_turn.mean(),
                       mean_true=positives.mean(), mean_other=turn[~is_turn].mean(),
                       p10_true=np.quantile(positives,.1), p50_true=np.median(positives), p90_true=np.quantile(positives,.9),
                       max_true=positives.max(), rank_second_true=int(((p.argsort(1)[:,-2]==1)&is_turn).sum()),
                       ap=average_precision_score(is_turn,turn), auroc=roc_auc_score(is_turn,turn),
                       true_was_turn=int(((current[ids[take]]==1)&is_turn).sum()))
            stages.append(row)
            matrix = confusion_matrix(y,label,labels=[0,1,2])
            for i,name in enumerate(MODES):
                for j,pred in enumerate(MODES):
                    confusion.append(dict(role=split,model=model,truth=name,predicted=pred,n=matrix[i,j]))
        err = {key: np.linalg.norm(path[take]-truth[take],axis=2)*100 for key,(_,path) in predictions.items()}
        for e in ["ALL", *sorted(set(env))]:
            select = np.ones(len(env),bool) if e=="ALL" else env==e
            for name in ("B3","B4"):
                ea=err[name][select]
                visual.append(dict(role=split,env=e,model=name,n=len(ea),path_cm=equal_env(ea.mean(1),env[select]),
                                   endpoint_cm=equal_env(ea[:,-1],env[select])))
        g=meta.iloc[ids[take]][KEY].reset_index(drop=True)
        g["truth"]=y
        g["origin_phase"]=current[ids[take]]
        for k, (phase, _) in predictions.items():
            g[k+"_turn_prob"]=phase[take,1]
            g[k+"_phase"]=phase[take].argmax(1)
        g["B3_path_cm"]=err["B3"].mean(1)
        g["B4_path_cm"]=err["B4"].mean(1)
        g["role"]=split
        samples.append(g)
    for sample,g in meta[meta.hand.eq("left")].groupby("sample",sort=False):
        with np.load(KIT/"cache/literature_features"/(sample+".npz")) as f:
            stamps=f["t"]; shape=f["rgb"].shape
        table=pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=["t_s","frame_index"],
                              filters=[("sample","==",sample)]).sort_values("frame_index")
        t=table.t_s.to_numpy()
        vi=np.searchsorted(stamps,t,side="right")-1
        valid=vi>=0
        delay=t[valid]-stamps[vi[valid]]
        assert (delay>=0).all()
        timing.append(dict(sample=sample,role=roles[g.index[0]],frames=len(t),image_frames=len(stamps),
            height=shape[1],width=shape[2],fps=1/np.median(np.diff(stamps)),
            unavailable=int((~valid).sum()),age_median_s=np.median(delay),age_p95_s=np.quantile(delay,.95)))
    # For this diagnosis only, measure record/environment structure of the archived visual embeddings.
    embedding=np.load(cache/"vision.npy",mmap_mode="r")
    record_rows=[]
    for sample,g in meta[meta.hand.eq("left")].groupby("sample",sort=False):
        ix=g.index.to_numpy()[::6]
        record_rows.append((sample,g.env.iat[0],roles[ix[0]],embedding[ix].mean(0)))
    prototypes=np.stack([row[3] for row in record_rows])
    e=np.array([row[1] for row in record_rows])
    distances=np.linalg.norm(prototypes[:,None]-prototypes[None,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    use=np.array([(e==v).sum()>1 for v in e])
    closest=distances.argmin(1)
    identity=pd.DataFrame(dict(sample=[r[0] for r in record_rows],env=e,role=[r[2] for r in record_rows],
               other_record_in_env=use,nearest_record_env=e[closest],hit=e[closest]==e,
               random_baseline=[((e==v).sum()-1)/(len(e)-1) for v in e]))
    for frame,filename in ((pd.DataFrame(stages),"v1_stage_diagnosis.csv"),(pd.DataFrame(confusion),"v1_confusion.csv"),
          (pd.DataFrame(visual),"v1_vision_by_environment.csv"),(pd.concat(samples),"v1_window_diagnosis.csv.gz"),
          (pd.DataFrame(timing),"camera_alignment.csv"),(identity,"v1_visual_environment_probe.csv")):
        frame.to_csv(OUT/filename,index=False)
    count=np.bincount(next_phase[train],minlength=3)
    print("Actual fit loss labels dwell/turn/translate:", count.tolist(),flush=True)
    (OUT/"v1_training_counts.json").write_text(json.dumps(dict(zip(MODES,count.tolist()))),encoding="utf8")
    print(pd.DataFrame(stages).query("model=='B4'")[["role","n","turns","called","tp","mean_true","mean_other","ap","auroc"]].to_string(index=False))
    print(pd.DataFrame(visual).query("env=='ALL'").pivot(index="role",columns="model",values="path_cm").to_string())
    print("Nearest other recorded video is in same environment:",identity[use].hit.mean(),"random:",identity[use].random_baseline.mean())


if __name__=="__main__":
    diagnose()
