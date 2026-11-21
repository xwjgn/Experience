"""Matched architecture, batches and seeds. Change only the visual inputs, fitting each control from scratch."""
import numpy as np
import pandas as pd
import torch

from settings import WORK, OUT, PREVIOUS, equal_env
import qsm_train
from qsm_data import reproducible_torch


@torch.inference_mode()
def issue(net,data,ids):
    coordinates,phase=[],[]
    for block in np.array_split(ids,max(1,int(np.ceil(len(ids)/384)))):
        ix=torch.as_tensor(block,device=data.dev)
        p,_,s,_=net(*data.args(ix,"B4"))
        coordinates.append(p.cpu().numpy())
        phase.append(s.softmax(1).cpu().numpy())
    return np.concatenate(coordinates),np.concatenate(phase)


def controls():
    reproducible_torch()
    qsm_train.CACHE=WORK/"cache"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data=qsm_train.Data(device)
    meta=data.meta
    roles=data.spec["environments"]
    train=np.flatnonzero(meta.env.isin(roles["fit"]) & meta.score)
    full=np.load(WORK/"cache"/"mask.npy",mmap_mode="r").all(1)
    ids=np.flatnonzero(full & np.load(WORK/"cache"/"grid.npy"))
    env=meta.env.to_numpy()[ids]
    role=meta.split.to_numpy()[ids].astype(object)
    for key,group in roles.items():
        role[np.isin(env,group)]=key
    aligned=data.vision
    allrows=torch.arange(len(meta),device=device)
    older=aligned[torch.maximum(data.start,allrows-30)]
    # Freeze the initial one-second visual history then retain it, using an as-of value before that time.
    static=aligned[torch.minimum(allrows,data.start+30)]
    zero=torch.zeros_like(aligned)
    truth=np.load(WORK/"cache"/"y.npy",mmap_mode="r")[ids]
    golden=np.load(PREVIOUS/"cache"/"B4_issues.npz")
    reference=golden["path"][np.searchsorted(golden["ids"],ids)]
    tolerance=None
    rows=[]
    for seed in (9317,9431):
        qsm_train.SEED=seed
        for target,features in (("aligned",aligned),("zero",zero)):
            qsm_train.OUT=WORK/f"control_{seed}_{target}"
            qsm_train.OUT.mkdir(exist_ok=True)
            data.vision=features
            model=qsm_train.fit(data,"B4",train)
            inputs={"aligned":aligned,"one_second_lag":older,"first_second":static,"zero":zero} if target=="aligned" else {"zero":zero}
            for intervention,value in inputs.items():
                data.vision=value
                path,phase=issue(model,data,ids)
                if target=="aligned" and intervention=="aligned" and seed==9317:
                    tolerance=float(np.abs(reference-path).max())
                    np.testing.assert_allclose(path,reference,atol=6e-5,rtol=2e-5)
                err=100*np.linalg.norm(path-truth,axis=2)
                for group in ("fit","gate","validation","test"):
                    use=role==group
                    rows.append(dict(seed=seed,train_input=target,issue_input=intervention,role=group,n=int(use.sum()),
                        path_cm=equal_env(err[use].mean(1),env[use]),endpoint_cm=equal_env(err[use,-1],env[use]),
                        predicted_turns=int((phase[use].argmax(1)==1).sum())))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/"visual_interventions.csv",index=False)
    print("Maximum freshly replicated vs archived B4 path difference (m)",tolerance)
    print(out.query("role=='validation' or role=='test'").groupby(["role","train_input","issue_input"]).path_cm.mean().to_string())


if __name__=="__main__":
    controls()
