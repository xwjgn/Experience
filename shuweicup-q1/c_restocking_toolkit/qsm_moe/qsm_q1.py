"""Issue a weighted distribution on entire 30-frame paths and score the independent QSM models."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from qsm_data import OUT,CACHE,write_json
from qsm_train import VARIANTS

KEY=["split","env","sample","hand","origin"]


def macro(values,env):
    return float(pd.Series(np.asarray(values)).groupby(np.asarray(env)).mean().mean())


def interpolate(knots):
    anchors=np.concatenate((np.zeros_like(knots[:,:,:1]),knots),axis=2)
    step=np.arange(1,31)
    left=(step-1)//10
    weight=((step-10*left)/10.).astype("float32")[None,None,:,None]
    return anchors[:,:,left]*(1-weight)+anchors[:,:,left+1]*weight


def geometry(a,b,truth):
    """Euclidean distance of flattened joint paths, normalized to metres per horizon."""
    n,k=b.shape[:2]
    ab=np.empty((n,k),np.float32); within=np.empty_like(ab); by=np.empty_like(ab)
    for start in range(0,n,384):
        sl=slice(start,min(n,start+384))
        def dist(x,y):
            return np.sqrt(np.square(x-y).sum(-1).mean(-1))
        ab[sl]=dist(b[sl],a[sl,None])
        by[sl]=dist(b[sl],truth[sl,None])
        for j in range(k):
            within[sl,j]=dist(b[sl],b[sl,j:j+1]).mean(1)
    ay=np.sqrt(np.square(a-truth).sum(2).mean(1))
    coeff=np.column_stack((ay,by.mean(1)-ay-ab.mean(1),ab.mean(1)-within.mean(1)/2))
    assert coeff[:,2].min()>-1e-5
    return ab,within,coeff


def fit_gate(x,coef,env):
    weights=1/pd.Series(env).map(pd.Series(env).value_counts()).to_numpy()
    weights/=weights.sum()
    mu=np.average(x,axis=0,weights=weights)
    sd=np.maximum(np.sqrt(np.average((x-mu)**2,axis=0,weights=weights)),.01)
    design=np.column_stack((np.ones(len(x)),np.clip((x-mu)/sd,-5,5)))
    l,c=100*coef[:,1],100*coef[:,2]
    constant=float(np.clip(-np.dot(weights,l)/max(2*np.dot(weights,c),1e-9),0,1))
    def objective(theta):
        p=expit(design@theta)
        penalty=.04*np.sum(theta[1:]**2)
        grad=design.T@(weights*(l+2*c*p)*p*(1-p))
        grad[1:]+=.08*theta[1:]
        return np.dot(weights,l*p+c*p**2)+penalty,grad
    fit=minimize(objective,np.zeros(design.shape[1]),jac=True,method="L-BFGS-B",
                 bounds=[(-9,9)]+[(-4,4)]*(design.shape[1]-1))
    if not fit.success:
        raise RuntimeError(fit.message)
    return dict(mu=mu.tolist(),sd=sd.tolist(),theta=fit.x.tolist(),constant=constant)


def gate_value(x,model):
    return expit(model["theta"][0]+np.clip((x-model["mu"])/model["sd"],-5,5)@model["theta"][1:])


def medoid(a,b,ab,within,g):
    cost=np.column_stack((g*ab.mean(1),(1-g[:,None])*ab+g[:,None]*within))
    i=cost.argmin(1)
    center=a.copy()
    ix=np.flatnonzero(i)
    center[ix]=b[ix,i[ix]-1]
    spread=(1-g[:,None])*np.square(a-center).sum(2)
    for k in range(b.shape[1]):
        spread+=g[:,None]/b.shape[1]*np.square(b[:,k]-center).sum(2)
    return center,i,np.sqrt(np.maximum(spread,0))


def rank95(x):
    k=int(np.ceil((len(x)+1)*.95))
    if k>len(x):
        raise ValueError("Insufficient calibration windows")
    return float(np.partition(x,k-1)[k-1])


def calfactors(scores,cls):
    base=rank95(scores)
    factors=[rank95(scores[cls==s]) if (cls==s).sum()>=25 else base for s in range(3)]
    return dict(global_factor=base,stage_factors=factors,n=len(scores),
                stage_n=[int(sum(cls==s)) for s in range(3)])


def feedback_scales(meta,phase,err,scale,ok,reference):
    """At a grid tick, only the endpoint issued 30 frames ago can update its phase's risk level."""
    origin=meta.origin.to_numpy()
    level=np.full(len(meta),.05)
    for ix in meta.groupby(["sample","hand"],sort=False).indices.values():
        slots=ix[(origin[ix]>=10)&((origin[ix]-10)%30==0)]
        risk=np.full(3,.05)
        for j,now in enumerate(slots):
            if j and ok[now]:
                prev=slots[j-1]
                assert origin[now]-origin[prev]==30
                s=phase[prev]
                q=reference[int(np.ceil((len(reference)-1)*(1-risk[s]))),-1]
                miss=err[prev,-1]>q*scale[prev,-1]
                risk[s]=np.clip(risk[s]+.04*(.05-float(miss)),.01,.20)
            end=slots[j+1] if j+1<len(slots) else ix[-1]+1
            level[now:end]=risk[phase[now:end]]
    idx=np.ceil((len(reference)-1)*(1-level)).astype(int)
    return scale*reference[idx]


def load_issues(kind):
    with np.load(CACHE/(kind+"_issues.npz")) as f:
        return SimpleNamespace(**{k:f[k] for k in f.files})


def score():
    b4=load_issues("B4")
    ids=b4.ids
    meta=pd.read_parquet(CACHE/"meta.parquet").iloc[ids].reset_index(drop=True)
    env=meta.env.to_numpy()
    def data(key):
        return np.load(CACHE/(key+".npy"),mmap_mode="r")[ids]
    truth,mask=data("y"),data("mask")
    full=mask.all(1); grid=full & data("grid")
    phases=b4.stage.argmax(1)
    recent=data("recent"); present=data("mode"); later=data("next_mode")
    prob,state,phys=data("prob"),data("state"),data("physics")
    x=np.column_stack((prob,~data("ok"),recent,b4.stage[:,0],b4.stage[:,1],b4.change,
                       state[:,0],state[:,1],np.log(1e-5+phys[:,1,6]),b4.scale[:,-1]))
    train=meta.split.eq("train").to_numpy()
    val=meta.split.eq("validation").to_numpy()
    test=meta.split.eq("test").to_numpy()
    roles=json.loads((CACHE/"ready.json").read_text(encoding="utf8"))["environments"]
    development,calibration=roles["gate"],roles["calibration"]
    assert not (set(development)|set(calibration)|set(roles["fit"])) & set(env[~train])
    generator=interpolate(data("knots"))
    a=b4.path
    ab,within,coef=geometry(a,generator,truth)
    devmask=train & grid & np.isin(env,development)
    def fit(where): return fit_gate(x[where],coef[where],env[where])
    gate=fit(devmask)
    noqual_idx=[3,4,5,6,7,9]
    motion_gate=fit_gate(x[devmask][:,noqual_idx],coef[devmask],env[devmask])
    weights=gate_value(x,gate)
    motion_w=gate_value(x[:,noqual_idx],motion_gate)
    constant=np.full(len(x),gate["constant"])
    center,branch,spread=medoid(a,generator,ab,within,weights)
    error=np.linalg.norm(center-truth,axis=2)
    scale=np.sqrt(b4.scale**2+.25*spread**2)
    scores=(error/scale).max(1)
    cal=train & grid & np.isin(env,calibration)
    factors=calfactors(scores[cal],phases[cal])
    reference=np.sort(error[devmask]/scale[devmask],axis=0)
    np.save(OUT/"q1_feedback_reference.npy",reference)
    adaptive=feedback_scales(meta,phases,error,scale,data("ok"),reference)
    first_track=next(ix for ix in meta.groupby(["sample","hand"],sort=False).indices.values() if val[ix[0]])
    sub=first_track
    origin=meta.origin.to_numpy()[sub]
    change=error[sub].copy()
    change[origin[:,None]+np.arange(1,31)>103]=1000.
    check=feedback_scales(meta.iloc[sub].reset_index(drop=True),phases[sub],change,scale[sub],data("ok")[sub],reference)
    np.testing.assert_array_equal(check[origin<=103],adaptive[sub][origin<=103])
    adaptive_factor=rank95((error[cal]/adaptive[cal]).max(1))
    write_json(OUT/"q1_locked.json",dict(fit_envs=roles["fit"],development_envs=development,calibration_envs=calibration,
        gate_training_windows=int(devmask.sum()),gate=gate,motion_gate=motion_gate,calibration=factors,
        feedback_factor=adaptive_factor,feedback_future_invariance=True,feedback_initial_alpha=.05,feedback_gain=.04,
        mode="stage-conditioned or pooled 95% scores, per window maximum; class fallback below 25",
        fit_splits="10 official training environments for experts, 3 for gate, 3 further for outer calibration",
        point_readout="weighted sample medoid under joint path distance; no access to future"))
    static=np.full(len(meta),factors["global_factor"])
    stagef=np.array(factors["stage_factors"])[phases]
    point=[]; envrows=[]; subgroup=[]; bandrows=[]; proper=[]
    logs=meta.loc[grid & ~train,KEY+["slow","end_bin"]].copy()
    ep=meta.loc[mask[:,-1] & ~train,KEY].copy()
    def points(name,err):
        ep[name]=100*err[ep.index,-1]
        logs[name+"_path_cm"]=100*err[logs.index].mean(1)
        for spl,part in (("validation",val),("test",test)):
            last=part & mask[:,-1]; paths=part & full; windows=part & grid
            def value(v,ix): return macro(v[ix]*100,env[ix])
            row=dict(method=name,split=spl,n=int(last.sum()),path_n=int(paths.sum()),spaced_n=int(windows.sum()),
                     endpoint_macro_cm=value(err[:,-1],last),path_macro_cm=value(err.mean(1),paths),
                     spaced_path_cm=value(err.mean(1),windows))
            for h in (1,3,10,30):
                ix=part & mask[:,h-1]
                row[f"frame{h}_cm"]=value(err[:,h-1],ix)
            point.append(row)
            for e in sorted(set(env[part])):
                select=env==e
                envrows.append(dict(method=name,split=spl,env=e,n=int((last&select).sum()),
                    endpoint_cm=100*err[last&select,-1].mean(),path_cm=100*err[paths&select].mean()))
            slow=meta.slow.to_numpy()
            for label,select in (("slow_to_fast",slow & (meta.end_bin.to_numpy()==2)),("slow_to_slow",slow & (meta.end_bin.to_numpy()==0)),
                         ("recent_flag",recent),("recent_clear",~recent),("future_turn_proxy",later==1),("current_dwell",present==0)):
                ix=last & select
                if ix.any():
                    subgroup.append(dict(method=name,split=spl,group=label,n=int(ix.sum()),pooled_endpoint_cm=float(100*err[ix,-1].mean())))
    points("hold",np.linalg.norm(truth,axis=2))
    for kind in VARIANTS:
        raw=b4 if kind=="B4" else load_issues(kind)
        np.testing.assert_array_equal(ids,raw.ids)
        points(kind,np.linalg.norm(raw.path-truth,axis=2))
    points("B5",error)
    for label,w in (("dynamics_only",np.zeros(len(x))),("diffusion_only",np.ones(len(x))),("fixed_25pct",np.full(len(x),.25)),
                    ("fitted_constant",constant),("motion_gate",motion_w),("quality_stage_gate",weights)):
        es=(coef[:,0]+coef[:,1]*w+coef[:,2]*w*w)*100
        for spl,part in (("validation",val),("test",test)):
            ix=part & grid
            proper.append(dict(method=label,split=spl,n=int(ix.sum()),energy_score_cm=macro(es[ix],env[ix]),
                        pooled_energy_cm=es[ix].mean(),diffusion_weight=w[ix].mean()))
        logs[label+"_energy_cm"]=es[logs.index]
    for name,bands in (("pooled",scale*static[:,None]),("stage",scale*stagef[:,None]),("adaptive",adaptive*adaptive_factor)):
        cover=(error<=bands).all(1)
        logs[name+"_covered"]=cover[logs.index]
        logs[name+"_radius_cm"]=100*bands[logs.index].mean(1)
        for spl,part in (("validation",val),("test",test)):
            for label,select in (("all",np.ones(len(meta),bool)),("predicted_change",b4.change>=.5),("recent_flag",recent),
                             *((s,phases==i) for i,s in enumerate(("dwell","turn","translate")))):
                ix=part & grid & select
                if ix.any():
                    bandrows.append(dict(split=spl,band=name,group=label,n=int(ix.sum()),covered=int(cover[ix].sum()),
                        path_coverage=cover[ix].mean(),macro_coverage=macro(cover[ix],env[ix]),radius_cm=100*bands[ix].mean(),
                        frame_coverage=(error[ix]<=bands[ix]).mean()))
    logs["weight"]=weights[logs.index];logs["medoid_expert"]=branch[logs.index]
    logs["B5_max_error_cm"]=100*error[logs.index].max(1)
    logs["stage"]=phases[logs.index];logs["change_risk"]=b4.change[logs.index];logs["recent_flag"]=recent[logs.index]
    logs["stage_at_origin"]=present[logs.index];logs["actual_mode_at_10"]=later[logs.index]
    for table,name in ((pd.DataFrame(point),"q1_points.csv"),(pd.DataFrame(envrows),"q1_environment.csv"),
           (pd.DataFrame(subgroup),"q1_groups.csv"),(pd.DataFrame(proper),"q1_energy.csv"),(pd.DataFrame(bandrows),"q1_bands.csv"),
           (logs,"q1_issued_windows.csv.gz"),(ep,"q1_issued_endpoints.csv.gz")):
        table.to_csv(OUT/name,index=False)
    # A complete set of weighted supports, reference and issued sphere for every held-out nonoverlap window.
    ix=logs.index.to_numpy()
    np.savez_compressed(OUT/"q1_issued_trajectories.npz",canonical_row=ids[ix],dynamics=a[ix],diffusion_knots=data("knots")[ix],
            diffusion_total_weight=weights[ix],medoid=branch[ix],center=center[ix],truth=truth[ix],
            stage_radius=(scale*stagef[:,None])[ix],radius=(adaptive*adaptive_factor)[ix])
    qa=[]
    for spl,part in (("validation",val),("test",test)):
        ix=part & grid
        qa.append(dict(split=spl,phase_accuracy=(phases[ix]==later[ix]).mean(),
            persistence_accuracy=(present[ix]==later[ix]).mean(),phase_change_Brier=np.mean((b4.change[ix]-(present[ix]!=later[ix]))**2),
            medoid_diffusion_fraction=(branch[ix]>0).mean(),n=int(sum(ix))))
    pd.DataFrame(qa).to_csv(OUT/"q1_phase.csv",index=False)
    print(pd.DataFrame(point).query("method in ['B3','B4','B5']").to_string(index=False),flush=True)


if __name__=="__main__":
    score()
