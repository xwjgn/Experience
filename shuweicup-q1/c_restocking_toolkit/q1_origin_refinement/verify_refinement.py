"""Compare the public online entry point against the frozen per-window evaluation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from wrist_history import M4_DIR,OUT,SIDES,load_record,observe,protect_weights,read,save
from alarm_rules import alarms_on_record
from refine import WristRefinement


def check_delay():
    phase=np.zeros(70,int)
    state=SimpleNamespace(quality=np.ones(70,bool),phase=phase)
    settings=[(.24,24)]
    prior=np.linspace(.05,.5,31)
    def thresholds(times):
        return next(iter(alarms_on_record(state,np.array(times),[.9,.1],prior,settings).values()))
    assert thresholds([10,17])[0]==thresholds([10,17])[1]
    normal=thresholds([10,40])
    assert normal[1]!=normal[0]
    state.quality[15]=False
    assert thresholds([10,40])[1]==normal[0]
    state.quality[15]=True
    state.phase[20]=1
    assert thresholds([10,40])[1]==normal[0]


def verify():
    check_delay()
    table=pd.read_csv(OUT/"760个起点逐窗回放.csv")
    select=pd.concat([table[table.previous_additional_miss],
                      table.groupby(["env","hand","previous_routed"],sort=False).head(1),
                      table.groupby(["dataset","hand"],sort=False).head(1)]).drop_duplicates("row")
    system=WristRefinement()
    with np.load(M4_DIR/"M3_issued.npz") as archive:
        lookup=pd.Index(archive["row"])
        pos=lookup.get_indexer(select.row)
        assert (pos>=0).all()
        center,radius,pturn=(archive["center"][pos],archive["radius"][pos],archive["phase"][pos,1])
    max_budget=max_score=max_error_cm=0.
    for j,r in enumerate(select.itertuples()):
        count=r.origin+1
        complete=load_record(r.dataset,r.sample,count+30)
        prefix=load_record(r.dataset,r.sample,count)
        m=observe(prefix,r.hand,system.cut)
        rot=m.rotation[-1].as_matrix()
        world=m.wrist[-1]+np.einsum("hi,ji->hj",center[j]*m.mirror,rot)
        base=dict(origin=r.origin,hand=r.hand,radius_m=radius[j],center_world=world,phase_probabilities={"turn":pturn[j]})
        factor=.8 if r.previous_routed else 1.
        new=system.issue(prefix,r.hand,base,factor)
        assert new["last_observed_frame"]==r.origin and new["turn_alert"]==r.online_alert and new["fused_turn_alert"]==r.fused_alert
        max_score=max(max_score,abs(new["turn_score"]-r.turn_score),abs(new["turn_threshold"]-r.online_threshold))
        truth=complete.joints[count:,SIDES.index(r.hand),0]
        mixed=np.asarray(new["mixed_center_world"])
        loss=np.linalg.norm(mixed-truth,axis=1)
        max_error_cm=max(max_error_cm,abs(100*loss.mean()-r.new_ade_cm))
        for key in ("compact","same_window_budget"):
            b=np.asarray(new["mixed_center_radii_m"][key])
            assert bool(np.all(loss<=b))==getattr(r,"mixed_center_"+key+"_hit")
            np.testing.assert_allclose(100*b.mean(),getattr(r,"mixed_center_"+key+"_radius_cm"),atol=1.e-5,rtol=0)
        budget=np.asarray(new["m4_center_radii_m"]["same_window_budget"])
        assert bool(np.all(np.linalg.norm(world-truth,axis=1)<=budget))==r.same_window_budget_hit
        max_budget=max(max_budget,abs(budget.mean()-factor*radius[j].mean()))
        # Changing all later joint locations must leave every predictor at the origin identical.
        changed=complete.joints.copy()
        changed[count:]+=10
        future=observe(replace(complete,joints=changed),r.hand,system.cut)
        np.testing.assert_allclose(m.tab.iloc[-1],future.tab.iloc[count-1],rtol=0,atol=1.e-7)
    assert max_score<3.e-6 and max_error_cm<2.e-4 and max_budget<1.e-10
    assert select.env.nunique()==4 and select.dataset.nunique()==3 and set(select.hand)==set(SIDES)
    save(OUT/"online_verification.json",dict(prefixes=len(select),old14_verified=int(select.previous_additional_miss.sum()),
         environments=select.env.nunique(),datasets=select.dataset.nunique(),max_score_difference=max_score,
         max_ade_difference_cm=max_error_cm,max_mean_radius_difference_m=max_budget,delayed_label_tests=True,
         earlier_inputs_unchanged_after_future_perturbation=True,old_weights_loaded=0))
    print("Verified",len(select),"original prefixes including every one of the 14 earlier failures.",flush=True)


if __name__=="__main__":
    protect_weights()
    verify()
