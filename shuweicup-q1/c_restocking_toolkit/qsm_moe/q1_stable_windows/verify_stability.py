"""Compare raw-prefix and batch forecasts, including histories interrupted between receipts."""
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from stable_config import OUT
from config import read_json,write_json
from motion import Inputs
from issue_stable import StableWristForecaster
from phase_rule import apply_rule
from origin_signals import PastErrors,raw_signals
from stable_data import FixedReference,measurements,track
from stability_router import Policy
from features import rows as read_prefix


def corruption_check(ref,d):
    group = next(iter(d.meta[d.val].groupby(["sample","hand"],sort=False)))[1]
    ids = group.index.to_numpy()
    sample,hand = group["sample"].iat[0],group.hand.iat[0]
    t,z,head,valid = measurements(track(sample),hand)
    cut = min(137,len(t)-32)
    origins = np.arange(10,cut+1,3)
    signals = raw_signals(t,z,head,valid,origins)
    altered_z,altered_head,altered_ok = z.copy(),head.copy(),valid.copy()
    altered_z[cut+1:] += 5.
    altered_head[cut+1:] -= 2.
    altered_ok[cut+1:] = False
    other = raw_signals(t,altered_z,altered_head,altered_ok,origins)
    truncated = raw_signals(t[:cut+1],z[:cut+1],head[:cut+1],valid[:cut+1],origins)
    for name in signals:
        np.testing.assert_array_equal(signals[name],other[name])
        np.testing.assert_array_equal(signals[name],truncated[name])
    error = np.linalg.norm(ref.center[ids]-d.y[ids],axis=2)
    radii = ref.radius(ids,error,valid)
    changed = error.copy()
    future = np.arange(len(ids))[:,None]+np.arange(1,31)>cut
    changed[future] = 1000.
    left,right = PastErrors(),PastErrors()
    for end in range(40,cut+1,30):
        old = end-30
        left.receive(end,error[old],radii[old],valid[old+1:end+1])
        right.receive(end,changed[old],radii[old],altered_ok[old+1:end+1])
        assert left.values(end)==right.values(end)
    return dict(sample=sample,hand=hand,cutoff=int(cut),prefixes_checked=len(origins),future_changes_affect_prior_features=False)


def verify():
    d = Inputs()
    ref = FixedReference(d)
    compact = read_json(OUT/"phase_rule.json")
    frozen = read_json(OUT/"frozen_before_test.json")
    rule = Policy(**frozen["policies"]["phase_only"])
    router = joblib.load(OUT/"origin_router.joblib")
    online = StableWristForecaster()
    assert online.name=="phase_only"
    prim = StableWristForecaster("conservative")
    cautious = StableWristForecaster("recognition")
    audit = []
    count = 0
    for split,allow in (("validation",d.val),("test",d.test)):
        a = ref.windows(allow)
        scores = router.scores(a.x)
        phase_r,selected,_ = rule.issue(router,a.x,a.radius,scores)
        if compact["simple"]:
            r,c,_ = apply_rule(compact,a.x,a.radius)
            np.testing.assert_array_equal(r,phase_r)
            np.testing.assert_array_equal(c,selected)
            assert online.router is None
            count += len(r)
            if np.any(selected):
                i = np.flatnonzero(selected)[0]
                for changes in ({"valid":0.},{"recent_valid":.5},{"speed5":.25,"direction_cos5":.3},
                                {"last_path_ratio":1.1,"history_count":1,"history_age":1.},{"turn_probability":.50}):
                    scenario = a.x.iloc[[i]].assign(**changes)
                    assert not apply_rule(compact,scenario,a.radius[[i]])[1][0]
        main,_,_ = Policy(**frozen["policies"]["conservative"]).issue(router,a.x,a.radius,scores)
        cautious_r,_,_ = Policy(**frozen["policies"]["recognition"]).issue(router,a.x,a.radius,scores)
        for env in sorted(a.meta.env.unique()):
            env_idx = a.meta.env.eq(env).to_numpy()
            tested = 0
            for flag in (True,False):
                eligible = np.flatnonzero(env_idx & (selected==flag) & (a.meta.origin.to_numpy()>=40))
                if not len(eligible):
                    continue
                row = eligible[np.argmin(a.meta.origin.to_numpy()[eligible])]
                meta = a.meta.iloc[row]
                s,hand,o = meta["sample"],meta["hand"],int(meta["origin"])
                result = online.issue(s,hand,o)
                assert result["last_read_frame"]==o and result["last_matured_frame"]==o
                assert (result["band_decision"]=="tighten")==flag
                np.testing.assert_allclose(result["radius_m"],phase_r[row],atol=1e-5,rtol=0)
                np.testing.assert_allclose(result["previous_observed_path_ratio"],a.x.last_path_ratio.iat[row],atol=2e-4,rtol=0)
                np.testing.assert_allclose(result["reference_radius_m"],a.radius[row],atol=1e-5,rtol=0)
                g = read_prefix(s,o)
                _,z,_,_ = measurements(g,hand)
                rot = Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].iloc[-1]).as_matrix()
                center = (np.array(result["center_world"])-z[-1])@rot
                np.testing.assert_allclose(center,ref.center[a.ids[row]],atol=1e-5,rtol=0)
                if tested==0:
                    check = prim.issue(s,hand,o)
                    np.testing.assert_allclose(check["radius_m"],main[row],atol=1e-5,rtol=0)
                    check = cautious.issue(s,hand,o)
                    np.testing.assert_allclose(check["radius_m"],cautious_r[row],atol=1e-5,rtol=0)
                    off = online.issue(s,hand,o+7)
                    assert off["last_matured_frame"]==o and off["last_read_frame"]==o+7
                    earlier = online.issue(s,hand,o-1)
                    assert earlier["last_matured_frame"]==(o-30 if o>=70 else None)
                    repeat = online.issue(s,hand,o)
                    np.testing.assert_array_equal(repeat["radius_m"],result["radius_m"])
                tested += 1
                audit.append(dict(split=split,env=env,sample=s,hand=hand,origin=o,tightened=bool(flag),
                           max_online_radius_difference_m=float(np.abs(result["radius_m"]-phase_r[row]).max())))
    check = corruption_check(ref,d)
    write_json(OUT/"verification.json",dict(compact_rule_matches_full_router_on=count,raw_prefixes=audit,
               future_edit_check=check,phase_policy_uses_additional_classifiers=online.router is not None))
    print("Verified",count,"paired compact decisions and",len(audit),"raw prefixes.",flush=True)


if __name__=="__main__":
    verify()
