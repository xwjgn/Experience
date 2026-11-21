"""Score the frozen primary and secondary decisions at identical issue times."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from stable_config import OUT, WORK, ARCHIVED, TRANSFER
from config import KEY, digest, joined, read_json, write_json
from motion import Inputs
from stability_router import Policy
from stable_data import FixedReference


def describe(df):
    turn = df.turn
    stable = df.stable
    chosen = df.routed
    original_hit = df.v3_hit
    before = df.v3_radius_cm
    def average(values):
        return float(values.mean()) if len(values) else None
    return dict(n=len(df), covered=int(df.hit.sum()), coverage=df.hit.mean(), radius_cm=df.radius_cm.mean(),
        mean_saving_cm=(before-df.radius_cm).mean(), overall_ade_cm=df.ade_cm.mean(), turns=int(turn.sum()),
        turn_covered=int(df.hit[turn].sum()),turn_coverage=average(df.hit[turn]),turn_ade_cm=average(df.ade_cm[turn]),
        turn_radius_cm=average(df.radius_cm[turn]),nonturn_radius_cm=average(df.radius_cm[~turn]),
        routed=int(chosen.sum()),routed_turns=int((chosen&turn).sum()),routed_turn_covered=int((chosen&turn&df.hit).sum()),
        padded=int(df.padded.sum()),actual_stable=int(stable.sum()),routed_stable=int((stable&chosen).sum()),
        stable_precision=average(stable[chosen]),stable_recall=average(chosen[stable]),
        routed_coverage=average(df.hit[chosen]),other_coverage=average(df.hit[~chosen]),
        gained=int((df.hit&~original_hit).sum()),lost=int((original_hit&~df.hit).sum()),
        gained_turn=int((df.hit&~original_hit&turn).sum()),lost_turn=int((~df.hit&original_hit&turn).sum()))


def historical_table(df, bounds):
    rows = []
    for split,part in df.groupby("split",sort=False):
        for prefix in ("original","v3","previous_transfer"):
            hit = part[prefix+"_hit"]
            radius = part[prefix+"_radius_cm"]
            turns = part.turn
            rows.append(dict(split=split,model=prefix,n=len(part),covered=int(hit.sum()),coverage=hit.mean(),
                       turns=int(turns.sum()),turn_covered=int(hit[turns].sum()),turn_coverage=hit[turns].mean(),
                       turn_ade_cm=part.loc[turns,prefix+"_ade_cm"].mean(),radius_cm=radius.mean(),
                       turn_radius_cm=radius[turns].mean(),nonturn_radius_cm=radius[~turns].mean()))
        v = bounds[bounds.split==split].set_index("group")
        rows.append(dict(split=split,model="previous_fixed_control",n=int(v.at["all","windows"]),
              covered=int(v.at["all","without_feedback_covered"]),coverage=v.at["all","without_feedback_covered"]/v.at["all","windows"],
              turns=int(v.at["turn","windows"]),turn_covered=int(v.at["turn","without_feedback_covered"]),
              turn_coverage=v.at["turn","without_feedback_covered"]/v.at["turn","windows"],
              turn_ade_cm=part.loc[part.turn,"v3_ade_cm"].mean(),radius_cm=v.at["all","without_feedback_cm"],
              turn_radius_cm=v.at["turn","without_feedback_cm"],nonturn_radius_cm=v.at["nonturn","without_feedback_cm"]))
    return rows


def score():
    lock_path = OUT/"frozen_before_test.json"
    stamp = digest(lock_path)
    lock = read_json(lock_path)
    assert lock["router_sha256"]==digest(OUT/"origin_router.joblib")
    assert lock["qsm_center_sha256"]==digest(WORK/"corrected_center.npy")
    model = joblib.load(OUT/"origin_router.joblib")
    d = Inputs()
    ref = FixedReference(d)
    archived = pd.read_csv(ARCHIVED/"paired_windows.csv.gz",usecols=KEY+["turn","v3_path_cm","v3_hit","v3_radius_cm",
                                                                        "original_path_cm","original_hit","original_radius_cm"])
    archived = archived.rename(columns={"v3_path_cm":"v3_ade_cm","original_path_cm":"original_ade_cm"})
    prior = pd.read_csv(TRANSFER/"paired_windows.csv.gz")
    prior = prior[prior.model=="selected"][KEY+["hit","radius_cm","ade_cm"]].rename(
            columns={c:"previous_transfer_"+c for c in ("hit","radius_cm","ade_cm")})
    archived = joined(archived,prior)
    table = historical_table(archived,pd.read_csv(TRANSFER/"feedback_diagnosis.csv"))
    raw,per_env,heads,group_diagnosis,keys,issued = [],[],[],[],[],[]
    with np.load(ARCHIVED/"issued_grid.npz") as hs:
        for split,allowed in (("validation",d.val),("test",d.test)):
            data = ref.windows(allowed)
            order = np.searchsorted(hs["row"],data.ids)
            np.testing.assert_array_equal(hs["row"][order],data.ids)
            np.testing.assert_array_equal(ref.center[data.ids],hs["center"][order])
            np.testing.assert_array_equal(data.radius,hs["radius"][order])
            previous = joined(data.meta,archived[archived.split==split])
            np.testing.assert_array_equal(data.turn,previous.turn)
            base_hit = (data.error<=data.radius).all(1)
            np.testing.assert_array_equal(base_hit,previous.v3_hit)
            ade = 100*data.error.mean(1)
            np.testing.assert_allclose(ade,previous.v3_ade_cm,atol=2e-5,rtol=0)
            scores = model.scores(data.x)
            ready = {name:Policy(**spec).issue(model,data.x,data.radius,scores) for name,spec in lock["policies"].items()}
            scale = ready["conservative"][0].mean()/data.radius.mean()
            phase_scale = ready["phase_only"][0].mean()/data.radius.mean()
            empty = np.zeros(len(data.x),bool)
            ready["uniform_same_mean"] = (data.radius*scale,empty,empty)
            ready["uniform_phase_same_mean"] = (data.radius*phase_scale,empty,empty)
            ready["uniform_from_validation"] = (data.radius*lock["uniform_validation_factor"],empty,empty)
            keys.append(data.ids)
            issued.append(ready["conservative"][0])
            for name,target,pred in (("stable",data.stable,scores["stable"]),("turn",data.turn,scores["turn"]),
                                     ("fragile_turn",data.turn & (data.ratio>.72),scores["fragile"])):
                heads.append(dict(split=split,head=name,prevalence=target.mean(),roc_auc=roc_auc_score(target,pred),
                                  average_precision=average_precision_score(target,pred)))
            heads.append(dict(split=split,head="tail",empirical_below_predicted_90th=(data.ratio<=scores["tail"]).mean()))
            for name,(radius,routed,padded) in ready.items():
                hits = (data.error<=radius).all(1)
                record = previous.copy()
                record["model"] = name
                record["stable"] = data.stable
                record["routed"],record["padded"] = routed,padded
                record["hit"],record["radius_cm"],record["ade_cm"] = hits,100*radius.mean(1),ade
                record["worst_ratio"] = (data.error/radius).max(1)
                record["newly_lost"] = base_hit & ~hits
                record["v3_worst_ratio"] = data.ratio
                record["retained_radius_fraction"] = radius.mean(1)/data.radius.mean(1)
                for f in ("turn_probability","quality_risk","last_path_ratio","history_count","speed5","speed30","excursion_recent"):
                    record[f] = data.x[f].to_numpy()
                record["stable_score"] = scores["stable"]
                record["turn_score"] = scores["turn"]
                record["tail_score"] = scores["tail"]
                raw.append(record)
                summary = describe(record)
                table.append(dict(split=split,model=name,**summary))
                for env,part in record.groupby("env"):
                    per_env.append(dict(split=split,model=name,env=env,**describe(part)))
                for group,mask in (("all",np.ones(len(hits),bool)),("true_turn",data.turn),("other",~data.turn),
                                   ("actually_stable",data.stable),("routed",routed),("not_routed",~routed)):
                    if np.any(mask):
                        part = record.loc[mask]
                        group_diagnosis.append(dict(split=split,model=name,group=group,**describe(part)))
    assert stamp==digest(lock_path)
    results = pd.DataFrame(table)
    results.to_csv(OUT/"comparison.csv",index=False)
    logs = pd.concat(raw,ignore_index=True)
    logs.to_csv(OUT/"paired_windows.csv.gz",index=False)
    tested = logs[logs.split=="test"]
    main_test = tested[tested.model=="conservative"]
    turn_log = main_test[main_test.turn][KEY+["original_ade_cm","original_hit","original_radius_cm","v3_ade_cm","v3_hit","v3_radius_cm"]]
    for tag in ("conservative","recognition","phase_only","uniform_phase_same_mean"):
        changes = ["routed","radius_cm","hit","newly_lost"]
        sub = tested[tested.turn & (tested.model==tag)][KEY+changes]
        turn_log = joined(turn_log,sub.rename(columns={c:tag+"_"+c for c in changes}))
    turn_log.to_csv(OUT/"99个转向窗口逐窗核对.csv",index=False,encoding="utf-8-sig")
    tested[tested.newly_lost & tested.model.isin(("conservative","recognition","phase_only"))].to_csv(
        OUT/"收紧后新增漏失.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(per_env).to_csv(OUT/"by_environment.csv",index=False)
    pd.DataFrame(group_diagnosis).to_csv(OUT/"by_group.csv",index=False)
    pd.DataFrame(heads).to_csv(OUT/"origin_head_quality.csv",index=False)
    np.savez_compressed(OUT/"issued_grid.npz",row=np.concatenate(keys),radius=np.concatenate(issued))
    write_json(OUT/"score_audit.json",dict(frozen_sha256=stamp,old_checkpoints_loaded=0,
            all_centers_bit_identical_to_v3=True,all_reference_radii_bit_identical_to_v3=True,
            grids_by_split=logs[logs.model=="conservative"].split.value_counts().to_dict()))
    print(results[results.split=="test"][["model","n","covered","turn_covered","turn_ade_cm","radius_cm","routed","routed_turns","lost_turn","stable_precision"]].round(4).to_string(index=False),flush=True)


if __name__=="__main__":
    score()
