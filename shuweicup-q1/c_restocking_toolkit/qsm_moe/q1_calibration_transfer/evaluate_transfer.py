"""Paired evaluation after both validation decisions have been saved."""
import numpy as np
import pandas as pd

from transfer_config import OUT, WORK, PREVIOUS, ORIGINAL
from calibration import Panel, SHAPES, incumbent, replay, load_priors
from config import KEY, digest, joined, read_json, write_json
from motion import Inputs
from select_bands import get_radius


def summary(df):
    turns = df[df.turn]
    return dict(windows=len(df), covered=int(df.hit.sum()), coverage=df.hit.mean(),
                mean_radius_cm=df.radius_cm.mean(), ade_cm=df.ade_cm.mean(), turns=len(turns),
                turn_ade_cm=turns.ade_cm.mean(), turn_covered=int(turns.hit.sum()),
                turn_coverage=turns.hit.mean(), turn_radius_cm=turns.radius_cm.mean())


def score():
    frozen = OUT/"frozen_before_test.json"
    stamp = digest(frozen)
    lock = read_json(frozen)
    if digest(WORK/"corrected_center.npy") != lock["center_sha256"]:
        raise ValueError("Frozen centers changed")
    priors = load_priors()
    assert all(digest(OUT/(s+"_training_prior.npz"))==lock["priors_sha256"][s] for s in SHAPES)
    reference = pd.read_csv(PREVIOUS/"paired_windows.csv.gz")
    d = Inputs()
    tables, horizons, origin_losses = [], [], []
    diffs, checks, issued = [], [], []
    with np.load(OUT/"issued_scales.npz") as widths, np.load(PREVIOUS/"issued_grid.npz") as archived:
        for split, allowed in (("validation", d.val), ("test", d.test)):
            panel = Panel(d, allowed, widths)
            mask = panel.complete
            ref_radius = incumbent(panel)
            replays = {s:replay(panel,s,priors[s]) for s in SHAPES}
            recipes = {**lock["controls"],"current":dict(kind="incumbent"),
                       "selected":lock["selected"]["recipe"],"matched":lock["matched"]["recipe"]}
            radii = {key:get_radius(r,panel,ref_radius,replays,priors) for key,r in recipes.items()}
            alternate = lock["alternate_validation_calibration"]
            if split=="validation":
                factor = np.array([alternate["validation_folds"][e] for e in panel.env])
                radii["gru_val_calibration"] = replays["gru"]["endpoint"]*factor[:,None]
            else:
                radii["gru_val_calibration"] = get_radius(alternate["recipe"],panel,ref_radius,replays,priors)

            kept = panel.ids[mask]
            locations = np.searchsorted(archived["row"], kept)
            np.testing.assert_array_equal(archived["row"][locations],kept)
            np.testing.assert_allclose(panel.center[mask],archived["center"][locations],rtol=0,atol=3e-6)
            np.testing.assert_allclose(ref_radius[mask],archived["radius"][locations],rtol=0,atol=3e-6)
            checks.append(dict(split=split,windows=len(kept),
                               centers_bit_identical=bool(np.array_equal(panel.center[mask],archived["center"][locations])),
                               max_center_difference_m=float(np.abs(panel.center[mask]-archived["center"][locations]).max()),
                               max_incumbent_radius_difference_m=float(np.abs(ref_radius[mask]-archived["radius"][locations]).max())))
            issued.append((kept, radii["selected"][mask]))
            base = panel.meta.loc[mask,KEY].reset_index(drop=True)
            base["turn"] = panel.turn[mask]
            old = joined(base,reference.loc[reference.split==split].drop(columns=["turn"]))
            np.testing.assert_array_equal(base.turn,old.actual_turn)
            orig = base.assign(model="original",hit=old.original_hit,radius_cm=old.original_radius_cm,ade_cm=old.original_path_cm)
            tables.append(orig)
            for key, rad in radii.items():
                error, radius = panel.error[mask], rad[mask]
                hit = (error<=radius).all(1)
                table = base.assign(model=key,hit=hit, radius_cm=100*radius.mean(1),ade_cm=100*error.mean(1))
                tables.append(table)
                for h in (1,3,10,20,30):
                    horizons.append(dict(model=key,split=split,horizon=h,radius_cm=100*radius[:,h-1].mean(),
                                         coverage=(error[:,h-1]<=radius[:,h-1]).mean()))
                if key in ("selected","matched","gru_endpoint_95","gru_static_95","gru_val_calibration"):
                    old_hit = old.v3_hit.to_numpy(bool)
                    diffs.append(dict(model=key, split=split, gained=int((hit & ~old_hit).sum()),
                                lost=int((~hit & old_hit).sum()),turn_gained=int((hit & ~old_hit & base.turn).sum()),
                                turn_lost=int((~hit & old_hit & base.turn).sum()),
                                paired_difference_cm=(table.radius_cm-old.v3_radius_cm).mean()))
                    old_turn = base.turn.to_numpy(bool)
                    for tag, subgroup in (("all", np.ones(len(base),bool)), ("turn",old_turn)):
                        window_loss = ~hit & subgroup
                        origin_losses.append(dict(model=key,split=split,group=tag,
                                                  misses=int(window_loss.sum()),
                                                  mean_miss_excess_cm=float(100*np.maximum(error-radius,0)[window_loss].max(1).mean())
                                                  if np.any(window_loss) else 0.))
    data = pd.concat(tables,ignore_index=True)
    metrics = pd.DataFrame([dict(split=sp,model=m,**summary(g)) for (sp,m),g in data.groupby(["split","model"],sort=False)])
    env = pd.DataFrame([dict(split=sp,model=m,env=e,**summary(g)) for (sp,m,e),g in data.groupby(["split","model","env"],sort=False)])
    data.to_csv(OUT/"paired_windows.csv.gz",index=False)
    metrics.to_csv(OUT/"comparison.csv",index=False)
    env.to_csv(OUT/"by_environment.csv",index=False)
    pd.DataFrame(horizons).to_csv(OUT/"by_horizon.csv",index=False)
    pd.DataFrame(diffs).to_csv(OUT/"paired_changes.csv",index=False)
    pd.DataFrame(origin_losses).to_csv(OUT/"miss_distance.csv",index=False)
    np.savez_compressed(OUT/"selected_grid.npz",row=np.concatenate([r for r,_ in issued]),
                        radius=np.concatenate([b for _,b in issued]))
    old_bands = pd.read_csv(ORIGINAL/"band_coverage.csv")
    old_bands[(old_bands.method=="anchored_damped") & (old_bands.sampling=="grid") &
              (old_bands.scope=="all") & (old_bands.env=="all")].to_csv(OUT/"original_band_audit.csv",index=False)
    assert stamp==digest(frozen)
    write_json(OUT/"paired_audit.json", dict(frozen_sha256=stamp, center_checks=checks, old_checkpoints_loaded=0))
    print(metrics[metrics.split=="test"][["model","covered","turn_covered","mean_radius_cm","turn_radius_cm","turn_ade_cm"]].round(4).to_string(index=False))


if __name__=="__main__":
    score()
