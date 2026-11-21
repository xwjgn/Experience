"""Check release-to-world predictions against the scored arrays using the public online interface."""
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.special import expit

from forecast import Forecaster, sample_paths
from hand_features import encode
from provenance import ReadAudit
from release_reader import SIDES, load_record
from study_paths import DATA, MODES, OUT, SEEDS, read, save
from trajectory_bands import BandMemory, play


def locations():
    meta = pd.read_parquet(DATA/"windows.parquet")
    with np.load(OUT/"M3_issued.npz") as file:
        records = meta.iloc[file["row"]].reset_index(drop=True)
    within = records[records.origin >= 100]
    chosen = within.groupby(["split", "env", "hand"], sort=True).head(1)
    extra = within.groupby(["dataset", "hand"], sort=True).head(1)
    return pd.concat((chosen, extra)).drop_duplicates(), records


def check_feedback(prior, recipe):
    rng = np.random.default_rng(137)
    meta = pd.DataFrame({"sample":"example", "hand":"left", "origin":10+30*np.arange(6)})
    prob = rng.dirichlet([2., 1., 2.], size=len(meta))
    scale = rng.uniform(.03, .07, (len(meta), 30))
    errors = rng.uniform(.03, .09, scale.shape)
    good = np.ones(scale.shape, dtype=bool)
    expected = play(meta, errors, scale, prob, good, prior, recipe)
    future = meta.origin.to_numpy()[:, None]+np.arange(1, 31) > 100
    errors[future] = 1.e4
    good[future] = False
    replay = play(meta, errors, scale, prob, good, prior, recipe)
    np.testing.assert_array_equal(replay[meta.origin <= 100], expected[meta.origin <= 100])
    state = BandMemory(prior, recipe)
    issued = state.issue(scale[0], prob[0])
    old_alpha = state.alpha.copy()
    good[0, -1] = False
    state.receive(errors[0], scale[0], prob[0], issued, good[0])
    np.testing.assert_array_equal(state.alpha, old_alpha)
    assert not state.past


def check_score_tables():
    totals = pd.read_csv(OUT/"comparison.csv").set_index(["split", "model", "band"])
    issues = pd.read_csv(OUT/"paired_issues.csv.gz")
    for split, part in issues.groupby("split"):
        grid = part[part.model == "M3"]
        assert not grid.duplicated(["env", "sample", "hand", "origin"]).any()
        for name, band, prefix in (("published_original", "archived", "original_"),
                                   ("published_qsm_low_turn", "archived", "qsm_"),
                                   ("published_quality_stage", "cafht_cptc", "lsttn_"),
                                   ("M3", "joint_stage", "")):
            m = totals.loc[(split, name, band)]
            turns = grid[grid.turn]
            assert len(grid) == m.n and len(turns) == m.turns
            assert grid[prefix+"hit"].sum() == m.covered and turns[prefix+"hit"].sum() == m.turn_covered
            np.testing.assert_allclose([grid[prefix+"ade_cm"].mean(), turns[prefix+"ade_cm"].mean(),
                                        grid[prefix+"radius_cm"].mean()],
                                       [m.ade_cm, m.turn_ade_cm, m.radius_cm], rtol=0, atol=3.e-5)
    horizons = pd.read_csv(OUT/"by_horizon.csv")
    assert set(horizons.horizon) == {1, 3, 10, 30}
    assert (horizons[["mean_error_cm", "turn_mean_error_cm", "mean_radius_cm"]] >= 0).all().all()
    assert (horizons[horizons.horizon == 30].median_seconds.between(.98, 1.02)).all()


def verify():
    check_score_tables()
    selected, all_rows = locations()
    checked = []
    maxima = {}
    flag_checks = 0
    sources = set()
    for model, archive in (("M2", "M2"), ("M4", "M3")):
        system = Forecaster(model)
        check_feedback(system.reference, system.recipe)
        maximum = 0.
        with np.load(OUT/(archive+"_issued.npz")) as disk:
            stored = {k:disk[k] for k in disk.files}
        for ix, r in selected.iterrows():
            end = int(r.origin)+1
            extended = load_record(r.dataset, r["sample"], end+6)
            prefix = load_record(r.dataset, r["sample"], end)
            sources.add(r.dataset)
            full_feature, _ = encode(extended, r.hand)
            short_feature, (_, wrist, axes, flip) = encode(prefix, r.hand)
            for name, block in short_feature.items():
                np.testing.assert_array_equal(block, full_feature[name][:end])
            issued = system.issue_prefix(prefix, r.hand)
            assert issued["last_read_frame"] == r.origin
            assert issued["last_complete_observation"] is None or issued["last_complete_observation"] <= r.origin
            np.testing.assert_allclose(sum(issued["component_weights"]), 1., atol=1.e-6)
            assert np.all(np.asarray(issued["radius_m"]) > 0)
            def local(world):
                return np.einsum("...j,ji->...i", np.asarray(world)-wrist[-1], axes[-1])*flip
            for key, arr in [("center", local(issued["center_world"])),
                             ("means", local(issued["component_means_world"])), ("radius", issued["radius_m"]),
                             ("sd", issued["component_axis_sigma_m"]), ("weights", issued["component_weights"]),
                             ("phase", [issued["phase_probabilities"][m] for m in MODES])]:
                diff = np.max(np.abs(arr-stored[key][ix]))
                maximum = max(maximum, float(diff))
                np.testing.assert_allclose(arr, stored[key][ix], rtol=0, atol=1.e-5, err_msg=f"{model}/{key}/{ix}")
            spec = system.spec["quality"]
            archived_risk = expit(spec["slope"]*stored["qlogit"][ix]+spec["intercept"])
            np.testing.assert_allclose(issued["observation_flag_probability"], archived_risk, rtol=0, atol=1.e-6)
            assert issued["turn_alert"] == (stored["phase"][ix,1] >= system.spec["turn_cut"])
            changed_flags = prefix.flags.copy()
            side = SIDES.index(r.hand)
            changed_flags[-1,side] = 1 if changed_flags[-1,side] == 0 else 0
            changed = system.issue_prefix(replace(prefix, flags=changed_flags), r.hand)
            np.testing.assert_allclose(changed["observation_flag_probability"], issued["observation_flag_probability"], rtol=0, atol=1.e-7)
            flag_checks += 1
            checked.append(dict(model=model, sample=r["sample"], hand=r.hand, origin=int(r.origin), env=r.env, split=r.split))
        maxima[model] = maximum
        example = selected.iloc[0]
        # Off-grid issue with eight missing input frames, then a fully observed public-API issue.
        partial = load_record(example.dataset, example["sample"], 48)
        missing = partial.joints.copy()
        missing[-8:, SIDES.index(example.hand)] = np.nan
        flags = partial.flags.copy()
        flags[-8:, SIDES.index(example.hand)] = 1
        damaged = system.issue_prefix(replace(partial, joints=missing, flags=flags), example.hand)
        assert damaged["origin"] == 47 and np.isfinite(damaged["center_world"]).all()
        assert np.all(np.asarray(damaged["radius_m"]) > 0)
        paths = sample_paths(damaged, 7, seed=553)
        assert paths.shape == (7, 30, 3) and np.isfinite(paths).all()
        np.testing.assert_array_equal(paths, sample_paths(damaged, 7, seed=553))
        short = system.issue(example["sample"], example.hand, 10)
        assert short["last_complete_observation"] is None
        assert short["model"] == model and short["point_model"] == archive
        assert short["last_read_frame"] == 10 and len(short["radius_m"]) == 30
        assert np.isfinite(short["future_time_s"]).all() and np.all(np.diff(short["future_time_s"]) > 0)

    assert selected.env.nunique() == all_rows.env.nunique() == 8 and set(selected.hand) == set(SIDES)
    assert sources == set(all_rows.dataset)
    fits = [read(OUT/f"{kind}_{seed}_fit.json") for kind in ("M1","M2","M2_uniform","M3") for seed in SEEDS]
    assert all(x["prior_weights_read"] == 0 for x in fits)
    signatures = {f"{x['kind']}/{x['seed']}":x["initial_parameter_hash"] for x in fits}
    assert len(set(signatures.values())) == len(fits)
    summary = dict(independent_initializations=signatures, official_datasets_checked=sorted(sources),
                   archived_and_new_scores_recomputed_from_paired_windows=True,
                   environment_count=selected.env.nunique(), prefix_comparisons=len(checked),
                   max_absolute_difference=maxima, supplied_flag_changes_without_quality_logit_change=flag_checks,
                   future_feature_comparisons_pass=True, unavailable_feedback_checks_pass=True,
                   off_grid_missing_observation_forecast_pass=True, predictive_draws_reproducible=True,
                   checks=checked)
    save(OUT/"online_parity.json", summary)
    print(f"Verified {len(checked)} raw-prefix predictions across eight environments; max differences {maxima}", flush=True)


if __name__ == "__main__":
    auditor = ReadAudit()
    with auditor.during("online_replay"):
        verify()
    save(OUT/"online_file_reads.json", auditor.result())
