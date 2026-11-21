from config import make_dirs, save

def register():
    make_dirs()
    save("protocol.json",dict(
        priority=["pooled mean 30-frame Euclidean error on all genuine +10-frame turns of the non-overlapping grid",
                  "95% simultaneous coverage of all 30 three-dimensional positions"],
        groups="future turning labels used for training supervision, validation subgroup scores and retrospective evaluation only",
        training="10 expert-fit, 3 gate/scale, 3 outer-calibration environments from official training; 4 validation, 4 retrospective test",
        checkpoint_policy="fresh reference architecture, fresh turn adapters and fresh calibration; no earlier checkpoint loaded",
        point_candidates=["no correction","reduced original-direction extrapolation","turn-specialized robust vector regression",
                          "training turn analogues in a wrist-velocity coordinate frame","combined corrections"],
        point_selection="Validation weighted 0.5 spaced pooled turn ADE + 0.25 dense pooled turn ADE + 0.25 spaced environment-macro turn ADE. "
                        "Validation overall ADE <= reference +0.05 cm and FDE <= reference +0.20 cm. "
                        "Gate-training spaced turn ADE <= reference +0.10 cm.",
        uncertainty="A score is the maximum of the 30 Euclidean errors divided by their issue-time scale. Candidates use static, "
                    "delayed endpoint, completed full-path residual feedback, or an envelope. "
                    "Outer quantiles only from three training calibration environments. Use instantaneous and resolved-prefix data only.",
        coverage_selection="Smallest validation mean radius passing each condition: pooled one-sided 90% Wilson lower bound >= .95, "
                           "each validation environment empirical coverage >=.95, and true-turn subgroup empirical coverage >=.95. "
                           "If no candidate passes all three, minimize the maximum coverage shortfall then mean radius. "
                           "Wilson is a robustness screen; intervals from the same environment are dependent.",
        comparison="All 99 test turns (75 onsets, 24 already turning), all 760 test windows, 4 test environments; "
                   "also validation 75/401, mean radius, FDE and ADE on all eligible origins. No retrospective switching.",
        older_sensitivity="The pre-existing v2 endpoint-feedback alternative reached 95.13% on test but 93.27% on validation; these historical tests are known."
    ))

if __name__=="__main__":
    register()
