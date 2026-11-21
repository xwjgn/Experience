"""Reproduce the complete wrist-origin refinement. Use --replay-only with this study's frozen models."""
import argparse
from wrist_history import protect_weights


def main(replay_only=False):
    protect_weights()
    if not replay_only:
        from fit_stage import fit
        from freeze_choices import freeze
        from path_expert import freeze_path
        fit()
        freeze()
        freeze_path()
    from score_replay import score
    from verify_refinement import verify
    from report_results import report
    score()
    verify()
    report()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-only",action="store_true")
    main(parser.parse_args().replay_only)
