"""Train from scratch, freeze and compare. --replay-only uses just this study's saved checkpoints."""
import argparse
from gain_records import dataset,isolate


def main(replay_only=False):
    isolate()
    if not replay_only:
        from comparison_reference import fit_reference
        from freeze_gain import freeze
        from train_gain import fit_networks
        train=dataset("train")
        fit_networks(train)
        reference=fit_reference(train)
        freeze(train,reference)
        del train,reference
    from score_gain import score
    from verify_gain import verify
    from bayesian_report import report
    score()
    verify()
    report()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-only",action="store_true")
    main(parser.parse_args().replay_only)
