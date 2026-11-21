"""Reproduce the independent wrist models, freeze before test, then score against archived result tables."""
from importlib.metadata import version
from platform import python_version

from study_paths import OUT, ROOT, save
from provenance import ReadAudit


def run():
    audit = ReadAudit()
    with audit.during("decode"):
        from build_observations import prepare
        prepare()
    with audit.during("random_fit"):
        from fit_wrist import train
        train()
    with audit.during("freeze"):
        from lock_candidates import freeze
        freeze()
    previous = ROOT/"分析报告"/"第一问质量感知LSTTN与阶段概率预测"
    comparator_tables = [previous/name for name in ("comparison.csv","current_kalman_anchor.csv","paired_windows.csv.gz")]
    with audit.during("score", comparator_tables):
        from heldout_score import score
        score()
    with audit.during("online_replay"):
        from verify_independence import verify
        verify()
    save(OUT/"run_file_reads.json", audit.result())
    save(OUT/"runtime.json", {"python":python_version(), **{p:version(p) for p in
        ("numpy","pandas","pyarrow","scipy","scikit-learn","torch")}})
    from report_results import report
    report()


if __name__ == "__main__":
    run()
