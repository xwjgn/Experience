"""Recompute the public wrist-forecast comparison from per-window errors, using only Python's standard library."""
import argparse
import csv
import io
import math
import sys
from pathlib import Path
from statistics import fmean

RESULTS = Path(__file__).resolve().parent / "results"
METHODS = ("hold", "archived_original", "m4", "fusion_reference", "fusion_covariance")
FIELDS = ("method", "windows", "turn_windows", "ade_cm", "turn_ade_cm", "mean_radius_cm",
          "covered", "coverage_pct", "turn_covered", "turn_coverage_pct")


def load_windows(path):
    windows = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            paired = windows.setdefault(row["window_id"], {})
            if method not in METHODS or method in paired:
                raise ValueError(f"Unknown or duplicate method for {row['window_id']}: {method}")
            if row["turn"] not in ("0", "1") or not row["environment_group"]:
                raise ValueError("Missing evaluation group")
            if method == "hold":
                if row["whole_path_covered"] or row["mean_radius_cm"]:
                    raise ValueError("The persistence baseline has no calibrated radius")
            elif row["whole_path_covered"] not in ("0", "1") or not math.isfinite(float(row["mean_radius_cm"])) or float(row["mean_radius_cm"]) <= 0:
                raise ValueError("Invalid coverage or radius")
            if not math.isfinite(float(row["ade_cm"])) or float(row["ade_cm"]) < 0:
                raise ValueError("Invalid 3-D displacement error")
            paired[method] = row
    for key, paired in windows.items():
        groups = {(row["environment_group"], row["turn"]) for row in paired.values()}
        if set(paired) != set(METHODS) or len(groups) != 1:
            raise ValueError(f"Unpaired methods or inconsistent labels at {key}")
    if not windows:
        raise ValueError("Empty comparison")
    return windows


def comparison(windows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    for method in METHODS:
        rows = [window[method] for window in windows.values()]
        turns = [row for row in rows if row["turn"] == "1"]
        def average(items, name):
            return f"{fmean(float(row[name]) for row in items):.6f}" if items else ""
        result = dict(method=method, windows=len(rows), turn_windows=len(turns),
                      ade_cm=average(rows, "ade_cm"), turn_ade_cm=average(turns, "ade_cm"))
        if method != "hold":
            covered = sum(int(row["whole_path_covered"]) for row in rows)
            turn_covered = sum(int(row["whole_path_covered"]) for row in turns)
            result.update(mean_radius_cm=average(rows, "mean_radius_cm"),
                          covered=covered, coverage_pct=f"{100 * covered / len(rows):.6f}",
                          turn_covered=turn_covered,
                          turn_coverage_pct=f"{100 * turn_covered / len(turns):.6f}" if turns else "")
        writer.writerow(result)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare the recomputed table with results/summary.csv.")
    args = parser.parse_args()
    windows = load_windows(RESULTS / "paired_windows.csv")
    result = comparison(windows)
    if args.check:
        if (RESULTS / "summary.csv").read_text(encoding="utf-8") != result:
            raise SystemExit("Stored summary differs from the paired-window calculation")
        print(f"Matched {len(METHODS)} methods on {len(windows)} windows.")
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
