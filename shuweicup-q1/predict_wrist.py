"""Issue the fused 30-step forecast using a local competition workspace and its fitted models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


def forecast(project, sample, hand, origin):
    toolkit = project / "c_restocking_toolkit"
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}

    def run(relative, *options):
        command = [sys.executable, str(toolkit / relative), sample, hand, str(origin), *options]
        output = subprocess.check_output(command, cwd=project, env=environment, encoding="utf-8")
        return json.loads(output)

    with TemporaryDirectory(prefix="wrist-forecast-") as directory:
        m4 = run("q1_transformer_lstm/forecast.py", "--model", "M4")
        gate = run("q1_origin_study/issue_origin.py")
        m4_file = Path(directory) / "m4.json"
        m4_file.write_text(json.dumps(m4, allow_nan=False), encoding="utf-8")
        refined = run("q1_origin_refinement/refine.py", "--m4-json", str(m4_file),
                      "--previous-factor", str(gate["low_turn_radius_factor"]))
        reference = Path(directory) / "reference.json"
        reference.write_text(json.dumps(refined, allow_nan=False), encoding="utf-8")
        issued = run("q1_bayesian_gain_covariance/forecast_bayesian.py", "--reference-json", str(reference))
    return {"sample": sample, "hand": hand, "origin": origin, "coordinate_frame": issued["coordinate_frame"],
            "time_s": m4["future_time_s"], "center_m": issued["hybrid_center_world"], "radius_m": issued["hybrid_radius_m"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", help="Identifier in the local record manifest")
    parser.add_argument("hand", choices=["left", "right"])
    parser.add_argument("origin", type=int, help="Zero-based issue frame")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the toolkit, local data and fitted experiment outputs")
    args = parser.parse_args()
    print(json.dumps(forecast(args.project.resolve(), args.sample, args.hand, args.origin),
                     allow_nan=False, ensure_ascii=False))
