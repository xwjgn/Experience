#!/usr/bin/env python3
"""Read the three raw archives into one canonical per-frame table (needs pyarrow).

Sources, read directly from the ZIP files without extracting videos:

* EgoDemo.zip: LeRobot v3 layout. data/chunk-000/file-000.parquet holds per-frame
  observation.state.hand_{left,right}_world [21, 3] and *_rotation [21, 4]
  (component names declared in meta/info.json), head pose and `timestamp` (s);
  bad_frame_ratio.json lists per-frame left/right bad reasons; the record's
  episode_uuid is in meta/episodes/chunk-000/file-000.parquet.
* EgoPro.zip, EgoStandard.zip: MCAP topics exported to CSV. tables/pose_{left,
  right}_hand.csv hold transforms[0..20].pos.{x,y,z} / quat.{w,x,y,z};
  `_offset_from_first_log_s` is the record time (s);
  annotation_bad_frame_pose_hand.csv has one row per pose message with
  problem_type[i] = "<hand>:<reason>"; episode_uuid is task_info.episode_uuid in
  session_metadata.csv. The folder name equals the `sample` column of
  source_segment_index.csv.

Joint order j00..j20 follows the names declared in EgoDemo meta/info.json
(Wrist, Thumb MCP/PIP/DIP/Tip, Index ..., Little ...). The CSV sources do not
name their transforms; the same order is supported by matching median bone
lengths (see adapter_report.json), not by a field dictionary.

Coordinates are metres in each record's own world frame. Frames of different
records do not share an origin or a store layout, so the frame is tagged per
record. Quality is copied as reported ("ok", "occlusion", "near edge"); no
numerical quality score is invented and no values are filled or smoothed.

Outputs (outside the data folder):
  frames.parquet   one row per record frame, all joints, wrist/head poses, quality
  wrist_tracks.csv one row per record frame, wrist positions and quality only
  records.csv      per-record provenance (archive member CRC32, counts, split)
  segments.csv     the 349 source segments mapped to frame ranges and prefix ends
  field_map.json   source file/field, unit and transform for every output column
  adapter_report.json / adapter_report.md  validation against the published counts
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ADAPTER_VERSION = "1.1.0"  # 1.1: head-left camera pose and intrinsics
TASK_DIR = "Restocking_Tasks_and_Constraints"
DATASETS = ("EgoDemo", "EgoPro", "EgoStandard")
# Published in the problem statement, Table 1.
EXPECTED_FRAMES = {"EgoDemo": 16414, "EgoPro": 17642, "EgoStandard": 32014}
EXPECTED_RECORDS = {"EgoDemo": 31, "EgoPro": 14, "EgoStandard": 24}
JOINTS = ("Wrist", "ThumbMCP", "ThumbPIP", "ThumbDIP", "ThumbTip", "IndexMCP", "IndexPIP", "IndexDIP",
          "IndexTip", "MiddleMCP", "MiddlePIP", "MiddleDIP", "MiddleTip", "RingMCP", "RingPIP", "RingDIP",
          "RingTip", "LittleMCP", "LittlePIP", "LittleDIP", "LittleTip")
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
PREFIX_SECONDS = (0.2, 0.6, 1.0)
HANDS = ("left", "right")


def _read_dicts(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _csv_rows(archive: zipfile.ZipFile, member: str) -> list[list[str]]:
    with archive.open(member) as stream:
        return list(csv.reader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")))


def _member(archive: zipfile.ZipFile, name: str) -> dict:
    info = archive.getinfo(name)
    return {"member": name, "crc32": f"{info.CRC:08x}", "bytes": info.file_size}


def _quat_wxyz(values, names) -> tuple:
    lookup = dict(zip(names, values))
    return tuple(float(lookup[k]) for k in ("qw", "qx", "qy", "qz"))


def _empty_frame(n: int) -> dict:
    columns = {"frame_index": [], "t_s": []}
    for hand in HANDS:
        for j in range(21):
            for axis in "xyz":
                columns[f"{hand}_j{j:02d}_{axis}"] = []
        for comp in "wxyz":
            columns[f"{hand}_wrist_q{comp}"] = []
        columns[f"{hand}_quality"] = []
    for axis in "xyz":
        columns[f"head_{axis}"] = []
    for comp in "wxyz":
        columns[f"head_q{comp}"] = []
    for key in CAMERA_COLUMNS:
        columns[key] = []
    return columns


# Head-left camera: per-frame pose (camera -> world, OpenCV axes: x right, y down,
# z forward; checked by projecting ok-frame wrists, 99-100 % fall inside the image)
# and per-record pinhole intrinsics. NaN when a source has no camera stream.
CAMERA_COLUMNS = ("cam_x", "cam_y", "cam_z", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
                  "cam_fx", "cam_fy", "cam_cx", "cam_cy", "cam_width", "cam_height")


def read_egodemo(archive: zipfile.ZipFile, sample: str) -> tuple[dict, dict]:
    base = f"EgoDemo/{sample}/"
    info = json.loads(archive.read(base + "meta/info.json"))
    features = info["features"]
    data_members = sorted(n for n in archive.namelist() if n.startswith(base + "data/") and n.endswith(".parquet"))
    if len(data_members) != 1:
        raise ValueError(f"{sample}: expected one data parquet, found {data_members}")
    table = pq.read_table(io.BytesIO(archive.read(data_members[0])))
    episode = pq.read_table(io.BytesIO(archive.read(base + "meta/episodes/chunk-000/file-000.parquet"))).to_pylist()
    if len(episode) != 1:
        raise ValueError(f"{sample}: expected one episode row")
    bad = json.loads(archive.read(base + "bad_frame_ratio.json"))
    reasons = {(d["frame"], hand): d.get(f"{hand}_bad_reason") for d in bad["hand_bad_frame_details"] for hand in HANDS}
    names = {}
    for hand in HANDS:
        joint_names = features[f"observation.state.hand_{hand}_world"]["names"][0]
        if [n[len(hand):] for n in joint_names] != list(JOINTS):
            raise ValueError(f"{sample}: unexpected {hand} joint names {joint_names}")
        names[hand] = features[f"observation.state.hand_{hand}_world_rotation"]["names"][-1]
    head_rot_names = features["observation.state.head_world_rotation"]["names"][-1]
    cols = _empty_frame(table.num_rows)
    data = {name: table.column(name).to_pylist() for name in (
        "timestamp", "frame_index", "observation.state.hand_left_world", "observation.state.hand_right_world",
        "observation.state.hand_left_world_rotation", "observation.state.hand_right_world_rotation",
        "observation.state.head_world", "observation.state.head_world_rotation")}
    cam_pos_key, cam_rot_key = "observation.state.head_left_camera_position", "observation.state.head_left_camera_rotation"
    has_camera = cam_pos_key in table.column_names and cam_rot_key in table.column_names
    if has_camera:
        cam_pos, cam_rot = table.column(cam_pos_key).to_pylist(), table.column(cam_rot_key).to_pylist()
        cam_rot_names = features[cam_rot_key]["names"]
        intr = episode[0].get("camera_intrinsics/head_left") or [math.nan] * 4
        shape = features.get("observation.images.head_left", {}).get("shape", [math.nan, math.nan])
        intrinsics = (float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3]), float(shape[1]), float(shape[0]))
    for i in range(table.num_rows):
        frame = data["frame_index"][i]
        cols["frame_index"].append(frame)
        cols["t_s"].append(float(data["timestamp"][i]))
        for hand in HANDS:
            points = data[f"observation.state.hand_{hand}_world"][i]
            for j, point in enumerate(points):
                for axis, value in zip("xyz", point):
                    cols[f"{hand}_j{j:02d}_{axis}"].append(float(value))
            for comp, value in zip("wxyz", _quat_wxyz(data[f"observation.state.hand_{hand}_world_rotation"][i][0], names[hand])):
                cols[f"{hand}_wrist_q{comp}"].append(value)
            cols[f"{hand}_quality"].append(reasons.get((frame, hand)) or "ok")
        for axis, value in zip("xyz", data["observation.state.head_world"][i][0]):
            cols[f"head_{axis}"].append(float(value))
        for comp, value in zip("wxyz", _quat_wxyz(data["observation.state.head_world_rotation"][i][0], head_rot_names)):
            cols[f"head_q{comp}"].append(value)
        if has_camera:
            camera = list(map(float, cam_pos[i])) + list(_quat_wxyz(cam_rot[i], cam_rot_names)) + list(intrinsics)
        else:
            camera = [math.nan] * len(CAMERA_COLUMNS)
        for key, value in zip(CAMERA_COLUMNS, camera):
            cols[key].append(value)
    unmatched = [d["frame"] for d in bad["hand_bad_frame_details"] if d["frame"] not in set(cols["frame_index"])]
    meta = {"episode_uuid": episode[0]["episode_uuid"], "environment_id": episode[0].get("environment_id"),
            "declared_fps": info["fps"], "bad_frames_not_in_data": len(unmatched),
            "frames_without_camera": 0 if has_camera else table.num_rows,
            "sources": [_member(archive, data_members[0]), _member(archive, base + "bad_frame_ratio.json"),
                        _member(archive, base + "meta/episodes/chunk-000/file-000.parquet")]}
    return cols, meta


def read_csv_source(archive: zipfile.ZipFile, dataset: str, sample: str) -> tuple[dict, dict]:
    base = f"{dataset}/{sample}/"
    session = {r[0]: r[1] for r in _csv_rows(archive, base + "session_metadata.csv")[1:] if len(r) >= 2}
    hands = {hand: _csv_rows(archive, base + f"tables/pose_{hand}_hand.csv") for hand in HANDS}
    head = _csv_rows(archive, base + "tables/pose_head.csv")
    bad = _csv_rows(archive, base + "tables/annotation_bad_frame_pose_hand.csv")
    n = len(hands["left"]) - 1
    if len(hands["right"]) - 1 != n or len(bad) - 1 != n:
        raise ValueError(f"{sample}: pose/bad-frame row counts differ")
    log_col = hands["left"][0].index("_log_time_ns")
    for name, rows in (("right", hands["right"]), ("bad", bad)):
        if [r[log_col] for r in rows[1:]] != [r[log_col] for r in hands["left"][1:]]:
            raise ValueError(f"{sample}: {name} log times do not match left-hand log times")
    head_index = {r[log_col]: r for r in head[1:]}
    head_header = head[0]
    names = set(archive.namelist())
    cam_member, intr_member = base + "tables/pose_headcam.csv", base + "tables/sensor_camera_head_left_intrinsic.csv"
    cam_index, cam_cols, intrinsics = {}, None, (math.nan,) * 6
    if cam_member in names and intr_member in names:
        cam_rows = _csv_rows(archive, cam_member)
        ch = cam_rows[0]
        cam_cols = [ch.index(f"transforms[0].pos.{a}") for a in "xyz"] + [ch.index(f"transforms[0].quat.{c}") for c in "wxyz"]
        cam_index = {r[ch.index("_log_time_ns")]: r for r in cam_rows[1:]}
        intr_rows = _csv_rows(archive, intr_member)
        ih = intr_rows[0]
        def med(key):
            return statistics.median(float(r[ih.index(key)]) for r in intr_rows[1:])
        intrinsics = (med("K[0]"), med("K[4]"), med("K[2]"), med("K[5]"), med("width"), med("height"))
    offset_col = hands["left"][0].index("_offset_from_first_log_s")
    cols = _empty_frame(n)
    index = {}
    for hand in HANDS:
        header = hands[hand][0]
        if int(hands[hand][1][header.index("transforms.length")]) != 21:
            raise ValueError(f"{sample}: {hand} hand does not have 21 transforms")
        index[hand] = ([[header.index(f"transforms[{j}].pos.{a}") for a in "xyz"] for j in range(21)],
                       [header.index(f"transforms[0].quat.{c}") for c in "wxyz"])
    bad_header = bad[0]
    type_cols = [i for i, h in enumerate(bad_header) if h.startswith("problem_type[")]
    is_bad_col = bad_header.index("is_bad")
    head_pos = [head_header.index(f"transforms[0].pos.{a}") for a in "xyz"]
    head_quat = [head_header.index(f"transforms[0].quat.{c}") for c in "wxyz"]
    missing_head = missing_camera = 0
    for i in range(1, n + 1):
        cols["frame_index"].append(i - 1)
        cols["t_s"].append(float(hands["left"][i][offset_col]))
        flags = {}
        if bad[i][is_bad_col] == "True":
            for c in type_cols:
                if c < len(bad[i]) and bad[i][c]:
                    side, _, reason = bad[i][c].partition(":")
                    flags[side.strip()] = reason.strip()
        for hand in HANDS:
            row = hands[hand][i]
            positions, quat = index[hand]
            for j, triple in enumerate(positions):
                for axis, c in zip("xyz", triple):
                    cols[f"{hand}_j{j:02d}_{axis}"].append(float(row[c]))
            for comp, c in zip("wxyz", quat):
                cols[f"{hand}_wrist_q{comp}"].append(float(row[c]))
            cols[f"{hand}_quality"].append(flags.get(hand, "ok"))
        head_row = head_index.get(hands["left"][i][log_col])
        if head_row is None:
            missing_head += 1
        for key, idx in list(zip(("head_x", "head_y", "head_z"), head_pos)) + list(zip(("head_qw", "head_qx", "head_qy", "head_qz"), head_quat)):
            cols[key].append(float(head_row[idx]) if head_row else math.nan)
        cam_row = cam_index.get(hands["left"][i][log_col])
        if cam_row is None:
            missing_camera += 1
        camera = ([float(cam_row[c]) for c in cam_cols] if cam_row else [math.nan] * 7) + list(intrinsics)
        for key, value in zip(CAMERA_COLUMNS, camera):
            cols[key].append(value)
    meta = {"episode_uuid": session.get("task_info.episode_uuid"), "environment_id": session.get("task_info.environment_id"),
            "declared_fps": float(session.get("devices[0].fps") or "nan"), "frames_without_head_pose": missing_head,
            "frames_without_camera": missing_camera,
            "sources": [_member(archive, base + f"tables/pose_{h}_hand.csv") for h in HANDS]
                       + [_member(archive, base + "tables/annotation_bad_frame_pose_hand.csv"),
                          _member(archive, base + "tables/pose_head.csv"), _member(archive, base + "session_metadata.csv")]}
    return cols, meta


def _bone_medians(cols: dict, hand: str, step: int = 10) -> list[float]:
    out = []
    for a, b in BONES:
        lengths = [math.dist([cols[f"{hand}_j{a:02d}_{x}"][i] for x in "xyz"], [cols[f"{hand}_j{b:02d}_{x}"][i] for x in "xyz"])
                   for i in range(0, len(cols["t_s"]), step)]
        out.append(statistics.median(lengths))
    return out


def convert(data: str | Path, out: str | Path) -> dict:
    root = Path(data)
    root = root if (root / "Raw_Data").is_dir() else root.parent
    out = Path(out)
    if out.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must be outside the data folder")
    out.mkdir(parents=True, exist_ok=True)
    split_by_uuid = {r["episode_uuid"]: r for r in _read_dicts(root / TASK_DIR / "record_split.csv")}
    segments = _read_dicts(root / TASK_DIR / "source_segment_index.csv")
    sample_uuid = {}
    for s in segments:
        sample_uuid.setdefault((s["dataset"], s["sample"]), set()).add(s["episode_uuid"])
    writer = None
    records, issues, bone_table, frames_by_source = [], [], {}, Counter()
    wrist_rows = []
    times: dict = {}
    try:
        for dataset in DATASETS:
            with zipfile.ZipFile(root / "Raw_Data" / f"{dataset}.zip") as archive:
                samples = sorted({n.split("/")[1] for n in archive.namelist() if n.count("/") >= 2})
                loaded = [(sample, read_egodemo(archive, sample) if dataset == "EgoDemo"
                           else read_csv_source(archive, dataset, sample)) for sample in samples]
            for sample, (cols, meta) in loaded:
                uuid = meta["episode_uuid"]
                split = split_by_uuid.get(uuid)
                if split is None:
                    issues.append({"severity": "error", "record": sample, "issue": "episode_uuid not in record_split.csv"})
                    continue
                if meta["environment_id"] and meta["environment_id"] != split["environment_id"]:
                    issues.append({"severity": "error", "record": sample, "issue": "environment differs from record_split.csv"})
                if sample_uuid.get((dataset, sample), {uuid}) != {uuid}:
                    issues.append({"severity": "error", "record": sample, "issue": "sample maps to another episode_uuid in source_segment_index.csv"})
                t = cols["t_s"]
                steps = [b - a for a, b in zip(t, t[1:])]
                nonincreasing = sum(1 for s in steps if s <= 0)
                nonfinite = sum(1 for k, v in cols.items() if k not in ("left_quality", "right_quality")
                                and k not in CAMERA_COLUMNS
                                for x in v if isinstance(x, float) and not math.isfinite(x))
                if meta.get("frames_without_camera"):
                    issues.append({"severity": "warning", "record": sample,
                                   "issue": f"{meta['frames_without_camera']} frames without head-left camera pose"})
                if nonincreasing:
                    issues.append({"severity": "warning", "record": sample, "issue": f"{nonincreasing} non-increasing time steps"})
                if nonfinite:
                    issues.append({"severity": "warning", "record": sample, "issue": f"{nonfinite} non-finite numeric values"})
                for hand in HANDS:
                    bone_table.setdefault(dataset, []).append(_bone_medians(cols, hand))
                n = len(t)
                frames_by_source[dataset] += n
                times[uuid] = t
                quality = {hand: Counter(cols[f"{hand}_quality"]) for hand in HANDS}
                records.append({
                    "dataset": dataset, "sample": sample, "episode_uuid": uuid, "environment_id": split["environment_id"],
                    "scene_id": split["scene_id"], "operator_id": split["operator_id"], "split": split["split"],
                    "frames": n, "duration_s": t[-1] - t[0] if n else 0.0, "median_dt_s": statistics.median(steps) if steps else None,
                    "max_gap_s": max(steps) if steps else None, "declared_fps": meta["declared_fps"],
                    "nonincreasing_steps": nonincreasing,
                    "left_ok": quality["left"]["ok"], "left_occlusion": quality["left"]["occlusion"], "left_near_edge": quality["left"]["near edge"],
                    "right_ok": quality["right"]["ok"], "right_occlusion": quality["right"]["occlusion"], "right_near_edge": quality["right"]["near edge"],
                    "coordinate_frame": f"{dataset}:{sample}:world", "adapter_version": ADAPTER_VERSION,
                    "source_members": json.dumps(meta["sources"], ensure_ascii=False),
                })
                ident = {"dataset": [dataset] * n, "sample": [sample] * n, "episode_uuid": [uuid] * n,
                         "environment_id": [split["environment_id"]] * n, "split": [split["split"]] * n,
                         "coordinate_frame": [f"{dataset}:{sample}:world"] * n}
                def _typed(key, values):
                    if key.endswith("_quality") or key == "frame_index":
                        return pa.array(values)
                    return pa.array(values, pa.float64() if key == "t_s" else pa.float32())
                batch = pa.table({**ident, **{k: _typed(k, v) for k, v in cols.items()}})
                if writer is None:
                    writer = pq.ParquetWriter(out / "frames.parquet", batch.schema, compression="zstd")
                writer.write_table(batch)
                for i in range(n):
                    wrist_rows.append([dataset, sample, uuid, split["split"], cols["frame_index"][i], f"{t[i]:.6f}"]
                                      + [f"{cols[f'{h}_j00_{a}'][i]:.6f}" for h in HANDS for a in "xyz"]
                                      + [cols["left_quality"][i], cols["right_quality"][i]])
    finally:
        if writer is not None:
            writer.close()
    with (out / "wrist_tracks.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["dataset", "sample", "episode_uuid", "split", "frame_index", "t_s", "left_wrist_x", "left_wrist_y",
                    "left_wrist_z", "right_wrist_x", "right_wrist_y", "right_wrist_z", "left_quality", "right_quality"])
        w.writerows(wrist_rows)
    with (out / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    # Map every source segment to record frames and to the 0.2/0.6/1.0 s prefix ends.
    seg_rows, seg_problems = [], 0
    for s in segments:
        t = times.get(s["episode_uuid"])
        start, end = float(s["start_s"]), float(s["end_s"])
        row = {k: s[k] for k in ("unit_id", "dataset", "sample", "episode_uuid", "split", "family_id",
                                 "source_segment_index", "start_s", "end_s", "duration_s", "eligible")}
        if t is None:
            seg_problems += 1
            row.update(first_frame="", end_frame="", frames="", **{f"prefix_{p}s_end_frame": "" for p in PREFIX_SECONDS})
        else:
            import bisect
            first = bisect.bisect_left(t, start - 1e-6)
            last = bisect.bisect_left(t, end - 1e-6)
            row.update(first_frame=first, end_frame=last, frames=last - first,
                       **{f"prefix_{p}s_end_frame": bisect.bisect_left(t, start + p - 1e-6) for p in PREFIX_SECONDS})
            if end > t[-1] + 0.1 or last <= first:
                seg_problems += 1
                issues.append({"severity": "warning", "record": s["sample"],
                               "issue": f"segment {s['source_segment_index']} [{start}, {end}] outside record time 0..{t[-1]:.2f}"})
        seg_rows.append(row)
    with (out / "segments.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=list(seg_rows[0]))
        w.writeheader()
        w.writerows(seg_rows)
    bones = {ds: [round(statistics.median(v[k] for v in rows) * 100, 2) for k in range(len(BONES))]
             for ds, rows in bone_table.items()}
    reference = bones.get("EgoDemo")
    bone_gap = {ds: round(max(abs(a - b) for a, b in zip(v, reference)), 2) for ds, v in bones.items()} if reference else {}
    if any(g > 1.0 for g in bone_gap.values()):
        issues.append({"severity": "error", "record": "*", "issue": f"bone-length profiles differ by > 1 cm: {bone_gap}"})
    counts_ok = dict(frames_by_source) == EXPECTED_FRAMES
    record_counts = Counter(r["dataset"] for r in records)
    report = {
        "adapter_version": ADAPTER_VERSION, "data_root": str(root),
        "frames_by_source": dict(frames_by_source), "expected_frames_table1": EXPECTED_FRAMES,
        "frame_counts_match_table1": counts_ok,
        "records_by_source": dict(record_counts), "expected_records": EXPECTED_RECORDS,
        "records_match": dict(record_counts) == EXPECTED_RECORDS,
        "records_by_split": dict(Counter(r["split"] for r in records)),
        "quality_frames": {f"{h}_{q}": sum(r[f"{h}_{q}"] for r in records) for h in HANDS for q in ("ok", "occlusion", "near_edge")},
        "segments": len(seg_rows), "segments_not_mapped": seg_problems,
        "median_bone_length_cm": bones, "max_bone_profile_gap_vs_egodemo_cm": bone_gap, "bones": BONES,
        "issues": issues, "errors": sum(i["severity"] == "error" for i in issues),
    }
    (out / "adapter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "field_map.json").write_text(json.dumps(field_map(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "adapter_report.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def field_map() -> dict:
    demo = "EgoDemo/<sample>/"
    csvsrc = "<EgoPro|EgoStandard>/<sample>/tables/"
    return {
        "adapter_version": ADAPTER_VERSION,
        "joint_order": {f"j{j:02d}": name for j, name in enumerate(JOINTS)},
        "joint_order_basis": "Names declared in EgoDemo meta/info.json; CSV transforms[0..20] order supported by matching bone lengths.",
        "columns": {
            "t_s": {"unit": "s", "EgoDemo": f"{demo}data/chunk-000/file-000.parquet:timestamp",
                    "csv": f"{csvsrc}pose_left_hand.csv:_offset_from_first_log_s", "transform": "none"},
            "frame_index": {"EgoDemo": "frame_index", "csv": "row order of pose_left_hand.csv (0-based)"},
            "<hand>_jNN_<x|y|z>": {"unit": "m", "EgoDemo": "observation.state.hand_<hand>_world[NN]",
                                   "csv": "pose_<hand>_hand.csv:transforms[NN].pos.<x|y|z>", "transform": "float32 copy"},
            "<hand>_wrist_q<w|x|y|z>": {"EgoDemo": "observation.state.hand_<hand>_world_rotation[0] reordered by declared names",
                                        "csv": "pose_<hand>_hand.csv:transforms[0].quat.<w|x|y|z>", "transform": "reorder to w,x,y,z"},
            "head_<x|y|z>, head_q<w|x|y|z>": {"unit": "m", "EgoDemo": "observation.state.head_world, head_world_rotation",
                                              "csv": "pose_head.csv:transforms[0], matched by _log_time_ns", "transform": "NaN if no matching row"},
            "<hand>_quality": {"EgoDemo": f"{demo}bad_frame_ratio.json:hand_bad_frame_details[frame].<hand>_bad_reason",
                               "csv": "annotation_bad_frame_pose_hand.csv:problem_type[i] = '<hand>:<reason>' when is_bad",
                               "transform": "'ok' when no reason is reported"},
            "coordinate_frame": {"transform": "record-specific world frame; not shared between records"},
            "cam_x..cam_qz": {"unit": "m", "EgoDemo": "observation.state.head_left_camera_position / _rotation (names declared)", "csv": "pose_headcam.csv:transforms[0], matched by _log_time_ns", "transform": "camera -> world; OpenCV axes (x right, y down, z forward)"},
            "cam_fx, cam_fy, cam_cx, cam_cy, cam_width, cam_height": {"unit": "px", "EgoDemo": "meta/episodes camera_intrinsics/head_left and images.head_left shape", "csv": "sensor_camera_head_left_intrinsic.csv: K[0], K[4], K[2], K[5], width, height (record median)", "transform": "pinhole; zero distortion as published"},
        },
    }


def report_markdown(report: dict) -> str:
    lines = ["# 原始轨迹适配报告", "",
             f"适配器版本 {report['adapter_version']}。原始压缩包只读，未解压视频。", "",
             "| 数据源 | 记录数 | 预期 | 帧数 | 题面表 1 |", "|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        lines.append(f"| {ds} | {report['records_by_source'].get(ds, 0)} | {report['expected_records'][ds]} | "
                     f"{report['frames_by_source'].get(ds, 0)} | {report['expected_frames_table1'][ds]} |")
    lines += ["", f"帧数与题面表 1 一致：{'是' if report['frame_counts_match_table1'] else '否'}；记录数一致：{'是' if report['records_match'] else '否'}。",
              f"划分：{report['records_by_split']}。", "",
              "质量标记帧数（按手）：" + "，".join(f"{k} {v}" for k, v in report["quality_frames"].items()) + "。", "",
              f"349 个来源片段中未能映射到帧的：{report['segments_not_mapped']}。", "",
              "关节顺序核对：各来源中位骨段长度与 EgoDemo 的最大差（cm）：" + json.dumps(report["max_bone_profile_gap_vs_egodemo_cm"]) + "。", "",
              f"错误 {report['errors']} 条，提示 {len(report['issues']) - report['errors']} 条。"]
    for issue in report["issues"][:50]:
        lines.append(f"- {issue['severity']} / {issue['record']}：{issue['issue']}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=here.parent / "Restocking_Problem_and_Data_EN")
    parser.add_argument("--out", type=Path, default=here / "cache" / "canonical")
    args = parser.parse_args(argv)
    report = convert(args.data, args.out)
    print(f"Frames {report['frames_by_source']} match Table 1: {report['frame_counts_match_table1']}; "
          f"records match: {report['records_match']}; errors: {report['errors']}")
    print(f"Output: {args.out.resolve()}")
    return 0 if report["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
