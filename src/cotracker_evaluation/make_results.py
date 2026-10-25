"""Summarize verified tracking results and render meeting comparison videos."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cotracker_evaluation.comparison_io import (
    DEFAULT_MANIFEST,
    SOURCE_BUNDLES,
    ffmpeg_executable,
    input_path,
    load_manifest,
    sha256_file,
    source_bundles,
    write_json_atomic,
)


CLASS_COLORS = (
    (165, 165, 165), (255, 95, 95), (255, 185, 55), (255, 90, 210),
    (170, 100, 255), (140, 210, 255), (235, 215, 110), (125, 245, 230),
    (235, 155, 110), (105, 225, 100), (120, 175, 255),
)
IDENTIFIER_FIELDS = {"class_id", "class_name", "case_id", "scene", "condition", "condition_label"}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _write_csv(path, rows, first_columns):
    columns = list(first_columns)
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _case_metadata(case):
    return {key: case[key] for key in ("case_id", "scene", "condition", "condition_label")}


def _verified_case(case, directory, fingerprint):
    """A partial tracking file alone is never evidence of a completed case."""
    marker_path = directory / "complete.json"
    if not marker_path.exists():
        failure_path = directory / "failure.json"
        detail = "tracking completion marker is missing"
        if failure_path.exists():
            failure = _read_json(failure_path)
            detail = str(failure.get("error", failure.get("message", failure)))
            return None, None, "failed", detail
        return None, None, "missing", detail
    try:
        marker = _read_json(marker_path)
        if marker.get("status") != "complete" or marker.get("case_id") != case["case_id"]:
            raise ValueError("completion identity differs")
        if marker.get("run_fingerprint") != fingerprint:
            raise ValueError("completion belongs to another tracking configuration")
        expected_inputs = {key: case[key] for key in ("video_sha256", "labels_sha256")}
        if marker.get("input_hashes") != expected_inputs:
            raise ValueError("completion input hashes differ")
        for filename in ("metrics.json", "tracks.npz"):
            if sha256_file(directory / filename) != marker["artifacts"][filename]:
                raise ValueError(f"saved {filename} hash differs")
        metrics = _read_json(directory / "metrics.json")
        if not isinstance(metrics["per_class"], list) or not isinstance(metrics["per_query"], list):
            raise ValueError("metrics must contain per-class and per-query rows")
        return metrics, marker, "complete", "verified tracking outputs"
    except (OSError, KeyError, TypeError, ValueError) as error:
        return None, None, "invalid", str(error)


def _paired_tables(cases, class_rows, classes):
    rows_by_case = {}
    numeric_fields = set()
    for row in class_rows:
        rows_by_case[(row["case_id"], row["class_id"])] = row
        for key, value in row.items():
            if key not in IDENTIFIER_FIELDS and (value is None or _number(value) is not None):
                numeric_fields.add(key)
    baselines = {case["scene"]: case for case in cases if case["condition"] == "baseline"}
    paired_rows = []
    for case in cases:
        baseline = baselines.get(case["scene"])
        for class_record in classes:
            class_id = class_record["class_id"]
            current = rows_by_case.get((case["case_id"], class_id), {})
            control = rows_by_case.get((baseline["case_id"], class_id), {}) if baseline else {}
            for metric in sorted(numeric_fields):
                value = _number(current.get(metric))
                baseline_value = _number(control.get(metric))
                supported = value is not None and baseline_value is not None
                paired_rows.append({
                    **_case_metadata(case), "class_id": class_id, "class_name": class_record["class_name"],
                    "metric": metric, "condition_value": value, "baseline_value": baseline_value,
                    "delta": value - baseline_value if supported else None,
                    "paired_available": supported,
                })
    groups = {}
    for row in paired_rows:
        key = (row["condition"], row["class_id"], row["metric"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for group in groups.values():
        paired = [row for row in group if row["paired_available"]]
        first = group[0]
        summaries.append({
            "condition": first["condition"], "condition_label": first["condition_label"],
            "class_id": first["class_id"], "class_name": first["class_name"], "metric": first["metric"],
            "condition_mean_on_paired_scenes": _mean([row["condition_value"] for row in paired]),
            "baseline_mean_on_paired_scenes": _mean([row["baseline_value"] for row in paired]),
            "mean_paired_delta": _mean([row["delta"] for row in paired]),
            "paired_scene_count": len(paired),
            "condition_scene_count": sum(row["condition_value"] is not None for row in group),
            "planned_scene_count": len(group),
        })
    return paired_rows, summaries


def _font(size):
    for filename in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(filename, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _frames(path, settings):
    """Decode one frame at a time, leaving the original geometry untouched."""
    width, height = settings["width"], settings["height"]
    command = [ffmpeg_executable(), "-v", "error", "-threads", "2", "-filter_threads", "1",
               "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = width * height * 3
    count = 0
    try:
        while True:
            data = process.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                raise ValueError("decoded RGB frame size differs from the manifest")
            count += 1
            yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        message = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg decoding failed: {message[-1000:]}")
        if count != settings["frame_count"]:
            raise ValueError(f"expected {settings['frame_count']} frames, decoded {count}")
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def _write_video(path, frames, width, height, settings):
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.mp4")
    command = [ffmpeg_executable(), "-v", "error", "-y", "-threads", "2", "-filter_threads", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(settings["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-threads", "2", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    count = 0
    try:
        for frame in frames:
            array = np.asarray(frame, dtype=np.uint8)
            if array.shape != (height, width, 3):
                raise ValueError("rendered frame has an unexpected shape")
            process.stdin.write(array.tobytes())
            count += 1
        process.stdin.close()
        message = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg encoding failed: {message[-1000:]}")
        if count != settings["frame_count"]:
            raise ValueError("rendered video did not include every source frame")
        temporary.replace(path)
    finally:
        if hasattr(frames, "close"):
            frames.close()
        if not process.stdin.closed:
            process.stdin.close()
        process.stderr.close()
        if process.poll() is None:
            process.terminate()
            process.wait()


def _overlay_frames(video, archive, case, classes, settings):
    width, height = settings["width"], settings["height"]
    header = 112
    tracks = archive["forward_tracks"]
    visibility = archive["forward_visibility"]
    queries = archive["queries"]
    query_classes = archive["query_classes"]
    anchors = queries[:, 0].astype(int)
    font = _font(18)
    for frame_index, frame in enumerate(_frames(video, settings)):
        canvas = Image.new("RGB", (width, height + header), (20, 23, 30))
        canvas.paste(Image.fromarray(frame), (0, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 7), f"{case['scene']} | {case['condition_label']} | frame {frame_index}", font=font, fill="white")
        for position, class_record in enumerate(classes):
            color = CLASS_COLORS[class_record["class_id"] % len(CLASS_COLORS)]
            column, row = position % 4, position // 4
            draw.text((12 + column * width // 4, 36 + row * 22), class_record["class_name"], font=font, fill=color)
        points = tracks[frame_index]
        valid = visibility[frame_index].astype(bool) & (anchors <= frame_index)
        valid &= np.isfinite(points).all(axis=1)
        valid &= (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
        for query_index in np.flatnonzero(valid):
            color = CLASS_COLORS[int(query_classes[query_index]) % len(CLASS_COLORS)]
            for later in range(max(anchors[query_index] + 1, frame_index - 7), frame_index + 1):
                pair = tracks[later - 1:later + 1, query_index]
                if not visibility[later - 1:later + 1, query_index].all() or not np.isfinite(pair).all():
                    continue
                if ((pair[:, 0] < 0) | (pair[:, 0] >= width) | (pair[:, 1] < 0) | (pair[:, 1] >= height)).any():
                    continue
                draw.line([(float(point[0]), float(point[1]) + header) for point in pair], fill=color, width=2)
            x, y = points[query_index]
            draw.ellipse((x - 2, y + header - 2, x + 2, y + header + 2), fill=color)
        yield canvas


def _comparison_frames(baseline_video, video, case, settings):
    width, height = settings["width"], settings["height"]
    baseline_frames = _frames(baseline_video, settings)
    condition_frames = _frames(video, settings)
    font = _font(22)
    try:
        for frame_index in range(settings["frame_count"]):
            canvas = Image.new("RGB", (width * 2, height + 64), (20, 23, 30))
            canvas.paste(Image.fromarray(next(baseline_frames)), (0, 64))
            canvas.paste(Image.fromarray(next(condition_frames)), (width, 64))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), "Base Wan", fill="white", font=font)
            draw.text((width + 12, 8), case["condition_label"], fill="white", font=font)
            draw.text((12, 37), f"{case['scene']} | frame {frame_index} | {frame_index / settings['fps']:.2f} s", fill="white", font=_font(16))
            yield canvas
        if next(baseline_frames, None) is not None or next(condition_frames, None) is not None:
            raise ValueError("comparison input contains extra frames")
    finally:
        baseline_frames.close()
        condition_frames.close()


def _render_case(case, baseline, marker, output, input_root, classes, settings,
                 allowed_bundles=SOURCE_BUNDLES):
    video = input_path(input_root, case["video_relative_path"], allowed_bundles=allowed_bundles)
    if sha256_file(video) != case["video_sha256"]:
        raise ValueError("source RGB video hash differs")
    media_directory = output / "videos"
    media_directory.mkdir(exist_ok=True)
    overlays = media_directory / f"{case['case_id']}__tracks.mp4"
    tasks = [("overlay", overlays)]
    if baseline and case["condition"] != "baseline":
        tasks.append(("comparison", media_directory / f"{case['case_id']}__vs_baseline.mp4"))
    for kind, path in tasks:
        inputs = {"kind": kind, "video_sha256": case["video_sha256"], "artifacts": marker["artifacts"],
                  "run_fingerprint": marker["run_fingerprint"], "frame_count": settings["frame_count"], "fps": settings["fps"]}
        if kind == "comparison":
            inputs["baseline_video_sha256"] = baseline["video_sha256"]
        marker_path = path.with_suffix(".json")
        if path.exists() and marker_path.exists():
            saved = _read_json(marker_path)
            if saved.get("inputs") == inputs and saved.get("sha256") == sha256_file(path):
                continue
        if kind == "overlay":
            with np.load(output / "cases" / case["case_id"] / "tracks.npz", allow_pickle=False) as archive:
                frames = _overlay_frames(video, archive, case, classes, settings)
                _write_video(path, frames, settings["width"], settings["height"] + 112, settings)
        else:
            baseline_video = input_path(input_root, baseline["video_relative_path"], allowed_bundles=allowed_bundles)
            if sha256_file(baseline_video) != baseline["video_sha256"]:
                raise ValueError("baseline RGB video hash differs")
            frames = _comparison_frames(baseline_video, video, case, settings)
            _write_video(path, frames, settings["width"] * 2, settings["height"] + 64, settings)
        write_json_atomic(marker_path, {"inputs": inputs, "sha256": sha256_file(path)})


def _summary_plots(output, summaries, classes, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    foreground = [item for item in classes if item["class_id"] != 0]
    conditions = list(dict.fromkeys(case["condition"] for case in cases if case["condition"] != "baseline"))
    if not conditions or not foreground:
        return
    labels = {case["condition"]: case["condition_label"] for case in cases}
    for metric, scale, description in (
        ("original_class_persistence", 100, "Original-class persistence change (percentage points)"),
        ("cycle_error_pixel_median", 1, "Median reverse-cycle error change (pixels)"),
    ):
        values = np.full((len(conditions), len(foreground)), np.nan)
        counts = np.zeros(values.shape, dtype=int)
        for row in summaries:
            if row["metric"] != metric or row["condition"] not in conditions:
                continue
            class_ids = [item["class_id"] for item in foreground]
            if row["class_id"] not in class_ids or row["mean_paired_delta"] is None:
                continue
            position = (conditions.index(row["condition"]), class_ids.index(row["class_id"]))
            values[position] = row["mean_paired_delta"] * scale
            counts[position] = row["paired_scene_count"]
        figure, axis = plt.subplots(figsize=(13, max(4, len(conditions) * .48 + 2)))
        finite = np.abs(values[np.isfinite(values)])
        limit = max(float(finite.max()), .001) if finite.size else 1
        palette = plt.get_cmap("coolwarm").copy()
        palette.set_bad("#e5e7eb")
        chart = axis.imshow(np.ma.masked_invalid(values), cmap=palette, vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(foreground)), [item["class_name"] for item in foreground], rotation=35, ha="right")
        axis.set_yticks(range(len(conditions)), [labels[item] for item in conditions])
        for row in range(len(conditions)):
            for column in range(len(foreground)):
                label = f"{values[row, column]:.2f}\nn={counts[row, column]}" if np.isfinite(values[row, column]) else "N/A"
                axis.text(column, row, label, ha="center", va="center", fontsize=7)
        axis.set_title(description + "\nEqual scene weighting; n = supported paired scenes; diagnostic only")
        figure.colorbar(chart, ax=axis, shrink=.8)
        figure.tight_layout()
        temporary = output / f"{metric}.tmp.{os.getpid()}.png"
        figure.savefig(temporary, dpi=140)
        plt.close(figure)
        temporary.replace(output / f"{metric}.png")


def build_results(manifest, output_dir, input_root, render_videos=True):
    """Build exploratory summaries without changing tracking or historical evidence."""
    manifest_path = Path(manifest) if isinstance(manifest, (str, Path)) else None
    manifest = load_manifest(manifest_path) if manifest_path is not None else manifest
    allowed_bundles = source_bundles(manifest)
    output = Path(output_dir).resolve()
    input_root = Path(input_root).resolve(strict=True)
    if output == input_root:
        raise ValueError("report output cannot be the historical input root")
    for name in set(SOURCE_BUNDLES).union(allowed_bundles):
        historical = (input_root / name).resolve()
        if output == historical or historical in output.parents:
            raise ValueError("report output cannot be inside a historical bundle")
    recorded_run = _read_json(output / "run_manifest.json")
    if manifest_path is not None:
        expected_hash = recorded_run.get("configuration", {}).get("manifest_sha256")
        if sha256_file(manifest_path) != expected_hash:
            raise ValueError("Report manifest differs from the verified tracking configuration.")
    fingerprint = recorded_run["run_fingerprint"]
    cases, settings = manifest["cases"], manifest["settings"]
    classes = [{"class_id": class_id, "class_name": name} for class_id, name in enumerate(manifest["classes"])]
    class_rows, query_rows, availability, verified = [], [], [], []
    expected_classes = {item["class_id"] for item in classes}
    for case in cases:
        directory = output / "cases" / case["case_id"]
        metrics, marker, status, detail = _verified_case(case, directory, fingerprint)
        if metrics is not None:
            actual_classes = [row.get("class_id") for row in metrics["per_class"]]
            if set(actual_classes) != expected_classes or len(actual_classes) != len(expected_classes):
                metrics, status, detail = None, "invalid", "per-class rows do not match every declared class"
        media_status = "pending" if render_videos and metrics is not None else "not_requested"
        availability.append({**_case_metadata(case), "status": status, "detail": detail, "media_status": media_status})
        if metrics is None:
            continue
        class_rows.extend({**row, **_case_metadata(case)} for row in metrics["per_class"])
        query_rows.extend({**row, **_case_metadata(case)} for row in metrics["per_query"])
        verified.append((case, marker, availability[-1]))
    identity = ["case_id", "scene", "condition", "condition_label", "class_id", "class_name"]
    _write_csv(output / "per_class_metrics.csv", class_rows, identity)
    _write_csv(output / "per_query_metrics.csv", query_rows, identity + ["query_index"])
    paired, summaries = _paired_tables(cases, class_rows, classes)
    _write_csv(output / "paired_scene_deltas.csv", paired, identity + ["metric", "condition_value", "baseline_value", "delta", "paired_available"])
    _write_csv(output / "per_class_summary.csv", summaries, ["condition", "condition_label", "class_id", "class_name", "metric", "mean_paired_delta", "paired_scene_count"])
    historical = [{**_case_metadata(case), **case.get("existing_metrics", {}), "source": "historical pre-MP4 RGB teacher evaluation"} for case in cases]
    _write_csv(output / "existing_rgb_metrics.csv", historical, ["case_id", "scene", "condition", "condition_label", "source"])

    # Keep condition names in an operator-only key, outside the blank review sheet.
    order = list(cases)
    random.Random(20260909).shuffle(order)
    review_rows, key_rows = [], []
    for position, case in enumerate(order, 1):
        code = f"V{position:03d}"
        review_rows.append({"review_code": code, "bending": "", "melting": "", "splitting": "", "unexplained_disappearance": "", "legitimate_motion_or_occlusion": "", "notes": ""})
        key_rows.append({"review_code": code, **_case_metadata(case), "video_relative_path": case["video_relative_path"]})
    errors = []
    worksheet = output / "blinded_shape_review.csv"
    if not worksheet.exists():
        _write_csv(worksheet, review_rows, ["review_code"])
    key_path = output / "blinded_review_key_operator_only.csv"
    if not key_path.exists():
        _write_csv(key_path, key_rows, ["review_code", "case_id"])
    else:
        with key_path.open(newline="") as handle:
            existing_key = list(csv.DictReader(handle))
        if existing_key != key_rows:
            errors.append({"stage": "blinded_review", "error": "Existing operator key differs from the fixed review order; it was preserved. Check it before blinded review."})
    availability_columns = ["case_id", "scene", "condition", "condition_label", "status", "detail", "media_status"]
    _write_csv(output / "availability.csv", availability, availability_columns)

    if verified:
        try:
            _summary_plots(output, summaries, classes, cases)
        except Exception as error:
            errors.append({"stage": "plots", "error": str(error)})
    baselines = {case["scene"]: case for case in cases if case["condition"] == "baseline"}
    if render_videos:
        for index, (case, marker, row) in enumerate(verified, 1):
            print(f"[report media {index}/{len(verified)}] {case['case_id']}", flush=True)
            try:
                _render_case(case, baselines.get(case["scene"]), marker, output, input_root, classes, settings,
                             allowed_bundles=allowed_bundles)
                row["media_status"] = "complete"
            except Exception as error:
                row["media_status"] = "failed"
                errors.append({"case_id": case["case_id"], "stage": "videos", "error": str(error)})
            _write_csv(output / "availability.csv", availability, availability_columns)
    label_source = manifest.get("label_source", "Existing RGB-teacher predictions; not human ground truth or semantic-branch outputs.")
    summary = (
        f"# RGB tracking comparison\n\nVerified tracking results: **{len(verified)} / {len(cases)} videos**. "
        "See availability.csv for every planned case, including failures and missing results.\n\n"
        "per_class_summary.csv averages condition-minus-baseline differences equally over supported scenes. "
        "Support is metric-specific; missing values stay blank. Thousands of tracked points are not independent experimental replicates. "
        "Background is retained separately and excluded from foreground plots.\n\n"
        "The two plots show class persistence and reverse-cycle error changes with paired scene counts. "
        "They are separate diagnostics, not an overall geometry score. Check retention, reverse-return coverage and motion alongside them.\n\n"
        "Existing layout/IoU/boundary and RGB detail proxies remain separately labeled in existing_rgb_metrics.csv. "
        "They were measured before lossy MP4 encoding; tracking uses saved MP4s. "
        f"Label source: {label_source}\n\n"
        "Forward walking permits perspective growth and changing apparent speed. Low motion, coherent deformation, "
        "segmentation errors and tracker errors can all produce misleading consistency scores. This evaluation cohort "
        "does not establish improved 3D geometry or generalization. No planar-patch metric was added.\n\n"
        "For blinded shape review, an operator presents the original unlabeled RGB files in the order from "
        "blinded_review_key_operator_only.csv, showing only the review code. Keep that key and labeled comparisons hidden from the reviewer. "
        "Record bending, melting, splitting and unexplained disappearance, allowing legitimate motion/occlusion. "
        "Existing review notes and the operator key are preserved when rebuilding reports.\n\n"
        f"Rendering/report errors: {len(errors)}. Tracking completion markers and historical bundles are unchanged.\n"
    )
    (output / "SUMMARY.md").write_text(summary)
    result = {"verified_case_count": len(verified), "planned_case_count": len(cases), "render_videos_requested": render_videos,
              "errors": errors, "scientific_result": False}
    write_json_atomic(output / "results_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    result = build_results(args.manifest, args.output_dir, args.input_root, render_videos=not args.no_videos)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
