"""CoTracker subprocess retaining numerical JSON/CSV only, never track arrays."""

import argparse
import csv
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from cotracker_evaluation import comparison_io as io
from cotracker_evaluation.run_cotracker import load_predictor, model_provenance, package_versions
from cotracker_evaluation.tracking_metrics import (
    CLASS_NAMES, _check_tracks, _nearest_pixels, _visible_coordinates,
)
from rgb_geometry_pilot.protocol import safe_path
from rgb_geometry_pilot.reward import validate_labels
from rgb_geometry_pilot.track import load_inputs
from rgb_geometry_rl_v2 import track as reward_tracking
from rgb_geometry_evaluation import track as evaluation_tracking


def expected_class_ids(expected_classes):
    """Expected classes come from the prompt plan, never the observed predictions."""
    result = []
    for value in expected_classes:
        if isinstance(value, str):
            if value not in CLASS_NAMES:
                raise ValueError("Unknown expected source class: " + value)
            value = CLASS_NAMES.index(value)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= 10:
            raise ValueError("Expected classes must be genuine source labels 1 through 10.")
        if int(value) in result:
            raise ValueError("Expected classes must not contain duplicates.")
        result.append(int(value))
    if not result:
        raise ValueError("At least one predefined expected class is required.")
    return result


def lifetime_classes(labels, arrays, anchors, historical_points=None):
    """Use the existing car reward formula for every genuine source class.

    Later visible unknown labels count as mismatches. Invisible, nonfinite and
    out-of-image observations are censored. A class change never ends a track.
    Each point has equal weight; an unobserved point receives zero, not omission.
    """
    labels = validate_labels(labels)
    tracks, visibility, queries = _check_tracks(
        arrays["forward_tracks"], arrays["forward_visibility"], arrays["queries"]
    )
    classes = np.asarray(arrays["query_classes"])
    frame_count, height, width = labels.shape
    if tracks.shape[0] != frame_count or classes.shape != (len(queries),):
        raise ValueError("Label frames, tracks and query classes must align.")
    if classes.dtype.kind not in "iu" or not np.isin(classes, np.arange(1, 11)).all():
        raise ValueError("Only genuine source classes 1 through 10 can seed points.")
    if not np.isin(queries[:, 0], anchors).all():
        raise ValueError("Unexpected point-anchor frame.")
    if not _visible_coordinates(queries[:, 1:], np.ones(len(queries), dtype=bool), width, height).all():
        raise ValueError("Query anchors must lie inside the image.")
    pixels = _nearest_pixels(queries[:, 1:], width, height)
    anchored = labels[queries[:, 0].astype(int), pixels[:, 1], pixels[:, 0]]
    if not np.array_equal(anchored, classes):
        raise ValueError("Query classes do not match their original RGB-derived labels.")
    observable = _visible_coordinates(tracks, visibility, width, height)
    points = []
    for query_index, query in enumerate(queries):
        class_id = int(classes[query_index])
        anchor = int(query[0])
        later = observable[anchor + 1:, query_index]
        frames = np.flatnonzero(later) + anchor + 1
        pixels = _nearest_pixels(tracks[frames, query_index], width, height)
        observed = labels[frames, pixels[:, 1], pixels[:, 0]]
        visible_count = len(frames)
        possible_count = frame_count - anchor - 1
        matching = int(np.count_nonzero(observed == class_id))
        missing = np.flatnonzero(~later)
        continuous = int(missing[0]) if len(missing) else possible_count
        later_marked = visibility[anchor + 1:, query_index]
        finite = np.isfinite(tracks[anchor + 1:, query_index]).all(axis=1)
        row = {
            "query_index": query_index, "class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "anchor_frame": anchor, "query_x": float(query[1]), "query_y": float(query[2]),
            "visible_observation_count": visible_count, "matching_observation_count": matching,
            "unknown_observation_count": int(np.count_nonzero((observed == 255) | (observed == 0))),
            "possible_observation_count": possible_count,
            "censored_observation_count": possible_count - visible_count,
            "invisible_observation_count": int(np.count_nonzero(~later_marked)),
            "nonfinite_observation_count": int(np.count_nonzero(later_marked & ~finite)),
            "out_of_frame_observation_count": int(np.count_nonzero(later_marked & finite & ~later)),
            "censoring": 1 - visible_count / possible_count,
            "score": matching / visible_count if visible_count else 0.0,
            "observed": bool(visible_count), "observable_seconds": visible_count / 16,
            "continuous_visible_seconds": continuous / 16,
            "first_observed_frame": int(frames[0]) if visible_count else None,
            "last_observed_frame": int(frames[-1]) if visible_count else None,
        }
        for observed_class in range(1, 11):
            row["observed_" + CLASS_NAMES[observed_class]] = int(np.count_nonzero(observed == observed_class))
        if historical_points is not None:
            old = historical_points[query_index]
            for name, value in old.items():
                if name.startswith(("retention_", "continuous_retention_", "cycle_")):
                    row[name] = value
        points.append(row)
    per_class = []
    for class_id in range(1, 11):
        selected = [point for point in points if point["class_id"] == class_id]
        counts_by_anchor = {str(anchor): sum(point["anchor_frame"] == anchor for point in selected)
                            for anchor in anchors}
        if any(count > 16 for count in counts_by_anchor.values()):
            raise ValueError("At most sixteen points per class per anchor are permitted.")
        count = len(selected)
        visible = sum(point["visible_observation_count"] for point in selected)
        per_class.append({
            "class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "score": float(np.mean([point["score"] for point in selected])) if count else 0.0,
            "query_count": count, "availability": count / (len(anchors) * 16),
            "supported": count > 0 and visible > 0,
            "support_reason": "measured" if visible else ("no_later_visible_observations" if count else "no_class_queries"),
            "visible_observation_count": visible,
            "matching_observation_count": sum(point["matching_observation_count"] for point in selected),
            "unknown_observation_count": sum(point["unknown_observation_count"] for point in selected),
            "zero_observation_queries": sum(not point["observed"] for point in selected),
            "censoring": float(np.mean([point["censoring"] for point in selected])) if count else None,
            "queries_by_anchor": counts_by_anchor,
            "class_pixel_fraction": float(np.mean(labels == class_id)),
        })
    return per_class, points


def analyze(predictor, frames, labels, expected_classes, evaluation=False, device="cuda:0"):
    """Reuse the historical tracker calls, changing only numerical retention."""
    expected = expected_class_ids(expected_classes)
    module = evaluation_tracking if evaluation else reward_tracking
    arrays, metrics = module.analyze_tracks(predictor, frames, labels, device=device)
    if evaluation:
        per_class = [dict(row) for row in metrics["per_class"] if row["class_id"] != 0]
        points = []
        for row in per_class:
            for point in row.pop("per_query"):
                point = dict(point)
                histogram = point.pop("observed_label_counts")
                for name, count in histogram.items():
                    point["observed_" + name] = count
                points.append(point)
        points.sort(key=lambda row: row["query_index"])
    else:
        per_class, points = lifetime_classes(
            labels, arrays, module.SETTINGS["anchor_frames"],
            metrics["historical_metrics"]["per_query"],
        )
    scores = {row["class_id"]: row["score"] for row in per_class}
    support = {row["class_id"]: row["supported"] for row in per_class}
    broader = float(np.mean([scores[class_id] for class_id in expected]))
    car = float(scores[2])
    if not evaluation and not np.isclose(car, metrics["reward"], rtol=0, atol=1e-12):
        raise ValueError("Generalized lifetime formula disagrees with the unchanged car reward.")
    metrics.update(
        per_class=per_class, car_reward=car, broader_reward=broader, reward=broader,
        expected_class_ids=expected, expected_class_names=[CLASS_NAMES[value] for value in expected],
        car_supported=bool(support[2]),
        broader_supported=any(support[class_id] for class_id in expected),
        supported=any(support[class_id] for class_id in expected),
        expected_class_support_complete=all(support[class_id] for class_id in expected),
        measured_expected_class_count=sum(bool(support[class_id]) for class_id in expected),
        reward_definition="equal expected-class mean of equal point lifetime persistence; missing class zero",
        anchor_frames=list(module.SETTINGS["anchor_frames"]),
        evaluation=bool(evaluation), geometry_certified=False, appearance_certified=False,
    )
    if evaluation:
        metrics["overall_score"] = broader
    return metrics, points


def write_csv(path, rows):
    """Nested numerical summaries are JSON strings inside valid CSV cells."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["query_index"]
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-npz", type=Path, required=True)
    parser.add_argument("--labels-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-classes", required=True, help="Comma-separated source class IDs")
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args(argv)
    for name in ("rgb_npz", "labels_npz", "output_dir", "cotracker_repo", "checkpoint"):
        setattr(args, name, safe_path(getattr(args, name).expanduser().absolute()))
    expected = expected_class_ids([int(value) for value in args.expected_classes.split(",")])
    frames, labels = load_inputs(args.rgb_npz, args.labels_npz)
    hashes = {"rgb": io.sha256_file(args.rgb_npz), "labels": io.sha256_file(args.labels_npz)}
    provenance = model_provenance(args.cotracker_repo, args.checkpoint)
    module = evaluation_tracking if args.evaluation else reward_tracking
    sources = module.scoring_source_hashes()
    sources["rgb_geometry_campaign/track.py"] = io.sha256_file(Path(__file__).resolve())
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Tracking requires one allocated visible CUDA GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    predictor, _ = load_predictor(args.cotracker_repo, args.checkpoint)
    metrics, points = analyze(predictor, frames, labels, expected, args.evaluation)
    torch.cuda.synchronize()
    if hashes != {"rgb": io.sha256_file(args.rgb_npz), "labels": io.sha256_file(args.labels_npz)}:
        raise RuntimeError("Temporary scoring inputs changed during tracking.")
    if model_provenance(args.cotracker_repo, args.checkpoint) != provenance:
        raise RuntimeError("CoTracker sources or checkpoint changed during tracking.")
    current_sources = module.scoring_source_hashes()
    current_sources["rgb_geometry_campaign/track.py"] = io.sha256_file(Path(__file__).resolve())
    if current_sources != sources:
        raise RuntimeError("Scoring sources changed during tracking.")
    metrics["seconds"] = time.monotonic() - started
    metrics["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
    metrics["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    metrics["peak_host_rss_bytes"] = int(host_peak if sys.platform == "darwin" else host_peak * 1024)
    io.write_json_atomic(args.output_dir / "metrics.json", metrics)
    write_csv(args.output_dir / "points.csv", points)
    write_csv(args.output_dir / "classes.csv", metrics["per_class"])
    marker = {
        "status": "complete", "input_hashes": hashes, "model": provenance,
        "scoring_source_sha256": sources, "versions": package_versions(),
        "artifacts": {name: io.sha256_file(args.output_dir / name)
                      for name in ("metrics.json", "points.csv", "classes.csv")},
        "temporary_arrays_retained": False,
    }
    io.write_json_atomic(args.output_dir / "complete.json", marker)
    print(json.dumps({"status": "complete", "car_reward": metrics["car_reward"],
                      "broader_reward": metrics["broader_reward"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
