"""One immutable CoTracker scoring pass in the separate CoTracker environment.

Launch with that environment's Python; this module never loads Wan or YOLO.
Only explicitly supplied local CoTracker sources/checkpoints are used.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from cotracker_evaluation import comparison_io as io
from cotracker_evaluation.run_cotracker import load_predictor, model_provenance, track_video
from cotracker_evaluation.tracking_metrics import (
    CLASS_NAMES, build_reverse_queries, compute_metrics, select_queries,
)
from rgb_geometry_pilot.reward import BACKGROUND_CLASSES, motion_cohort, motion_score, validate_labels


SETTINGS = {"anchor_frames": [0, 32, 64], "points_per_class": 16,
            "cycle_horizon": 16, "retention_horizons": [8, 16], "fps": 16.0}


def scoring_source_hashes():
    source = Path(__file__).resolve().parents[1]
    names = ("rgb_geometry_pilot/track.py", "rgb_geometry_pilot/reward.py",
             "cotracker_evaluation/tracking_metrics.py", "cotracker_evaluation/run_cotracker.py")
    return {name: io.sha256_file(source / name) for name in names}


def load_inputs(rgb_path, labels_path, expected_shape=(81, 480, 832)):
    with np.load(rgb_path, allow_pickle=False) as arrays:
        frames = arrays["frames"].copy()
        rgb_order = arrays["frame_indices"].copy()
    with np.load(labels_path, allow_pickle=False) as arrays:
        labels = arrays["class_ids"].copy()
        label_order = arrays["frame_indices"].copy()
        names = arrays["class_names"].tolist()
    validate_labels(labels)
    if names != list(CLASS_NAMES):
        raise ValueError("Tracking requires the declared historical eleven-class name mapping")
    if frames.dtype != np.uint8 or frames.shape != (*labels.shape, 3):
        raise ValueError("RGB must be uint8 and share labels' [T,H,W] coordinates")
    if tuple(labels.shape) != tuple(expected_shape):
        raise ValueError(f"Expected frozen experiment dimensions {tuple(expected_shape)}")
    expected_order = np.arange(len(frames))
    if (not np.issubdtype(rgb_order.dtype, np.integer) or not np.issubdtype(label_order.dtype, np.integer)
            or not np.array_equal(rgb_order, expected_order) or not np.array_equal(label_order, expected_order)):
        raise ValueError("RGB and labels must declare original consecutive frame order")
    return frames, labels


def load_reference(reference_tracks):
    """Never accept an uncertified or modified reference tracking artifact."""
    path = Path(reference_tracks).resolve(strict=True)
    if path.name != "tracks.npz":
        raise ValueError("Reference must be the certified tracks.npz artifact")
    directory = path.parent
    marker = json.loads((directory / "complete.json").read_text())
    if marker.get("status") != "complete" or set(marker.get("artifacts", {})) != {"tracks.npz", "metrics.json"}:
        raise ValueError("Incomplete reference tracking receipt")
    if marker.get("settings") != SETTINGS or marker.get("scoring_source_sha256") != scoring_source_hashes():
        raise ValueError("Reference tracking settings/source differ from this frozen scorer")
    for filename, expected_hash in marker["artifacts"].items():
        if io.sha256_file(directory / filename) != expected_hash:
            raise ValueError(f"Reference tracking hash mismatch: {filename}")
    metrics = json.loads((directory / "metrics.json").read_text())
    if marker.get("reference") is not None:
        raise ValueError("Motion reference must have its own independent reference cohort")
    with np.load(path, allow_pickle=False) as arrays:
        queries = arrays["bg_queries"].copy()
    if not np.array_equal(queries, np.asarray(metrics["cohort"]["queries"]).reshape(-1, 3)):
        raise ValueError("Reference queries do not match the frozen cohort")
    return {"queries": queries, "cohort": metrics["cohort"], "model": marker["model"],
            "tracks_sha256": marker["artifacts"]["tracks.npz"],
            "metrics_sha256": marker["artifacts"]["metrics.json"], "path": str(path)}


def analyze_tracks(predictor, frames, labels, reference=None, device="cuda:0"):
    """CPU device is exclusively for tiny mocked-predictor contract tests."""
    validate_labels(labels)
    if frames.dtype != np.uint8 or frames.shape != (*labels.shape, 3):
        raise ValueError("Tracking needs aligned uint8 RGB and label frames")
    own_queries, own_classes = select_queries(
        labels, SETTINGS["anchor_frames"], SETTINGS["points_per_class"], CLASS_NAMES)
    car_queries = own_queries[own_classes == 2]
    if reference is None:
        bg_queries = own_queries[np.isin(own_classes, BACKGROUND_CLASSES)]
    else:
        bg_queries = np.asarray(reference["queries"], dtype=np.float32).reshape(-1, 3)
        if list(labels.shape) != reference["cohort"]["label_shape"]:
            raise ValueError("Candidate frames do not match the frozen motion reference")
    # Preserve the full historical query set, since queries can interact inside
    # the predictor. Extract the car metric only after the unchanged scoring.
    if len(own_queries):
        forward_tracks, forward_visibility = track_video(predictor, frames, own_queries, device=device)
    else:
        forward_tracks = np.empty((len(frames), 0, 2), dtype=np.float32)
        forward_visibility = np.empty((len(frames), 0), dtype=bool)
    car_tracks = forward_tracks[:, own_classes == 2]
    car_visibility = forward_visibility[:, own_classes == 2]
    if len(bg_queries):
        bg_tracks, bg_visibility = track_video(predictor, frames, bg_queries, device=device)
    else:
        bg_tracks = np.empty((len(frames), 0, 2), dtype=np.float32)
        bg_visibility = np.empty((len(frames), 0), dtype=bool)

    # Reverse inference is a fresh call on the reversed candidate RGB frames.
    reverse_queries, reverse_indices = build_reverse_queries(
        forward_tracks, forward_visibility, own_queries, frames.shape[2], frames.shape[1], SETTINGS["cycle_horizon"])
    if len(reverse_queries):
        reverse_tracks, reverse_visibility = track_video(predictor, frames[::-1], reverse_queries, device=device)
    else:
        reverse_tracks = np.empty((len(frames), 0, 2), dtype=np.float32)
        reverse_visibility = np.empty((len(frames), 0), dtype=bool)
    historical = compute_metrics(
        labels, forward_tracks, forward_visibility, own_queries, own_classes,
        reverse_tracks=reverse_tracks, reverse_visibility=reverse_visibility,
        reverse_query_indices=reverse_indices, fps=SETTINGS["fps"],
        cycle_horizon=SETTINGS["cycle_horizon"], retention_horizons=SETTINGS["retention_horizons"])
    car_diagnostics = historical["per_class"][2]
    # Lifetimes supplement the unchanged historical persistence and retention.
    lifetimes = []
    for query_index, query in enumerate(car_queries):
        anchor = int(query[0])
        positions = car_tracks[anchor + 1:, query_index]
        visible = car_visibility[anchor + 1:, query_index] & np.isfinite(positions).all(axis=1)
        visible &= (positions[:, 0] >= 0) & (positions[:, 0] < frames.shape[2])
        visible &= (positions[:, 1] >= 0) & (positions[:, 1] < frames.shape[1])
        missing = np.flatnonzero(~visible)
        lifetimes.append(int(missing[0]) if len(missing) else len(visible))
    car_diagnostics["continuous_visible_lifetime_frames"] = lifetimes
    cohort = (motion_cohort(labels, bg_tracks, bg_visibility, bg_queries)
              if reference is None else reference["cohort"])
    motion = motion_score(labels, bg_tracks, bg_visibility, bg_queries, cohort)
    metrics = {"P": car_diagnostics["original_class_persistence"],
               "car_diagnostics": car_diagnostics, "historical_metrics": historical,
               "cohort": cohort, "background_motion": motion,
               "yolo_unknown_fraction": float(np.mean(labels == 255))}
    arrays = {"forward_tracks": forward_tracks, "forward_visibility": forward_visibility,
              "queries": own_queries, "query_classes": own_classes,
              "reverse_tracks": reverse_tracks, "reverse_visibility": reverse_visibility,
              "reverse_queries": reverse_queries, "reverse_query_indices": reverse_indices,
              "bg_queries": bg_queries, "bg_tracks": bg_tracks, "bg_visibility": bg_visibility}
    return arrays, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-npz", type=Path, required=True)
    parser.add_argument("--labels-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-tracks", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args(argv)
    paths = {"rgb": args.rgb_npz.resolve(strict=True), "labels": args.labels_npz.resolve(strict=True)}
    input_hashes = {name: io.sha256_file(path) for name, path in paths.items()}
    frames, labels = load_inputs(paths["rgb"], paths["labels"])
    reference = load_reference(args.reference_tracks) if args.reference_tracks is not None else None
    provenance = model_provenance(args.cotracker_repo, args.checkpoint)
    source_hashes = scoring_source_hashes()
    if reference is not None and reference["model"] != provenance:
        raise ValueError("Candidate CoTracker source/checkpoint must match its frozen reference")
    if args.check_inputs:
        print(json.dumps({"status": "inputs_valid", "input_hashes": input_hashes,
                          "shape": list(labels.shape), "model": provenance}, allow_nan=False))
        return 0
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Tracking requires exactly one visible allocated CUDA GPU")
    # An existing output, even partial, is evidence and must never be replaced.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    predictor, torch = load_predictor(args.cotracker_repo, args.checkpoint)
    arrays, metrics = analyze_tracks(predictor, frames, labels, reference=reference)
    torch.cuda.synchronize()
    for name, path in paths.items():
        if io.sha256_file(path) != input_hashes[name]:
            raise RuntimeError("Scoring input changed during tracking")
    if model_provenance(args.cotracker_repo, args.checkpoint) != provenance:
        raise RuntimeError("CoTracker source/checkpoint changed during scoring")
    if scoring_source_hashes() != source_hashes:
        raise RuntimeError("Scorer source changed during tracking")
    if reference is not None:
        checked = load_reference(args.reference_tracks)
        if checked["tracks_sha256"] != reference["tracks_sha256"] or checked["metrics_sha256"] != reference["metrics_sha256"]:
            raise RuntimeError("Frozen reference changed during scoring")
    with (args.output_dir / "tracks.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with (args.output_dir / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    reference_receipt = None if reference is None else {
        name: reference[name] for name in ("path", "tracks_sha256", "metrics_sha256")}
    marker = {"status": "complete", "input_paths": {name: str(path) for name, path in paths.items()},
              "input_hashes": input_hashes, "model": provenance, "settings": SETTINGS,
              "scoring_source_sha256": source_hashes,
              "reference": reference_receipt, "seconds": time.monotonic() - started,
              "gpu_runtime": getattr(predictor, "comparison_gpu_runtime", {}),
              "artifacts": {name: io.sha256_file(args.output_dir / name) for name in ("tracks.npz", "metrics.json")}}
    io.write_json_atomic(args.output_dir / "complete.json", marker)
    print(json.dumps({"status": "complete", "P": metrics["P"], "background_motion": metrics["background_motion"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
