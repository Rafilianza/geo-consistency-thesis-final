"""Score one generated RGB clip in the separate CoTracker Python environment."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from cotracker_evaluation import comparison_io as io
from cotracker_evaluation.run_cotracker import load_predictor, model_provenance, package_versions, track_video
from cotracker_evaluation.tracking_metrics import build_reverse_queries
from rgb_geometry_pilot.protocol import safe_path
from rgb_geometry_pilot.reward import validate_labels
from rgb_geometry_pilot.track import load_inputs
from rgb_geometry_evaluation.metrics import (
    BACKGROUND_CLASSES, SETTINGS, background_motion, class_persistence,
    one_second_photo_diagnostic, select_evaluation_queries,
)


def scoring_source_hashes():
    source = Path(__file__).resolve().parents[1]
    names = (
        "rgb_geometry_evaluation/track.py", "rgb_geometry_evaluation/metrics.py",
        "rgb_geometry_pilot/track.py", "rgb_geometry_pilot/reward.py",
        "rgb_geometry_pilot/protocol.py", "rgb_geometry_qualification/diagnostics.py",
        "cotracker_evaluation/tracking_metrics.py", "cotracker_evaluation/run_cotracker.py",
        "cotracker_evaluation/comparison_io.py",
    )
    return {name: io.sha256_file(source / name) for name in names}


def _track_or_empty(predictor, frames, queries, device):
    if len(queries):
        return track_video(predictor, frames, queries, device=device)
    return (np.empty((len(frames), 0, 2), dtype=np.float32),
            np.empty((len(frames), 0), dtype=bool))


def analyze_tracks(predictor, frames, labels, device="cuda:0"):
    """Evaluate every class; CPU is useful only with a mocked tracker in tests."""
    labels = validate_labels(labels)
    if frames.dtype != np.uint8 or frames.shape != (*labels.shape, 3):
        raise ValueError("Tracking requires aligned uint8 RGB and original label frames")
    if len(frames) != SETTINGS["frame_count"]:
        raise ValueError("Shared evaluation requires exactly 81 frames")
    queries, query_classes = select_evaluation_queries(labels)
    forward_tracks, forward_visibility = _track_or_empty(predictor, frames, queries, device)

    # Use a separate candidate-local background call, independent of car support.
    bg_queries = queries[np.isin(query_classes, BACKGROUND_CLASSES)]
    bg_tracks, bg_visibility = _track_or_empty(predictor, frames, bg_queries, device)
    reverse_queries, reverse_indices = build_reverse_queries(
        forward_tracks, forward_visibility, queries, frames.shape[2], frames.shape[1],
        horizon=SETTINGS["cycle_horizon"])
    reverse_tracks, reverse_visibility = _track_or_empty(predictor, frames[::-1], reverse_queries, device)
    per_class = class_persistence(
        labels, forward_tracks, forward_visibility, queries, query_classes,
        reverse_tracks, reverse_visibility, reverse_indices)
    metrics = {
        "format": SETTINGS["format"], "settings": SETTINGS, "per_class": per_class,
        "background_motion": background_motion(labels, bg_tracks, bg_visibility, bg_queries),
        "animated_photo": one_second_photo_diagnostic(frames, bg_tracks, bg_visibility, bg_queries),
        "yolo_unknown_fraction": float(np.mean((labels == 255) | (labels == 0))),
        "geometry_certified": False, "appearance_certified": False,
        "overall_score": None,
        "overall_score_note": "Aggregate only after joining the prespecified expected-class list for this case.",
    }
    arrays = {
        "forward_tracks": forward_tracks, "forward_visibility": forward_visibility,
        "queries": queries, "query_classes": query_classes,
        "reverse_tracks": reverse_tracks, "reverse_visibility": reverse_visibility,
        "reverse_queries": reverse_queries, "reverse_query_indices": reverse_indices,
        "bg_queries": bg_queries, "bg_tracks": bg_tracks, "bg_visibility": bg_visibility,
    }
    return arrays, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-npz", type=Path, required=True)
    parser.add_argument("--labels-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args(argv)
    for name in ("rgb_npz", "labels_npz", "output_dir", "cotracker_repo", "checkpoint"):
        setattr(args, name, safe_path(getattr(args, name).expanduser().absolute()))
    paths = {"rgb": args.rgb_npz.resolve(strict=True), "labels": args.labels_npz.resolve(strict=True)}
    input_hashes = {name: io.sha256_file(path) for name, path in paths.items()}
    frames, labels = load_inputs(paths["rgb"], paths["labels"])
    provenance = model_provenance(args.cotracker_repo, args.checkpoint)
    source_hashes = scoring_source_hashes()
    if args.check_inputs:
        print(json.dumps({"status": "inputs_valid", "input_hashes": input_hashes,
                          "shape": list(labels.shape), "model": provenance, "settings": SETTINGS}, allow_nan=False))
        return 0

    # Import Torch only in the dedicated tracker interpreter, never at import.
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Tracking requires exactly one visible allocated CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    predictor, torch = load_predictor(args.cotracker_repo, args.checkpoint)
    arrays, metrics = analyze_tracks(predictor, frames, labels)
    torch.cuda.synchronize()
    gpu_runtime = dict(getattr(predictor, "comparison_gpu_runtime", {}))
    gpu_runtime.update(peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                       peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
    for name, path in paths.items():
        if io.sha256_file(path) != input_hashes[name]:
            raise RuntimeError("Scoring input changed during tracking")
    if model_provenance(args.cotracker_repo, args.checkpoint) != provenance:
        raise RuntimeError("CoTracker source/checkpoint changed during scoring")
    if scoring_source_hashes() != source_hashes:
        raise RuntimeError("Scorer source changed during tracking")
    with (args.output_dir / "tracks.npz").open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with (args.output_dir / "metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=False)
    host_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    marker = {
        "status": "complete", "format": SETTINGS["format"], "settings": SETTINGS,
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_hashes": input_hashes, "model": provenance,
        "scoring_source_sha256": source_hashes, "versions": package_versions(),
        "seconds": time.monotonic() - started, "gpu_runtime": gpu_runtime,
        "host_peak_rss_bytes": int(host_peak if sys.platform == "darwin" else host_peak * 1024),
        "artifacts": {name: io.sha256_file(args.output_dir / name) for name in ("tracks.npz", "metrics.json")},
    }
    io.write_json_atomic(args.output_dir / "complete.json", marker)
    print(json.dumps({"status": "complete", "query_count": len(arrays["queries"]),
                      "scores": {row["class_name"]: row["score"] for row in metrics["per_class"]}}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
