"""Candidate-local lifetime scoring in the separate CoTracker interpreter.

Launch this module with the CoTracker environment's Python. It never loads
Wan or YOLO, never accepts reference coordinates, and never replaces output.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from cotracker_evaluation import comparison_io as io
from cotracker_evaluation.run_cotracker import load_predictor, model_provenance, package_versions, track_video
from cotracker_evaluation.tracking_metrics import build_reverse_queries, compute_metrics
from rgb_geometry_pilot import protocol as historical_protocol
from rgb_geometry_pilot.reward import BACKGROUND_CLASSES, motion_cohort, validate_labels
from rgb_geometry_pilot.track import load_inputs
from rgb_geometry_rl_v2.metrics import lifetime_car_reward, select_interior_queries
from rgb_geometry_rl_v2.photo import PHOTO_SAMPLE_STRIDE, PHOTO_VERSION, PHOTO_WINDOWS, windowed_photo_diagnostic


SETTINGS = {
    "format": "rgb_geometry_rl_v2_tracking_v2",
    "anchor_frames": [0, 32, 64], "points_per_class": 16,
    "car_source_label": 2, "car_query_erosion": "full-resolution radius-two square",
    "erosion_applies_to": "query selection only", "later_labels": "original hard YOLO labels",
    "visible_unknown_label": "mismatch", "unobserved_query_reward": 0.0,
    "empty_clip_reward": 0.0, "cycle_horizon": 16,
    "retention_horizons": [8, 16], "fps": 16.0,
    "background_queries": "candidate-local road/sidewalk/building; separate tracking call",
    "gross_displacement": "mean supported 17-position segment net_squared/(path+1e-8), image-diagonal units",
    "photo_frame_indices": list(range(1, 81)), "photo_sample_stride": PHOTO_SAMPLE_STRIDE,
    "photo_version": PHOTO_VERSION,
    "photo_windows": [{"anchor_frame": anchor, "first_target_frame": first, "last_target_frame": last}
                      for anchor, first, last in PHOTO_WINDOWS],
    "appearance_certified": False,
}


def scoring_source_hashes():
    source = Path(__file__).resolve().parents[1]
    names = (
        "rgb_geometry_rl_v2/track.py", "rgb_geometry_rl_v2/metrics.py", "rgb_geometry_rl_v2/photo.py",
        "rgb_geometry_pilot/track.py", "rgb_geometry_pilot/reward.py",
        "rgb_geometry_pilot/protocol.py",
        "rgb_geometry_qualification/diagnostics.py", "rgb_geometry_qualification/controls.py",
        "rgb_geometry_reward_check/controls.py", "cotracker_evaluation/tracking_metrics.py",
        "cotracker_evaluation/run_cotracker.py", "cotracker_evaluation/comparison_io.py",
    )
    return {name: io.sha256_file(source / name) for name in names}


def _track_or_empty(predictor, frames, queries, device):
    if len(queries):
        return track_video(predictor, frames, queries, device=device)
    return (np.empty((len(frames), 0, 2), dtype=np.float32),
            np.empty((len(frames), 0), dtype=bool))


def analyze_tracks(predictor, frames, labels, device="cuda:0"):
    """CPU is available for mocked tests only; the CLI requires one CUDA GPU."""
    labels = validate_labels(labels)
    if frames.dtype != np.uint8 or frames.shape != (*labels.shape, 3):
        raise ValueError("Tracking requires aligned uint8 RGB and original label frames")
    queries, query_classes = select_interior_queries(labels)

    # Keep all classes in the model call because queries can interact inside it.
    forward_tracks, forward_visibility = _track_or_empty(predictor, frames, queries, device)
    background_queries = queries[np.isin(query_classes, BACKGROUND_CLASSES)]
    background_tracks, background_visibility = _track_or_empty(predictor, frames, background_queries, device)
    reverse_queries, reverse_indices = build_reverse_queries(
        forward_tracks, forward_visibility, queries, frames.shape[2], frames.shape[1],
        SETTINGS["cycle_horizon"])
    reverse_tracks, reverse_visibility = _track_or_empty(predictor, frames[::-1], reverse_queries, device)

    lifetime = lifetime_car_reward(labels, forward_tracks, forward_visibility, queries, query_classes)
    historical = compute_metrics(
        labels, forward_tracks, forward_visibility, queries, query_classes,
        reverse_tracks=reverse_tracks, reverse_visibility=reverse_visibility,
        reverse_query_indices=reverse_indices, fps=SETTINGS["fps"],
        cycle_horizon=SETTINGS["cycle_horizon"], retention_horizons=SETTINGS["retention_horizons"])
    car_diagnostics = historical["per_class"][2]

    # Reuse the existing displacement statistic on this video's own support.
    cohort = motion_cohort(labels, background_tracks, background_visibility, background_queries)
    motion = {
        "supported": cohort["cohort_size"] > 0,
        "support_reason": "measured" if cohort["cohort_size"] else "no_supported_background_segments",
        "gross_displacement": cohort["reference_mean_motion"],
        "query_count": len(background_queries), "supported_segments": cohort["cohort_size"],
        "unsupported_segments": len(background_queries) - cohort["cohort_size"],
        "query_indices": cohort["query_indices"], "segment_displacements": cohort["reference_motion"],
        "segment_positions": cohort["segment_positions"], "candidate_local": True,
        "definition": SETTINGS["gross_displacement"],
    }

    # Re-anchor the photo diagnostic using each window's matching query cohort.
    photo = windowed_photo_diagnostic(frames, background_tracks, background_visibility, background_queries)
    photo_supported = photo["supported"]
    photo_residual = photo["mean_brightness_aligned_rgb_mae"]
    unsupported_reasons = []
    if not lifetime["supported"]:
        unsupported_reasons.append(lifetime["support_reason"])
    if lifetime["censoring"] is None:
        unsupported_reasons.append("car_censoring_unavailable")
    if not motion["supported"]:
        unsupported_reasons.append(motion["support_reason"])
    if not photo_supported:
        unsupported_reasons.append("windowed_photo_not_supported_on_all_80_later_frames")
    sentinel = {
        "supported": not unsupported_reasons, "unsupported_reasons": unsupported_reasons,
        "availability": lifetime["availability"], "censoring": lifetime["censoring"],
        "gross_displacement": motion["gross_displacement"],
        "photo_residual": photo_residual if photo_supported else None,
    }
    metrics = {
        "format": SETTINGS["format"], "reward": lifetime["reward"], "supported": lifetime["supported"],
        "lifetime_car": lifetime, "historical_P": car_diagnostics["original_class_persistence"],
        "historical_P_note": "Historical formula on newly selected queries, not a reproduction of historical evaluation",
        "car_diagnostics": car_diagnostics, "historical_metrics": historical,
        "background_motion": motion, "animated_photo": photo, "sentinel": sentinel,
        "yolo_unknown_fraction": float(np.mean(labels == 255)), "appearance_certified": False,
    }
    arrays = {
        "forward_tracks": forward_tracks, "forward_visibility": forward_visibility,
        "queries": queries, "query_classes": query_classes,
        "reverse_tracks": reverse_tracks, "reverse_visibility": reverse_visibility,
        "reverse_queries": reverse_queries, "reverse_query_indices": reverse_indices,
        "bg_queries": background_queries, "bg_tracks": background_tracks,
        "bg_visibility": background_visibility,
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
    # Apply the inherited reserved-path rule before reading scoring inputs.
    for name in ("rgb_npz", "labels_npz", "output_dir", "cotracker_repo", "checkpoint"):
        path = getattr(args, name).expanduser().absolute()
        setattr(args, name, historical_protocol.safe_path(path))
    paths = {"rgb": args.rgb_npz.resolve(strict=True), "labels": args.labels_npz.resolve(strict=True)}
    input_hashes = {name: io.sha256_file(path) for name, path in paths.items()}
    frames, labels = load_inputs(paths["rgb"], paths["labels"])
    provenance = model_provenance(args.cotracker_repo, args.checkpoint)
    source_hashes = scoring_source_hashes()
    if args.check_inputs:
        print(json.dumps({"status": "inputs_valid", "input_hashes": input_hashes,
                          "shape": list(labels.shape), "model": provenance}, allow_nan=False))
        return 0
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Tracking requires exactly one visible allocated CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # Initialize CUDA before reading allocator statistics in this fresh process.
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
    # ru_maxrss is bytes on macOS and KiB on the Linux scoring workers.
    host_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    host_peak_rss_source_units = "bytes" if sys.platform == "darwin" else "KiB"
    host_peak_rss_bytes = int(host_peak_rss if sys.platform == "darwin" else host_peak_rss * 1024)
    marker = {
        "status": "complete", "format": SETTINGS["format"],
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_hashes": input_hashes, "model": provenance, "settings": SETTINGS,
        "scoring_source_sha256": source_hashes, "versions": package_versions(),
        "seconds": time.monotonic() - started,
        "gpu_runtime": gpu_runtime,
        "host_peak_rss_bytes": host_peak_rss_bytes,
        "host_peak_rss_source_units": host_peak_rss_source_units,
        "artifacts": {name: io.sha256_file(args.output_dir / name) for name in ("tracks.npz", "metrics.json")},
    }
    io.write_json_atomic(args.output_dir / "complete.json", marker)
    print(json.dumps({"status": "complete", "reward": metrics["reward"],
                      "historical_P": metrics["historical_P"], "sentinel": metrics["sentinel"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
