#!/usr/bin/env python3
"""Track a manifest of saved RGB videos without loading Wan or training anything."""

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from cotracker_evaluation import comparison_io as io


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=io.DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/dhiazr/cotracker3/checkpoints/scaled_offline.pth"))
    parser.add_argument("--cotracker-repo", type=Path, default=Path("/home/dhiazr/cotracker3/co-tracker"))
    parser.add_argument("--check-inputs", action="store_true", help="Verify the manifest input files without importing Torch or loading a model")
    parser.add_argument("--resume", action="store_true", help="Verify completed files and continue this exact configuration")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the first manifest video, including reverse tracking")
    parser.add_argument("--no-render-videos", action="store_true", help="Build tables but defer comparison videos")
    return parser.parse_args()


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def model_provenance(repository, checkpoint):
    repository = Path(repository).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], text=True)
    source_paths = sorted((repository / "cotracker").rglob("*.py"))
    for name in ("hubconf.py", "setup.py", "pyproject.toml"):
        if (repository / name).is_file():
            source_paths.append(repository / name)
    if not (repository / "cotracker/predictor.py").is_file() or not source_paths:
        raise FileNotFoundError("The specified local CoTracker source is incomplete")
    return {
        "repository": str(repository), "git_revision": revision, "git_status": status,
        "source_sha256": {str(path.relative_to(repository)): io.sha256_file(path) for path in source_paths},
        "checkpoint": str(checkpoint), "checkpoint_sha256": io.sha256_file(checkpoint),
    }


def package_versions():
    versions = {"python": sys.version}
    for name in ("torch", "torchvision", "numpy", "cotracker", "imageio", "imageio-ffmpeg"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def configuration(args, manifest):
    return {
        "input_root": str(args.input_root.resolve(strict=True)),
        "manifest_sha256": io.sha256_file(args.manifest),
        "settings": manifest["settings"], "classes": manifest["classes"],
        "model": model_provenance(args.cotracker_repo, args.checkpoint),
        "versions": package_versions(), "package_sha256": io.package_hashes(),
        "predictor": {"offline": True, "window_len": 60, "backward_tracking": False,
                      "input_range": "RGB float32 0..255", "device": "cuda:0"},
    }


def configuration_fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_output_directory(output_dir, input_root, manifest=None):
    output_dir = Path(output_dir).resolve()
    input_root = Path(input_root).resolve(strict=True)
    if output_dir == input_root:
        raise ValueError("Use a fresh evaluation-output directory, not the input root itself")
    protected_bundles = set(io.SOURCE_BUNDLES).union(io.source_bundles(manifest))
    for bundle_name in protected_bundles:
        source_bundle = (input_root / bundle_name).resolve()
        if output_dir == source_bundle or source_bundle in output_dir.parents:
            raise ValueError("Evaluation output cannot be inside a historical result bundle")


def prepare_output(output_dir, configuration_value, resume):
    output_dir = Path(output_dir).resolve()
    fingerprint = configuration_fingerprint(configuration_value)
    if resume:
        recorded = read_json(output_dir / "run_manifest.json")
        if recorded.get("run_fingerprint") != fingerprint or recorded.get("configuration") != configuration_value:
            raise RuntimeError("Resume rejected: input, model, code or configuration fingerprint differs")
    else:
        # Atomic creation prevents concurrent fresh runs from replacing identity.
        output_dir.mkdir(parents=True, exist_ok=False)
        io.write_json_atomic(output_dir / "run_manifest.json", {
            "run_fingerprint": fingerprint, "configuration": configuration_value,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
    return fingerprint


def verify_completed_case(directory, case, fingerprint):
    marker_path = Path(directory) / "complete.json"
    if not marker_path.exists():
        return False
    marker = read_json(marker_path)
    expected_inputs = {name: case[name] for name in ("video_sha256", "labels_sha256")}
    if (marker.get("status") != "complete" or marker.get("case_id") != case["case_id"]
            or marker.get("run_fingerprint") != fingerprint or marker.get("input_hashes") != expected_inputs):
        raise RuntimeError(f"Invalid completion marker: {marker_path}")
    artifacts = marker.get("artifacts", {})
    if set(artifacts) != {"tracks.npz", "metrics.json"}:
        raise RuntimeError(f"Incomplete artifact list: {marker_path}")
    for filename, expected_hash in artifacts.items():
        path = Path(directory) / filename
        if not path.is_file() or io.sha256_file(path) != expected_hash:
            raise RuntimeError(f"Saved artifact hash mismatch: {path}")
    return True


def load_predictor(repository, checkpoint):
    # These imports are deliberately below the CPU-only --check-inputs path.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU tracking requires a CUDA allocation; CPU input validation uses --check-inputs")
    sys.path.insert(0, str(Path(repository).resolve()))
    from cotracker import predictor as predictor_module

    expected = (Path(repository) / "cotracker/predictor.py").resolve()
    if Path(predictor_module.__file__).resolve() != expected:
        raise RuntimeError("Imported CoTracker does not come from the requested local repository")
    torch.set_num_threads(2)
    predictor = predictor_module.CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60)
    predictor.eval().requires_grad_(False).to("cuda:0")
    predictor.comparison_gpu_runtime = {
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    return predictor, torch


def track_video(predictor, frames, queries, device="cuda:0"):
    """One independent full-video call; CPU device is for mocked-predictor tests only."""
    import numpy as np
    import torch

    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Tracking frames must be uint8 RGB [T,H,W,3]")
    if queries.ndim != 2 or queries.shape[1] != 3 or not len(queries):
        raise ValueError("Tracking needs at least one query in [N,3] (t,x,y) format")
    if not np.isfinite(queries).all():
        raise ValueError("Query coordinates must be finite")
    with torch.inference_mode():
        video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).unsqueeze(0)
        video = video.to(device=device, dtype=torch.float32)
        query_tensor = torch.from_numpy(np.ascontiguousarray(queries)).unsqueeze(0).to(device=device, dtype=torch.float32)
        tracks, visibility = predictor(video, queries=query_tensor, backward_tracking=False)
        expected_shape = (1, frames.shape[0], len(queries))
        if tuple(tracks.shape) != (*expected_shape, 2) or tuple(visibility.shape) != expected_shape:
            raise RuntimeError("Installed offline predictor output shapes differ from the verified API")
        if visibility.dtype != torch.bool:
            raise RuntimeError("Installed offline predictor must return Boolean visibility; refusing an implicit threshold change")
        return tracks[0].float().cpu().numpy().copy(), visibility[0].cpu().numpy().copy()


def save_tracks(path, arrays):
    import numpy as np

    temporary = Path(path).with_name("tracks.partial.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def analyze_case(predictor, case, manifest, input_root, directory):
    import numpy as np
    from cotracker_evaluation.tracking_metrics import select_queries, build_reverse_queries, compute_metrics

    settings = manifest["settings"]
    allowed_bundles = io.source_bundles(manifest)
    video_path = io.input_path(input_root, case["video_relative_path"], allowed_bundles=allowed_bundles)
    labels_path = io.input_path(input_root, case["labels_relative_path"], allowed_bundles=allowed_bundles)
    frames = io.read_rgb_video(video_path, settings)
    with np.load(labels_path, allow_pickle=False) as archive:
        labels = archive["class_ids"].copy()
    if labels.dtype != np.uint8 or labels.shape != frames.shape[:3]:
        raise ValueError("Saved labels and RGB frames must share uint8 [T,H,W] coordinates")
    queries, query_classes = select_queries(labels, settings["anchor_frames"], settings["points_per_class"], manifest["classes"])
    started = time.perf_counter()
    tracks, visibility = track_video(predictor, frames, queries)
    forward_seconds = time.perf_counter() - started
    reverse_queries, reverse_indices = build_reverse_queries(
        tracks, visibility, queries, frames.shape[2], frames.shape[1], settings["cycle_horizon"])
    started = time.perf_counter()
    if len(reverse_queries):
        reverse_tracks, reverse_visibility = track_video(predictor, frames[::-1], reverse_queries)
    else:
        reverse_tracks = np.empty((len(frames), 0, 2), dtype=np.float32)
        reverse_visibility = np.empty((len(frames), 0), dtype=bool)
    reverse_seconds = time.perf_counter() - started
    metrics = compute_metrics(
        labels, tracks, visibility, queries, query_classes,
        reverse_tracks=reverse_tracks, reverse_visibility=reverse_visibility, reverse_query_indices=reverse_indices,
        fps=settings["fps"], cycle_horizon=settings["cycle_horizon"], retention_horizons=settings["retention_horizons"])
    metrics.update({name: case[name] for name in ("case_id", "scene", "condition")})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_tracks(directory / "tracks.npz", {
        "forward_tracks": tracks, "forward_visibility": visibility, "queries": queries, "query_classes": query_classes,
        "reverse_tracks": reverse_tracks, "reverse_visibility": reverse_visibility,
        "reverse_queries": reverse_queries, "reverse_query_indices": reverse_indices,
    })
    io.write_json_atomic(directory / "metrics.json", metrics)
    return {"forward_seconds": forward_seconds, "reverse_seconds": reverse_seconds,
            "query_count": len(queries), "reverse_query_count": len(reverse_queries)}


def verify_unchanged_configuration(expected, args):
    if io.package_hashes() != expected["package_sha256"]:
        raise RuntimeError("Evaluation package changed during this run")
    if io.sha256_file(args.manifest) != expected["manifest_sha256"]:
        raise RuntimeError("Evaluation manifest changed during this run")
    if model_provenance(args.cotracker_repo, args.checkpoint) != expected["model"]:
        raise RuntimeError("CoTracker source/checkpoint changed during this run")


def execute(args, manifest, input_report, configuration_value, fingerprint):
    output = args.output_dir
    selected = manifest["cases"][:1] if args.smoke_only else manifest["cases"]
    # Reject corrupt completed outputs before loading a model or changing files.
    completed = []
    for case in manifest["cases"]:
        if verify_completed_case(output / "cases" / case["case_id"], case, fingerprint):
            completed.append(case["case_id"])
    io.write_json_atomic(output / "input_validation.json", input_report)
    io.write_json_atomic(output / "run_status.json", {
        "status": "running", "full_comparison_complete": False, "run_fingerprint": fingerprint,
        "completed_cases": completed, "expected_case_count": len(manifest["cases"]),
    })
    failures = []
    fatal_error = None
    predictor = torch = None
    run_started = time.perf_counter()
    try:
        pending = [case for case in selected if case["case_id"] not in completed]
        if pending:
            predictor, torch = load_predictor(args.cotracker_repo, args.checkpoint)
            io.write_json_atomic(output / "gpu_runtime.json", getattr(predictor, "comparison_gpu_runtime", {}))
        for position, case in enumerate(pending):
            directory = output / "cases" / case["case_id"]
            print(f"[{position + 1}/{len(pending)}] {case['case_id']}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            poisoned_cuda = False
            try:
                timing = analyze_case(predictor, case, manifest, args.input_root, directory)
                torch.cuda.synchronize()
                timing.update({"case_seconds": time.perf_counter() - started,
                               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})
                verify_unchanged_configuration(configuration_value, args)
                # Check the two consumed inputs again before certifying this case.
                for kind in ("video", "labels"):
                    input_path = io.input_path(args.input_root, case[f"{kind}_relative_path"],
                                               allowed_bundles=io.source_bundles(manifest))
                    if io.sha256_file(input_path) != case[f"{kind}_sha256"]:
                        raise RuntimeError(f"{kind} input changed during tracking")
                marker = {"status": "complete", "case_id": case["case_id"], "run_fingerprint": fingerprint,
                          "input_hashes": {name: case[name] for name in ("video_sha256", "labels_sha256")},
                          "artifacts": {name: io.sha256_file(directory / name) for name in ("tracks.npz", "metrics.json")},
                          "timing": timing, "gpu_runtime": getattr(predictor, "comparison_gpu_runtime", {})}
                io.write_json_atomic(directory / "complete.json", marker)
                completed.append(case["case_id"])
                if not (output / "gpu_smoke.json").exists():
                    smoke = {"case_id": case["case_id"], "first_successful_case": True, **timing,
                             "estimated_all_case_seconds": timing["case_seconds"] * len(manifest["cases"]),
                             "estimate_excludes_model_load_summary_render_and_variation_between_videos": True}
                    io.write_json_atomic(output / "gpu_smoke.json", smoke)
                    print(f"First successful full-video smoke passed: {timing['case_seconds']:.1f}s; "
                          f"rough {len(manifest['cases'])}-video tracking estimate {smoke['estimated_all_case_seconds'] / 3600:.2f}h", flush=True)
            except Exception as error:
                failure = {"case_id": case["case_id"], "error_type": type(error).__name__, "error": str(error),
                           "traceback": traceback.format_exc(), "settings_unchanged": True}
                failures.append(failure)
                directory.mkdir(parents=True, exist_ok=True)
                io.write_json_atomic(directory / "failure.json", failure)
                print(f"FAILED {case['case_id']}: {error}", file=sys.stderr, flush=True)
                poisoned_cuda = any(text in str(error).lower() for text in
                                    ("device-side assert", "illegal memory access", "unspecified launch failure"))
            gc.collect()
            if poisoned_cuda:
                raise RuntimeError("CUDA context may be unusable; remaining cases were not attempted")
            torch.cuda.empty_cache()
        verify_unchanged_configuration(configuration_value, args)
    except KeyboardInterrupt:
        fatal_error = "Interrupted; verified completed cases are preserved for --resume"
    except Exception:
        fatal_error = traceback.format_exc()
        print(fatal_error, file=sys.stderr)

    # A failed video does not prevent useful partial tables from being written.
    summary_error = None
    try:
        from cotracker_evaluation.make_results import build_results
        summary = build_results(args.manifest, output, args.input_root, render_videos=not args.no_render_videos)
        if not isinstance(summary, dict):
            raise RuntimeError("Summary builder did not return its completion report")
        if summary.get("errors"):
            summary_error = json.dumps(summary["errors"], allow_nan=False)
        if summary.get("verified_case_count") != len(completed):
            raise RuntimeError("Summary verified-case count differs from completed tracking cases")
    except Exception:
        summary_error = traceback.format_exc()
        print(summary_error, file=sys.stderr)
    try:
        verify_unchanged_configuration(configuration_value, args)
    except Exception:
        fatal_error = traceback.format_exc()
        print(fatal_error, file=sys.stderr)
    selected_ids = {case["case_id"] for case in selected}
    selected_complete = selected_ids.issubset(completed)
    success = selected_complete and not failures and fatal_error is None and summary_error is None
    status = {
        "status": ("smoke_complete" if args.smoke_only else "complete") if success else "incomplete",
        "full_comparison_complete": success and not args.smoke_only and len(completed) == len(manifest["cases"]),
        "smoke_only": args.smoke_only, "completed_cases": completed, "expected_case_count": len(manifest["cases"]),
        "failed_cases": failures, "fatal_error": fatal_error, "summary_error": summary_error,
        "run_fingerprint": fingerprint, "seconds": time.perf_counter() - run_started,
        "scientific_geometry_improvement_established": False,
    }
    io.write_json_atomic(output / "run_status.json", status)
    io.write_json_atomic(output / "failures.json", {"cases": failures, "fatal_error": fatal_error, "summary_error": summary_error})
    return 0 if success else 1


def main():
    args = parse_args()
    manifest = io.load_manifest(args.manifest)
    input_report = io.validate_inputs(manifest, args.input_root)
    if args.check_inputs:
        print(json.dumps(input_report, indent=2, allow_nan=False))
        return 0
    args.input_root = args.input_root.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    validate_output_directory(args.output_dir, args.input_root, manifest)
    configuration_value = configuration(args, manifest)
    fingerprint = prepare_output(args.output_dir, configuration_value, args.resume)
    # Keep the lock file; flock releases ownership automatically on process exit.
    with (args.output_dir / "run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args, manifest, input_report, configuration_value, fingerprint)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
