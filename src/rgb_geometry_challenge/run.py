"""Evaluate one saved variant or verify the full challenge campaign."""

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback

from . import inputs, protocol


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", required=True, choices=(*protocol.VARIANTS, "report"))
    result.add_argument("--reference-root", type=Path, required=True)
    result.add_argument("--lora-root", type=Path, required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--config", type=Path, default=protocol.CONFIG)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--check-output", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    config = protocol.load_config(args.config)
    protocol.runtime_args(args, config)
    args.output_root = inputs.validate_output(args)
    if args.check_output:
        return 0
    if args.plan:
        print(json.dumps({"experiment": args.experiment, "inference_only": True,
                          "training_rounds": 0, "maximum_concurrent_gpus": 4,
                          "videos_per_variant": 80, "total_videos": 800,
                          "cases": protocol.cases(config, "evaluation")}, indent=2))
        return 0
    if args.check_inputs:
        info = inputs.validate_inputs(args, config)
        available = inputs.check_free_space(args.output_root)
        print(json.dumps({"status": "inputs_valid", "identity": info["identity"],
                          "filesystem_free_bytes": available, "quota_checked": False}, indent=2))
        return 0
    if args.experiment != "report" and not os.environ.get("SLURM_JOB_ID", "").isdigit():
        arguments.error("GPU inference must run inside Slurm using the supplied workers.")

    directory = args.output_root / args.experiment
    if directory.exists() and any(path.name != "execution_log.csv" for path in directory.iterdir()):
        raise FileExistsError("Preserving the existing variant/report; use a fresh output root.")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError("Interrupted by signal " + str(signum))

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        inputs.check_cpu_identity(args)
        if args.experiment == "report":
            from . import reporting
            result = reporting.write_report(args, config)
        else:
            # Submission reserves 10 GB; allow the campaign to consume that space.
            inputs.check_free_space(args.output_root, minimum_bytes=1_000_000_000)
            info = inputs.validate_inputs(args, config)
            inputs.check_submitted_identity(args, info)
            from rgb_geometry_qualitative import artifacts
            encoder = artifacts.check_video_encoder()
            protocol.tracker_probe(args)
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Each worker needs exactly one visible CUDA GPU.")
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(config["settings"]["seed"])
            torch.cuda.manual_seed_all(config["settings"]["seed"])
            spec = protocol.experiment_spec(args.experiment)
            identity = dict(info["identity"], experiment=args.experiment, training_spec=spec)
            protocol.write_json(directory / "manifest.json", {
                "format": "rgb_geometry_challenge_run_v1", "config": config, "spec": spec,
                "identity": identity, "job_id": os.environ.get("SLURM_JOB_ID"),
                "python": sys.version, "platform": platform.platform(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "encoder": encoder,
                "inference_only": True, "historical_scores_reused": False,
                "comparison_fingerprint": protocol.fingerprint(config),
                "source_fingerprint": protocol.fingerprint(info["identity"]["sources"])})
            from . import experiments
            result = experiments.run_experiment(args, config, info, spec)
            result["resources"] = {
                "parent_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "parent_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        result["elapsed_seconds"] = time.monotonic() - started
        protocol.write_json(directory / "status.json", result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("status") == "complete" else 2
    except Exception as error:
        failure = {"status": "technical_failure", "experiment": args.experiment,
                   "error": str(error), "exception": type(error).__name__,
                   "traceback": traceback.format_exc(), "inference_only": True,
                   "completed_outputs_preserved": True,
                   "elapsed_seconds": time.monotonic() - started}
        protocol.write_json(directory / "failure.json", failure)
        protocol.write_json(directory / "status.json", failure)
        traceback.print_exc()
        return 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
