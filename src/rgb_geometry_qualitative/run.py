"""Run, resume or export one study arm into a fresh results directory."""

import json
import os
import platform
import signal
import sys
import time
import traceback

from . import protocol


def main(argv=None, study_protocol=protocol, report_writer=None):
    parser = study_protocol.parser()
    args = parser.parse_args(argv)
    if args.experiment in ("baseline", "report") and (args.resume_from or args.checkpoint):
        parser.error("Checkpoint loading is only available for trained policies.")
    config = study_protocol.load_config(args.config)
    study_protocol.runtime_args(args, config)
    args.output_root = study_protocol.validate_output(args)
    spec = study_protocol.experiment_spec(args.experiment)
    if args.check_output:
        return 0
    if args.plan:
        print(json.dumps({"experiment": args.experiment, "spec": spec,
                          "training_rounds": 0 if args.checkpoint else (10 if spec["trainable"] else 0),
                          "resume_from": str(args.resume_from) if args.resume_from else None,
                          "weights_saved": spec["trainable"] and not bool(args.checkpoint),
                          "final_rgb_videos": 20 if args.experiment != "report" else getattr(study_protocol, "REPORT_VIDEOS", 140),
                          "retention": config["retention"],
                          "evaluation_cases": study_protocol.cases(config, "evaluation")}, indent=2))
        return 0
    if args.check_inputs:
        info = study_protocol.validate_inputs(args, config)
        free_bytes = study_protocol.check_free_space(args.output_root)
        print(json.dumps({"status": "inputs_valid", "identity": info["identity"],
                          "config_sha256": study_protocol.file_hash(args.config),
                          "filesystem_free_bytes": free_bytes,
                          "quota_checked": False}, indent=2))
        return 0
    if args.experiment != "report" and not os.environ.get("SLURM_JOB_ID", "").isdigit():
        parser.error("GPU work must run inside a Slurm allocation using the provided wrappers.")

    directory = args.output_root / args.experiment
    # The launcher may have opened its CSV before this process starts.
    if directory.exists() and any(path.name != "execution_log.csv" for path in directory.iterdir()):
        raise FileExistsError("Use a fresh output root; prior attempts are never overwritten.")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError("Interrupted by signal " + str(signum))

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, interrupted)
        study_protocol.check_cpu_identity(args)
        if args.experiment == "report":
            if report_writer is None:
                from . import reporting
                result = reporting.write_report(args.output_root, config)
            else:
                result = report_writer(args, config)
        else:
            study_protocol.check_free_space(args.output_root)
            info = study_protocol.validate_inputs(args, config)
            study_protocol.check_submitted_identity(args, info)
            from . import artifacts
            encoder = artifacts.check_video_encoder()
            study_protocol.tracker_probe(args)
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Each arm requires exactly one visible CUDA GPU.")
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(config["settings"]["seed"])
            torch.cuda.manual_seed_all(config["settings"]["seed"])
            identity = dict(info["identity"], experiment=args.experiment, training_spec=spec)
            protocol.write_json(directory / "manifest.json", {
                "format": "rgb_geometry_qualitative_run_v1", "config": config, "spec": spec,
                "identity": identity, "job_id": os.environ.get("SLURM_JOB_ID"),
                "python": sys.version, "platform": platform.platform(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "encoder": encoder,
                "new_reproduction_run": True, "historical_scores_reused": False,
                "comparison_fingerprint": protocol.fingerprint(config),
                "source_fingerprint": protocol.fingerprint(info["identity"]["sources"]),
                "retention": config["retention"]})
            from . import experiments
            runtime = experiments.runtime_for(args, config, info)
            result = experiments.run_experiment(args, config, runtime, spec, identity)
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
                   "elapsed_seconds": time.monotonic() - started, "traceback": traceback.format_exc(),
                   "completed_checkpoints_preserved": True}
        protocol.write_json(directory / "failure.json", failure)
        protocol.write_json(directory / "status.json", failure)
        traceback.print_exc()
        return 1
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    sys.exit(main())
