"""Submit four inference-only GPU workers and one final CPU report."""

import argparse
import csv
from datetime import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
JOB_DIRECTORY = REPOSITORY / "jobs/rgb_geometry_challenge"
worker_spec = importlib.util.spec_from_file_location("challenge_fifo_worker", JOB_DIRECTORY / "worker.py")
queue_worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(queue_worker)
EXPERIMENTS = queue_worker.EXPERIMENTS
WORKERS = queue_worker.WORKERS


def campaign_plan():
    # Baseline is an ordinary queue item. All four GPU jobs start independently.
    plan = [{"experiment": name, "gpu": True, "afterany": []} for name in WORKERS]
    plan.append({"experiment": "report", "gpu": False, "afterany": WORKERS})
    return plan


def slurm_environment():
    """Shell defaults must not hide jobs or override the fixed GPU resources."""
    return {name: value for name, value in os.environ.items()
            if not name.startswith(("SQUEUE_", "SBATCH_"))}


def inspect_queue():
    """One scheduler query before submission; no scheduler polling loops."""
    command = ["squeue", "--all", "--local", "--me", "--noheader",
               "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED,STOPPED",
               "--format=%i|%j|%b|%P"]
    response = subprocess.run(command, env=slurm_environment(), check=True, text=True, capture_output=True)
    for line in response.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4 or not all(fields):
            raise RuntimeError("Unexpected squeue output; no jobs submitted: " + line)
        job_id, name, resources, partition = fields
        if name.startswith(("rgbgeomchallenge_", "rgbgeomqual", "rgbgeomcompare_", "rgbgeommethods_")):
            raise RuntimeError("A campaign job is already queued: " + job_id)
        if "gpu" in resources.lower() or "gpu" in partition.lower():
            raise RuntimeError("Another GPU job is queued or active: " + job_id + " (" + name + ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Completed baseline/RGB/Semantic qualitative run; read only.")
    parser.add_argument("--lora-root", type=Path, required=True,
                        help="Completed LoRA qualitative run; read only.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print a static plan without reading runs, writing files or calling Slurm.")
    parser.add_argument("--worker-hours", type=int, default=48,
                        help="Reservation per GPU worker, not a runtime estimate (1–168 hours).")
    args = parser.parse_args()
    if not 1 <= args.worker_hours <= 168:
        parser.error("--worker-hours must be between 1 and 168.")
    plan = campaign_plan()
    description = {
        "gpu_allocation_jobs": 4, "cpu_jobs": 1, "maximum_concurrent_gpus": 4,
        "training_experiments": 0, "inference_variants": 10,
        "prompts": 40, "seeds_per_prompt": 2, "videos_per_variant": 80, "final_videos": 800,
        "worker_hours": args.worker_hours, "automatic_retries": False,
        "baseline_barrier": False, "automatic_requeue": False,
        "reference_root": str(args.reference_root), "lora_root": str(args.lora_root),
        "retained_formats": ["json", "csv", "mp4"], "new_checkpoints": 0,
        "minimum_free_bytes": 10_000_000_000,
        "free_space_check": "filesystem availability, not user quota",
        "ordered_queue": EXPERIMENTS, "jobs": plan,
    }
    if args.plan:
        print(json.dumps(description, indent=2))
        return 0

    record_path = None
    job_ids = {}
    lock_handle = None
    try:
        for command in ("sbatch", "squeue"):
            if shutil.which(command) is None:
                raise RuntimeError("Required command is unavailable: " + command)
        for wrapper in ("worker.sh", "report.sh", "common.sh", "environment.sh"):
            if not (JOB_DIRECTORY / wrapper).is_file():
                raise RuntimeError("Missing wrapper: " + wrapper)
        sys.path.insert(0, str(REPOSITORY / "src"))
        try:
            from rgb_geometry_pilot.protocol import safe_path
        finally:
            sys.path.pop(0)
        reference_root = safe_path(args.reference_root, existing=True)
        lora_root = safe_path(args.lora_root, existing=True)
        config_path = safe_path(args.config or (REPOSITORY / "configs/rgb_geometry_challenge_v1.json"), existing=True)
        parent = safe_path(os.environ.get("RGB_GEOMETRY_CHALLENGE_RESULTS_PARENT",
                                         "/fs04/scratch2/gt69/dhiazgeo"), existing=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempt = "rgb_geometry_challenge_" + timestamp + "_" + str(os.getpid())
        output_root = safe_path(args.output_root or (parent / attempt))
        if not output_root.is_absolute() or output_root.exists() or not output_root.parent.is_dir():
            raise RuntimeError("Use a fresh absolute output root whose parent already exists.")
        free_bytes = shutil.disk_usage(output_root.parent).free
        if free_bytes < description["minimum_free_bytes"]:
            raise RuntimeError("At least 10 GB of free filesystem space is required; user quota is not checked.")
        config_bytes = config_path.read_bytes()
        json.loads(config_bytes)
        arguments = ["--reference-root", str(reference_root), "--lora-root", str(lora_root),
                     "--output-root", str(output_root), "--config", str(config_path)]
        environment = slurm_environment()
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = str(REPOSITORY / "src")
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["WAN_REPO_ROOT"] = str(REPOSITORY)

        # Hashes, checkpoint receipts and frozen inputs are checked before sbatch.
        command = [sys.executable, "-B", "-u", "-m", "rgb_geometry_challenge.run",
                   "--experiment", "baseline", "--check-inputs"] + arguments
        check = subprocess.run(command, cwd=REPOSITORY, env=environment,
                               check=True, text=True, capture_output=True)
        input_check = json.loads(check.stdout)
        if input_check.get("status") != "inputs_valid":
            raise RuntimeError("Input check did not report inputs_valid.")
        lock_path = parent / ".rgb_geometry_campaign_submit_lock.json"
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another campaign submission helper is active; nothing submitted.") from error
        lock_handle.seek(0)
        lock_handle.truncate()
        json.dump({"pid": os.getpid(), "started_at": datetime.now().isoformat()}, lock_handle)
        lock_handle.write("\n")
        lock_handle.flush()
        inspect_queue()
        if config_path.read_bytes() != config_bytes:
            raise RuntimeError("Config changed during precheck; no jobs submitted.")

        output_root.mkdir()
        queue_worker.initialize_queue(output_root)
        record_path = output_root / "submissions.csv"
        description["reference_root"] = str(reference_root)
        description["lora_root"] = str(lora_root)
        (output_root / "input_check.json").write_text(json.dumps(input_check, indent=2) + "\n")
        (output_root / "submission_plan.json").write_text(json.dumps(description, indent=2) + "\n")
        snapshot = output_root / "config.json"
        snapshot.write_bytes(config_bytes)
        arguments = ["--reference-root", str(reference_root), "--lora-root", str(lora_root),
                     "--output-root", str(output_root), "--config", str(snapshot)]
        with record_path.open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stage", "job_id", "dependency", "gpu_allocation"])
            handle.flush()
            for row in plan:
                dependency = ""
                if row["afterany"]:
                    dependency = "afterany:" + ":".join(job_ids[name] for name in row["afterany"])
                command = ["sbatch", "--parsable", "--no-requeue", "--kill-on-invalid-dep=yes"]
                if dependency:
                    command.append("--dependency=" + dependency)
                if row["gpu"]:
                    command += ["--job-name=rgbgeomchallenge_" + row["experiment"],
                                "--time=" + str(args.worker_hours) + ":00:00",
                                str(JOB_DIRECTORY / "worker.sh"), row["experiment"]]
                else:
                    command.append(str(JOB_DIRECTORY / "report.sh"))
                command += arguments
                response = subprocess.run(command, cwd=REPOSITORY, env=environment,
                                          check=True, text=True, capture_output=True)
                match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", response.stdout.strip())
                if match is None:
                    raise RuntimeError("Unexpected sbatch reply; inspect accepted jobs before retrying: " + repr(response.stdout))
                job_id = match.group(1)
                job_ids[row["experiment"]] = job_id
                writer.writerow([row["experiment"], job_id, dependency, int(row["gpu"])])
                handle.flush()
                os.fsync(handle.fileno())
                print(job_id + "\t" + row["experiment"], flush=True)
        print("Submitted four independent A100-80G FIFO workers and one CPU report job.")
        print("800 videos: 40 prompts x 2 seeds x 10 variants; inference from existing weights only.")
        print("Baseline and trained variants share the queue. There is no baseline or method barrier.")
        print("At most four GPUs run in parallel; do not submit unrelated GPU work alongside this campaign.")
        print("Reservation per worker: " + str(args.worker_hours) + " hours; runtime and QoS are unverified.")
        print("Results: " + str(output_root))
        print("Job IDs: " + str(record_path))
        print("Retained outputs: final RGB MP4s, raw tracking records, rewards and provenance JSON/CSV; no new weights.")
        print("Free filesystem space checked: " + str(free_bytes) + " bytes; user quota must be checked separately.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Challenge submission stopped: " + str(error), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        if record_path is not None:
            print("Accepted jobs remain queued. Inspect " + str(record_path), file=sys.stderr)
            print("No automatic cancellation or resubmission was attempted.", file=sys.stderr)
        if job_ids:
            print("Known accepted job IDs:", file=sys.stderr)
            for experiment, job_id in job_ids.items():
                print(job_id + "\t" + experiment, file=sys.stderr)
        return 1
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
