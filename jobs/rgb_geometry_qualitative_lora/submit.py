"""Submit three saved-video LoRA runs after the first qualitative study is complete."""

import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
JOB_DIRECTORY = REPOSITORY / "jobs/rgb_geometry_qualitative_lora"
EXPERIMENTS = ["flash_grpo_L", "rloo_L", "flow_grpo_L"]


def campaign_plan():
    plan = [{"experiment": name, "gpu": True, "afterany": []} for name in EXPERIMENTS]
    plan.append({"experiment": "report", "gpu": False, "afterany": EXPERIMENTS.copy()})
    return plan


def slurm_environment():
    """Ignore inherited Slurm defaults that could hide or resize queued jobs."""
    return {name: value for name, value in os.environ.items()
            if not name.startswith(("SQUEUE_", "SBATCH_"))}


def inspect_queue():
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
        if name.startswith(("rgbgeomqual", "rgbgeomcompare_", "rgbgeommethods_")):
            raise RuntimeError("A campaign job is already queued: " + job_id)
        if "gpu" in resources.lower() or "gpu" in partition.lower():
            raise RuntimeError("Another GPU job is queued or active: " + job_id + " (" + name + ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path,
                        help="Completed first-stage seven-variant results directory.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print the plan without reading inputs or calling Slurm.")
    parser.add_argument("--hours", type=int, default=30, help="Reservation per LoRA run, from 1 to 168 hours.")
    args = parser.parse_args()
    if not args.reference_root.is_absolute():
        parser.error("--reference-root must be an absolute path.")
    if args.output_root is not None and not args.output_root.is_absolute():
        parser.error("--output-root must be an absolute path.")
    if not 1 <= args.hours <= 168:
        parser.error("--hours must be between 1 and 168.")
    plan = campaign_plan()
    description = {
        "gpu_allocation_jobs": 3, "cpu_jobs": 1, "training_experiments": 3,
        "baseline_experiments": 0, "maximum_concurrent_gpus": 3,
        "hours_per_arm": args.hours, "automatic_retries": False,
        "automatic_phase_chaining": False, "reference_root": str(args.reference_root),
        "retained_formats": ["json", "csv", "pt", "mp4"],
        "new_final_videos": 60, "new_checkpoints": 33, "combined_variants": 10,
        "minimum_free_bytes": 5_000_000_000,
        "free_space_check": "filesystem availability, not user quota", "jobs": plan,
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
        for filename in ("arm.sh", "report.sh"):
            if not (JOB_DIRECTORY / filename).is_file():
                raise RuntimeError("Missing job wrapper: " + filename)
        sys.path.insert(0, str(REPOSITORY / "src"))
        try:
            from rgb_geometry_pilot.protocol import safe_path
        finally:
            sys.path.pop(0)
        # Guard every supplied path before reading any configuration or results.
        reference_root = safe_path(args.reference_root, existing=True)
        config_path = safe_path(args.config or (REPOSITORY / "configs/rgb_geometry_qualitative_lora_v1.json"),
                                existing=True)
        parent = safe_path(os.environ.get("RGB_GEOMETRY_QUALITATIVE_LORA_RESULTS_PARENT",
                                          "/home/dhiazr/gt69_scratch2/dhiazgeo"), existing=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempt = "rgb_geometry_qualitative_lora_" + timestamp + "_" + str(os.getpid())
        output_root = safe_path(args.output_root or (parent / attempt))
        if output_root.exists() or not output_root.parent.is_dir():
            raise RuntimeError("Use a fresh absolute output root whose parent already exists.")
        free_bytes = shutil.disk_usage(output_root.parent).free
        if free_bytes < description["minimum_free_bytes"]:
            raise RuntimeError("At least 5 GB of free filesystem space is required; user quota is not checked.")
        if not config_path.is_file():
            raise RuntimeError("Config is not a file: " + str(config_path))
        config_bytes = config_path.read_bytes()
        json.loads(config_bytes)
        arguments = ["--reference-root", str(reference_root), "--output-root", str(output_root),
                     "--config", str(config_path)]
        environment = slurm_environment()
        environment.pop("PYTHONHOME", None)
        environment.update({"PYTHONPATH": str(REPOSITORY / "src"), "PYTHONNOUSERSITE": "1",
                            "PYTHONDONTWRITEBYTECODE": "1", "WAN_REPO_ROOT": str(REPOSITORY)})
        command = [sys.executable, "-B", "-u", "-m", "rgb_geometry_qualitative_lora.run",
                   "--experiment", EXPERIMENTS[0], "--check-inputs"] + arguments
        check = subprocess.run(command, cwd=REPOSITORY, env=environment,
                               check=True, text=True, capture_output=True)
        input_check = json.loads(check.stdout)
        if input_check.get("status") != "inputs_valid":
            raise RuntimeError("Input check did not report inputs_valid.")
        # Share the earlier campaign's lock; leave its inode in place afterwards.
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
        record_path = output_root / "submissions.csv"
        description["reference_root"] = str(reference_root)
        (output_root / "input_check.json").write_text(json.dumps(input_check, indent=2) + "\n")
        (output_root / "submission_plan.json").write_text(json.dumps(description, indent=2) + "\n")
        snapshot = output_root / "config.json"
        snapshot.write_bytes(config_bytes)
        arguments = ["--reference-root", str(reference_root), "--output-root", str(output_root),
                     "--config", str(snapshot)]
        with record_path.open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stage", "job_id", "dependency", "gpu_allocation"])
            handle.flush()
            for row in plan:
                dependency = "afterany:" + ":".join(job_ids[name] for name in row["afterany"]) if row["afterany"] else ""
                command = ["sbatch", "--parsable", "--no-requeue", "--kill-on-invalid-dep=yes"]
                if dependency:
                    command.append("--dependency=" + dependency)
                if row["gpu"]:
                    command += ["--job-name=rgbgeomqualora_" + row["experiment"],
                                "--time=" + str(args.hours) + ":00:00",
                                str(JOB_DIRECTORY / "arm.sh"), row["experiment"]]
                else:
                    command.append(str(JOB_DIRECTORY / "report.sh"))
                command += arguments
                response = subprocess.run(command, cwd=REPOSITORY, env=environment,
                                          check=True, text=True, capture_output=True)
                match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", response.stdout.strip())
                if match is None:
                    raise RuntimeError("Unexpected sbatch reply; inspect the queue before retrying: " + repr(response.stdout))
                job_id = match.group(1)
                job_ids[row["experiment"]] = job_id
                writer.writerow([row["experiment"], job_id, dependency, int(row["gpu"])])
                handle.flush()
                os.fsync(handle.fileno())
                print(job_id + "\t" + row["experiment"], flush=True)
        print("Submitted three independent A100-80G LoRA jobs and one CPU report job.")
        print("No baseline or RGB/Semantic training jobs are submitted by this script.")
        print("Reservation per LoRA arm: " + str(args.hours) + " hours; runtime and QoS remain unverified.")
        print("Reference results: " + str(reference_root))
        print("New results: " + str(output_root))
        print("Job IDs: " + str(record_path))
        print("Retained new outputs: 33 adapter/recovery checkpoints, 60 final RGB videos, JSON/CSV records.")
        print("Free filesystem space checked: " + str(free_bytes) + " bytes; user quota must be checked separately.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("LoRA submission stopped: " + str(error), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        if record_path is not None:
            print("Already submitted jobs remain queued. Inspect " + str(record_path), file=sys.stderr)
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
