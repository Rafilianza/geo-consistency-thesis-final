"""Submit the saved-video study: one baseline, four FIFO workers and one report.

Submission and locking follow the existing comparison helper. This separate
entry point leaves the historical metrics-only campaign unchanged.
"""

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
JOB_DIRECTORY = REPOSITORY / "jobs/rgb_geometry_qualitative"
# Resolve this sibling explicitly so another campaign's worker module cannot
# be selected when several submission helpers are imported by the tests.
worker_spec = importlib.util.spec_from_file_location("qualitative_fifo_worker", JOB_DIRECTORY / "worker.py")
queue_worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(queue_worker)
EXPERIMENTS = queue_worker.EXPERIMENTS
METHODS = queue_worker.METHODS
WORKERS = queue_worker.WORKERS
initialize_queue = queue_worker.initialize_queue


def campaign_plan():
    """Workers share a FIFO; the report waits for every ancestor explicitly."""
    plan = [{"experiment": "baseline", "gpu": True, "afterok": [], "afterany": []}]
    for worker_name in WORKERS:
        plan.append({"experiment": worker_name, "gpu": True,
                     "afterok": ["baseline"], "afterany": []})
    plan.append({"experiment": "report", "gpu": False,
                 "afterok": [], "afterany": ["baseline"] + WORKERS})
    return plan


submission_plan = campaign_plan


def slurm_environment():
    """User shell defaults must not hide jobs or override fixed job resources."""
    return {name: value for name, value in os.environ.items()
            if not name.startswith(("SQUEUE_", "SBATCH_"))}


def inspect_queue():
    command = [
        "squeue", "--all", "--local", "--me", "--noheader",
        "--states=PENDING,RUNNING,CONFIGURING,COMPLETING,SUSPENDED,STOPPED",
        "--format=%i|%j|%b|%P",
    ]
    response = subprocess.run(command, env=slurm_environment(), check=True, text=True, capture_output=True)
    for line in response.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 4 or not all(fields):
            raise RuntimeError("Unexpected squeue output; no jobs submitted: " + line)
        job_id, name, resources, partition = fields
        if name.startswith(("rgbgeomqual_", "rgbgeomcompare_", "rgbgeommethods_")):
            raise RuntimeError("A campaign job is already queued: " + job_id)
        if "gpu" in resources.lower() or "gpu" in partition.lower():
            raise RuntimeError("Another GPU job is queued or active: " + job_id + " (" + name + ")")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--plan", action="store_true", help="Print the dependency plan without Slurm calls or files.")
    parser.add_argument("--worker-hours", type=int, default=48,
                        help="Slurm reservation for each multi-arm worker (1–168 hours).")
    args = parser.parse_args()
    if not 1 <= args.worker_hours <= 168:
        parser.error("--worker-hours must be between 1 and 168.")
    plan = campaign_plan()
    description = {
        "gpu_allocation_jobs": 5, "cpu_jobs": 1, "training_experiments": 6,
        "baseline_experiments": 1, "maximum_concurrent_gpus": 4,
        "worker_hours": args.worker_hours, "automatic_retries": False,
        "retained_formats": ["json", "csv", "pt", "mp4"],
        "final_videos": 140, "minimum_free_bytes": 5_000_000_000,
        "free_space_check": "filesystem availability, not user quota",
        "method_order": list(METHODS),
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
        for row in plan:
            wrapper = "worker.sh" if row["experiment"] in WORKERS else row["experiment"] + ".sh"
            if not (JOB_DIRECTORY / wrapper).is_file():
                raise RuntimeError("Missing wrapper: " + row["experiment"])

        # This standard-library-only guard rejects reserved names before probing
        # supplied paths, while preserving the existing safe scratch aliases.
        sys.path.insert(0, str(REPOSITORY / "src"))
        try:
            from rgb_geometry_pilot.protocol import safe_path
        finally:
            sys.path.pop(0)
        config_path = safe_path(args.config or (REPOSITORY / "configs/rgb_geometry_qualitative_v1.json"),
                                existing=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempt = "rgb_geometry_qualitative_" + timestamp + "_" + str(os.getpid())
        parent = safe_path(os.environ.get("RGB_GEOMETRY_QUALITATIVE_RESULTS_PARENT",
                                          "/home/dhiazr/gt69_scratch2/dhiazgeo"), existing=True)
        output_root = safe_path(args.output_root or (parent / attempt))
        if not output_root.is_absolute() or output_root.exists() or not output_root.parent.is_dir():
            raise RuntimeError("Use a fresh absolute output root whose parent already exists.")
        free_bytes = shutil.disk_usage(output_root.parent).free
        if free_bytes < description["minimum_free_bytes"]:
            raise RuntimeError("At least 5 GB of free filesystem space is required; user quota is not checked.")
        if not config_path.is_file():
            raise RuntimeError("Config is not a file: " + str(config_path))
        config_bytes = config_path.read_bytes()
        json.loads(config_bytes)
        config_args = ["--config", str(config_path)]
        arguments = ["--output-root", str(output_root)] + config_args
        environment = slurm_environment()
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = str(REPOSITORY / "src")
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["WAN_REPO_ROOT"] = str(REPOSITORY)

        # The runner checks all campaign inputs using only standard-library I/O.
        command = [sys.executable, "-B", "-u", "-m", "rgb_geometry_qualitative.run",
                   "--experiment", "baseline", "--check-inputs"] + arguments
        check = subprocess.run(command, cwd=REPOSITORY, env=environment,
                               check=True, text=True, capture_output=True)
        input_check = json.loads(check.stdout)
        if input_check.get("status") != "inputs_valid":
            raise RuntimeError("Input check did not report inputs_valid.")
        # Keep this inode: unlinking a lock would let another helper lock a
        # different inode while this submission is still using the old one.
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
        initialize_queue(output_root)
        record_path = output_root / "submissions.csv"
        (output_root / "input_check.json").write_text(json.dumps(input_check, indent=2) + "\n")
        (output_root / "submission_plan.json").write_text(json.dumps(description, indent=2) + "\n")
        snapshot = output_root / "config.json"
        snapshot.write_bytes(config_bytes)
        arguments = ["--output-root", str(output_root), "--config", str(snapshot)]
        with record_path.open("x", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stage", "job_id", "dependency", "gpu_allocation"])
            handle.flush()
            for row in plan:
                clauses = []
                for kind in ("afterok", "afterany"):
                    parents = row[kind]
                    if parents:
                        clauses.append(kind + ":" + ":".join(job_ids[parent] for parent in parents))
                dependency = ",".join(clauses)
                command = ["sbatch", "--parsable", "--no-requeue", "--kill-on-invalid-dep=yes"]
                if dependency:
                    command.append("--dependency=" + dependency)
                if row["experiment"] in WORKERS:
                    command += ["--job-name=rgbgeomqual_" + row["experiment"],
                                "--time=" + str(args.worker_hours) + ":00:00",
                                str(JOB_DIRECTORY / "worker.sh"), row["experiment"]]
                else:
                    command.append(str(JOB_DIRECTORY / (row["experiment"] + ".sh")))
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
        print("Submitted one baseline, four A100-80G FIFO workers and one CPU report job.")
        print("Six training experiments share four slots: Flash RGB/Semantic, RLOO RGB/Semantic, Flow RGB/Semantic.")
        print("Workers start only after the baseline succeeds; final videos use the same evaluation cases.")
        print("Workers take the next unstarted arm immediately; there are no method barriers.")
        print("Worker reservation: " + str(args.worker_hours) + " hours; actual runtime and QoS are unverified.")
        print("Do not submit unrelated GPU work alongside this campaign.")
        print("Results: " + str(output_root))
        print("Job IDs: " + str(record_path))
        print("Retained outputs: round checkpoints with optimizer state, 140 final RGB videos, JSON/CSV records.")
        print("Free filesystem space checked: " + str(free_bytes) + " bytes; user quota must be checked separately.")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("Campaign submission stopped: " + str(error), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            if error.stdout:
                print(error.stdout, file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        if record_path is not None:
            print("Already submitted jobs remain queued. Inspect " + str(record_path), file=sys.stderr)
            print("No automatic cancellation or resubmission was attempted.", file=sys.stderr)
        if job_ids:
            print("Known accepted job IDs (even if writing the CSV failed):", file=sys.stderr)
            for experiment, job_id in job_ids.items():
                print(job_id + "\t" + experiment, file=sys.stderr)
        return 1
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
