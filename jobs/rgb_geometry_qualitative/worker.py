"""Four single-GPU workers consume one ordered queue; claimed arms never retry.

The locking and cancellation design is copied from the historical comparison
worker so this new saving campaign does not change older jobs.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


METHODS = ("flash_grpo", "rloo", "flow_grpo")
EXPERIMENTS = [method + "_" + arm for method in METHODS for arm in ("R", "S")]
WORKERS = ["worker_" + str(number) for number in range(1, 5)]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_state(root, state):
    """Replace the whole receipt atomically while holding the separate lock."""
    temporary = root / ("scheduler_state_" + str(os.getpid()) + ".json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, root / "scheduler_state.json")


@contextmanager
def locked_state(root):
    # Never unlink this lock: all workers must lock the same inode.
    with (root / "scheduler_lock.json").open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        state = json.loads((root / "scheduler_state.json").read_text())
        yield state


def initialize_queue(root):
    """Called once by submission, before any Slurm job is submitted."""
    with (root / "scheduler_lock.json").open("x") as handle:
        handle.write("{}\n")
    state = {
        "method_order": list(METHODS), "maximum_concurrent_gpus": 4,
        "automatic_retries": False, "created_utc": timestamp(),
        "workers": {name: {"status": "pending"} for name in WORKERS},
        "arms": [{"experiment": name, "status": "pending"} for name in EXPERIMENTS],
    }
    write_state(root, state)


def run_queue(root, worker_name, command_builder):
    """Run each claimed process to its exit before this worker claims another."""
    if worker_name not in WORKERS:
        raise ValueError("Use one of the four fixed worker names.")
    with locked_state(root) as state:
        worker = state["workers"][worker_name]
        if worker["status"] != "pending":
            raise RuntimeError("This worker has already started; automatic restarts are forbidden.")
        worker.update(status="running", started_utc=timestamp(),
                      job_id=os.environ.get("SLURM_JOB_ID", ""), pid=os.getpid())
        write_state(root, state)

    process = None
    received_signal = None
    signal_time = None
    worker_error = None
    old_handlers = {}

    def stop_worker(signum, frame):
        nonlocal received_signal, signal_time
        received_signal = signum
        signal_time = time.monotonic()
        if process is not None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, stop_worker)
        while received_signal is None:
            # Persist the claim and launch under one lock, so a later arm cannot
            # launch ahead of an earlier unclaimed Flash/RLOO arm.
            experiment = None
            with locked_state(root) as state:
                if received_signal is not None:
                    break
                pending = [arm for arm in state["arms"] if arm["status"] == "pending"]
                if not pending:
                    break
                arm = pending[0]
                experiment = arm["experiment"]
                arm.update(status="running", worker=worker_name, started_utc=timestamp(),
                           job_id=os.environ.get("SLURM_JOB_ID", ""))
                write_state(root, state)
                try:
                    command = command_builder(experiment)
                    if received_signal is not None:
                        arm.update(status="interrupted", finished_utc=timestamp())
                        write_state(root, state)
                        break
                    process = subprocess.Popen(command, start_new_session=True)
                    if received_signal is not None:
                        stop_worker(received_signal, None)
                except OSError as error:
                    arm.update(status="failed", finished_utc=timestamp(), returncode=None,
                               error=type(error).__name__ + ": " + str(error))
                    write_state(root, state)
                    print(experiment + ": launch failed: " + str(error), flush=True)
                    continue

            print(worker_name + " started " + experiment, flush=True)
            while process.poll() is None:
                if signal_time is not None and time.monotonic() - signal_time > 15:
                    # The existing CSV launcher forwards TERM to its child and
                    # kills an unresponsive child after ten seconds.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            returncode = process.returncode
            process = None
            with locked_state(root) as state:
                arm = next(arm for arm in state["arms"] if arm["experiment"] == experiment)
                status = "complete" if returncode == 0 else "failed"
                if received_signal is not None:
                    status = "interrupted"
                arm.update(status=status, finished_utc=timestamp(), returncode=returncode)
                write_state(root, state)
            print(experiment + ": " + status, flush=True)
    except Exception as error:
        worker_error = type(error).__name__ + ": " + str(error)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        with locked_state(root) as state:
            worker = state["workers"][worker_name]
            status = "failed" if worker_error is not None else "complete"
            if received_signal is not None:
                status = "interrupted"
            worker.update(status=status, finished_utc=timestamp(),
                          received_signal=received_signal, error=worker_error)
            if worker_error is not None:
                for arm in state["arms"]:
                    if arm.get("worker") == worker_name and arm["status"] == "running":
                        arm.update(status="failed", finished_utc=timestamp(), error=worker_error)
            write_state(root, state)
    return 128 + received_signal if received_signal is not None else 0


def finalize_queue(root):
    """Called by the report only after all workers are terminal in Slurm."""
    with locked_state(root) as state:
        for entry in list(state["workers"].values()) + state["arms"]:
            if entry["status"] == "running":
                entry.update(status="interrupted_or_worker_lost", reconciled_utc=timestamp())
            elif entry["status"] == "pending":
                entry.update(status="not_started", reconciled_utc=timestamp())
        state["finalized_utc"] = timestamp()
        write_state(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKERS)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from rgb_geometry_qualitative import protocol
    config = protocol.load_config(args.config)
    protocol.runtime_args(args, config)
    args.output_root = protocol.validate_output(args)
    if not (args.output_root / "input_check.json").is_file():
        parser.error("The queue needs a submission input/provenance receipt.")
    protocol.check_cpu_identity(args)
    if args.finalize:
        finalize_queue(args.output_root)
        return 0
    if args.worker is None:
        parser.error("Provide --worker or --finalize.")
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        parser.error("Run workers only through the supplied Slurm submission helper.")
    baseline_status = args.output_root / "baseline/status.json"
    if not baseline_status.is_file() or json.loads(baseline_status.read_text()).get("status") != "complete":
        parser.error("Workers require a completed baseline receipt before claiming any training arm.")

    def command_builder(experiment):
        return [sys.executable, "-B", "-u", "-m", "rgb_geometry_campaign.launcher",
                "--output-dir", str(args.output_root / experiment), "--",
                sys.executable, "-B", "-u", "-m", "rgb_geometry_qualitative.run",
                "--experiment", experiment, "--output-root", str(args.output_root),
                "--config", str(args.config)]

    return run_queue(args.output_root, args.worker, command_builder)


if __name__ == "__main__":
    sys.exit(main())
