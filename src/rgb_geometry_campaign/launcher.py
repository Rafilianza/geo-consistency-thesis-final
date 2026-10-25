"""Capture setup and runtime output as genuine CSV, without retained raw logs."""

import argparse
import codecs
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


LOG_COLUMNS = ["time_utc", "elapsed_seconds", "job_id", "event", "stream", "message"]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def launch(output_dir, command):
    """Stream both pipes without deadlock, and preserve the child's exit status."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "launcher_status.json"
    if result_path.exists():
        raise FileExistsError("Preserving a previous launch; use a fresh campaign root: " + str(directory))
    log_path = directory / "execution_log.csv"
    started = time.monotonic()
    started_utc = timestamp()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    process = None
    received_signal = None
    signal_time = None
    old_handlers = {}
    error_text = None
    returncode = 1

    with log_path.open("a+", newline="", encoding="utf-8") as handle:
        # A second launch may not write interleaved records to the same stage.
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result_path.exists():
            raise FileExistsError("A previous launch finished; use a fresh campaign root.")
        if os.fstat(handle.fileno()).st_size:
            raise FileExistsError("Preserving an earlier execution log; use a fresh campaign root.")
        writer = csv.writer(handle)
        writer.writerow(LOG_COLUMNS)

        def record(event, stream, message):
            writer.writerow([timestamp(), round(time.monotonic() - started, 6), job_id, event, stream, message])
            handle.flush()

        def forward_signal(signum, frame):
            nonlocal received_signal, signal_time
            received_signal = signum
            if signal_time is None:
                signal_time = time.monotonic()
            record("signal", "launcher", str(signum))
            if process is not None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

        record("start", "launcher", json.dumps(command))
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                old_handlers[signum] = signal.signal(signum, forward_signal)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       bufsize=0, start_new_session=True)
            if received_signal is not None:
                os.killpg(process.pid, received_signal)
            # Read chunks, not readline(): progress bars may never emit a newline,
            # and one pipe must not block draining the other pipe.
            with selectors.DefaultSelector() as selector:
                decoders = {}
                for stream_name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                    selector.register(pipe, selectors.EVENT_READ, stream_name)
                    decoders[stream_name] = codecs.getincrementaldecoder("utf-8")(errors="replace")
                while selector.get_map():
                    if signal_time is not None and time.monotonic() - signal_time > 10:
                        # A dead shell can leave a TERM-ignoring descendant
                        # holding these pipes open. Kill the whole group even
                        # when the direct child has already exited.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        signal_time = None
                    for key, _ in selector.select(timeout=0.25):
                        block = os.read(key.fileobj.fileno(), 16384)
                        stream_name = key.data
                        if block:
                            text = decoders[stream_name].decode(block)
                            if text:
                                record("output", stream_name, text)
                        else:
                            tail = decoders[stream_name].decode(b"", final=True)
                            if tail:
                                record("output", stream_name, tail)
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
            returncode = process.wait()
            if received_signal is not None and returncode == 0:
                returncode = -received_signal
        except Exception as error:
            error_text = type(error).__name__ + ": " + str(error)
            record("error", "launcher", error_text)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        finally:
            for signum, old_handler in old_handlers.items():
                signal.signal(signum, old_handler)
            if process is not None:
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
            record("exit", "launcher", str(returncode))
            os.fsync(handle.fileno())

        result = {
            "status": "complete" if returncode == 0 else "technical_failure",
            "returncode": returncode, "command": command, "job_id": job_id,
            "started_utc": started_utc, "finished_utc": timestamp(),
            "elapsed_seconds": time.monotonic() - started,
            "received_signal": received_signal, "error": error_text,
        }
        with result_path.open("x", encoding="utf-8") as result_handle:
            json.dump(result, result_handle, indent=2, allow_nan=False)
            result_handle.write("\n")
        if returncode != 0:
            # Preserve a more specific failure report produced by the experiment.
            try:
                with (directory / "failure.json").open("x", encoding="utf-8") as failure_handle:
                    json.dump(dict(result, component="launcher"), failure_handle, indent=2, allow_nan=False)
                    failure_handle.write("\n")
            except FileExistsError:
                pass
    return returncode if returncode >= 0 else 128 - returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("Provide a command after --.")
    if not args.output_dir.is_absolute():
        parser.error("Use an absolute output directory.")
    try:
        return launch(args.output_dir, command)
    except (OSError, ValueError) as error:
        print("Campaign logger failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
