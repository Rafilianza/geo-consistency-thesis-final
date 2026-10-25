"""Keep small trainable checkpoints and verified RGB previews, never frozen models."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile

from rgb_geometry_pilot.protocol import file_hash, safe_path


CHECKPOINT_FORMAT = "rgb_geometry_qualitative_checkpoint_v1"
VIDEO_FORMAT = "rgb_geometry_qualitative_video_v1"
VIDEO_SHAPE = (81, 480, 832, 3)


def _checked_path(value, existing=False):
    # Reject reserved names and symlinks before resolving or opening their targets.
    path = Path(value).expanduser()
    reserved = r"(?:seed[ _=-]?|(?:^|[/_.-])s[ _=-]?)0*47(?:\D|$)"
    if re.search(reserved, str(path), re.I):
        raise ValueError("Reserved data is outside this experiment.")
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError("Artifact paths must not contain symlinks.")
    return safe_path(path, existing=existing)


def _prepare_output(value, suffix):
    path = _checked_path(value)
    if path.suffix != suffix:
        raise ValueError(f"Expected a {suffix} artifact.")
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"Preserving an existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _temporary_path(path):
    descriptor, name = tempfile.mkstemp(prefix=".partial_", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _publish(temporary, path):
    # A hard link publishes a complete file atomically and refuses to overwrite.
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _save_receipt(path, receipt):
    sidecar = path.with_suffix(".json")
    temporary = _temporary_path(sidecar)
    try:
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        _publish(temporary, sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def _policy_api():
    from rgb_geometry_campaign import policy
    return policy


def _rng_state():
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(saved):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    numpy_state = saved["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(saved["torch_cpu"])
    if saved["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("This training checkpoint needs CUDA RNG restoration.")
        torch.cuda.set_rng_state_all(saved["torch_cuda"])


def _validate_policy(saved, runtime, completed_round):
    import torch
    if type(completed_round) is not int or not 0 <= completed_round <= 10:
        raise ValueError("Checkpoint round must be an integer from zero to ten.")
    if saved["arm"] not in ("R", "S", "L") or saved["arm"] != runtime["arm"]:
        raise ValueError("Only the matching R, S or L trainable policy can be restored.")
    version = saved["parameter_version"]
    if type(version) is not int or version < 0:
        raise ValueError("Invalid policy parameter version.")
    # The existing policy API selects head tensors or LoRA matrices only.
    expected = _policy_api().state_dict(runtime)["parameters"]
    if set(saved["parameters"]) != set(expected):
        raise ValueError("Checkpoint does not contain exactly the trainable policy tensors.")
    for name, value in saved["parameters"].items():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != expected[name].shape or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid trainable policy tensor: {name}")


def save_checkpoint(path, runtime, optimizer, completed_round, identity, progress):
    """Save only R/S heads or L adapters, optimizer and RNG after a completed round."""
    import torch
    path = _prepare_output(path, ".pt")
    identity = json.loads(json.dumps(identity, allow_nan=False))
    progress = json.loads(json.dumps(progress, allow_nan=False))
    saved_policy = _policy_api().state_dict(runtime)
    _validate_policy(saved_policy, runtime, completed_round)
    payload = {"format": CHECKPOINT_FORMAT, "identity": identity,
               "completed_round": completed_round, "progress": progress,
               "policy": saved_policy, "optimizer": optimizer.state_dict() if optimizer else None,
               "rng": _rng_state()}
    temporary = _temporary_path(path)
    try:
        torch.save(payload, temporary)
        receipt = {"format": CHECKPOINT_FORMAT, "path": str(path),
                   "sha256": file_hash(temporary), "completed_round": completed_round,
                   "parameter_version": saved_policy["parameter_version"], "identity": identity}
        _publish(temporary, path)
        _save_receipt(path, receipt)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)


def restore_checkpoint(path, runtime, optimizer, identity, restore_training=True):
    """Load verified trainable weights; inference leaves optimizer and RNG unchanged."""
    path = _checked_path(path, existing=True)
    sidecar = _checked_path(path.with_suffix(".json"), existing=True)
    receipt = json.loads(sidecar.read_text())
    if receipt.get("format") != CHECKPOINT_FORMAT or file_hash(path) != receipt.get("sha256"):
        raise ValueError("Checkpoint receipt or SHA256 verification failed.")
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported checkpoint format.")
    expected_identity = json.loads(json.dumps(identity, allow_nan=False))
    if payload["identity"] != expected_identity or receipt["identity"] != expected_identity:
        raise ValueError("Checkpoint identity differs from this run.")
    _validate_policy(payload["policy"], runtime, payload["completed_round"])
    if (receipt["completed_round"] != payload["completed_round"]
            or receipt["parameter_version"] != payload["policy"]["parameter_version"]):
        raise ValueError("Checkpoint receipt does not match its policy state.")
    if restore_training and optimizer is not None and payload["optimizer"] is None:
        raise ValueError("Training continuation needs the saved optimizer state.")
    _policy_api().load_state_dict(runtime, payload["policy"])
    if restore_training and optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
        optimizer.zero_grad(set_to_none=True)
        _restore_rng(payload["rng"])
    return payload


def check_video_encoder():
    """Check a CPU H.264 encoder without loading any research model."""
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    else:
        executable = shutil.which("ffmpeg")
        if executable is None or shutil.which("ffprobe") is None:
            raise RuntimeError("Install imageio-ffmpeg, or FFmpeg and ffprobe, for video export.")
    result = subprocess.run([executable, "-hide_banner", "-encoders"], check=True,
                            capture_output=True, text=True, timeout=30)
    if "libx264" not in result.stdout:
        raise RuntimeError("FFmpeg has no libx264 encoder; no codec fallback is allowed.")
    return {"executable": executable, "codec": "libx264"}


def _validate_frames(frames, expected_shape=VIDEO_SHAPE):
    import numpy as np
    values = np.asarray(frames)
    if values.dtype != np.uint8 or values.shape != expected_shape:
        raise ValueError(f"Expected uint8 RGB frames with shape {expected_shape}.")
    return np.ascontiguousarray(values)


def _encode_video(path, frames, fps):
    executable = check_video_encoder()["executable"]
    command = [executable, "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{frames.shape[2]}x{frames.shape[1]}",
               "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE) as process:
        try:
            for frame in frames:
                process.stdin.write(frame.tobytes())
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.stdin = None
        _, error = process.communicate(timeout=120)
        if process.returncode:
            raise RuntimeError("H.264 video encoding failed: " + error.decode(errors="replace"))


def _verify_video(path, expected_shape, fps):
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        import imageio_ffmpeg
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(reader)
            width, height = metadata["size"]
            actual_fps = metadata["fps"]
            frame_count = sum(1 for _ in reader)
        finally:
            reader.close()
    else:
        command = [shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                   "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        metadata = json.loads(result.stdout)["streams"][0]
        width, height = metadata["width"], metadata["height"]
        numerator, denominator = metadata["avg_frame_rate"].split("/")
        actual_fps = float(numerator) / float(denominator)
        frame_count = int(metadata["nb_read_frames"])
    if ((width, height) != (expected_shape[2], expected_shape[1])
            or abs(actual_fps - fps) > 0.01 or frame_count != expected_shape[0]):
        raise ValueError("Encoded video failed its decoded dimensions, frame count or FPS check.")


def save_video(path, frames, case, policy_digest, fps=16):
    """Save one lossy RGB preview; scores must be calculated from original pixels."""
    if fps != 16:
        raise ValueError("The registered evaluation videos use 16 FPS.")
    path = _prepare_output(path, ".mp4")
    frames = _validate_frames(frames)
    case = json.loads(json.dumps(case, allow_nan=False))
    check_video_encoder()
    source_digest = hashlib.sha256()
    for frame in frames:
        source_digest.update(frame.tobytes())
    temporary = _temporary_path(path)
    try:
        _encode_video(temporary, frames, fps)
        _verify_video(temporary, frames.shape, fps)
        receipt = {"format": VIDEO_FORMAT, "path": str(path),
                   "video_sha256": file_hash(temporary), "source_rgb_sha256": source_digest.hexdigest(),
                   "case": case, "case_id": case.get("case_id", case.get("id")),
                   "policy_digest": policy_digest, "frames": len(frames),
                   "height": frames.shape[1], "width": frames.shape[2], "fps": fps,
                   "encoding": "H.264 libx264, yuv420p, CRF 18",
                   "scoring_pixels": "original uint8 RGB frames before lossy MP4 encoding"}
        _publish(temporary, path)
        _save_receipt(path, receipt)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)
