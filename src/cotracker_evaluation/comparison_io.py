"""File and video helpers for repeatable, explicitly listed RGB comparisons."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIRECTORY.parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs/cotracker_evaluation/reference_rgb_20260909.json"
SOURCE_BUNDLES = (
    "reference_rgb_correction_59869249",
    "reference_rgb_strength_59884661",
    "additional_rgb_r4_59900883",
    "additional_rgb_r5_59900884",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        chunk = handle.read(8 * 1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(8 * 1024 * 1024)
    return digest.hexdigest()


def write_json_atomic(path, data):
    """Replace a JSON file only after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def package_hashes():
    """Fingerprint evaluator code; the runner hashes the selected manifest separately."""
    paths = list(PACKAGE_DIRECTORY.glob("*.py"))
    paths.append(REPOSITORY_ROOT / "jobs/cotracker_evaluation/run_cotracker_rgb.sh")
    return {str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path) for path in sorted(paths)}


def source_bundles(manifest=None):
    """A future batch names its input folders; old manifests keep the original four."""
    bundles = SOURCE_BUNDLES if manifest is None else manifest.get("source_bundles", SOURCE_BUNDLES)
    if not isinstance(bundles, (list, tuple)) or not bundles:
        raise ValueError("source_bundles must contain distinct result-folder names.")
    for name in bundles:
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
            raise ValueError("Each source bundle must be a plain folder name.")
        reject_reserved_path(name)
    if len(set(bundles)) != len(bundles):
        raise ValueError("source_bundles must contain distinct result-folder names.")
    return tuple(bundles)


def reject_reserved_path(path):
    if re.search(r"(?:^|[/_.-])(?:seed|s)[_-]?0*47(?=$|[/_.-])", str(path), re.IGNORECASE):
        raise ValueError("Reserved final-evaluation data is outside this evaluator's scope.")


def input_path(input_root, relative_path, allowed_bundles=SOURCE_BUNDLES):
    """Resolve only files in the batch's explicit source folders, without scanning."""
    reject_reserved_path(input_root)
    reject_reserved_path(relative_path)
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Expected a relative file path inside a saved result bundle.")
    if relative.parts[0] not in allowed_bundles:
        raise ValueError("Input is outside the explicitly listed result bundles.")
    root = Path(input_root).resolve(strict=True)
    reject_reserved_path(root)
    path = (root / relative).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError("Input must be a file inside the chosen result root.")
    resolved_relative = path.relative_to(root)
    reject_reserved_path(resolved_relative)
    if resolved_relative.parts[0] not in allowed_bundles:
        raise ValueError("Input symlink points outside the explicitly listed result bundles.")
    return path


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported input-manifest version.")
    cases = manifest.get("cases", [])
    case_ids = [case["case_id"] for case in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("Manifest must contain unique video cases.")
    if any(re.fullmatch(r"[a-z0-9_]+", case_id) is None for case_id in case_ids):
        raise ValueError("Invalid case identifier.")
    from .tracking_metrics import CLASS_NAMES
    if tuple(manifest["classes"]) != tuple(CLASS_NAMES):
        raise ValueError("Expected the existing eleven-class label mapping.")
    bundles = source_bundles(manifest)
    scene_conditions = [(case["scene"], case["condition"]) for case in cases]
    if len(scene_conditions) != len(set(scene_conditions)):
        raise ValueError("Each scene/condition must occur once; use separate manifests for different seeds or cohorts.")
    baseline_scenes = {case["scene"] for case in cases if case["condition"] == "baseline"}
    if {case["scene"] for case in cases} != baseline_scenes:
        raise ValueError("Every scene needs its matched baseline in the same manifest.")
    if manifest.get("case_count", len(cases)) != len(cases):
        raise ValueError("case_count does not match the listed cases.")
    settings = manifest["settings"]
    if min(settings["width"], settings["height"], settings["frame_count"]) < 2 or settings["fps"] <= 0:
        raise ValueError("Invalid video dimensions or frame rate.")
    for entry in manifest["source_files"]:
        if re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
            raise ValueError("Invalid source-file hash.")
        relative = Path(entry["relative_path"])
        reject_reserved_path(relative)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] not in bundles:
            raise ValueError("Source-file records must stay inside the listed result bundles.")
    return manifest


def validate_inputs(manifest, input_root):
    """Verify every recorded source file without loading models or writing data."""
    total_bytes = 0
    verified = {}
    bundles = source_bundles(manifest)
    for entry in manifest["source_files"]:
        relative = entry["relative_path"]
        if relative in verified:
            raise ValueError("Duplicate source-file record: " + relative)
        path = input_path(input_root, relative, allowed_bundles=bundles)
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            raise ValueError("Saved input hash differs from the reviewed artifact: " + relative)
        verified[relative] = actual_hash
        total_bytes += path.stat().st_size
    for case in manifest["cases"]:
        for field in ("video", "labels"):
            relative = case[field + "_relative_path"]
            if verified.get(relative) != case[field + "_sha256"]:
                raise ValueError("Case input is missing from the verified source list: " + relative)
    return {
        "input_files_present": True,
        "input_hashes_match": True,
        "checked_file_count": len(verified),
        "unique_video_count": len(manifest["cases"]),
        "total_checked_bytes": total_bytes,
        "input_root": str(Path(input_root).resolve()),
        "model_weights_loaded": False,
        "gpu_execution_tested": False,
    }


def ffmpeg_executable():
    # M3 has the bundled binary in imageio-ffmpeg. Local tests can use FFmpeg.
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg is missing; activate the prepared envcotrack environment.")
    return executable


def video_metadata(path):
    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        import imageio_ffmpeg
        frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(frames)
        finally:
            frames.close()
        return {"width": metadata["size"][0], "height": metadata["size"][1], "fps": metadata["fps"]}
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("Video metadata needs imageio-ffmpeg or a local ffprobe binary.")
    result = subprocess.run([
        executable, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return {"width": stream["width"], "height": stream["height"], "fps": float(numerator) / float(denominator)}


def read_rgb_video(path, settings):
    """Decode all original frames without resizing or changing their time order."""
    import numpy as np

    metadata = video_metadata(path)
    for name in ("width", "height"):
        if metadata[name] != settings[name]:
            raise ValueError("Unexpected video " + name + ": " + str(path))
    if abs(metadata["fps"] - settings["fps"]) > 0.01:
        raise ValueError("Video FPS differs from the fixed evaluation setting.")
    result = subprocess.run([
        ffmpeg_executable(), "-nostdin", "-v", "error", "-threads", "2",
        "-i", str(path), "-map", "0:v:0", "-an", "-vsync", "0",
        "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True)
    shape = (settings["frame_count"], settings["height"], settings["width"], 3)
    if len(result.stdout) != int(np.prod(shape)):
        raise ValueError("Decoded frame count differs from the expected full video.")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(shape).copy()
