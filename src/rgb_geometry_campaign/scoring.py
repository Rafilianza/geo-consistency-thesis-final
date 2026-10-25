"""Generated-RGB decoding, in-memory YOLO labels and temporary-only tracking I/O."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from rgb_geometry_campaign.track import expected_class_ids


SOURCE_NAMES = ["background", "person", "car", "bus", "bicycle", "truck", "road",
                "sidewalk", "building", "vegetation", "sky"]
CODEC_NAMES = ["other", "car", "bus", "truck", "road", "sidewalk", "building", "vegetation", "sky"]
CODEC_MAPPING = np.asarray([255, 0, 1, 2, 0, 3, 4, 5, 6, 7, 8], dtype=np.uint8)
FRAME_SHAPE = (81, 480, 832)


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def offload_for_scoring(runtime):
    """Make space for VAE, teacher and the isolated tracker, one stage at a time."""
    pipeline = runtime["pipeline"]
    modules = [pipeline.transformer, pipeline.text_encoder, pipeline.vae]
    modules += [runtime.get(name) for name in ("semantic", "codec", "head", "observer", "campaign_observer")]
    for module in modules:
        if module is not None and hasattr(module, "to"):
            module.to("cpu")
    runtime["teacher"].to("cpu")
    runtime["embeddings"] = {
        prompt: (positive.cpu(), negative.cpu())
        for prompt, (positive, negative) in runtime.get("embeddings", {}).items()
    }
    if runtime["device"].type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def decode_rgb(runtime, latent):
    """Decode a newly generated final latent without encoding or saving a video."""
    if tuple(latent.shape) != (1, 16, 21, 60, 104) or not bool(torch.isfinite(latent).all()):
        raise ValueError("Expected a finite final Wan latent [1,16,21,60,104].")
    offload_for_scoring(runtime)
    pipeline, device = runtime["pipeline"], runtime["device"]
    try:
        pipeline.vae.to(device)
        video = pipeline._decode_latents_to_video(latent.to(device))
        decoded = np.asarray(pipeline.video_processor.postprocess_video(video, output_type="np")[0])
        if (decoded.shape != (*FRAME_SHAPE, 3) or not np.isfinite(decoded).all()
                or decoded.min() < 0 or decoded.max() > 1):
            raise ValueError("VAE returned nonfinite or misaligned RGB pixels.")
        return np.rint(decoded * 255).astype(np.uint8)
    finally:
        offload_for_scoring(runtime)


@torch.no_grad()
def semantic_arrays_from_rgb(teacher, frame, mapping):
    """Same native-logit/public-map checks as the historical YOLO scorer, in RAM.

    An RGB array replaces lossless PNG + cv2.imread. Ultralytics receives the
    identical BGR byte order it historically obtained from that PNG.
    """
    from yolo.yolo_labelmap_concepts import (
        extract_semantic_label_map, group_native_probabilities, remap_native_label_map,
    )
    image_bgr = np.ascontiguousarray(frame[:, :, ::-1])
    predictor = teacher.predictor
    input_tensor = predictor.preprocess([image_bgr])
    raw_output = predictor.model(input_tensor)
    public_result = predictor.postprocess(raw_output, input_tensor, [image_bgr])[0]
    public_native = extract_semantic_label_map(public_result)
    if public_native is None:
        raise RuntimeError("Teacher did not return a semantic class map.")
    logits = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
    if (not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[0] != 1
            or not logits.is_floating_point() or logits.shape[1] != len(teacher.names)):
        raise RuntimeError("Expected native floating-point semantic logits [1,C,H,W].")
    scores = logits.detach().float()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("YOLO returned nonfinite semantic logits.")
    if scores.shape[2:] != input_tensor.shape[2:]:
        scores = F.interpolate(scores, input_tensor.shape[2:], mode="bilinear")
    predictor_module = sys.modules.get(type(predictor).__module__)
    predictor_ops = getattr(predictor_module, "ops", None)
    if predictor_ops is None:
        raise RuntimeError("Cannot access the installed predictor's mask-resizing operation.")
    scores = predictor_ops.scale_masks(scores, image_bgr.shape[:2])
    native = scores[0].argmax(dim=0).cpu().numpy().astype(np.int64)
    if native.shape != public_native.shape:
        raise RuntimeError("Raw YOLO logits and its public semantic map have different shapes.")
    agreement = native == public_native
    if float(np.mean(agreement)) < 0.9999:
        raise RuntimeError("Raw YOLO logits do not reproduce its public semantic map.")
    label_map, foreground = remap_native_label_map(native, mapping)
    probabilities = torch.softmax(scores, dim=1)
    grouped = group_native_probabilities(probabilities, mapping, len(SOURCE_NAMES))
    target = torch.from_numpy(label_map.astype(np.int64)).to(grouped.device)[None, None]
    confidence = grouped.gather(1, target)[0, 0].cpu().numpy().astype(np.float16)
    if not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
        raise FloatingPointError("YOLO probabilities are not finite values in [0,1].")
    # The global parity check allows tiny rounding differences. Those individual
    # pixels remain unknown instead of acquiring a disputed tracking label.
    return {"label_map": label_map, "mapped_foreground_mask": foreground,
            "agreement_mask": agreement.astype(np.uint8),
            "numerical_valid_mask": np.ones_like(label_map, dtype=np.uint8),
            "teacher_target_probability": confidence}


@torch.no_grad()
def label_frames(runtime, frames):
    """Label the generated uint8 RGB, preserving raw tracking and codec taxonomies."""
    from rgb_geometry_pilot.runtime import valid_raw_labels

    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.shape != (*FRAME_SHAPE, 3):
        raise ValueError("YOLO requires generated uint8 RGB [81,480,832,3].")
    offload_for_scoring(runtime)
    teacher, device = runtime["teacher"], runtime["device"]
    teacher.to(device)
    teacher.model.requires_grad_(False).eval()
    raw_labels = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
    codec_labels = np.full(FRAME_SHAPE, 255, dtype=np.uint8)
    valid_count, confidence_sum = 0, 0.0
    try:
        for frame_index, frame in enumerate(frames):
            if getattr(teacher, "predictor", None) is None:
                teacher.predict(
                    source=np.ascontiguousarray(frame[:, :, ::-1]), device=str(device),
                    conf=0.25, verbose=False, save=False, save_txt=False, save_crop=False, show=False,
                )
                teacher.model.requires_grad_(False).eval()
            arrays = semantic_arrays_from_rgb(teacher, frame, runtime["mapping"])
            original = np.asarray(arrays["label_map"])
            raw, valid = valid_raw_labels(arrays, FRAME_SHAPE[1:])
            raw_labels[frame_index] = raw
            codec_labels[frame_index] = np.where(valid, CODEC_MAPPING[original], 255)
            confidence = arrays["teacher_target_probability"]
            valid_count += int(valid.sum())
            confidence_sum += float(confidence[valid].astype(np.float64).sum())
    finally:
        offload_for_scoring(runtime)
    metrics = {
        "frame_count": len(frames), "rgb_uint8_sha256": array_hash(frames),
        "raw_labels_sha256": array_hash(raw_labels), "codec_labels_sha256": array_hash(codec_labels),
        "valid_pixels": valid_count, "total_pixels": int(raw_labels.size),
        "yolo_coverage": valid_count / raw_labels.size,
        "mean_source_label_probability_on_valid": confidence_sum / valid_count if valid_count else None,
        "source_class_names": SOURCE_NAMES, "codec_class_names": CODEC_NAMES,
        "source_to_codec": CODEC_MAPPING.tolist(),
        "validity": "numerical_valid & native_public_agreement & mapped_foreground & original_label != 0",
        "labels_source": "YOLO on this same generated RGB; no supplied dataset mask",
    }
    return {"raw_labels": raw_labels, "codec_labels": codec_labels,
            "valid": codec_labels != 255, "metrics": metrics}


def agreement_counts(predicted_labels, target_codec_labels):
    """All valid teacher pixels count; no confidence-based evaluation filtering."""
    predicted = np.asarray(predicted_labels)
    target = np.asarray(target_codec_labels)
    if (predicted.shape != target.shape or predicted.dtype.kind not in "iu"
            or target.dtype.kind not in "iu" or not np.isin(predicted, np.arange(9)).all()
            or not np.isin(target, list(range(9)) + [255]).all()):
        raise ValueError("Expected aligned nine-class predictions and targets with optional ignore255.")
    valid = target != 255
    pairs = target[valid].astype(np.int64) * 9 + predicted[valid]
    confusion = np.bincount(pairs, minlength=81).reshape(9, 9)
    return agreement_from_confusion(confusion, int(target.size), np.bincount(predicted.ravel(), minlength=9))


def agreement_from_confusion(confusion, total_pixels, predicted_pixels=None):
    confusion = np.asarray(confusion)
    if confusion.shape != (9, 9) or confusion.dtype.kind not in "iu" or np.any(confusion < 0):
        raise ValueError("Expected nonnegative integer nine-class confusion counts.")
    target_counts = confusion.sum(axis=1)
    prediction_counts = confusion.sum(axis=0)
    intersection = np.diag(confusion)
    union = target_counts + prediction_counts - intersection
    rows = []
    for class_id, name in enumerate(CODEC_NAMES):
        rows.append({
            "class_id": class_id, "class_name": name, "target_pixels": int(target_counts[class_id]),
            "predicted_pixels_on_valid": int(prediction_counts[class_id]),
            "intersection_pixels": int(intersection[class_id]), "union_pixels": int(union[class_id]),
            "iou": float(intersection[class_id] / union[class_id]) if union[class_id] else None,
            "teacher_supported": bool(target_counts[class_id]),
        })
    supported = [row["iou"] for row in rows[1:] if row["teacher_supported"]]
    # A fixed eight-class denominator prevents an absent class being silently
    # removed. Both-empty contributes zero and is also explicitly unsupported.
    score = sum(row["iou"] or 0.0 for row in rows[1:]) / 8
    return {
        "confusion": confusion.tolist(), "per_class": rows,
        "foreground_miou": score, "fixed_foreground_miou": score, "selection_score": score,
        "supported_foreground_miou": float(np.mean(supported)) if supported else None,
        "foreground_classes": CODEC_NAMES[1:], "valid_pixels": int(confusion.sum()),
        "total_pixels": int(total_pixels), "ignored_pixels": int(total_pixels - confusion.sum()),
        "predicted_pixels": None if predicted_pixels is None else np.asarray(predicted_pixels).tolist(),
        "no_supported_foreground": not bool(supported),
        "geometry_certified": False,
    }


@torch.no_grad()
def teacher_agreement(runtime, prediction, target_codec_labels):
    """Decode eight normalized latent channels into nine classes and count agreement."""
    if prediction.ndim != 5 or prediction.shape[0] != 1 or prediction.shape[1] != 8:
        raise ValueError("Semantic prediction must have shape [1,8,T,H,W].")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Semantic prediction is nonfinite.")
    codec, device = runtime["codec"], runtime["device"]
    codec.to(device)
    try:
        raw = codec.denormalize(prediction.to(device).float())
        parts = []
        for first, last, logits in codec.decode_chunks(raw, chunk_frames=9):
            if logits.shape[1] != 9 or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Frozen codec must emit nine finite class logits.")
            parts.append(logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8))
        return agreement_counts(np.concatenate(parts), target_codec_labels)
    finally:
        codec.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def tracker_environment():
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _write_process_log(path, text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    with Path(path).open("x", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "stream", "message"])
        for line in (text or "").splitlines():
            writer.writerow([datetime.now(timezone.utc).isoformat(), "tracker", line])


def track_frames(args, frames, labels, directory, expected_classes, evaluation=False):
    """Use a separate interpreter; NPZ exchange files exist only in owned temporary storage."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "tracking"
    if target.exists():
        raise FileExistsError("Tracking requires a fresh case; historical outputs are never overwritten.")
    expected = expected_class_ids(expected_classes)
    if (frames.dtype != np.uint8 or frames.shape != (*FRAME_SHAPE, 3)
            or labels.shape != FRAME_SHAPE or labels.dtype.kind not in "iu"):
        raise ValueError("Tracking requires aligned full-resolution generated RGB and source labels.")
    if not np.isin(labels, list(range(11)) + [255]).all():
        raise ValueError("Tracking source labels must be 0..10 or ignore255.")
    with tempfile.TemporaryDirectory(prefix="rgb_campaign_tracker_", dir=os.environ.get("SLURM_TMPDIR")) as temporary:
        rgb_path = Path(temporary) / "rgb.npz"
        labels_path = Path(temporary) / "labels.npz"
        order = np.arange(len(frames), dtype=np.int64)
        np.savez(rgb_path, frames=frames, frame_indices=order)
        np.savez(labels_path, class_ids=labels, frame_indices=order, class_names=np.asarray(SOURCE_NAMES))
        from cotracker_evaluation.comparison_io import sha256_file
        hashes = {"rgb": sha256_file(rgb_path), "labels": sha256_file(labels_path)}
        command = [str(args.track_python), "-B", "-m", "rgb_geometry_campaign.track",
                   "--rgb-npz", str(rgb_path), "--labels-npz", str(labels_path),
                   "--output-dir", str(target), "--cotracker-repo", str(args.cotracker_repo),
                   "--checkpoint", str(args.cotracker_weights),
                   "--expected-classes", ",".join(str(value) for value in expected)]
        if evaluation:
            command.append("--evaluation")
        try:
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=tracker_environment(), text=True, check=False,
                                     timeout=getattr(args, "tracking_timeout_seconds", 1800))
        except subprocess.TimeoutExpired as error:
            _write_process_log(directory / "tracking_log.csv", error.stdout)
            raise RuntimeError("CoTracker timed out; no reward was assigned.") from error
        _write_process_log(directory / "tracking_log.csv", process.stdout)
        if process.returncode:
            raise RuntimeError("CoTracker failed; see tracking_log.csv. No reward was assigned.")
        if hashes != {"rgb": sha256_file(rgb_path), "labels": sha256_file(labels_path)}:
            raise RuntimeError("Temporary tracker inputs changed during subprocess execution.")
        marker = json.loads((target / "complete.json").read_text())
        expected_artifacts = {"metrics.json", "points.csv", "classes.csv"}
        if (marker.get("status") != "complete" or marker.get("input_hashes") != hashes
                or set(marker.get("artifacts", {})) != expected_artifacts):
            raise RuntimeError("Tracker receipt is incomplete or belongs to different generated pixels.")
        for name, expected_hash in marker["artifacts"].items():
            if sha256_file(target / name) != expected_hash:
                raise RuntimeError("A numerical tracking artifact failed its hash check.")
        return json.loads((target / "metrics.json").read_text())
