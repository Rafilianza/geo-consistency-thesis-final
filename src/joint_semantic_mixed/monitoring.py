"""Prompt-only checks: generate, compare with frozen YOLO, and save small previews."""

import csv
import hashlib
import importlib.metadata
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from codec_retraining.data import CLASS_NAMES, LABEL_MAPPING, PALETTE, SOURCE_NAMES, map_labels
from joint_semantic_mixed.common import (
    MONITOR_PROMPT, RGB_SHAPE, check_tensor, file_hash, fingerprint, load_codec, read_json, write_json,
)
from joint_semantic_mixed.examples import JointSemanticObserver, generate_rgb, save_videos, semantic_labels


def monitoring_plan(manifest, updates, interval, final_prompt):
    """Keep prompt text fixed, but record a different seed for every check."""
    prompts = {record["prompt"] for record in manifest["records"]}
    if MONITOR_PROMPT in prompts or final_prompt in prompts or final_prompt == MONITOR_PROMPT or not final_prompt.strip():
        raise ValueError("Monitoring and final new prompts must differ from each other and the dataset.")
    validation = [record for record in manifest["records"] if record["split"] == "val"]
    # Prefer a multi-vehicle validation prompt when one was accepted.
    multi = [record for record in validation if record["scene"].startswith("mixed_")]
    record = (multi or validation)[0]
    checkpoints = list(range(interval, updates + 1, interval))
    if not checkpoints or checkpoints[-1] != updates:
        checkpoints.append(updates)
    plan = []
    for check_number, update in enumerate(checkpoints):
        first_seed = 52 + 2 * check_number
        cases = [
            {"case_id": "validation_prompt", "prompt": record["prompt"], "seed": first_seed,
             "role": "validation_prompt_monitor", "source_clip_id": record["clip_id"]},
            {"case_id": "additional_prompt", "prompt": MONITOR_PROMPT, "seed": first_seed + 1,
             "role": "reused_monitoring_prompt_not_final_test"},
        ]
        plan.append({"update": update, "cases": cases})
    return plan


def teacher_provenance(model_path):
    path = Path(model_path)
    if not path.is_absolute() or re.search(r"(?:seed|_s)0*47(?:\D|$)", str(path), re.I):
        raise ValueError("Use an absolute local YOLO weight path outside reserved data.")
    path = path.resolve(strict=True)
    if re.search(r"(?:seed|_s)0*47(?:\D|$)", str(path), re.I) or path.suffix != ".pt" or not path.is_file():
        raise ValueError("Expected existing local semantic YOLO .pt weights.")
    return {"path": str(path), "sha256": file_hash(path),
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "label_mapping": LABEL_MAPPING, "validity": "numerical_valid & agreement & mapped_foreground; unknown ignored",
            "score_source": "all decoded RGB frames rounded to uint8, before lossy MP4/JPEG compression",
            "low_confidence": "reported, not thresholded; no new confidence threshold",
            "interpretation": "YOLO agreement, not human ground truth, RGB geometry, or an optimization reward"}


@torch.no_grad()
def cache_prompt_embeddings(pipeline, cases, device):
    from video_generation.gen_dataset import NEGATIVE_PROMPT
    pipeline.text_encoder.to(device)
    embeddings = {}
    for case in cases:
        prompt = case["prompt"]
        if prompt in embeddings:
            continue
        if len(pipeline.tokenizer(prompt, truncation=False)["input_ids"]) > 512:
            raise ValueError("Prompt exceeds 512 tokens.")
        embeddings[prompt] = pipeline.encode_prompt(
            prompt=prompt, negative_prompt=NEGATIVE_PROMPT, do_classifier_free_guidance=True,
            num_videos_per_prompt=1, max_sequence_length=512, device=device,
        )
    pipeline.text_encoder.to("cpu")
    return embeddings


def create_context(args, device, embeddings, provenance):
    # Load YOLO on CPU. It only occupies the GPU while scoring decoded frames.
    from ultralytics import YOLO
    from yolo.yolo_labelmap_concepts import build_label_space
    if file_hash(Path(provenance["path"])) != provenance["sha256"]:
        raise ValueError("YOLO weights changed since this run was configured.")
    teacher = YOLO(provenance["path"])
    teacher.model.requires_grad_(False).eval()
    label_space, mapping = build_label_space(SOURCE_NAMES[1:], teacher.names, False)
    if label_space != dict(enumerate(SOURCE_NAMES)) or set(mapping.values()) != set(range(1, 11)):
        raise ValueError("YOLO must support the same ten source concepts as the dataset.")
    return {"teacher": teacher, "mapping": mapping, "codec": load_codec(args.weights_dir / "frozen_codec.pt", "cpu"),
            "embeddings": embeddings, "device": device}


@torch.no_grad()
def generate_pair(pipeline, semantic, case, context):
    """Reuse loaded Wan, keeping the large decoder/YOLO phases off its GPU."""
    device = context["device"]
    pipeline.vae.to("cpu")
    pipeline.transformer.to(device)
    observer = JointSemanticObserver(pipeline, semantic, case["seed"], device)
    result = generate_rgb(pipeline, case, device, observer, context["embeddings"][case["prompt"]], "latent")
    rgb_latent = check_tensor(result.final_latent, RGB_SHAPE)
    del result
    if observer.steps != 50:
        raise ValueError("Expected 50 semantic generation steps.")
    pipeline.transformer.to("cpu")
    torch.cuda.empty_cache()
    pipeline.vae.to(device)
    video = pipeline._decode_latents_to_video(rgb_latent.to(device))
    frames = pipeline.video_processor.postprocess_video(video, output_type="np")[0]
    del video, rgb_latent
    pipeline.vae.to("cpu")
    context["codec"].to(device)
    labels = semantic_labels(context["codec"], observer.state)
    context["codec"].to("cpu")
    del observer
    torch.cuda.empty_cache()
    if labels.shape != (81, 480, 832) or np.asarray(frames).shape != (81, 480, 832, 3):
        raise ValueError("Expected 81 aligned RGB and semantic frames at 832x480.")
    return frames, labels


def agreement_metrics(confusion, total_pixels, joint_pixels, confidence_sum):
    target_pixels = confusion.sum(axis=1)
    predicted_pixels = confusion.sum(axis=0)
    intersection = np.diag(confusion)
    union = target_pixels + predicted_pixels - intersection
    class_iou = [float(intersection[index] / union[index]) if union[index] else None for index in range(9)]
    main = [class_iou[index] for index in range(1, 9) if class_iou[index] is not None]
    vehicles = [class_iou[index] for index in (1, 2, 3) if class_iou[index] is not None]
    valid_pixels = int(confusion.sum())
    return {"main_miou": float(np.mean(main)) if main else None,
            "vehicle_miou": float(np.mean(vehicles)) if vehicles else None,
            "yolo_coverage": valid_pixels / total_pixels,
            "mean_source_label_probability_on_valid": confidence_sum / valid_pixels if valid_pixels else None,
            "valid_pixels": valid_pixels, "total_pixels": total_pixels,
            "per_class_iou": dict(zip(CLASS_NAMES, class_iou)),
            "yolo_valid_class_pixels": target_pixels.tolist(),
            "joint_valid_class_pixels": predicted_pixels.tolist(),
            "joint_all_class_pixels": joint_pixels.tolist(),
            "missing_yolo_classes": [name for index, name in enumerate(CLASS_NAMES) if not target_pixels[index]],
            "confusion_yolo_rows_joint_columns": confusion.tolist()}


def save_comparison(path, rgb, joint, yolo, caption):
    import matplotlib
    font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    title_font = ImageFont.truetype(str(font_path), 20)
    caption_font = ImageFont.truetype(str(font_path), 16)
    yolo_colors = PALETTE[np.where(yolo == 255, 9, yolo)]
    panel = Image.fromarray(np.concatenate([rgb, PALETTE[joint], yolo_colors], axis=1))
    draw = ImageDraw.Draw(panel)
    width = rgb.shape[1]
    draw.rectangle((0, 0, panel.width, 53), fill="black")
    for column, title in enumerate(("Generated RGB", "Joint segmentation", "YOLO (black = ignored)")):
        draw.text((column * width + 5, 3), title, fill="white", font=title_font)
    draw.text((5, 30), caption, fill="white", font=caption_font)
    panel.save(path, quality=90)


@torch.no_grad()
def score_frames(frames, joint_labels, context, directory, caption):
    from yolo.yolo_labelmap_concepts import semantic_uncertainty_forward
    teacher = context["teacher"]
    teacher.to(context["device"])
    teacher.model.requires_grad_(False).eval()
    confusion = np.zeros((9, 9), dtype=np.int64)
    joint_pixels = np.zeros(9, dtype=np.int64)
    confidence_sum = 0.0
    total_pixels = 0
    rgb_hash = hashlib.sha256()
    # One disposable lossless PNG lets us reuse the dataset's verified teacher path.
    temporary_root = os.environ.get("SLURM_TMPDIR") or None
    try:
        with tempfile.TemporaryDirectory(prefix="joint_yolo_", dir=temporary_root) as temporary:
            frame_path = Path(temporary) / "frame.png"
            for frame_index, frame in enumerate(frames):
                pixels = np.asarray(frame)
                joint = joint_labels[frame_index]
                if pixels.shape != (*joint.shape, 3) or not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                    raise ValueError("Invalid or misaligned RGB/semantic frame.")
                if joint.min() < 0 or joint.max() > 8:
                    raise ValueError("Expected nine-class joint labels.")
                rgb = np.rint(pixels * 255).astype(np.uint8)
                rgb_hash.update(rgb.tobytes())
                Image.fromarray(rgb).save(frame_path)
                if getattr(teacher, "predictor", None) is None:
                    teacher.predict(str(frame_path), device=str(context["device"]), conf=0.25, verbose=False)
                    teacher.model.requires_grad_(False).eval()
                arrays, _, _ = semantic_uncertainty_forward(teacher, frame_path, context["mapping"], 11, uncertainty_version=2)
                valid = arrays["numerical_valid_mask"] & arrays["agreement_mask"] & arrays["mapped_foreground_mask"]
                yolo, valid = map_labels(arrays["label_map"], valid)
                if yolo.shape != joint.shape:
                    raise ValueError("YOLO mask is not aligned with the generated frame.")
                pairs = yolo[valid].astype(np.int64) * 9 + joint[valid]
                confusion += np.bincount(pairs, minlength=81).reshape(9, 9)
                joint_pixels += np.bincount(joint.reshape(-1), minlength=9)
                confidence_sum += float(arrays["teacher_target_probability"][valid].astype(np.float64).sum())
                total_pixels += joint.size
                if frame_index in (0, len(frames) // 2, len(frames) - 1):
                    save_comparison(directory / f"comparison_{frame_index:05d}.jpg", rgb, joint, yolo, caption)
    finally:
        teacher.to("cpu")
        torch.cuda.empty_cache()
    metrics = agreement_metrics(confusion, total_pixels, joint_pixels, confidence_sum)
    metrics["rgb_uint8_sha256"] = rgb_hash.hexdigest()
    metrics["frame_count"] = len(frames)
    return metrics


def evaluate_cases(pipeline, semantic, cases, context, root, update, epoch, config_id, keep_videos=False):
    """Publish each complete example once; resume verifies and skips it."""
    root.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256()
    for name, tensor in sorted(semantic.state_dict().items()):
        model_hash.update(name.encode())
        model_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    results = []
    was_training = semantic.training
    semantic.eval()
    devices = [context["device"].index or 0] if context["device"].type == "cuda" else []
    completed = False
    try:
        with torch.random.fork_rng(devices=devices):
            for case in cases:
                directory = root / case["case_id"]
                metadata = dict(case, update=update, epoch=epoch, cnn_state_sha256=model_hash.hexdigest(),
                                config_fingerprint=config_id, semantic_noise_seed=100000 + case["seed"],
                                keep_videos=keep_videos, rgb_feedback=False)
                if directory.exists():
                    receipt = read_json(directory / "complete.json")
                    if receipt["metadata"] != metadata:
                        raise ValueError("Existing prompt check belongs to different settings or CNN weights.")
                    for name, expected in receipt["artifacts"].items():
                        if file_hash(directory / name) != expected:
                            raise ValueError(f"Changed prompt-check artifact: {directory / name}")
                    results.append(read_json(directory / "scores.json"))
                    continue
                attempt = root / f".{case['case_id']}.{uuid.uuid4().hex[:12]}.partial"
                attempt.mkdir()
                started = time.monotonic()
                print(f"Prompt check update {update}: {case['case_id']}, seed {case['seed']}", flush=True)
                frames, labels = generate_pair(pipeline, semantic, case, context)
                caption = f"CNN update {update} | epoch {epoch:g} | seed {case['seed']}"
                metrics = score_frames(frames, labels, context, attempt, caption)
                if keep_videos:
                    save_videos(attempt, frames, labels)
                del frames, labels
                result = dict(metadata, metrics=metrics, seconds=time.monotonic() - started)
                write_json(attempt / "scores.json", result)
                artifacts = {path.name: file_hash(path) for path in attempt.iterdir() if path.is_file()}
                write_json(attempt / "complete.json", {"metadata": metadata, "artifacts": artifacts})
                attempt.rename(directory)
                results.append(result)
                print(f"YOLO agreement: mIoU={metrics['main_miou']}; vehicles={metrics['vehicle_miou']}; coverage={metrics['yolo_coverage']:.3f}", flush=True)
        completed = True
    finally:
        context["teacher"].to("cpu")
        context["codec"].to("cpu")
        pipeline.vae.to("cpu")
        pipeline.text_encoder.to("cpu")
        pipeline.transformer.to(context["device"] if completed else "cpu")
        pipeline.scheduler.set_timesteps(50, device=context["device"])
        semantic.train(was_training)
        torch.cuda.empty_cache()
    write_json(root / "scores.json", results)
    return results


def save_progress(root, plan, previous_root=None, previous_update=0):
    rows = []
    for checkpoint in plan:
        for case in checkpoint["cases"]:
            source_root = previous_root if previous_root is not None and checkpoint["update"] <= previous_update else root
            directory = source_root / f"update_{checkpoint['update']:06d}" / case["case_id"]
            if not (directory / "complete.json").exists():
                continue
            receipt = read_json(directory / "complete.json")
            if file_hash(directory / "scores.json") != receipt["artifacts"]["scores.json"]:
                raise ValueError(f"Changed prompt-check scores: {directory}")
            result = read_json(directory / "scores.json")
            metrics = result["metrics"]
            row = {key: result[key] for key in ("update", "epoch", "case_id", "seed")}
            row.update({key: metrics[key] for key in ("main_miou", "vehicle_miou", "yolo_coverage", "mean_source_label_probability_on_valid")})
            row.update({f"iou_{name}": value for name, value in metrics["per_class_iou"].items()})
            rows.append(row)
    write_json(root / "agreement_history.json", rows)
    if not rows:
        return
    temporary = root / "agreement_history.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "agreement_history.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ("Mean IoU (8 main classes)", "Vehicle mean IoU", "YOLO valid pixel fraction")
    for axis, key, label in zip(axes, ("main_miou", "vehicle_miou", "yolo_coverage"), labels):
        for case_id in dict.fromkeys(row["case_id"] for row in rows):
            selected = [row for row in rows if row["case_id"] == case_id]
            values = [np.nan if row[key] is None else row[key] for row in selected]
            axis.plot([row["epoch"] for row in selected], values, marker="o", label=case_id.replace("_", " "))
        axis.set(ylabel=label, ylim=(0, 1))
        axis.legend()
    axes[-1].set_xlabel("Epoch")
    figure.suptitle("Generated-mask agreement: changing seeds, evaluation only")
    figure.tight_layout()
    figure.savefig(root / "agreement.png", dpi=140)
    plt.close(figure)
