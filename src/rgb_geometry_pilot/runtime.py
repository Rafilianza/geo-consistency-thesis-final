"""Load the frozen incumbent once, then decode and score lossless RGB frames.

The caller validates accepted input records and owns experiment receipts.
These helpers retain the original eleven-class YOLO space for tracking and
the separate nine-class joint output for semantic agreement.
"""

from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np
from PIL import Image
import torch

from codec_retraining.data import CLASS_NAMES, LABEL_MAPPING, SOURCE_NAMES, map_labels
from joint_semantic_mixed import common, monitoring
from joint_semantic_mixed.examples import save_videos, semantic_labels
from middle_feature_semantic.phase1_semantic_models import FeatureSemanticFlowModel


FRAME_SHAPE = (81, 480, 832)
SELECTED_SEMANTIC_UPDATE = 4600


def offload_runtime(runtime):
    """Release model GPU memory before VAE/YOLO phases or a tracker subprocess."""
    pipeline = runtime["pipeline"]
    for module in (pipeline.transformer, pipeline.text_encoder, pipeline.vae,
                   runtime["semantic"], runtime["codec"]):
        module.to("cpu")
    runtime["teacher"].to("cpu")
    runtime["embeddings"] = {
        prompt: (positive.cpu(), negative.cpu())
        for prompt, (positive, negative) in runtime["embeddings"].items()
    }
    torch.cuda.empty_cache()


def prepare_generation(runtime):
    """Restore only the components used by joint latent generation/training."""
    pipeline, device = runtime["pipeline"], runtime["device"]
    pipeline.vae.to("cpu")
    pipeline.text_encoder.to("cpu")
    runtime["codec"].to("cpu")
    runtime["teacher"].to("cpu")
    pipeline.transformer.to(device)
    runtime["semantic"].to(device)
    runtime["embeddings"] = {
        prompt: (positive.to(device), negative.to(device))
        for prompt, (positive, negative) in runtime["embeddings"].items()
    }
    pipeline.scheduler.set_timesteps(50, device=device)
    torch.cuda.empty_cache()


def load_runtime(args, cases, input_info):
    """Load verified local weights only; the actual run requires one Slurm GPU."""
    device = common.require_gpu()
    semantic_run = Path(args.semantic_run)
    prepared = input_info["prepared"]
    training_config = input_info["training_config"]
    completion = input_info["completion"]
    for name, expected in (("prepared.json", prepared), ("training_config.json", training_config),
                           ("training_complete.json", completion)):
        if common.read_json(semantic_run / name) != expected:
            raise ValueError(f"Semantic input metadata changed after validation: {name}")
    if common.file_hash(semantic_run / "prepared.json") != training_config["prepared_sha256"]:
        raise ValueError("Prepared semantic inputs no longer match training.")
    if str(Path(args.model_dir).resolve()) != training_config["model_dir"]:
        raise ValueError("Use the same Wan model directory as the incumbent semantic CNN.")
    if common.runtime() != training_config["runtime"]:
        raise ValueError("Use the incumbent semantic training runtime versions.")
    if common.wan_inventory(Path(args.model_dir)) != training_config["model_inventory"]:
        raise ValueError("Frozen Wan files differ from the semantic training inventory.")
    for name in ("frozen_codec.pt", "normalization_stats.json"):
        if common.file_hash(semantic_run / name) != prepared["weight_files"][name]:
            raise ValueError(f"Frozen semantic representation changed: {name}")
    best_path = semantic_run / "semantic_best.pt"
    if common.file_hash(best_path) != completion["best_weights_sha256"]:
        raise ValueError("The selected semantic CNN checkpoint changed.")
    saved = torch.load(best_path, map_location="cpu", weights_only=True)
    if saved["config_fingerprint"] != common.fingerprint(training_config):
        raise ValueError("The semantic CNN checkpoint belongs to different training settings.")
    if saved["update"] != completion["best_update"] or saved["update"] != SELECTED_SEMANTIC_UPDATE:
        raise ValueError("Use the selected update 4600 / epoch 57.5 semantic CNN.")
    if saved["validation_loss"] != completion["best_validation_loss"]:
        raise ValueError("The selected semantic loss and completion receipt disagree.")

    # Normalization is part of the frozen codec, not refitted on evaluation data.
    codec = common.load_codec(semantic_run / "frozen_codec.pt", "cpu")
    statistics = common.read_json(semantic_run / "normalization_stats.json")
    train_ids = [record["clip_id"] for record in prepared["records"] if record["split"] == "train"]
    if statistics["fit_split"] != "train" or statistics["training_clip_ids"] != train_ids:
        raise ValueError("Normalization must come from these accepted training clips only.")
    mean = torch.tensor(statistics["mean"], dtype=torch.float32)
    standard_deviation = torch.tensor(statistics["std"], dtype=torch.float32)
    if (mean.shape != (8,) or standard_deviation.shape != (8,)
            or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(standard_deviation).all())
            or bool((standard_deviation < 1e-6).any())):
        raise ValueError("Expected eight finite normalization means and positive standard deviations.")
    if (not bool(codec.normalization_fitted)
            or not torch.equal(codec.latent_channel_mean.cpu(), mean)
            or not torch.equal(codec.latent_channel_std.cpu(), standard_deviation)):
        raise ValueError("The codec's embedded normalization differs from its frozen statistics.")
    teacher_info = monitoring.teacher_provenance(args.yolo_model)
    if teacher_info != training_config["yolo"]:
        raise ValueError("Use the same YOLO weights, mapping and runtime as semantic training.")

    # Imports that construct GPU models are intentionally delayed until validation.
    from ultralytics import YOLO
    from yolo.yolo_labelmap_concepts import build_label_space
    teacher = YOLO(teacher_info["path"])
    teacher.model.requires_grad_(False).eval()
    teacher.to("cpu")
    label_space, mapping = build_label_space(SOURCE_NAMES[1:], teacher.names, False)
    if label_space != dict(enumerate(SOURCE_NAMES)) or set(mapping.values()) != set(range(1, 11)):
        raise ValueError("YOLO does not support the original ten foreground concepts.")
    semantic = FeatureSemanticFlowModel()
    semantic.load_state_dict(saved["model_state_dict"], strict=True)
    semantic.requires_grad_(False).eval()
    del saved
    pipeline = common.load_wan(Path(args.model_dir), device)
    # The shared cache deduplicates prompts across accepted records and cases.
    embeddings = monitoring.cache_prompt_embeddings(pipeline, prepared["records"] + list(cases), device)
    runtime = {"pipeline": pipeline, "semantic": semantic, "codec": codec, "device": device,
               "prepared": prepared, "embeddings": embeddings, "teacher": teacher,
               "mapping": mapping, "teacher_provenance": teacher_info,
               "semantic_update": SELECTED_SEMANTIC_UPDATE}
    prepare_generation(runtime)
    return runtime


def valid_raw_labels(arrays, expected_shape):
    """Keep person/bicycle distinct; invalid or unknown pixels become 255."""
    original = np.asarray(arrays["label_map"])
    if (original.shape != expected_shape or not np.issubdtype(original.dtype, np.integer)
            or np.any(original < 0) or np.any(original > 10)):
        raise ValueError("Expected original YOLO label IDs 0 through 10 on the RGB grid.")
    valid = original != 0
    for name in ("numerical_valid_mask", "agreement_mask", "mapped_foreground_mask"):
        mask = np.asarray(arrays[name])
        if mask.shape != expected_shape or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"Expected an aligned binary {name}.")
        valid = valid & mask.astype(bool)
    labels = np.where(valid, original, 255).astype(np.uint8)
    return labels, valid


@torch.no_grad()
def score_rgb(runtime, frames, directory, joint_labels=None):
    """Save lossless uint8 frames and original-space YOLO labels for tracking."""
    from yolo.yolo_labelmap_concepts import semantic_uncertainty_forward

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.shape != (*FRAME_SHAPE, 3):
        raise ValueError("Expected 81 lossless uint8 RGB frames at 832x480.")
    if joint_labels is not None:
        joint_labels = np.asarray(joint_labels)
        if (joint_labels.shape != FRAME_SHAPE or not np.issubdtype(joint_labels.dtype, np.integer)
                or np.any(joint_labels < 0) or np.any(joint_labels > 8)):
            raise ValueError("Joint labels must contain aligned IDs 0 through 8.")
        joint_labels = joint_labels.astype(np.uint8, copy=False)
    paths = {"frames": str(directory / "frames.npz"), "labels": str(directory / "labels.npz"),
             "rgb_video": str(directory / "generated_rgb.mp4"), "scores": str(directory / "rgb_scores.json")}
    paths["rgb"] = paths["frames"]
    if any(Path(path).exists() for path in paths.values()):
        raise FileExistsError("RGB scoring requires a fresh case directory.")
    offload_runtime(runtime)
    teacher, device = runtime["teacher"], runtime["device"]
    teacher.to(device)
    teacher.model.requires_grad_(False).eval()
    labels = np.empty(FRAME_SHAPE, dtype=np.uint8)
    confusion = np.zeros((9, 9), dtype=np.int64)
    joint_pixels = np.zeros(9, dtype=np.int64)
    confidence_sum, valid_count = 0.0, 0
    try:
        temporary_root = os.environ.get("SLURM_TMPDIR") or None
        with tempfile.TemporaryDirectory(prefix="geometry_yolo_", dir=temporary_root) as temporary:
            frame_path = Path(temporary) / "frame.png"
            for frame_index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_path)
                if getattr(teacher, "predictor", None) is None:
                    teacher.predict(str(frame_path), device=str(device), conf=0.25, verbose=False)
                    teacher.model.requires_grad_(False).eval()
                arrays, _, _ = semantic_uncertainty_forward(
                    teacher, frame_path, runtime["mapping"], 11, uncertainty_version=2,
                )
                raw, valid = valid_raw_labels(arrays, FRAME_SHAPE[1:])
                labels[frame_index] = raw
                confidence = np.asarray(arrays["teacher_target_probability"])
                if (confidence.shape != FRAME_SHAPE[1:] or not np.isfinite(confidence).all()
                        or np.any(confidence < 0) or np.any(confidence > 1)):
                    raise ValueError("YOLO target probabilities must be aligned and finite in [0,1].")
                valid_count += int(valid.sum())
                confidence_sum += float(confidence[valid].astype(np.float64).sum())
                if joint_labels is not None:
                    reduced, reduced_valid = map_labels(arrays["label_map"], valid)
                    joint = joint_labels[frame_index]
                    pairs = reduced[reduced_valid].astype(np.int64) * 9 + joint[reduced_valid]
                    confusion += np.bincount(pairs, minlength=81).reshape(9, 9)
                    joint_pixels += np.bincount(joint.reshape(-1), minlength=9)
    finally:
        offload_runtime(runtime)

    # All scoring used the same pixels saved here, before lossy MP4 compression.
    frame_indices = np.arange(len(frames), dtype=np.int64)
    np.savez_compressed(paths["frames"], frames=frames, frame_indices=frame_indices)
    label_arrays = {"class_ids": labels, "class_names": np.asarray(SOURCE_NAMES),
                    "frame_indices": frame_indices}
    if joint_labels is not None:
        label_arrays["joint_labels"] = joint_labels
        label_arrays["joint_class_names"] = np.asarray(CLASS_NAMES)
    np.savez_compressed(paths["labels"], **label_arrays)
    if joint_labels is not None:
        save_videos(directory, frames.astype(np.float32) / 255.0, joint_labels)
        paths["semantic_video"] = str(directory / "generated_semantic.mp4")
        paths["comparison_video"] = str(directory / "rgb_and_semantic.mp4")
    else:
        import imageio.v2 as imageio
        with imageio.get_writer(paths["rgb_video"], fps=16, codec="libx264", pixelformat="yuv420p", macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
    total_pixels = int(np.prod(FRAME_SHAPE))
    metrics = {"frame_count": len(frames), "rgb_uint8_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
               "valid_pixels": valid_count, "total_pixels": total_pixels,
               "yolo_coverage": valid_count / total_pixels,
               "class_names": SOURCE_NAMES, "joint_class_names": CLASS_NAMES,
               "unknown_label": 255, "label_mapping_to_joint": LABEL_MAPPING,
               "validity": "numerical_valid & agreement & mapped_foreground & original_label != 0",
               "mean_source_label_probability_on_valid": confidence_sum / valid_count if valid_count else None}
    if joint_labels is not None:
        metrics["agreement"] = monitoring.agreement_metrics(confusion, total_pixels, joint_pixels, confidence_sum)
    common.write_json(Path(paths["scores"]), metrics)
    return {"frames": frames, "raw_labels": labels, "joint_labels": joint_labels,
            "paths": paths, "metrics": metrics}


@torch.no_grad()
def decode_and_label(runtime, result, directory):
    """Decode final scheduler states, then run the same lossless scoring path."""
    pipeline, device = runtime["pipeline"], runtime["device"]
    rgb = common.check_tensor(result["rgb_latent"], common.RGB_SHAPE)
    semantic = common.check_tensor(result["semantic_latent"], common.SEMANTIC_SHAPE)
    offload_runtime(runtime)
    try:
        pipeline.vae.to(device)
        video = pipeline._decode_latents_to_video(rgb.to(device))
        decoded = np.asarray(pipeline.video_processor.postprocess_video(video, output_type="np")[0])
        if (decoded.shape != (*FRAME_SHAPE, 3) or not np.isfinite(decoded).all()
                or decoded.min() < 0 or decoded.max() > 1):
            raise ValueError("VAE decoding returned invalid RGB frames.")
        frames = np.rint(decoded * 255).astype(np.uint8)
        del video, decoded
        pipeline.vae.to("cpu")
        torch.cuda.empty_cache()
        runtime["codec"].to(device)
        labels = semantic_labels(runtime["codec"], semantic.to(device))
    finally:
        offload_runtime(runtime)
    return score_rgb(runtime, frames, directory, labels)
