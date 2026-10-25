#!/usr/bin/env python3
"""Prepare and retrain the separate nine-class codec. Never loads Wan or YOLO."""

import argparse
import csv
from contextlib import nullcontext
import fcntl
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from codec_retraining.data import (
    CLASS_NAMES, DEFAULT_DATA, DEFAULT_RUN, DEFAULT_MIXED_DATA, DEFAULT_MIXED_RUN, LABEL_MAPPING, PALETTE, SCHEMA, file_hash,
    fingerprint, load_training_data, prepare, read_json, reviewed_records, source_hashes, validate_prepared, write_json,
)
from codec_retraining.model import LargeClassCodec
from phase3.semantic_codec import SemanticCodec

DEFAULT_BASELINE = "/home/dhiazr/gt69/dhiazgeo/wanti2v/outputs/phase3_runs/semantic_codec_full_58588651/frozen_codec.pt"


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:12]}.partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def precision_context(device, precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and precision == "bf16" else nullcontext()


def class_weights(clips, device):
    counts = np.zeros(9, dtype=np.int64)
    for clip in clips:
        counts += np.bincount(clip["labels"][clip["valid"]], minlength=9)
    counts = torch.tensor(counts, dtype=torch.float32, device=device)
    present = counts > 0
    weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    weights = (weights / weights[present].mean()).clamp(0.25, 4)
    weights[~present] = 0
    return weights, counts


def reconstruction_loss(codec, labels, valid, weights, chunk_frames):
    """Weighted cross entropy + soft Dice, summed over a complete clip.

    scatter_add groups correct-class probabilities by their target class. It
    avoids building a full-resolution one-hot tensor for all nine classes.
    """
    latent = codec.encode(labels, valid)
    if not bool(torch.isfinite(latent).all()):
        raise FloatingPointError("Nonfinite semantic latent.")
    counts = torch.bincount(labels[valid], minlength=9).float()
    if not bool(counts.sum() > 0):
        raise ValueError("A training clip has no valid target pixels.")
    cross_entropy = latent.new_zeros((), dtype=torch.float32)
    intersection = latent.new_zeros(9, dtype=torch.float32)
    predicted_mass = latent.new_zeros(9, dtype=torch.float32)
    for start, end, logits in codec.decode_chunks(latent, chunk_frames):
        logits = logits.float()
        targets = labels[:, start:end]
        mask = valid[:, start:end]
        targets = torch.where(mask, targets, torch.full_like(targets, 255))
        cross_entropy = cross_entropy + F.cross_entropy(logits, targets, weight=weights, ignore_index=255, reduction="sum")
        probabilities = logits.softmax(dim=1)
        predicted_mass = predicted_mass + (probabilities * mask.unsqueeze(1)).sum(dim=(0, 2, 3, 4))
        safe_targets = targets.clamp_max(8)
        correct_probability = probabilities.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        intersection = intersection.scatter_add(0, targets[mask], correct_probability[mask])
    cross_entropy = cross_entropy / (counts * weights).sum().clamp_min(1)
    dice_per_class = (2 * intersection + 1e-6) / (predicted_mass + counts + 1e-6)
    dice_loss = 1 - dice_per_class[counts > 0].mean()
    loss = cross_entropy + dice_loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite reconstruction loss.")
    return loss, cross_entropy, dice_loss


def metrics_from_confusion(confusion):
    # The tenth prediction column counts the legacy codec's unknown/background
    # outputs as errors on valid targets, instead of silently discarding them.
    target_count = confusion.sum(axis=1)
    predicted_count = confusion[:, :9].sum(axis=0)
    intersection = np.diag(confusion[:, :9])
    union = target_count + predicted_count - intersection
    iou = [float(intersection[index] / union[index]) if union[index] else None for index in range(9)]
    main_scores = [value for value in iou[1:] if value is not None]
    return {"main_miou": float(np.mean(main_scores)) if main_scores else None, "per_class_iou": dict(zip(CLASS_NAMES, iou)),
            "target_pixels": dict(zip(CLASS_NAMES, target_count.tolist())),
            "missing_target_classes": [CLASS_NAMES[index] for index in range(1, 9) if not target_count[index]],
            "invalid_prediction_pixels": int(confusion[:, 9].sum()), "confusion": confusion.tolist()}


def save_preview(path, targets, predictions, valid):
    canvas = Image.new("RGB", (832, 3 * 260), "white")
    draw = ImageDraw.Draw(canvas)
    for row, frame_index in enumerate((0, 40, 80)):
        for column, (name, frames) in enumerate((("Target", targets), ("Reconstruction", predictions))):
            labels = frames[frame_index].copy()
            labels[~valid[frame_index]] = 9
            pixels = PALETTE[np.minimum(labels, 9)]
            preview = Image.fromarray(pixels).resize((416, 240), Image.Resampling.NEAREST)
            canvas.paste(preview, (column * 416, row * 260 + 20))
            draw.text((column * 416 + 5, row * 260 + 3), f"{name}, frame {frame_index}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=85)


@torch.no_grad()
def evaluate(codec, clips, device, precision, chunk_frames, legacy=False, preview_dir=None, loss_weights=None):
    codec.eval()
    confusion = np.zeros((9, 10), dtype=np.int64)
    per_clip = {}
    loss_values = []
    if legacy and loss_weights is not None:
        raise ValueError("Nine-class reconstruction loss is only defined for the new codec.")
    mapping = torch.tensor([9 if value == 255 else value for value in LABEL_MAPPING], device=device)
    for clip in clips:
        input_labels = clip["original"] if legacy else clip["labels"]
        labels = torch.as_tensor(input_labels, device=device, dtype=torch.long).unsqueeze(0)
        targets = torch.as_tensor(clip["labels"], device=device, dtype=torch.long).unsqueeze(0)
        valid = torch.as_tensor(clip["valid"], device=device).unsqueeze(0)
        clip_confusion = np.zeros((9, 10), dtype=np.int64)
        saved_predictions = []
        with precision_context(device, precision):
            latent = codec.encode(labels, valid)
            if not bool(torch.isfinite(latent).all()):
                raise FloatingPointError("Nonfinite validation latent.")
            for start, end, logits in codec.decode_chunks(latent, chunk_frames):
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite validation logits.")
                prediction = logits.argmax(dim=1)
                if legacy:
                    prediction = mapping[prediction]
                mask = valid[:, start:end]
                pairs = targets[:, start:end][mask] * 10 + prediction[mask]
                clip_confusion += torch.bincount(pairs, minlength=90).reshape(9, 10).cpu().numpy()
                if preview_dir is not None:
                    saved_predictions.append(prediction[0].cpu().numpy().astype(np.uint8))
        confusion += clip_confusion
        per_clip[clip["clip_id"]] = metrics_from_confusion(clip_confusion)
        if loss_weights is not None:
            # Reuse the exact training objective in a separate no-gradient pass.
            # The weights are fixed from training pixels, never fitted on validation.
            with precision_context(device, precision):
                losses = reconstruction_loss(codec, targets, valid, loss_weights, chunk_frames)
            values = [float(value) for value in losses]
            loss_values.append(values)
            per_clip[clip["clip_id"]].update(dict(zip(("loss", "cross_entropy", "dice_loss"), values)))
        if preview_dir is not None:
            predictions = np.concatenate(saved_predictions)
            save_preview(preview_dir / f"{clip['clip_id']}.jpg", clip["labels"], predictions, clip["valid"])
    result = metrics_from_confusion(confusion)
    if sum(list(result["target_pixels"].values())[1:]) == 0:
        raise ValueError("Validation has no valid targets for any of the eight main classes.")
    result["per_clip"] = per_clip
    if loss_values:
        averages = np.mean(loss_values, axis=0).tolist()
        result.update(dict(zip(("loss", "cross_entropy", "dice_loss"), averages)))
    return result


def save_history(run_dir, history):
    """Save identical epoch records in JSON and CSV; unmeasured values stay empty."""
    fields = ["epoch", "training_loss", "training_cross_entropy", "training_dice_loss",
              "validation_loss", "validation_cross_entropy", "validation_dice_loss",
              "validation_miou", "seconds"]
    write_json(run_dir / "history.json", history)
    path = run_dir / "history.csv"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:12]}.partial")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def plot_history(run_dir, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for filename, title, fields in (
        ("loss.png", "Codec training and validation loss", ("training_loss", "validation_loss")),
        ("validation.png", "Validation reconstruction: eight main classes", ("validation_miou",)),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        for field in fields:
            records = [row for row in history if row.get(field) is not None]
            axis.plot([row["epoch"] for row in records], [row[field] for row in records], label=field.replace("_", " "))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("mIoU" if filename == "validation.png" else "Loss")
        axis.set_title(title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / filename, dpi=140)
        plt.close(figure)


def training_device(args):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_ARRAY_JOB_ID"):
        raise RuntimeError("Submit the training wrapper as one GPU job, without an array.")
    if os.environ.get("SLURM_NTASKS", "1") != "1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Training requires one task and exactly one visible CUDA GPU.")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use --precision fp32 on a GPU without bfloat16 support.")
    return torch.device("cuda")


def train(args):
    device = training_device(args)
    run_dir = args.run_dir
    manifest = read_json(run_dir / "prepared.json")
    validate_prepared(manifest)
    selected, rejected = reviewed_records(manifest, run_dir / "review.csv")
    baseline_path = args.baseline_checkpoint.resolve(strict=True)
    config = {"schema": SCHEMA, "class_names": CLASS_NAMES, "label_mapping": LABEL_MAPPING,
              "prepared_sha256": file_hash(run_dir / "prepared.json"), "review_sha256": file_hash(run_dir / "review.csv"),
              "sources": source_hashes(manifest.get("dataset_kind", "original")), "epochs": args.epochs, "learning_rate": args.learning_rate,
              "precision": args.precision, "seed": args.seed, "chunk_frames": args.chunk_frames,
              "evaluate_every": args.evaluate_every, "baseline_checkpoint": str(baseline_path),
              "baseline_sha256": file_hash(baseline_path), "initialization": "new random nine-class codec",
              "accepted": {split: [record["clip_id"] for record in records] for split, records in selected.items()},
              "rejected": rejected, "torch_version": str(torch.__version__)}
    if (run_dir / "config.json").exists():
        if not args.resume or read_json(run_dir / "config.json") != config:
            raise ValueError("Existing training found. Use --resume with unchanged settings/reviews/sources, or a new prepared run folder.")
    elif args.resume:
        raise ValueError("No existing training configuration to resume.")
    if manifest["schema"] != SCHEMA or manifest["sources"] != config["sources"]:
        raise ValueError("Prepared schema or sources differ from current training code.")
    if (run_dir / "complete.json").exists():
        print("This run already completed. Inspect complete.json and the validation previews.")
        return
    write_json(run_dir / "config.json", config)
    clips = load_training_data(manifest, selected)
    print(f"Training {len(clips['train'])} clips; validating {len(clips['val'])}; rejected {len(rejected)}.", flush=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    codec = LargeClassCodec().to(device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.learning_rate, weight_decay=0)
    weights, counts = class_weights(clips["train"], device)
    if not bool(counts[1:].sum() > 0):
        raise ValueError("Training has no valid targets for any of the eight main classes.")
    write_json(run_dir / "training_class_counts.json", {"classes": CLASS_NAMES, "valid_pixels": counts.cpu().tolist(), "loss_weights": weights.cpu().tolist()})
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU test fixture"
    print(f"Device: {device_name}; codec parameters: {sum(parameter.numel() for parameter in codec.parameters())}", flush=True)
    start_epoch, best_metric, history = 1, -1.0, []
    if args.resume and (run_dir / "latest.pt").exists():
        checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        if checkpoint["config_fingerprint"] != fingerprint(config):
            raise ValueError("Checkpoint belongs to a different training configuration.")
        codec.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng"])
        start_epoch = checkpoint["epoch"] + 1
        best_metric, history = checkpoint["best_metric"], checkpoint["history"]

    def checkpoint_payload(epoch):
        return {"schema": SCHEMA, "class_names": CLASS_NAMES, "config_fingerprint": fingerprint(config),
                "model_state_dict": {name: value.detach().cpu() for name, value in codec.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "best_metric": best_metric,
                "history": history, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}

    if not history:
        initial = evaluate(codec, clips["val"], device, args.precision, args.chunk_frames, loss_weights=weights)
        best_metric = initial["main_miou"]
        history.append({"epoch": 0, "training_loss": None, "training_cross_entropy": None,
                        "training_dice_loss": None, "validation_loss": initial["loss"],
                        "validation_cross_entropy": initial["cross_entropy"],
                        "validation_dice_loss": initial["dice_loss"], "validation_miou": best_metric,
                        "seconds": None})
        save_checkpoint(run_dir / "best.pt", checkpoint_payload(0))
        save_checkpoint(run_dir / "latest.pt", checkpoint_payload(0))
    save_history(run_dir, history)
    plot_history(run_dir, history)
    if not (run_dir / "baseline_validation.json").exists():
        baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
        if baseline_payload.get("num_classes") != 11:
            raise ValueError("Baseline must be the original frozen 11-class codec.")
        baseline = SemanticCodec().to(device)
        baseline.load_state_dict(baseline_payload["model_state_dict"])
        baseline.requires_grad_(False)
        baseline_metrics = evaluate(baseline, clips["val"], device, args.precision, args.chunk_frames,
                                    legacy=True, preview_dir=run_dir / "previews_baseline")
        write_json(run_dir / "baseline_validation.json", baseline_metrics)
        del baseline, baseline_payload
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.monotonic()
        codec.train()
        order = list(clips["train"])
        random.Random(args.seed + epoch).shuffle(order)
        totals = np.zeros(3, dtype=np.float64)
        for clip in order:
            labels = torch.as_tensor(clip["labels"], device=device, dtype=torch.long).unsqueeze(0)
            valid = torch.as_tensor(clip["valid"], device=device).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(device, args.precision):
                loss, cross_entropy, dice_loss = reconstruction_loss(codec, labels, valid, weights, args.chunk_frames)
            loss.backward()
            for component in (codec.encoder, codec.decoder):
                gradients = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
                if not gradients or not any(bool(gradient.abs().max() > 0) for gradient in gradients):
                    raise RuntimeError("Gradients must reach both encoder and decoder.")
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            totals += [float(loss.detach()), float(cross_entropy.detach()), float(dice_loss.detach())]
            del labels, valid, loss, cross_entropy, dice_loss
        values = totals / len(order)
        row = {"epoch": epoch, "training_loss": float(values[0]), "training_cross_entropy": float(values[1]),
               "training_dice_loss": float(values[2]), "validation_miou": None,
               "validation_loss": None, "validation_cross_entropy": None, "validation_dice_loss": None}
        metrics = None
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            metrics = evaluate(codec, clips["val"], device, args.precision, args.chunk_frames, loss_weights=weights)
            row["validation_miou"] = metrics["main_miou"]
            row["validation_loss"] = metrics["loss"]
            row["validation_cross_entropy"] = metrics["cross_entropy"]
            row["validation_dice_loss"] = metrics["dice_loss"]
        row["seconds"] = time.monotonic() - epoch_started
        history.append(row)
        if metrics is not None and metrics["main_miou"] > best_metric:
            best_metric = metrics["main_miou"]
            save_checkpoint(run_dir / "best.pt", checkpoint_payload(epoch))
        save_checkpoint(run_dir / "latest.pt", checkpoint_payload(epoch))
        save_history(run_dir, history)
        if metrics is not None:
            write_json(run_dir / "latest_validation.json", metrics)
            plot_history(run_dir, history)
        print(f"Epoch {epoch}/{args.epochs}: training_loss={row['training_loss']:.5f}, validation_loss={row['validation_loss']}, validation_miou={row['validation_miou']}, seconds={row['seconds']:.1f}", flush=True)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    codec.load_state_dict(best["model_state_dict"])
    final = evaluate(codec, clips["val"], device, args.precision, args.chunk_frames,
                     preview_dir=run_dir / "previews_best", loss_weights=weights)
    write_json(run_dir / "best_validation.json", final)
    save_checkpoint(run_dir / "best_weights.pt", {"schema": SCHEMA, "class_names": CLASS_NAMES,
                    "label_mapping": LABEL_MAPPING, "model_state_dict": best["model_state_dict"],
                    "epoch": best["epoch"], "config_fingerprint": fingerprint(config),
                    "normalization_fitted": False, "note": "Reconstruction codec only; not compatible with historical CNN/RL checkpoints or latent caches."})
    plot_history(run_dir, history)
    baseline_metrics = read_json(run_dir / "baseline_validation.json")
    write_json(run_dir / "complete.json", {"status": "training_complete_requires_reconstruction_review",
        "epochs": args.epochs, "selected_epoch": best["epoch"], "training_clips": len(clips["train"]),
        "validation_clips": len(clips["val"]), "baseline_main_miou": baseline_metrics["main_miou"],
        "new_main_miou": final["main_miou"], "main_miou_change": final["main_miou"] - baseline_metrics["main_miou"],
        "missing_validation_target_classes": final["missing_target_classes"],
        "this_invocation_training_and_finalization_seconds": time.monotonic() - started,
        "best_weights_sha256": file_hash(run_dir / "best_weights.pt"),
        "note": "Coarse YOLO-mask reconstruction, not RGB geometry or human-label accuracy. No normalization/latent cache/CNN training performed."})
    print(f"Completed. Weights and reports: {run_dir}. Review before freezing or creating latent caches.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "check", "train", "plot"))
    parser.add_argument("--dataset-kind", choices=("original", "mixed"), default="original",
                        help="Select the fixed dataset plan and default input/output folders.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path, default=Path(DEFAULT_BASELINE))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--chunk-frames", type=int, default=9)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is None:
        args.data_root = Path(DEFAULT_MIXED_DATA if args.dataset_kind == "mixed" else DEFAULT_DATA)
    if args.run_dir is None:
        args.run_dir = Path(DEFAULT_MIXED_RUN if args.dataset_kind == "mixed" else DEFAULT_RUN)
    for path in (args.data_root, args.run_dir, args.baseline_checkpoint):
        if not path.is_absolute() or re.search(r"(?:seed|_s)0*47(?:\D|$)", str(path), re.IGNORECASE):
            parser.error("Use absolute paths outside reserved seed-47 data.")
    if args.seed == 47 or min(args.epochs, args.evaluate_every, args.chunk_frames) < 1 or args.learning_rate <= 0:
        parser.error("Use positive training settings and a non-reserved seed.")
    if args.stage == "prepare":
        prepare(args.data_root, args.run_dir, args.dataset_kind)
    elif args.stage == "plot":
        history = read_json(args.run_dir / "history.json")
        save_history(args.run_dir, history)
        plot_history(args.run_dir, history)
    elif args.stage == "check":
        manifest = read_json(args.run_dir / "prepared.json")
        validate_prepared(manifest)
        selected, rejected = reviewed_records(manifest, args.run_dir / "review.csv")
        if manifest["sources"] != source_hashes(manifest.get("dataset_kind", "original")):
            raise ValueError("Sources changed after preparation.")
        print(f"Reviewed: {len(selected['train'])} train / {len(selected['val'])} validation; {len(rejected)} rejected.")
        print("Review check only; training rechecks actual input files before use.")
    else:
        # Kernel lock disappears on process exit; no stale-lock deletion needed.
        with (args.run_dir / ".training.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is training in this folder.") from error
            train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
