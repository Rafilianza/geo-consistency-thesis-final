#!/usr/bin/env python3
"""Create the separate 100-clip walking-POV dataset; no model training.

Run check/status on a CPU. The Slurm wrapper runs generate and label in separate
processes so Wan is released before YOLO starts. Completed clips are verified on
resume. Interrupted attempts remain in attempts/ for inspection.
"""

import argparse
import csv
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from video_generation.gen_dataset import (
    CORE_CONCEPTS, FPS, GUIDANCE, HEIGHT, NEGATIVE_PROMPT, NUM_FRAMES,
    NUM_STEPS, FLOW_SHIFT, WIDTH,
)

DEFAULT_ROOT = "/home/dhiazr/gt69_scratch2/dhiazgeo/semantic_dataset_100_v1"
DEFAULT_MODEL = "/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_YOLO = "/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Yolo26/yolo26x-sem.pt"
DEFAULT_PROMPTS = REPOSITORY / "configs/datasets/walking_pov_100_v1.json"
SETTINGS = {
    "height": HEIGHT, "width": WIDTH, "num_frames": NUM_FRAMES, "fps": FPS,
    "num_inference_steps": NUM_STEPS, "guidance_scale": GUIDANCE,
    "flow_shift": FLOW_SHIFT, "scheduler": "UniPCMultistepScheduler",
    "negative_prompt": NEGATIVE_PROMPT, "core_concepts": CORE_CONCEPTS,
    "max_sequence_length": 512, "uncertainty_version": 2,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Replace small metadata atomically; a failed write keeps its temporary file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_clips(path):
    """Keep the prompt split fixed before any images or scores are seen."""
    specification = path if isinstance(path, dict) else read_json(path)
    if specification["dataset_id"] != "walking_pov_100_v1":
        raise ValueError("Unexpected dataset_id; use a new script/version for a different experiment.")
    if specification["train_seeds"] != [42, 43, 44, 45]:
        raise ValueError("Training seeds must be 42, 43, 44, 45.")
    if specification["validation_seeds"] != [46, 48, 49, 50]:
        raise ValueError("Validation seeds must be 46, 48, 49, 50; seed 47 is reserved.")
    scenes = set()
    prompt_texts = set()
    clips = []
    for entry in specification["prompts"]:
        scene = entry["scene"]
        prompt = entry["prompt"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scene) or scene in scenes:
            raise ValueError(f"Invalid or repeated scene: {scene}")
        if prompt in prompt_texts or not 80 <= len(prompt.split()) <= 100:
            raise ValueError(f"{scene}: use a unique prompt of 80–100 words.")
        if "first-person" not in prompt.lower() or "forward" not in prompt.lower() or "walk" not in prompt.lower():
            raise ValueError(f"{scene}: missing the forward-walking POV instruction.")
        if entry["split"] not in ("train", "val") or entry["focus_class"] not in ("car", "bus", "truck"):
            raise ValueError(f"{scene}: invalid split or focus class.")
        scenes.add(scene)
        prompt_texts.add(prompt)
        seed_key = "train_seeds" if entry["split"] == "train" else "validation_seeds"
        for seed in specification[seed_key]:
            clips.append(dict(entry, seed=seed, clip_id=f"{scene}_s{seed}"))
    if len(scenes) != 25 or len(clips) != 100:
        raise ValueError("Expected 25 prompts and 100 clips.")
    if sum(clip["split"] == "train" for clip in clips) != 80:
        raise ValueError("Expected 80 training clips and 20 validation clips.")
    return specification, clips


def model_inventory(directory):
    """Record weight file sizes/times, not an expensive full Wan weight hash."""
    if not (directory / "model_index.json").is_file():
        raise FileNotFoundError(directory / "model_index.json")
    for component in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
        if not (directory / component).is_dir():
            raise FileNotFoundError(directory / component)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".safetensors", ".bin", ".json", ".model"):
            details = path.stat()
            record = {"bytes": details.st_size, "mtime_ns": details.st_mtime_ns}
            if path.suffix == ".json":
                record["sha256"] = file_hash(path)
            files[str(path.relative_to(directory))] = record
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError(f"No local Wan weights found in {directory}")
    return files


def make_plan(args, specification):
    model_directory = Path(args.model_dir).resolve(strict=True)
    yolo_path = Path(args.yolo_model).resolve(strict=True)
    sources = [
        Path(__file__).resolve(), REPOSITORY / "src/video_generation/gen_dataset.py",
        REPOSITORY / "src/video_generation/my_wan_pipeline.py",
        REPOSITORY / "src/yolo/yolo_labelmap_concepts.py",
        REPOSITORY / "src/seg_decoder/audit_uncertainty_v2.py",
    ]
    return {
        "dataset_id": specification["dataset_id"], "specification": specification,
        "settings": SETTINGS, "model_directory": str(model_directory),
        "model_inventory": model_inventory(model_directory),
        "model_inventory_note": "Wan weight sizes/mtimes and JSON hashes; not full weight hashes.",
        "yolo_model": str(yolo_path), "yolo_sha256": file_hash(yolo_path),
        "sources": {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sources},
    }


def check_root(root, plan):
    if root.is_symlink():
        raise ValueError("The dataset folder itself must not be a symlink; a parent storage link is allowed.")
    plan_path = root / "dataset_plan.json"
    if plan_path.is_file():
        if read_json(plan_path) != plan:
            raise ValueError("Saved dataset plan differs from these prompts, sources or models. Use a new output folder.")
    elif root.exists():
        # Only our exact initialization leftovers may exist before the plan.
        leftovers = {
            path.name for path in root.iterdir()
            if not re.fullmatch(r"initial_plan_[a-f0-9]{12}\.json\.partial", path.name)
        }
        if leftovers - {".dataset.lock", "dataset_plan.json.tmp"}:
            raise ValueError("Refusing an existing nonempty folder without this dataset's plan.")


def verified_receipt(directory, stage):
    marker = directory / f"{stage}_complete.json"
    if not marker.is_file():
        return False
    receipt = read_json(marker)
    for relative_path, expected_hash in receipt["artifacts"].items():
        path = directory / relative_path
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Artifact leaves its clip directory: {path}")
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError(f"Completed artifact changed or is missing: {path}")
    return True


def save_receipt(directory, stage, paths, details):
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Cannot mark an empty or missing artifact complete: {path}")
    artifacts = {str(path.relative_to(directory)): file_hash(path) for path in paths}
    write_json(directory / f"{stage}_complete.json", {
        "completed_utc": timestamp(), "artifacts": artifacts, **details,
    })


def new_attempt(root, clip_id, stage):
    directory = root / "attempts" / f"{clip_id}_{stage}_{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True)
    return directory


def runtime(stage):
    packages = ["torch", "diffusers", "transformers", "numpy", "Pillow", "imageio", "imageio-ffmpeg"]
    if stage == "label":
        packages.append("ultralytics")
    return {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in packages}}


def check_runtime(root, stage):
    current = runtime(stage)
    path = root / f"{stage}_runtime.json"
    if path.exists() and read_json(path) != current:
        raise ValueError(f"{stage} environment changed; use the original environment or a new output folder.")
    if not path.exists():
        write_json(path, current)


def report_space(root):
    """Report available space without deciding whether the filesystem can write."""
    parent = root
    while not parent.exists():
        parent = parent.parent
    free_gib = shutil.disk_usage(parent).free / 1024**3
    print(f"Filesystem-reported free: {free_gib:.2f} GiB (information only; no minimum enforced).")
    print("This is NOT your project quota; check user_info. Actual write errors will stop the job.")


def generate(root, clips, plan):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from video_generation.my_wan_pipeline import MyWanPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires an allocated GPU; use the Slurm wrapper.")
    check_runtime(root, "generate")
    pending = []
    for clip in clips:
        destination = root / "clips" / clip["clip_id"]
        if verified_receipt(destination, "generation"):
            print(f"[skip RGB] {clip['clip_id']}", flush=True)
        elif destination.exists():
            raise ValueError(f"Incomplete or unrelated final clip folder: {destination}; preserved for inspection.")
        else:
            pending.append(clip)
    if not pending:
        return

    # Load once for the entire selected batch. No controller, LoRA or semantic model.
    print(f"Loading frozen Wan on {torch.cuda.get_device_name(0)}", flush=True)
    vae = AutoencoderKLWan.from_pretrained(plan["model_directory"], subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipeline = MyWanPipeline.from_pretrained(plan["model_directory"], vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=FLOW_SHIFT)
    for component in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
        component.requires_grad_(False)
        component.eval()
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    # Detect prompt truncation before spending time on a video.
    for clip in pending:
        token_ids = pipeline.tokenizer(clip["prompt"], truncation=False)["input_ids"]
        if len(token_ids) > SETTINGS["max_sequence_length"]:
            raise ValueError(f"Prompt exceeds 512 tokens: {clip['clip_id']}")

    for clip_index, clip in enumerate(pending, start=1):
        destination = root / "clips" / clip["clip_id"]
        attempt = new_attempt(root, clip["clip_id"], "generation")
        started = time.monotonic()
        print(f"[RGB {clip_index}/{len(pending)}] {clip['clip_id']}", flush=True)
        result = pipeline(
            prompt=clip["prompt"], negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT, width=WIDTH, num_frames=NUM_FRAMES,
            num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE,
            generator=torch.Generator(device="cuda").manual_seed(clip["seed"]),
            max_sequence_length=512, verbose=False,
        )
        frames = result.frames[0]
        latent = result.final_latent
        if len(frames) != NUM_FRAMES or tuple(latent.shape) != (1, 16, 21, 60, 104) or not torch.isfinite(latent).all():
            raise ValueError(f"Invalid frame count or RGB latent for {clip['clip_id']}")
        torch.save(latent.cpu(), attempt / "final_latent.pt")
        (attempt / "frames").mkdir()
        for frame_index, frame in enumerate(frames):
            frame_array = np.asarray(frame)
            if frame_array.shape != (HEIGHT, WIDTH, 3) or not np.isfinite(frame_array).all():
                raise ValueError(f"Invalid RGB frame {frame_index}")
            if frame_array.min() < 0 or frame_array.max() > 1:
                raise ValueError(f"Expected RGB pixels in [0,1], got another range at frame {frame_index}.")
            pixels = np.clip(np.rint(frame_array * 255), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(attempt / "frames" / f"{frame_index:05d}.jpg", quality=95)
        export_to_video(frames, str(attempt / "video.mp4"), fps=FPS)
        metadata = dict(clip, **SETTINGS, latent_shape=list(latent.shape), latent_dtype=str(latent.dtype),
                        frames_dir=str(destination / "frames"), mp4=str(destination / "video.mp4"),
                        final_latent=str(destination / "final_latent.pt"), created_utc=timestamp(),
                        job_id=os.environ.get("SLURM_JOB_ID"), seconds=time.monotonic() - started)
        write_json(attempt / "meta.json", metadata)
        files = [attempt / "meta.json", attempt / "video.mp4", attempt / "final_latent.pt"]
        files.extend(sorted((attempt / "frames").glob("*.jpg")))
        save_receipt(attempt, "generation", files, {"clip_id": clip["clip_id"]})
        destination.parent.mkdir(exist_ok=True)
        attempt.rename(destination)
        print(f"[saved RGB] {clip['clip_id']}: {metadata['seconds']:.1f}s", flush=True)
        del result, frames, latent


def label(root, clips, plan):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from yolo.yolo_labelmap_concepts import build_label_space, semantic_uncertainty_forward
    from seg_decoder.audit_uncertainty_v2 import validate_frame

    if not torch.cuda.is_available():
        raise RuntimeError("Labelling requires an allocated GPU; use the Slurm wrapper.")
    check_runtime(root, "label")
    pending = []
    for clip in clips:
        directory = root / "clips" / clip["clip_id"]
        if not verified_receipt(directory, "generation"):
            raise ValueError(f"Generate this clip first: {clip['clip_id']}")
        if verified_receipt(directory, "labels"):
            print(f"[skip labels] {clip['clip_id']}", flush=True)
        else:
            pending.append(clip)
    if not pending:
        return

    model = YOLO(plan["yolo_model"])
    model.model.requires_grad_(False)
    model.model.eval()
    label_space, mapping = build_label_space(CORE_CONCEPTS, model.names, False)
    if set(mapping.values()) != set(range(1, 11)):
        raise ValueError("The local YOLO teacher does not cover all ten foreground concepts.")
    first_frame = root / "clips" / pending[0]["clip_id"] / "frames/00000.jpg"
    model.predict(str(first_frame), device=0, conf=0.25, verbose=False)

    # The existing teacher helper checks logits against YOLO's public prediction.
    # Store both hard labels and v2 uncertainty from that SAME prediction.
    for clip_index, clip in enumerate(pending, start=1):
        directory = root / "clips" / clip["clip_id"]
        attempt = new_attempt(root, clip["clip_id"], "labels")
        for folder in ("labelmaps", "labelmaps_uncertainty_v2"):
            (attempt / folder).mkdir()
        agreements = []
        class_pixels = np.zeros(11, dtype=np.int64)
        class_frames = np.zeros(11, dtype=np.int64)
        valid_pixels = 0
        print(f"[labels {clip_index}/{len(pending)}] {clip['clip_id']}", flush=True)
        for frame_index in range(NUM_FRAMES):
            arrays, agreement, _ = semantic_uncertainty_forward(
                model, directory / "frames" / f"{frame_index:05d}.jpg", mapping, 11,
                uncertainty_version=2,
            )
            labels = arrays["label_map"]
            if labels.max() > 10:
                raise ValueError("Teacher produced a label outside the stored class schema.")
            baseline_path = attempt / "labelmaps" / f"{frame_index:05d}.npz"
            uncertainty_path = attempt / "labelmaps_uncertainty_v2" / baseline_path.name
            np.savez_compressed(baseline_path, label_map=labels)
            np.savez_compressed(uncertainty_path, **arrays)
            validate_frame(uncertainty_path, baseline_path, (HEIGHT, WIDTH))
            frame_counts = np.bincount(labels.reshape(-1), minlength=11)
            class_pixels += frame_counts
            class_frames += frame_counts > 0
            valid = arrays["numerical_valid_mask"] & arrays["agreement_mask"] & arrays["mapped_foreground_mask"]
            valid_pixels += int(valid.sum())
            agreements.append(agreement)
        metadata = {
            "mode": "semantic", "model": plan["yolo_model"], "model_sha256": plan["yolo_sha256"],
            "conf": 0.25, "all_classes": False, "n_frames": NUM_FRAMES,
            "label_space": {str(key): value for key, value in label_space.items()},
            "note": "YOLO pseudo-labels, not ground truth. Hard labels and uncertainty share the same prediction.",
        }
        write_json(attempt / "labelmaps/label_space.json", dict(metadata, output_subdir="labelmaps", save_uncertainty=False))
        write_json(attempt / "labelmaps_uncertainty_v2/label_space.json", dict(
            metadata, output_subdir="labelmaps_uncertainty_v2", save_uncertainty=True, uncertainty_version=2,
            min_reconstruction_agreement=min(agreements),
            array_contract={key: str(value.dtype) for key, value in arrays.items()},
        ))
        write_json(attempt / "label_statistics.json", {
            "class_pixels": {label_space[index]: int(count) for index, count in enumerate(class_pixels)},
            "frames_with_class": {label_space[index]: int(count) for index, count in enumerate(class_frames)},
            "valid_pixel_fraction": valid_pixels / (NUM_FRAMES * HEIGHT * WIDTH),
            "focus_class": clip["focus_class"],
            "focus_class_present": bool(class_pixels[CORE_CONCEPTS.index(clip["focus_class"]) + 1]),
            "note": "Presence may be only a few erroneous pixels; inspect counts, masks and RGB before accepting.",
        })
        files = sorted(attempt.rglob("*.npz")) + sorted(attempt.rglob("*.json"))
        save_receipt(attempt, "labels", files, {"clip_id": clip["clip_id"]})
        # Preserve any incomplete publication from an earlier interrupted job.
        for name in ("labelmaps", "labelmaps_uncertainty_v2", "label_statistics.json", "labels_complete.json"):
            target = directory / name
            if target.exists():
                preserved = new_attempt(root, clip["clip_id"], "incomplete_publication")
                target.rename(preserved / name)
            (attempt / name).rename(target)
        print(f"[saved labels] {clip['clip_id']}", flush=True)


def write_tables(root, clips):
    """Manifest covers all planned clips; pending review is never an approval."""
    rows = []
    for clip in clips:
        directory = root / "clips" / clip["clip_id"]
        statistics_path = directory / "label_statistics.json"
        statistics = read_json(statistics_path) if statistics_path.exists() else {}
        rows.append(dict(clip, n_frames=NUM_FRAMES, latent_path=str(directory / "final_latent.pt"),
                         frames_dir=str(directory / "frames"), video_path=str(directory / "video.mp4"),
                         labelmaps_dir=str(directory / "labelmaps"),
                         uncertainty_dir=str(directory / "labelmaps_uncertainty_v2"),
                         focus_class_present=statistics.get("focus_class_present", ""),
                         focus_class_pixels=statistics.get("class_pixels", {}).get(clip["focus_class"], ""),
                         generation_complete=(directory / "generation_complete.json").is_file(),
                         labels_complete=(directory / "labels_complete.json").is_file()))
    temporary = root / "dataset.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "dataset.csv")
    review_path = root / "review.csv"
    if not review_path.exists():
        temporary_review = root / f"review_{uuid.uuid4().hex[:12]}.csv.tmp"
        with temporary_review.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["clip_id", "split", "rgb_quality_ok", "forward_walk_ok", "mask_quality_ok", "notes"])
            writer.writeheader()
            for clip in clips:
                writer.writerow({"clip_id": clip["clip_id"], "split": clip["split"],
                                 "rgb_quality_ok": "pending", "forward_walk_ok": "pending", "mask_quality_ok": "pending", "notes": ""})
        temporary_review.rename(review_path)
    summary = {"planned": len(clips), "train": 80, "validation": 20,
               "generated": sum(row["generation_complete"] for row in rows),
               "labelled": sum(row["labels_complete"] for row in rows),
               "manual_review": "See review.csv; completion does not establish quality or suitability for training."}
    summary["output_size_gib"] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) / 1024**3
    write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "generate", "label", "status"])
    parser.add_argument("--out-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--yolo-model", type=Path, default=Path(DEFAULT_YOLO))
    parser.add_argument("--limit", type=int, default=100, help="Process the first N planned clips (e.g. 4), without changing the 100-clip plan.")
    args = parser.parse_args(argv)
    if not args.out_root.is_absolute() or not 1 <= args.limit <= 100:
        parser.error("Use an absolute output root and --limit between 1 and 100.")
    if re.search(r"(?:seed|_s)0*47(?:\D|$)", str(args.out_root), re.IGNORECASE):
        parser.error("Seed 47 paths are reserved.")
    root = args.out_root
    if args.stage == "status":
        if not (root / "dataset_plan.json").is_file():
            raise FileNotFoundError(root / "dataset_plan.json")
        _, clips = load_clips(read_json(root / "dataset_plan.json")["specification"])
        for clip in clips:
            directory = root / "clips" / clip["clip_id"]
            verified_receipt(directory, "generation")
            verified_receipt(directory, "labels")
        generated = sum((root / "clips" / clip["clip_id"] / "generation_complete.json").is_file() for clip in clips)
        labelled = sum((root / "clips" / clip["clip_id"] / "labels_complete.json").is_file() for clip in clips)
        print(f"Verified {generated}/100 RGB clips and {labelled}/100 label exports. Review: {root / 'review.csv'}")
        return 0
    specification, clips = load_clips(args.prompts)
    plan = make_plan(args, specification)
    check_root(root, plan)
    print(f"Output path: {root}\nResolved storage path: {root.resolve()}")
    report_space(root)
    print(f"Plan: 80 train + 20 validation; this invocation selects {args.limit} clips.")
    if args.stage == "check":
        return 0

    root.mkdir(parents=True, exist_ok=True)
    # Kernel lock is released even if Slurm terminates the process; no stale lock removal.
    with (root / ".dataset.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is writing this dataset. Do not submit duplicates.") from error
        plan_path = root / "dataset_plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Dataset plan changed while waiting for the lock.")
        if not plan_path.exists():
            temporary_plan = root / "dataset_plan.json.tmp"
            if temporary_plan.exists():
                # Retain an interrupted first metadata write before retrying.
                temporary_plan.rename(root / f"initial_plan_{uuid.uuid4().hex[:12]}.json.partial")
            write_json(plan_path, plan)
        write_tables(root, clips)
        try:
            if args.stage == "generate":
                generate(root, clips[:args.limit], plan)
            else:
                label(root, clips[:args.limit], plan)
        finally:
            print(json.dumps(write_tables(root, clips), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
