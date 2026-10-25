#!/usr/bin/env python3
"""Generate 60 new moving-vehicle clips and reference 40 existing clips.

check/status use only the standard library. The wrapper runs Wan and YOLO in
separate GPU processes. Old clips are read in place, never copied or generated.
"""

import argparse
from collections import Counter
import csv
import fcntl
import importlib.util
from pathlib import Path
import re
import uuid

REPOSITORY = Path(__file__).resolve().parents[2]
# Load the local file explicitly; an installed 'datasets' package is unrelated.
module_spec = importlib.util.spec_from_file_location(
    "original_walking_dataset", REPOSITORY / "src/datasets/create_walking_pov_100.py"
)
base = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(base)

DEFAULT_ROOT = "/home/dhiazr/gt69_scratch2/dhiazgeo/semantic_dataset_mixed_100_v1"
DEFAULT_SOURCE_ROOT = base.DEFAULT_ROOT
DEFAULT_PROMPTS = REPOSITORY / "configs/datasets/walking_pov_mixed_100_v1.json"
SETTINGS = base.SETTINGS
DATASET_ID = "walking_pov_mixed_100_v1"


def load_clips(path):
    """Keep the 40/60 selection, seeds and prompt split explicit."""
    specification = path if isinstance(path, dict) else base.read_json(path)
    if specification["dataset_id"] != DATASET_ID:
        raise ValueError("Expected walking_pov_mixed_100_v1.")
    original_spec, originals = base.load_clips(base.DEFAULT_PROMPTS)
    original_scenes = {entry["scene"]: entry for entry in original_spec["prompts"]}
    if specification["train_seeds"] != [42, 43, 44, 45] or specification["validation_seeds"] != [46, 48, 49, 50]:
        raise ValueError("Use the approved seeds; seed 47 is reserved.")
    scenes, texts, clips = set(), set(), []
    for entry in specification["prompts"]:
        scene, prompt = entry["scene"], entry["prompt"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", scene) or scene in scenes or prompt in texts:
            raise ValueError(f"Invalid or repeated scene/prompt: {scene}")
        if entry["split"] not in ("train", "val"):
            raise ValueError(f"Invalid split: {scene}")
        if not 80 <= len(prompt.split()) <= 100:
            raise ValueError(f"{scene}: expected an 80–100 word prompt.")
        if any(word not in prompt.lower() for word in ("first-person", "walking steadily forward", "eye level")):
            raise ValueError(f"{scene}: expected forward-walking POV.")
        vehicles = entry["vehicle_classes"]
        if not vehicles or set(vehicles) - {"car", "bus", "truck"}:
            raise ValueError(f"{scene}: unexpected vehicle classes.")
        if any(type(count) is not int or count < 1 for count in vehicles.values()):
            raise ValueError(f"{scene}: invalid vehicle counts.")
        if sum(vehicles.values()) != entry["intended_vehicle_count"]:
            raise ValueError(f"{scene}: vehicle counts disagree.")
        seed_key = "train_seeds" if entry["split"] == "train" else "validation_seeds"
        seeds = specification[seed_key]
        if entry["source"] == "reuse_existing":
            original = original_scenes.get(scene)
            if original is None or any(entry[key] != original[key] for key in ("scene", "split", "prompt")):
                raise ValueError(f"{scene}: reuse must match its original prompt and split.")
            if vehicles != {original["focus_class"]: 1}:
                raise ValueError(f"{scene}: expected one original vehicle class.")
            expected_ids = [f"{scene}_s{seed}" for seed in seeds]
            if entry["source_clip_ids"] != expected_ids or entry["source_dataset_id"] != original_spec["dataset_id"]:
                raise ValueError(f"{scene}: reuse provenance changed.")
        elif entry["source"] == "generate_new":
            if scene in original_scenes or entry["intended_vehicle_count"] < 3:
                raise ValueError(f"{scene}: new prompts need new scene IDs and at least three vehicles.")
        else:
            raise ValueError(f"{scene}: unexpected source.")
        if re.search(r"\b(parked|stopped|waits|waiting)\b", prompt.lower()):
            raise ValueError(f"{scene}: the proposed vehicles should be moving.")
        scenes.add(scene)
        texts.add(prompt)
        for seed in seeds:
            # focus_class is a compatibility hint for the old label statistics;
            # all classes are exported. It is not an instance-count measurement.
            clips.append(dict(entry, seed=seed, clip_id=f"{scene}_s{seed}", focus_class=next(iter(vehicles))))
    counts = Counter((clip["source"], clip["split"]) for clip in clips)
    expected = {("reuse_existing", "train"): 32, ("reuse_existing", "val"): 8,
                ("generate_new", "train"): 48, ("generate_new", "val"): 12}
    if dict(counts) != expected or len(scenes) != 25:
        raise ValueError("Expected 40 reused + 60 new clips, with an 80/20 split.")
    return specification, clips


def clip_directory(root, source_root, clip):
    parent = source_root if clip["source"] == "reuse_existing" else root
    directory = parent / "clips" / clip["clip_id"]
    if (parent / "clips").is_symlink() or directory.is_symlink():
        raise ValueError(f"Expected an actual clip directory, not a link: {directory}")
    return directory


def verify_clip(directory, clip, require_complete=True):
    """Check receipts, identity and expected files without loading a model."""
    hashes = {}
    for stage in ("generation", "labels"):
        if not base.verified_receipt(directory, stage):
            if require_complete:
                raise ValueError(f"{clip['clip_id']}: missing completed {stage} files.")
            continue
        receipt = base.read_json(directory / f"{stage}_complete.json")
        if receipt.get("clip_id") != clip["clip_id"]:
            raise ValueError(f"{clip['clip_id']}: receipt belongs to another clip.")
        if stage == "generation":
            expected = {"meta.json", "video.mp4", "final_latent.pt"}
            expected.update(f"frames/{index:05d}.jpg" for index in range(SETTINGS["num_frames"]))
        else:
            expected = {"label_statistics.json", "labelmaps/label_space.json", "labelmaps_uncertainty_v2/label_space.json"}
            for folder in ("labelmaps", "labelmaps_uncertainty_v2"):
                expected.update(f"{folder}/{index:05d}.npz" for index in range(SETTINGS["num_frames"]))
        if set(receipt["artifacts"]) != expected:
            raise ValueError(f"{clip['clip_id']}: {stage} receipt has an unexpected file set.")
        hashes[stage] = base.file_hash(directory / f"{stage}_complete.json")
    if "generation" in hashes:
        metadata = base.read_json(directory / "meta.json")
        for key in ("clip_id", "scene", "split", "seed", "prompt"):
            if metadata.get(key) != clip[key]:
                raise ValueError(f"{clip['clip_id']}: metadata changed at {key}.")
        for key, value in SETTINGS.items():
            if metadata.get(key) != value:
                raise ValueError(f"{clip['clip_id']}: generation setting differs at {key}.")
    return hashes


def check_source(source_root, clips):
    """Only inspect the retained 40 clips; removed clips are never required."""
    original_spec, _ = base.load_clips(base.DEFAULT_PROMPTS)
    source_plan = base.read_json(source_root / "dataset_plan.json")
    if source_plan.get("specification") != original_spec or source_plan.get("settings") != SETTINGS:
        raise ValueError("Source dataset plan/settings differ from the completed original dataset.")
    receipts = {}
    for clip in clips:
        if clip["source"] == "reuse_existing":
            directory = clip_directory(source_root, source_root, clip)
            receipts[clip["clip_id"]] = verify_clip(directory, clip)
    return receipts


def make_plan(args, specification, clips):
    source_root = args.source_root.resolve(strict=True)
    root = args.out_root.resolve()
    if root == source_root or root in source_root.parents or source_root in root.parents:
        raise ValueError("Use separate, non-nested roots for old and new datasets.")
    receipts = check_source(source_root, clips)
    plan = base.make_plan(args, specification)
    plan.update(source_root=str(source_root),
                source_dataset_plan_sha256=base.file_hash(source_root / "dataset_plan.json"),
                reused_receipt_hashes=receipts)
    plan["sources"][str(Path(__file__).resolve().relative_to(REPOSITORY))] = base.file_hash(Path(__file__).resolve())
    source_plan = base.read_json(source_root / "dataset_plan.json")
    if source_plan.get("yolo_sha256") != plan["yolo_sha256"]:
        raise ValueError("Use the same YOLO teacher weights as the original dataset.")
    if source_plan.get("model_inventory") != plan["model_inventory"]:
        raise ValueError("Wan model inventory differs from the original dataset.")
    return plan


def inspect_cohort(root, plan, clips):
    source_root = Path(plan["source_root"])
    if base.file_hash(source_root / "dataset_plan.json") != plan["source_dataset_plan_sha256"]:
        raise ValueError("The original dataset plan changed.")
    records = []
    for clip in clips:
        directory = clip_directory(root, source_root, clip)
        reused = clip["source"] == "reuse_existing"
        receipts = verify_clip(directory, clip, require_complete=reused)
        if reused and receipts != plan["reused_receipt_hashes"][clip["clip_id"]]:
            raise ValueError(f"A retained original clip changed: {clip['clip_id']}")
        records.append(dict(clip, directory=str(directory.resolve()), receipt_hashes=receipts))
    complete = all(len(record["receipt_hashes"]) == 2 for record in records)
    return {"schema": "walking_pov_mixed_cohort_v1", "dataset_id": DATASET_ID,
            "status": "complete" if complete else "partial", "records": records}


def write_tables(root, cohort):
    """Publish references, not copies. Manual review remains a separate step."""
    base.write_json(root / "cohort.json", cohort)
    rows = []
    for record in cohort["records"]:
        directory = Path(record["directory"])
        rows.append({"clip_id": record["clip_id"], "split": record["split"], "source": record["source"],
                     "intended_vehicle_count": record["intended_vehicle_count"], "directory": str(directory),
                     "video_path": str(directory / "video.mp4"),
                     "generation_complete": "generation" in record["receipt_hashes"],
                     "labels_complete": "labels" in record["receipt_hashes"]})
    temporary = root / "dataset.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "dataset.csv")
    review_path = root / "review.csv"
    if not review_path.exists():
        temporary = root / "review.csv.tmp"
        fields = ["clip_id", "split", "rgb_quality_ok", "forward_walk_ok", "mask_quality_ok", "vehicle_motion_count_ok", "notes"]
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in cohort["records"]:
                row = {field: "pending" for field in fields[2:-1]}
                row.update(clip_id=record["clip_id"], split=record["split"], notes="")
                writer.writerow(row)
        temporary.replace(review_path)
    summary = {"status": cohort["status"], "planned": 100, "train": 80, "validation": 20,
               "reused": 40, "new_planned": 60,
               "new_generated": sum(row["generation_complete"] for row in rows if row["source"] == "generate_new"),
               "new_labelled": sum(row["labels_complete"] for row in rows if row["source"] == "generate_new"),
               "manual_review": "Confirm vehicle_motion_count_ok: >=3 moving vehicles in each new clip; one in reused clips. Counts are not automatically verified.",
               "new_output_size_gib": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) / 1024**3}
    base.write_json(root / "summary.json", summary)
    print(summary, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "generate", "label", "status"])
    parser.add_argument("--out-root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--source-root", type=Path, default=Path(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--model-dir", type=Path, default=Path(base.DEFAULT_MODEL))
    parser.add_argument("--yolo-model", type=Path, default=Path(base.DEFAULT_YOLO))
    parser.add_argument("--limit", type=int, default=60, help="Process the first N NEW clips; default 60.")
    args = parser.parse_args(argv)
    if not args.out_root.is_absolute() or not args.source_root.is_absolute() or not 1 <= args.limit <= 60:
        parser.error("Use absolute dataset paths and --limit between 1 and 60.")
    for path in (args.out_root, args.source_root):
        if re.search(r"(?:seed|_s)0*47(?:\D|$)", str(path), re.IGNORECASE):
            parser.error("Seed 47 paths are reserved.")
    root = args.out_root
    if args.stage == "status":
        plan = base.read_json(root / "dataset_plan.json")
        _, clips = load_clips(plan["specification"])
        cohort = inspect_cohort(root, plan, clips)
        completed = sum(len(record["receipt_hashes"]) == 2 for record in cohort["records"])
        print(f"Verified {completed}/100 complete clips (40 reused + 60 new); {cohort['status']}.")
        return 0
    specification, clips = load_clips(args.prompts)
    plan = make_plan(args, specification, clips)
    base.check_root(root, plan)
    base.report_space(root)
    print(f"Retained 40 old clips verified. New outputs: {root}", flush=True)
    if args.stage == "check":
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".dataset.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.check_root(root, plan)
        plan_path = root / "dataset_plan.json"
        if not plan_path.exists():
            temporary = root / "dataset_plan.json.tmp"
            if temporary.exists():
                temporary.rename(root / f"initial_plan_{uuid.uuid4().hex[:12]}.json.partial")
            base.write_json(plan_path, plan)
        new_clips = [clip for clip in clips if clip["source"] == "generate_new"][:args.limit]
        write_tables(root, inspect_cohort(root, plan, clips))
        try:
            if args.stage == "generate":
                base.generate(root, new_clips, plan)
            else:
                base.label(root, new_clips, plan)
        finally:
            write_tables(root, inspect_cohort(root, plan, clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
