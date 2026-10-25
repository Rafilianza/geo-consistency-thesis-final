"""Prepare a fixed walking-POV split without copying or changing its data."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageDraw

from seg_decoder.audit_uncertainty_v2 import validate_frame

REPOSITORY = Path(__file__).resolve().parents[2]
# Load this exact local script; an installed Hugging Face "datasets" package
# must not shadow the repository's datasets folder.
specification = importlib.util.spec_from_file_location("walking_codec_inputs", REPOSITORY / "src/datasets/create_walking_pov_100.py")
dataset = importlib.util.module_from_spec(specification)
specification.loader.exec_module(dataset)
DEFAULT_PROMPTS, SETTINGS = dataset.DEFAULT_PROMPTS, dataset.SETTINGS
file_hash, load_clips = dataset.file_hash, dataset.load_clips
read_json, verified_receipt = dataset.read_json, dataset.verified_receipt
DEFAULT_DATA = "/home/dhiazr/gt69_scratch2/dhiazgeo/semantic_dataset_100_v1"
DEFAULT_RUN = "/home/dhiazr/codec_walking100_v1"
DEFAULT_MIXED_DATA = "/home/dhiazr/gt69_scratch2/dhiazgeo/semantic_dataset_mixed_100_v1"
DEFAULT_MIXED_RUN = "/home/dhiazr/codec_walking_mixed100_v1"
SCHEMA = "walking_codec_large9_v1"
CLASS_NAMES = ["other", "car", "bus", "truck", "road", "sidewalk", "building", "vegetation", "sky"]
SOURCE_NAMES = ["background", "person", "car", "bus", "bicycle", "truck", "road", "sidewalk", "building", "vegetation", "sky"]
# Original background is unknown, not a reliable training target for "other".
LABEL_MAPPING = [255, 0, 1, 2, 0, 3, 4, 5, 6, 7, 8]
REVIEW_FIELDS = ["clip_id", "split", "rgb_quality_ok", "forward_walk_ok", "mask_quality_ok", "notes"]
MIXED_REVIEW_FIELDS = REVIEW_FIELDS[:-1] + ["vehicle_motion_count_ok", "notes"]
PALETTE = np.array([[110, 110, 110], [0, 0, 142], [0, 60, 100], [0, 0, 70],
                    [128, 64, 128], [244, 35, 232], [70, 70, 70], [107, 142, 35],
                    [70, 130, 180], [0, 0, 0]], dtype=np.uint8)


def write_json(path, value):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:12]}.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hashes(dataset_kind="original"):
    paths = [
        "src/codec_retraining/data.py", "src/codec_retraining/model.py",
        "src/codec_retraining/run.py", "src/phase3/semantic_codec.py",
        "src/datasets/create_walking_pov_100.py", "src/video_generation/gen_dataset.py",
        "src/seg_decoder/audit_uncertainty_v2.py", "configs/datasets/walking_pov_100_v1.json",
    ]
    if dataset_kind == "mixed":
        paths += ["src/datasets/create_walking_pov_mixed_100.py", "configs/datasets/walking_pov_mixed_100_v1.json"]
    return {name: file_hash(REPOSITORY / name) for name in paths}


def planned_clips(dataset_kind):
    if dataset_kind == "original":
        return load_clips(DEFAULT_PROMPTS)
    if dataset_kind != "mixed":
        raise ValueError("Choose dataset kind original or mixed.")
    # Load the mixed generator only when requested; the original plan stays independent.
    module_specification = importlib.util.spec_from_file_location(
        "mixed_walking_codec_inputs", REPOSITORY / "src/datasets/create_walking_pov_mixed_100.py")
    mixed_dataset = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(mixed_dataset)
    return mixed_dataset.load_clips(mixed_dataset.DEFAULT_PROMPTS)


def mixed_records(data_root, clips):
    """Accept only the completed cohort, with old and new files in their declared roots."""
    plan = read_json(data_root / "dataset_plan.json")
    cohort = read_json(data_root / "cohort.json")
    if (cohort.get("schema") != "walking_pov_mixed_cohort_v1"
            or cohort.get("dataset_id") != "walking_pov_mixed_100_v1"
            or cohort.get("status") != "complete"):
        raise ValueError("The mixed dataset needs a completed 100-clip cohort.json.")
    source_root = Path(plan["source_root"])
    if not source_root.is_absolute() or re.search(r"(?:seed|_s)0*47(?:\D|$)", str(source_root), re.IGNORECASE):
        raise ValueError("The reused dataset root must be absolute and outside reserved data.")
    records = cohort.get("records", [])
    expected = {clip["clip_id"]: clip for clip in clips}
    if len(records) != 100 or {record["clip_id"] for record in records} != set(expected):
        raise ValueError("The mixed cohort must contain each of its 100 planned clips exactly once.")
    by_id = {}
    for record in records:
        clip = expected[record["clip_id"]]
        if any(record.get(key) != value for key, value in clip.items()):
            raise ValueError("Mixed cohort metadata differs from the planned split/prompts/seeds.")
        root = source_root if clip["source"] == "reuse_existing" else data_root
        expected_directory = root / "clips" / clip["clip_id"]
        if Path(record["directory"]) != expected_directory:
            raise ValueError("Mixed cohort clip path differs from its planned source folder.")
        if set(record.get("receipt_hashes", {})) != {"generation", "labels"}:
            raise ValueError("Mixed cohort needs generation and label receipt hashes.")
        by_id[record["clip_id"]] = record
    return by_id


def map_labels(original, valid):
    if original.min() < 0 or original.max() > 10:
        raise ValueError("Expected original class IDs 0 through 10.")
    labels = np.asarray(LABEL_MAPPING, dtype=np.uint8)[original]
    valid = valid.astype(bool) & (original != 0)
    labels[~valid] = 255
    return labels, valid


def verify_clip(directory, clip):
    for stage in ("generation", "labels"):
        if not verified_receipt(directory, stage):
            raise ValueError(f"{clip['clip_id']}: {stage} is incomplete. Let the dataset job finish first.")
    metadata = read_json(directory / "meta.json")
    for key in ("clip_id", "scene", "split", "seed", "prompt"):
        if metadata.get(key) != clip[key]:
            raise ValueError(f"{clip['clip_id']}: metadata differs at {key}.")
    for key in ("height", "width", "num_frames", "fps", "core_concepts"):
        if metadata.get(key) != SETTINGS[key]:
            raise ValueError(f"{clip['clip_id']}: unexpected {key}.")
    expected_frames = {f"{index:05d}" for index in range(81)}
    for folder, extension in (("frames", "jpg"), ("labelmaps", "npz"), ("labelmaps_uncertainty_v2", "npz")):
        actual = {path.stem for path in (directory / folder).glob(f"*.{extension}")}
        if actual != expected_frames:
            raise ValueError(f"{clip['clip_id']}: expected exactly 81 files in {folder}.")
    schema = read_json(directory / "labelmaps_uncertainty_v2/label_space.json")
    if schema.get("uncertainty_version") != 2:
        raise ValueError("Expected original uncertainty v2.")
    if schema.get("label_space") != {str(index): name for index, name in enumerate(SOURCE_NAMES)}:
        raise ValueError("Unexpected original class schema.")
    return {stage: file_hash(directory / f"{stage}_complete.json") for stage in ("generation", "labels")}


def load_clip(directory, frame_count=81, shape=(480, 832)):
    """Keep compact uint8 labels in RAM; use only original v2 validity flags."""
    original_frames, mapped_frames, valid_frames = [], [], []
    for frame_index in range(frame_count):
        name = f"{frame_index:05d}.npz"
        arrays = validate_frame(directory / "labelmaps_uncertainty_v2" / name,
                                directory / "labelmaps" / name, shape)
        original = arrays["label_map"]
        valid = arrays["numerical_valid_mask"] & arrays["agreement_mask"] & arrays["mapped_foreground_mask"]
        mapped, valid = map_labels(original, valid)
        original_frames.append(original.astype(np.uint8))
        mapped_frames.append(mapped)
        valid_frames.append(valid)
    labels = np.stack(mapped_frames)
    valid = np.stack(valid_frames)
    if not valid.any():
        raise ValueError(f"No valid supervised pixels in {directory}.")
    return {"original": np.stack(original_frames), "labels": labels, "valid": valid}


def save_review_preview(directory, output_path, frame_indices=(0, 40, 80), shape=(480, 832)):
    """Small RGB/mask contact sheet; the original MP4 is still needed for motion review."""
    canvas = Image.new("RGB", (832, len(frame_indices) * 260), "white")
    draw = ImageDraw.Draw(canvas)
    for row, frame_index in enumerate(frame_indices):
        name = f"{frame_index:05d}"
        arrays = validate_frame(directory / "labelmaps_uncertainty_v2" / f"{name}.npz",
                                directory / "labelmaps" / f"{name}.npz", shape)
        valid = arrays["numerical_valid_mask"] & arrays["agreement_mask"] & arrays["mapped_foreground_mask"]
        labels, _ = map_labels(arrays["label_map"], valid)
        with Image.open(directory / "frames" / f"{name}.jpg") as image:
            if image.size != (shape[1], shape[0]):
                raise ValueError(f"Unexpected RGB frame size in {directory}.")
            rgb = image.convert("RGB").resize((416, 240))
        mask = Image.fromarray(PALETTE[np.minimum(labels, 9)]).resize((416, 240), Image.Resampling.NEAREST)
        canvas.paste(rgb, (0, row * 260 + 20))
        canvas.paste(mask, (416, row * 260 + 20))
        draw.text((5, row * 260 + 3), f"RGB frame {frame_index}", fill="black")
        draw.text((421, row * 260 + 3), "Mapped target; black = ignored", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=85)


def prepare(data_root, run_dir, dataset_kind="original"):
    data_root = data_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    if run_dir == data_root or data_root in run_dir.parents or run_dir in data_root.parents:
        raise ValueError("Use a separate run folder outside the dataset.")
    specification, clips = planned_clips(dataset_kind)
    dataset_plan = read_json(data_root / "dataset_plan.json")
    if dataset_plan.get("specification") != specification or dataset_plan.get("settings") != SETTINGS:
        raise ValueError("Dataset plan differs from the selected fixed walking-POV plan.")
    cohort_records = mixed_records(data_root, clips) if dataset_kind == "mixed" else None
    if dataset_kind == "mixed":
        source_root = Path(dataset_plan["source_root"])
        if run_dir == source_root or source_root in run_dir.parents or run_dir in source_root.parents:
            raise ValueError("Use a separate run folder outside the reused dataset.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Preparation needs a new empty run folder. Existing preparations/reviews are preserved.")
    records = []
    counts = {"train": np.zeros(9, dtype=np.int64), "val": np.zeros(9, dtype=np.int64)}
    for clip_number, clip in enumerate(clips, start=1):
        directory = data_root / "clips" / clip["clip_id"]
        if cohort_records is not None:
            directory = Path(cohort_records[clip["clip_id"]]["directory"])
        print(f"[prepare {clip_number}/100] {clip['clip_id']}", flush=True)
        receipts = verify_clip(directory, clip)
        if cohort_records is not None and receipts != cohort_records[clip["clip_id"]]["receipt_hashes"]:
            raise ValueError(f"{clip['clip_id']}: receipts changed since mixed cohort completion.")
        arrays = load_clip(directory)
        clip_counts = np.bincount(arrays["labels"][arrays["valid"]], minlength=9)
        counts[clip["split"]] += clip_counts
        records.append(dict(clip, directory=str(directory), receipt_hashes=receipts,
                            valid_class_pixels=clip_counts.tolist()))
        del arrays
    manifest = {
        "schema": SCHEMA, "class_names": CLASS_NAMES, "label_mapping": LABEL_MAPPING,
        "validity_rule": "original numerical_valid_mask & agreement_mask & mapped_foreground_mask; original 0 ignored",
        "uncertainty_note": "Original 11-class confidence scores are preserved, not interpreted as recomputed nine-class scores.",
        "data_root": str(data_root), "dataset_plan_sha256": file_hash(data_root / "dataset_plan.json"),
        "dataset_kind": dataset_kind, "sources": source_hashes(dataset_kind), "records": records,
        "class_pixels": {split: value.tolist() for split, value in counts.items()},
    }
    if dataset_kind == "mixed":
        manifest["cohort_sha256"] = file_hash(data_root / "cohort.json")
    # Publish metadata only after all 100 complete clips have been checked.
    run_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        save_review_preview(Path(record["directory"]), run_dir / "review_previews" / f"{record['clip_id']}.jpg")
    write_json(run_dir / "review_colours.json", {name: PALETTE[index].tolist() for index, name in enumerate(CLASS_NAMES)})
    write_json(run_dir / "prepared.json", manifest)
    for split in ("train", "val"):
        write_json(run_dir / f"{split}_clips.json", [record["clip_id"] for record in records if record["split"] == split])
    with (run_dir / "review.csv").open("x", newline="") as handle:
        review_fields = MIXED_REVIEW_FIELDS if dataset_kind == "mixed" else REVIEW_FIELDS
        writer = csv.DictWriter(handle, fieldnames=review_fields)
        writer.writeheader()
        for clip in clips:
            row = {"clip_id": clip["clip_id"], "split": clip["split"], "notes": ""}
            for field in review_fields[2:-1]:
                row[field] = "pending"
            writer.writerow(row)
    print(f"Prepared 80 training / 20 validation clips in {run_dir}")
    print("Review the videos/masks and fill review.csv with yes/no. No training has run.")


def reviewed_records(manifest, review_path):
    with review_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {record["clip_id"]: record for record in manifest["records"]}
    if len(rows) != len(expected) or {row["clip_id"] for row in rows} != set(expected):
        raise ValueError("Review must contain each planned clip exactly once.")
    selected = {"train": [], "val": []}
    rejected = []
    review_fields = MIXED_REVIEW_FIELDS if manifest.get("dataset_kind", "original") == "mixed" else REVIEW_FIELDS
    for row in rows:
        record = expected[row["clip_id"]]
        if row["split"] != record["split"]:
            raise ValueError("Do not move clips between train and validation.")
        decisions = [row.get(field, "").strip().lower() for field in review_fields[2:-1]]
        if any(value not in ("yes", "no") for value in decisions):
            raise ValueError(f"Finish the actual review for {row['clip_id']}; use yes/no.")
        if all(value == "yes" for value in decisions):
            selected[record["split"]].append(record)
        else:
            rejected.append(record["clip_id"])
    if not selected["train"] or not selected["val"]:
        raise ValueError("Both splits need accepted clips.")
    return selected, rejected


def validate_prepared(manifest):
    dataset_kind = manifest.get("dataset_kind", "original")
    _, planned = planned_clips(dataset_kind)
    expected = {clip["clip_id"]: clip for clip in planned}
    records = manifest["records"]
    if manifest["schema"] != SCHEMA or manifest["class_names"] != CLASS_NAMES or manifest["label_mapping"] != LABEL_MAPPING:
        raise ValueError("Prepared class mapping differs from this codec.")
    if len(records) != 100 or {record["clip_id"] for record in records} != set(expected):
        raise ValueError("Prepared records must match the fixed 100-clip plan.")
    root = Path(manifest["data_root"])
    if not root.is_absolute():
        raise ValueError("Prepared dataset root must be absolute.")
    cohort_records = None
    if dataset_kind == "mixed":
        if file_hash(root / "cohort.json") != manifest["cohort_sha256"]:
            raise ValueError("Mixed cohort changed after preparation.")
        cohort_records = mixed_records(root, planned)
    for record in records:
        clip = expected[record["clip_id"]]
        if any(record[key] != value for key, value in clip.items()):
            raise ValueError("Prepared clip metadata differs from the planned split/prompts/seeds.")
        expected_directory = root / "clips" / record["clip_id"]
        if cohort_records is not None:
            cohort_record = cohort_records[record["clip_id"]]
            expected_directory = Path(cohort_record["directory"])
            if record["receipt_hashes"] != cohort_record["receipt_hashes"]:
                raise ValueError("Prepared receipts differ from the mixed cohort.")
        if Path(record["directory"]) != expected_directory:
            raise ValueError("Prepared clip path must stay in its planned dataset folder.")


def load_training_data(manifest, selected):
    validate_prepared(manifest)
    if manifest["sources"] != source_hashes(manifest.get("dataset_kind", "original")):
        raise ValueError("Sources changed after preparation. Preserve this run and prepare a new folder.")
    data_root = Path(manifest["data_root"])
    if file_hash(data_root / "dataset_plan.json") != manifest["dataset_plan_sha256"]:
        raise ValueError("Dataset plan changed after preparation.")
    result = {"train": [], "val": []}
    for split, records in selected.items():
        for record in records:
            directory = Path(record["directory"])
            if verify_clip(directory, record) != record["receipt_hashes"]:
                raise ValueError(f"{record['clip_id']}: receipts changed after preparation.")
            arrays = load_clip(directory)
            if split == "train":
                del arrays["original"]
            result[split].append(dict(arrays, clip_id=record["clip_id"]))
            print(f"[loaded {split}] {record['clip_id']}", flush=True)
    return result
