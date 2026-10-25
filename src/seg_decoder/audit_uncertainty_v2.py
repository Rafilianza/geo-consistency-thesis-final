#!/usr/bin/env python3
"""Validate thesis-space uncertainty and build train-only class CDFs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


NUM_BINS = 4096
REQUIRED_KEYS = {
    "label_map",
    "teacher_target_probability",
    "teacher_grouped_margin",
    "teacher_grouped_entropy",
    "numerical_valid_mask",
    "agreement_mask",
    "mapped_foreground_mask",
}
FLOAT_KEYS = {
    "teacher_target_probability",
    "teacher_grouped_margin",
    "teacher_grouped_entropy",
}
MASK_KEYS = {
    "numerical_valid_mask",
    "agreement_mask",
    "mapped_foreground_mask",
}


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"manifest contains no clips: {path}")
    return rows


def numbered_npz_files(directory: Path) -> List[Path]:
    return sorted(
        directory.glob("[0-9][0-9][0-9][0-9][0-9].npz"),
        key=lambda path: int(path.stem),
    )


def load_label_space(directory: Path) -> Tuple[Dict[int, str], Dict[str, object]]:
    path = directory / "label_space.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("uncertainty_version") != 2:
        raise ValueError(f"{path}: expected uncertainty_version=2")
    label_space = {
        int(key): str(value) for key, value in payload["label_space"].items()
    }
    return label_space, payload


def margin_bins(margin: np.ndarray) -> np.ndarray:
    scaled = np.floor((margin.astype(np.float32) + 1.0) * (NUM_BINS / 2.0))
    return np.clip(scaled, 0, NUM_BINS - 1).astype(np.int32)


def validate_frame(
    uncertainty_path: Path,
    baseline_path: Path,
    expected_shape: Optional[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    with np.load(uncertainty_path) as payload:
        missing = REQUIRED_KEYS - set(payload.files)
        if missing:
            raise KeyError(f"{uncertainty_path}: missing keys {sorted(missing)}")
        arrays = {key: payload[key] for key in REQUIRED_KEYS}
    shape = arrays["label_map"].shape
    if len(shape) != 2:
        raise ValueError(f"{uncertainty_path}: expected [H,W], got {shape}")
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"{uncertainty_path}: shape {shape} != {expected_shape}")
    for key, array in arrays.items():
        if array.shape != shape:
            raise ValueError(f"{uncertainty_path}: {key} shape differs from label")
        if not np.isfinite(array).all():
            raise ValueError(f"{uncertainty_path}: {key} contains non-finite values")
    if arrays["label_map"].dtype != np.uint16:
        raise TypeError(f"{uncertainty_path}: label_map must be uint16")
    for key in FLOAT_KEYS:
        if arrays[key].dtype != np.float16:
            raise TypeError(f"{uncertainty_path}: {key} must be float16")
    for key in MASK_KEYS:
        if arrays[key].dtype != np.uint8:
            raise TypeError(f"{uncertainty_path}: {key} must be uint8")
        if not np.isin(arrays[key], (0, 1)).all():
            raise ValueError(f"{uncertainty_path}: {key} is not binary")
    for key in ("teacher_target_probability", "teacher_grouped_entropy"):
        if float(arrays[key].min()) < 0.0 or float(arrays[key].max()) > 1.0:
            raise ValueError(f"{uncertainty_path}: {key} outside [0,1]")
    margin = arrays["teacher_grouped_margin"]
    if float(margin.min()) < -1.0 or float(margin.max()) > 1.0:
        raise ValueError(f"{uncertainty_path}: grouped margin outside [-1,1]")
    if not bool(arrays["numerical_valid_mask"].all()):
        raise ValueError(f"{uncertainty_path}: numerical_valid_mask must be all ones")
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    with np.load(baseline_path) as baseline:
        baseline_label = baseline["label_map"]
    if not np.array_equal(arrays["label_map"], baseline_label):
        raise ValueError(f"{uncertainty_path}: label_map differs from baseline")
    return arrays


def cdf_from_histogram(histogram: np.ndarray) -> Tuple[np.ndarray, float]:
    total = int(histogram.sum())
    if total == 0:
        raise RuntimeError("cannot build class CDF without training pixels")
    cumulative_before = np.cumsum(histogram) - histogram
    midrank = (cumulative_before + 0.5 * histogram) / total
    raw_weight = 0.5 + 0.5 * midrank
    raw_mean = float(np.sum(raw_weight * histogram) / total)
    return midrank.astype(np.float32), raw_mean


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--uncertainty-subdir", default="labelmaps_uncertainty_v2")
    parser.add_argument("--output", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--reliable-percentile", type=float, default=0.10)
    parser.add_argument("--expected-clips", type=int, default=20)
    parser.add_argument("--expected-frames", type=int, default=81)
    args = parser.parse_args()
    if not 0.0 < args.reliable_percentile < 1.0:
        parser.error("--reliable-percentile must be between zero and one")

    data_root = Path(args.data_root)
    manifest = Path(args.manifest) if args.manifest else data_root / "dataset.csv"
    output = (
        Path(args.output)
        if args.output
        else data_root / "uncertainty_v2_stats.json"
    )
    rows = read_manifest(manifest)
    if len(rows) != args.expected_clips:
        raise RuntimeError(f"expected {args.expected_clips} clips, found {len(rows)}")
    first_dir = Path(rows[0]["frames_dir"]).parent / args.uncertainty_subdir
    label_space, _ = load_label_space(first_dir)
    class_ids = sorted(label_space)
    if class_ids != list(range(len(class_ids))):
        raise ValueError(f"label ids must be contiguous, got {class_ids}")
    num_classes = len(class_ids)
    histograms = np.zeros((num_classes, NUM_BINS), dtype=np.int64)
    expected_shape: Optional[Tuple[int, int]] = None
    split_counts: Dict[str, Dict[str, object]] = {}

    print("[audit-v2] pass 1/2: validate files and build train-only class CDFs")
    for clip_index, row in enumerate(rows, start=1):
        clip_id = row["clip_id"]
        split = row["split"]
        clip_root = Path(row["frames_dir"]).parent
        uncertainty_dir = clip_root / args.uncertainty_subdir
        found_space, _ = load_label_space(uncertainty_dir)
        if found_space != label_space:
            raise ValueError(f"label-space mismatch: {uncertainty_dir}")
        files = numbered_npz_files(uncertainty_dir)
        if len(files) != args.expected_frames:
            raise RuntimeError(
                f"{clip_id}: expected {args.expected_frames} files, found {len(files)}"
            )
        summary = split_counts.setdefault(
            split,
            {
                "clips": 0,
                "frames": 0,
                "pixels": 0,
                "agreement_pixels": 0,
                "foreground_pixels": 0,
                "class_pixels": np.zeros(num_classes, dtype=np.int64),
            },
        )
        summary["clips"] += 1
        for path in files:
            arrays = validate_frame(
                path, clip_root / "labelmaps" / path.name, expected_shape
            )
            if expected_shape is None:
                expected_shape = arrays["label_map"].shape
            label = arrays["label_map"].astype(np.int64, copy=False)
            if label.size and int(label.max()) >= num_classes:
                raise ValueError(f"{path}: class id outside label space")
            agreement = arrays["agreement_mask"].astype(bool, copy=False)
            mapped_fg = arrays["mapped_foreground_mask"].astype(bool, copy=False)
            summary["frames"] += 1
            summary["pixels"] += label.size
            summary["agreement_pixels"] += int(agreement.sum())
            summary["foreground_pixels"] += int(mapped_fg.sum())
            summary["class_pixels"] += np.bincount(
                label.reshape(-1), minlength=num_classes
            )
            if split == args.train_split:
                bins = margin_bins(arrays["teacher_grouped_margin"])
                for class_id in class_ids:
                    selected = label == class_id
                    histograms[class_id] += np.bincount(
                        bins[selected], minlength=NUM_BINS
                    )
        print(f"[audit-v2]   {clip_index:02d}/{len(rows):02d} {clip_id}")

    percentiles = np.zeros_like(histograms, dtype=np.float32)
    raw_means = np.zeros(num_classes, dtype=np.float64)
    for class_id in class_ids:
        percentiles[class_id], raw_means[class_id] = cdf_from_histogram(
            histograms[class_id]
        )

    reliable_by_split: Dict[str, Dict[str, object]] = {}
    weight_by_class_sum = np.zeros(num_classes, dtype=np.float64)
    weight_by_class_count = np.zeros(num_classes, dtype=np.int64)
    print("[audit-v2] pass 2/2: verify weights and reliable coverage")
    for clip_index, row in enumerate(rows, start=1):
        split = row["split"]
        clip_root = Path(row["frames_dir"]).parent
        files = numbered_npz_files(clip_root / args.uncertainty_subdir)
        summary = reliable_by_split.setdefault(
            split,
            {
                "pixels": 0,
                "reliable_pixels": 0,
                "class_reliable_pixels": np.zeros(num_classes, dtype=np.int64),
            },
        )
        for path in files:
            with np.load(path) as payload:
                label = payload["label_map"].astype(np.int64, copy=False)
                bins = margin_bins(payload["teacher_grouped_margin"])
                agreement = payload["agreement_mask"].astype(bool, copy=False)
                numerical = payload["numerical_valid_mask"].astype(bool, copy=False)
            pixel_percentile = percentiles[label, bins]
            weights = (0.5 + 0.5 * pixel_percentile) / raw_means[label]
            reliable = (
                numerical
                & agreement
                & (pixel_percentile >= args.reliable_percentile)
            )
            summary["pixels"] += label.size
            summary["reliable_pixels"] += int(reliable.sum())
            summary["class_reliable_pixels"] += np.bincount(
                label[reliable], minlength=num_classes
            )
            if split == args.train_split:
                for class_id in class_ids:
                    selected = label == class_id
                    weight_by_class_sum[class_id] += float(
                        weights[selected].sum(dtype=np.float64)
                    )
                    weight_by_class_count[class_id] += int(selected.sum())
        print(f"[audit-v2]   {clip_index:02d}/{len(rows):02d} {row['clip_id']}")

    class_weight_means = weight_by_class_sum / weight_by_class_count
    if not np.allclose(class_weight_means, 1.0, atol=0.005):
        raise RuntimeError(
            f"per-class soft weights are not mean one: {class_weight_means.tolist()}"
        )

    serialised_splits = {}
    for split, summary in split_counts.items():
        reliable = reliable_by_split[split]
        pixels = int(summary["pixels"])
        serialised_splits[split] = {
            "clips": int(summary["clips"]),
            "frames": int(summary["frames"]),
            "pixels": pixels,
            "agreement_fraction": int(summary["agreement_pixels"]) / pixels,
            "mapped_foreground_fraction": int(summary["foreground_pixels"]) / pixels,
            "reliable_fraction": int(reliable["reliable_pixels"]) / pixels,
            "classes": {
                label_space[class_id]: {
                    "pixels": int(summary["class_pixels"][class_id]),
                    "reliable_pixels": int(
                        reliable["class_reliable_pixels"][class_id]
                    ),
                }
                for class_id in class_ids
            },
        }

    report = {
        "version": 2,
        "data_root": str(data_root),
        "manifest": str(manifest),
        "uncertainty_subdir": args.uncertainty_subdir,
        "file_contract_validated": True,
        "expected_shape": list(expected_shape) if expected_shape else None,
        "label_space": {str(key): value for key, value in label_space.items()},
        "threshold_source_split": args.train_split,
        "num_margin_bins": NUM_BINS,
        "margin_range": [-1.0, 1.0],
        "reliable_percentile": args.reliable_percentile,
        "weight_rule": "(0.5 + 0.5 * class_midrank_percentile) / class_raw_mean",
        "classes": {
            str(class_id): {
                "name": label_space[class_id],
                "train_histogram": histograms[class_id].tolist(),
                "bin_midrank_percentiles": percentiles[class_id].tolist(),
                "raw_weight_mean": float(raw_means[class_id]),
                "normalised_train_weight_mean": float(class_weight_means[class_id]),
            }
            for class_id in class_ids
        },
        "splits": serialised_splits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"[audit-v2] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
