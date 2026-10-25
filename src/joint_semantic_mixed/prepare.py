"""Freeze the selected codec, fit training-only normalization and cache masks."""

from pathlib import Path

import torch

from codec_retraining.data import load_clip, reviewed_records, validate_prepared, verify_clip
from joint_semantic_mixed.common import (
    CLASS_NAMES, FORMAT, LABEL_MAPPING, SEMANTIC_SHAPE, atomic_torch_save, check_tensor,
    file_hash, fingerprint, load_codec, load_prepared, read_json, source_hashes, write_json,
)


def channel_statistics(latents):
    """Population moments, weighted by voxel count; caller supplies training only."""
    total = torch.zeros(8, dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for latent in latents:
        values = check_tensor(latent, SEMANTIC_SHAPE).detach().cpu().double()
        total += values.sum(dim=(0, 2, 3, 4))
        squared += values.square().sum(dim=(0, 2, 3, 4))
        count += values.numel() // 8
    if count == 0:
        raise ValueError("No training latents for normalization.")
    mean = total / count
    std = (squared / count - mean.square()).clamp_min(0).sqrt()
    if not bool(torch.isfinite(std).all()) or bool((std < 1e-6).any()):
        raise ValueError("A semantic channel is nonfinite or has almost zero variance.")
    return mean.float(), std.float(), count


@torch.no_grad()
def encode_record(codec, record, device):
    directory = Path(record["directory"])
    if verify_clip(directory, record) != record["receipt_hashes"]:
        raise ValueError(f"Source clip changed: {record['clip_id']}")
    arrays = load_clip(directory)
    labels = torch.as_tensor(arrays["labels"], device=device, dtype=torch.long).unsqueeze(0)
    valid = torch.as_tensor(arrays["valid"], device=device).unsqueeze(0)
    return check_tensor(codec.encode(labels, valid).float(), SEMANTIC_SHAPE)


def prepare(args, device):
    source = args.codec_run
    original = read_json(source / "prepared.json")
    validate_prepared(original)
    if original.get("dataset_kind") != "mixed":
        raise ValueError("Use the completed mixed-dataset codec run.")
    selected, rejected = reviewed_records(original, source / "review.csv")
    config = read_json(source / "config.json")
    completion = read_json(source / "complete.json")
    weights = source / "best_weights.pt"
    if (file_hash(weights) != completion["best_weights_sha256"]
            or file_hash(source / "prepared.json") != config["prepared_sha256"]
            or file_hash(source / "review.csv") != config["review_sha256"]):
        raise ValueError("The codec weights, preparation or reviews differ from its completed training.")
    accepted = {split: [record["clip_id"] for record in records] for split, records in selected.items()}
    if accepted != config["accepted"]:
        raise ValueError("Use the same accepted clips as codec training.")
    # Do not put new data inside either source dataset.
    source_roots = {Path(record["directory"]).resolve().parents[1] for records in selected.values() for record in records}
    for root in source_roots:
        for output in (args.weights_dir.resolve(), args.outputs_dir.resolve()):
            if output == root or root in output.parents or output in root.parents:
                raise ValueError("Keep new weights/cache separate from source datasets.")
    plan = {"format": FORMAT, "codec_run": str(source.resolve()), "codec_sha256": file_hash(weights),
            "prepared_sha256": config["prepared_sha256"], "review_sha256": config["review_sha256"],
            "weights_dir": str(args.weights_dir.resolve()), "outputs_dir": str(args.outputs_dir.resolve()),
            "accepted": accepted, "sources": source_hashes(), "encoding_precision": "float32"}
    marker = args.weights_dir / "prepare_plan.json"
    if marker.exists():
        if not args.resume or read_json(marker) != plan:
            raise ValueError("Preparation exists. Use --resume with unchanged inputs, or new output folders.")
        if (args.weights_dir / "prepared.json").exists():
            load_prepared(args)
            print("Semantic preparation already complete and verified.")
            return
    else:
        for root in (args.weights_dir, args.outputs_dir):
            if root.exists() and any(root.iterdir()):
                raise ValueError(f"Expected a new empty output folder: {root}")
            root.mkdir(parents=True, exist_ok=True)
        write_json(marker, plan)
    codec = load_codec(weights, device)
    print(f"Fitting normalization on {len(selected['train'])} training clips only.", flush=True)
    training_latents = (encode_record(codec, record, device) for record in selected["train"])
    mean, std, count = channel_statistics(training_latents)
    codec.set_normalization_statistics(mean, std)
    statistics = {"mean": mean.tolist(), "std": std.tolist(), "voxel_count_per_channel": count,
                  "training_clip_ids": accepted["train"], "fit_split": "train", "source_codec_sha256": file_hash(weights)}
    write_json(args.weights_dir / "normalization_stats.json", statistics)
    payload = {"class_names": CLASS_NAMES, "label_mapping": LABEL_MAPPING,
               "model_state_dict": {key: value.cpu() for key, value in codec.state_dict().items()},
               "selected_epoch": completion["selected_epoch"], "source_codec_sha256": file_hash(weights),
               "format": FORMAT, "normalization_fitted": True}
    atomic_torch_save(args.weights_dir / "frozen_codec.pt", payload)
    records = []
    max_roundtrip_error = 0.0
    cache = args.outputs_dir / "semantic_latents"
    cache.mkdir(exist_ok=True)
    for split in ("train", "val"):
        for record in selected[split]:
            raw = encode_record(codec, record, device)
            normalized = codec.normalize(raw).cpu()
            path = cache / f"{record['clip_id']}.pt"
            atomic_torch_save(path, normalized)
            restored = torch.load(path, map_location=device, weights_only=True)
            restored = codec.denormalize(check_tensor(restored, SEMANTIC_SHAPE))
            error = float((restored - raw).abs().max())
            if not torch.allclose(restored, raw, rtol=1e-5, atol=1e-5):
                raise ValueError("Cached semantic latent failed its round-trip check.")
            max_roundtrip_error = max(max_roundtrip_error, error)
            rgb_path = Path(record["directory"]) / "final_latent.pt"
            records.append(dict(record, semantic_latent=str(path.resolve()), semantic_latent_sha256=file_hash(path),
                                rgb_latent=str(rgb_path), rgb_latent_sha256=file_hash(rgb_path)))
            print(f"[cached {len(records)}] {record['clip_id']}", flush=True)
    frozen_files = {name: file_hash(args.weights_dir / name) for name in ("frozen_codec.pt", "normalization_stats.json")}
    manifest = dict(plan, records=records, weight_files=frozen_files, rejected=rejected,
                    max_normalization_roundtrip_error=max_roundtrip_error,
                    review_note="Acceptance follows the researcher's CSV; visual quality is not independently verified here.")
    write_json(args.weights_dir / "prepared.json", manifest)
    print(f"Prepared {len(records)} semantic latents. Codec weights: {args.weights_dir}; caches: {cache}", flush=True)
