"""Continue a completed CNN run in new folders, reusing its frozen inputs."""

from pathlib import Path
import re
import shutil

import torch

from joint_semantic_mixed.common import (
    FORMAT, file_hash, fingerprint, load_prepared, read_json, source_hashes, write_json,
)

# Only these workflow files changed from the completed 8cea091 run.
# Model architecture, loss inputs, codec, dataset and sampling code must still match.
ORIGINAL_WORKFLOW = {
    "src/joint_semantic_mixed/run.py": "5e1fdea21fe702f33db4379907fe337c276a0f3834faf84ed52b2abb2d7ca915",
    "src/joint_semantic_mixed/training.py": "6e21c02c2f50de4ade5a3c371c86724cdaddeda262009901e88f76b740640822",
    "src/joint_semantic_mixed/monitoring.py": "3f78fa867a145eef79c4e76ff9f00972510ab5e12ce1ab1f8005d5e5bdec572b",
}


def check_parent_sources(previous, current):
    for name, expected in previous.items():
        if name not in current or (current[name] != expected and ORIGINAL_WORKFLOW.get(name) != expected):
            raise ValueError(f"Parent uses different scientific code: {name}")
    additions = set(current) - set(previous)
    if additions - {"src/joint_semantic_mixed/continuation.py"}:
        raise ValueError(f"Unexpected new source files: {sorted(additions)}")


def check_parent_settings(previous, current):
    # Paths/provenance, the longer budget and its preview schedule intentionally change.
    changed = {"prepared_sha256", "updates", "preview_plan", "continuation"}
    for name in set(previous) | set(current):
        if name not in changed and previous.get(name) != current.get(name):
            raise ValueError(f"Continuation must keep the parent's {name} unchanged.")


def prepare_continuation(args):
    parent = args.continue_from
    if not parent.is_absolute() or re.search(r"(?:seed|_s)0*47(?:\D|$)", str(parent), re.I):
        raise ValueError("Use an absolute parent path outside reserved data.")
    parent = parent.resolve(strict=True)
    if re.search(r"(?:seed|_s)0*47(?:\D|$)", str(parent), re.I):
        raise ValueError("Parent path resolves into reserved data.")
    manifest = read_json(parent / "prepared.json")
    previous = read_json(parent / "training_config.json")
    complete = read_json(parent / "training_complete.json")
    current_sources = source_hashes()
    if manifest["format"] != FORMAT:
        raise ValueError("Expected a prepared joint semantic CNN run.")
    check_parent_sources(manifest["sources"], current_sources)
    if previous["prepared_sha256"] != file_hash(parent / "prepared.json"):
        raise ValueError("Parent preparation changed after training.")
    if file_hash(parent / "semantic_best.pt") != complete["best_weights_sha256"]:
        raise ValueError("Parent best checkpoint changed after training.")
    saved = torch.load(parent / "semantic_latest.pt", map_location="cpu", weights_only=False)
    if saved["config_fingerprint"] != fingerprint(previous) or saved["update"] != complete["updates"]:
        raise ValueError("Parent latest checkpoint is not its completed training state.")
    if saved["history"][-1]["update"] != saved["update"]:
        raise ValueError("Parent checkpoint history is incomplete.")
    best = torch.load(parent / "semantic_best.pt", map_location="cpu", weights_only=True)
    if (best["config_fingerprint"] != fingerprint(previous)
            or best["update"] != complete["best_update"]
            or best["validation_loss"] != saved["best_validation_loss"]):
        raise ValueError("Parent best checkpoint and training state disagree.")
    training_clips = sum(record["split"] == "train" for record in manifest["records"])
    if (training_clips != complete["train_clips"] or args.updates <= saved["update"]
            or saved["update"] % training_clips or args.updates % training_clips
            or saved["update"] % args.preview_every):
        raise ValueError("Extend a completed whole-epoch prompt-check boundary to a larger whole-epoch budget.")
    for name in ("evaluate_every", "preview_every", "learning_rate"):
        if getattr(args, name) != previous[name]:
            raise ValueError(f"Keep the parent's {name} unchanged.")
    if str(args.model_dir.resolve()) != previous["model_dir"] or args.new_prompt != previous["final_new_prompt"]:
        raise ValueError("Keep the parent's Wan path and prompt unchanged.")
    if str(args.codec_run.resolve()) != manifest["codec_run"]:
        raise ValueError("Keep the parent's codec run unchanged.")

    # Reuse latent files in place. Never nest destinations inside source folders.
    source_roots = [parent, Path(manifest["outputs_dir"]).resolve()]
    source_roots += [Path(record["directory"]).resolve().parents[1] for record in manifest["records"]]
    for destination in (args.weights_dir.resolve(), args.outputs_dir.resolve()):
        for source in source_roots:
            if destination == source or source in destination.parents or destination in source.parents:
                raise ValueError("Continuation needs separate folders outside the parent and datasets.")
    for record in manifest["records"]:
        for key in ("semantic_latent", "rgb_latent"):
            if file_hash(Path(record[key])) != record[key + "_sha256"]:
                raise ValueError(f"Parent latent changed: {record['clip_id']}")
    for name, expected in manifest["weight_files"].items():
        if file_hash(parent / name) != expected:
            raise ValueError(f"Parent frozen codec/statistics changed: {name}")

    names = ("prepared.json", "training_config.json", "training_complete.json", "semantic_latest.pt", "semantic_best.pt")
    plan = {
        "parent_weights": str(parent), "parent_outputs": manifest["outputs_dir"],
        "parent_files": {name: file_hash(parent / name) for name in names},
        "parent_training_config": previous, "start_update": saved["update"],
        "total_updates": args.updates, "additional_epochs": (args.updates - saved["update"]) // training_clips,
        "weights_dir": str(args.weights_dir.resolve()), "outputs_dir": str(args.outputs_dir.resolve()),
        "sources": current_sources,
    }
    marker = args.weights_dir / "continuation.json"
    if marker.exists():
        if not args.resume or read_json(marker) != plan:
            raise ValueError("Continuation exists with different settings; resume unchanged.")
        if (args.weights_dir / "prepared.json").exists():
            load_prepared(args)
            return
    else:
        for destination in (args.weights_dir, args.outputs_dir):
            if destination.exists() and any(destination.iterdir()):
                raise ValueError(f"Expected a new empty continuation folder: {destination}")
        args.weights_dir.mkdir(parents=True, exist_ok=True)
        args.outputs_dir.mkdir(parents=True, exist_ok=True)
        write_json(marker, plan)
    for name in manifest["weight_files"]:
        shutil.copyfile(parent / name, args.weights_dir / name)
    new_manifest = dict(manifest, weights_dir=plan["weights_dir"], outputs_dir=plan["outputs_dir"],
                        sources=current_sources, continuation_parent=str(parent))
    write_json(args.weights_dir / "prepared.json", new_manifest)
    print(f"Continuing update {saved['update']} to {args.updates}; existing latent cache reused.", flush=True)


def initial_checkpoint(plan, config):
    parent = Path(plan["parent_weights"])
    for name in ("semantic_latest.pt", "semantic_best.pt"):
        if file_hash(parent / name) != plan["parent_files"][name]:
            raise ValueError(f"Parent checkpoint changed: {name}")
    saved = torch.load(parent / "semantic_latest.pt", map_location="cpu", weights_only=False)
    best = torch.load(parent / "semantic_best.pt", map_location="cpu", weights_only=True)
    # Only the new configuration identity changes; weights/optimizer/RNG stay intact.
    saved["config_fingerprint"] = fingerprint(config)
    best["config_fingerprint"] = fingerprint(config)
    saved["best_payload"] = best
    return saved
