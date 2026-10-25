"""Read-only checkpoint provenance checks, separate from the new prompt panel."""

import shutil
from pathlib import Path

from rgb_geometry_qualitative import artifacts
from rgb_geometry_qualitative import protocol as original
from rgb_geometry_qualitative_lora import protocol as lora
from . import protocol


safe_path = original.comparison.campaign.previous.safe_path
MINIMUM_FREE_BYTES = 10_000_000_000


def source_hashes():
    hashes = lora.source_hashes()
    for folder in ("src/rgb_geometry_challenge", "jobs/rgb_geometry_challenge"):
        for path in sorted((protocol.ROOT / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                hashes[str(path.relative_to(protocol.ROOT))] = protocol.file_hash(path)
    hashes[str(protocol.CONFIG.relative_to(protocol.ROOT))] = protocol.file_hash(protocol.CONFIG)
    return hashes


def validate_output(args):
    root = original.comparison.validate_output(args)
    args.reference_root = safe_path(args.reference_root)
    args.lora_root = safe_path(args.lora_root)
    for source in (args.reference_root, args.lora_root):
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Use a new output folder outside both saved studies.")
    if args.reference_root == args.lora_root:
        raise ValueError("The first study and LoRA study must be distinct folders.")
    return root


def check_free_space(root, minimum_bytes=MINIMUM_FREE_BYTES):
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    available = shutil.disk_usage(ancestor).free
    if available < minimum_bytes:
        raise RuntimeError(f"At least {minimum_bytes / 1e9:g} GB of filesystem free space is required; check user quota separately.")
    return available


def inspect_origins(reference_root, lora_root):
    """Verify saved receipts and all nine final checkpoint hashes, without torch.load.

    Old videos are not inference inputs. Their completed reports are checked,
    but their MP4 files are not read again. Original manifests are never edited.
    """
    roots = {"heads": safe_path(reference_root, existing=True),
             "lora": safe_path(lora_root, existing=True)}
    studies = {}
    for name, study in (("heads", original), ("lora", lora)):
        root = roots[name]
        config = study.load_config(root / "config.json")
        identity = protocol.read_json(root / "input_check.json")["identity"]
        if (identity["sources"] != study.source_hashes()
                or identity["config_sha256"] != protocol.file_hash(root / "config.json")):
            raise ValueError("Historical source/config identity changed: " + name)
        report = protocol.read_json(root / "report/status.json")
        expected_count = 140 if name == "heads" else 200
        if report.get("status") != "complete" or report.get("verified_videos") != expected_count:
            raise ValueError("The original study report is not complete: " + name)
        studies[name] = {"root": str(root), "config": config, "identity": identity,
                         "metadata_sha256": {filename: protocol.file_hash(root / filename)
                            for filename in ("config.json", "input_check.json", "report/status.json")}}

    head_identity = studies["heads"]["identity"]
    lora_identity = studies["lora"]["identity"]
    reference = lora_identity["reference"]
    if reference["input_identity"] != head_identity:
        raise ValueError("LoRA was trained against a different reference study.")
    for filename, digest in reference["artifacts"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid historical reference artifact path.")
        if protocol.file_hash(safe_path(roots["heads"] / relative, existing=True)) != digest:
            raise ValueError("Historical reference receipt changed: " + filename)
    excluded = {"sources", "config_sha256", "reference"}
    head_common = {key: value for key, value in head_identity.items() if key not in excluded}
    lora_common = {key: value for key, value in lora_identity.items() if key not in excluded}
    if head_common != lora_common:
        raise ValueError("The two studies do not share the same frozen inputs.")

    origins = {}
    for variant in protocol.VARIANTS:
        study_name = "lora" if variant.endswith("_L") else "heads"
        study = studies[study_name]
        root = roots[study_name]
        specification = protocol.experiment_spec(variant)
        identity = dict(study["identity"], experiment=variant, training_spec=specification)
        manifest = protocol.read_json(root / variant / "manifest.json")
        status = protocol.read_json(root / variant / "status.json")
        expected_rounds = 0 if variant == "baseline" else 10
        if manifest["identity"] != identity:
            raise ValueError("Original arm identity mismatch: " + variant)
        if (status.get("status") != "complete" or status.get("rounds") != expected_rounds
                or status.get("retained_videos") != 20
                or status.get("comparison_fingerprint") != protocol.fingerprint(study["config"])
                or status.get("source_fingerprint") != protocol.fingerprint(study["identity"]["sources"])):
            raise ValueError("Original arm completion receipt mismatch: " + variant)
        for field in ("experiment", "method", "arm", "condition"):
            if status.get(field) != specification[field]:
                raise ValueError("Original arm label mismatch: " + variant)
        origin = {"original_run": str(root), "identity": identity,
                  "policy_digest": status["final_policy_digest"],
                  "manifest_sha256": protocol.file_hash(root / variant / "manifest.json"),
                  "status_sha256": protocol.file_hash(root / variant / "status.json"),
                  "path": None, "sha256": None, "completed_round": expected_rounds}
        if variant != "baseline":
            checkpoint = artifacts._checked_path(root / variant / "checkpoints/round_10.pt", existing=True)
            receipt_path = artifacts._checked_path(checkpoint.with_suffix(".json"), existing=True)
            receipt = protocol.read_json(receipt_path)
            digest = protocol.file_hash(checkpoint)
            if (receipt.get("format") != artifacts.CHECKPOINT_FORMAT
                    or receipt.get("completed_round") != 10 or receipt.get("identity") != identity
                    or receipt.get("sha256") != digest):
                raise ValueError("Final checkpoint receipt/hash mismatch: " + variant)
            origin.update(path=str(checkpoint), sha256=digest,
                          receipt_sha256=protocol.file_hash(receipt_path),
                          original_m3_path=receipt["path"], parameter_version=receipt["parameter_version"])
        origins[variant] = origin
    return {"studies": studies, "origins": origins}


def validate_inputs(args, config):
    saved = inspect_origins(args.reference_root, args.lora_root)
    validation = {"provenance": config["provenance"], "methods": [],
                  "prompts": config["panels"]["evaluation"]}
    info = original.comparison.evaluation.validate_inputs(args, validation)
    old_identity = saved["studies"]["heads"]["identity"]
    for key in ("semantic_files", "model_inventory", "yolo_sha256", "cotracker", "input_paths"):
        if info["identity"][key] != old_identity[key]:
            raise ValueError("Current frozen input differs from the trained checkpoints: " + key)
    info["identity"].update(sources=source_hashes(), config_sha256=protocol.file_hash(args.config),
                            checkpoint_origins=saved["origins"],
                            original_studies=saved["studies"], inference_only=True)
    return info


def check_submitted_identity(args, info):
    path = args.output_root / "input_check.json"
    if not path.is_file() or protocol.read_json(path)["identity"] != info["identity"]:
        raise ValueError("Inputs changed after submission, or input_check.json is missing.")


def check_cpu_identity(args):
    identity = protocol.read_json(args.output_root / "input_check.json")["identity"]
    if (identity["sources"] != source_hashes()
            or identity["config_sha256"] != protocol.file_hash(args.config)):
        raise ValueError("Challenge source/config changed after submission.")
    saved = inspect_origins(args.reference_root, args.lora_root)
    if identity["checkpoint_origins"] != saved["origins"] or identity["original_studies"] != saved["studies"]:
        raise ValueError("Saved checkpoints or provenance changed after submission.")
    return identity
