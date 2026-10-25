"""Fixed three-LoRA extension; all planning and input checks use only the CPU."""

import copy
from pathlib import Path

from rgb_geometry_qualitative import protocol as shared

ROOT = shared.ROOT
CONFIG = ROOT / "configs/rgb_geometry_qualitative_lora_v1.json"
TRAIN_EXPERIMENTS = ("flash_grpo_L", "rloo_L", "flow_grpo_L")
EXPERIMENTS = (*TRAIN_EXPERIMENTS, "report")
REPORT_VIDEOS = 200
read_json = shared.read_json
write_json = shared.write_json
file_hash = shared.file_hash
fingerprint = shared.fingerprint
runtime_args = shared.runtime_args
cases = shared.cases
check_free_space = shared.check_free_space
tracker_probe = shared.tracker_probe
check_submitted_identity = shared.check_submitted_identity
safe_path = shared.comparison.campaign.previous.safe_path


def expected_config():
    config = copy.deepcopy(shared.expected_config())
    config["format"] = "rgb_geometry_qualitative_lora_v1"
    config["experiments"] = list(TRAIN_EXPERIMENTS)
    config["method_settings"]["rloo"]["lora_learning_rate"] = 1e-8
    config["reference_protocol"] = {"format": "rgb_geometry_qualitative_v1",
                                    "config_sha256": file_hash(shared.CONFIG)}
    config["retention"]["trainable_weights"] = "lora_adapters_only"
    return config


def reference_config(config):
    return shared.expected_config()


def load_config(path):
    path = safe_path(path, existing=True)
    config = read_json(path)
    if config != expected_config():
        raise ValueError("Use the fixed three-LoRA extension configuration.")
    return config


def parser():
    import argparse
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    result.add_argument("--reference-root", type=Path, required=True,
                        help="Completed baseline/RGB/Semantic study; read only.")
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--plan", action="store_true")
    result.add_argument("--check-inputs", action="store_true")
    result.add_argument("--check-output", action="store_true", help=argparse.SUPPRESS)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--resume-from", type=Path)
    mode.add_argument("--checkpoint", type=Path)
    return result


def experiment_spec(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown LoRA experiment: " + str(experiment))
    if experiment == "rloo_L":
        return {"experiment": experiment, "method": "rloo", "arm": "L",
                "policy": "lora", "condition": "none", "trainable": True, "gpu": True}
    return shared.comparison.experiment_spec(experiment)


def validate_output(args):
    root = shared.validate_output(args)
    reference = safe_path(args.reference_root)
    if root == reference or root in reference.parents or reference in root.parents:
        raise ValueError("LoRA output must be outside the first study's results directory.")
    args.reference_root = reference
    return root


def validate_reference(root):
    """Require completed phase-one receipts without changing that study."""
    root = safe_path(root, existing=True)
    config = shared.load_config(root / "config.json")
    identity = read_json(root / "input_check.json")["identity"]
    sources = shared.source_hashes()
    if identity["sources"] != sources or identity["config_sha256"] != file_hash(root / "config.json"):
        raise ValueError("First-stage code/configuration changed; preserve the original checkout.")
    common_fingerprint = fingerprint(sources)
    report_status = read_json(root / "report/status.json")
    summary = read_json(root / "report/summary.json")
    for record in (report_status, summary):
        if (record.get("status") != "complete" or record.get("verified_videos") != 140
                or not record.get("matched_source_identity")):
            raise ValueError("Finish the complete seven-variant video study before submitting LoRA.")
    if set(summary["methods"]) != set(shared.EXPERIMENTS[:-1]):
        raise ValueError("The first-stage report must contain exactly the seven planned variants.")
    planned = {(method, case["case_id"]) for method in shared.EXPERIMENTS[:-1]
               for case in cases(config, "evaluation")}
    video_rows = summary.get("videos", [])
    if len(video_rows) != 140 or {(row["method"], row["case_id"]) for row in video_rows} != planned:
        raise ValueError("The first-stage report must index all 140 planned videos.")
    for row in video_rows:
        relative = f"{row['method']}/videos/{row['case_id']}.mp4"
        if row.get("status") != "complete" or row.get("video_path") != relative:
            raise ValueError("First-stage video index is incomplete or mismatched.")
        if file_hash(root / relative) != row.get("video_sha256"):
            raise ValueError("A first-stage video changed after its report was written.")
    paths = ["config.json", "input_check.json", "report/status.json", "report/summary.json"]
    for experiment in shared.EXPERIMENTS[:-1]:
        status_path = f"{experiment}/status.json"
        manifest_path = f"{experiment}/manifest.json"
        status = read_json(root / status_path)
        manifest = read_json(root / manifest_path)
        spec = shared.experiment_spec(experiment)
        if manifest["identity"] != dict(identity, experiment=experiment, training_spec=spec):
            raise ValueError("First-stage arm has inconsistent model/source identity: " + experiment)
        expected_rounds = 0 if experiment == "baseline" else 10
        if (status.get("status") != "complete" or status.get("rounds") != expected_rounds
                or status.get("retained_videos") != 20
                or status.get("source_fingerprint") != common_fingerprint
                or status.get("comparison_fingerprint") != fingerprint(config)):
            raise ValueError("First-stage arm is incomplete or mismatched: " + experiment)
        for field in ("experiment", "method", "arm", "condition"):
            if status.get(field) != spec[field]:
                raise ValueError("First-stage arm label differs from its recipe.")
        if summary["methods"][experiment].get("status") != "complete":
            raise ValueError("First-stage report contains an incomplete arm.")
        paths.extend([status_path, manifest_path])
    return {"root": str(root), "config_sha256": identity["config_sha256"],
            "input_identity": identity, "common_source_fingerprint": common_fingerprint,
            "artifacts": {name: file_hash(root / name) for name in paths}, "phase1_videos": 140}


def source_hashes():
    hashes = shared.source_hashes()
    for folder in ("src/rgb_geometry_qualitative_lora", "jobs/rgb_geometry_qualitative_lora"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                hashes[str(path.relative_to(ROOT))] = file_hash(path)
    hashes[str(CONFIG.relative_to(ROOT))] = file_hash(CONFIG)
    return hashes


def validate_inputs(args, config):
    reference = validate_reference(args.reference_root)
    info = shared.validate_inputs(args, config)
    # The second stage changes only its extension code and protocol file.
    current_inputs = {key: value for key, value in info["identity"].items()
                      if key not in ("sources", "config_sha256")}
    prior_inputs = {key: value for key, value in reference["input_identity"].items()
                    if key not in ("sources", "config_sha256")}
    if current_inputs != prior_inputs:
        raise ValueError("LoRA must use the first study's exact frozen models and inputs.")
    info["identity"]["sources"] = source_hashes()
    info["identity"]["reference"] = reference
    return info


def check_cpu_identity(args):
    reference = validate_reference(args.reference_root)
    path = args.output_root / "input_check.json"
    if path.exists():
        identity = read_json(path)["identity"]
        if (identity["sources"] != source_hashes()
                or identity["config_sha256"] != file_hash(args.config)
                or identity.get("reference") != reference):
            raise ValueError("LoRA source, configuration or reference changed since submission.")
