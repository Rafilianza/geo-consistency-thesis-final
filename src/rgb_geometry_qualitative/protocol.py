"""Fixed seven-variant study; planning and input checks need no ML imports."""

import argparse
import copy
import shutil
from pathlib import Path

from rgb_geometry_rl_comparison import protocol as comparison

ROOT = comparison.ROOT
CONFIG = ROOT / "configs/rgb_geometry_qualitative_v1.json"
TRAIN_EXPERIMENTS = ("flash_grpo_R", "flash_grpo_S", "rloo_R", "rloo_S",
                     "flow_grpo_R", "flow_grpo_S")
EXPERIMENTS = ("baseline", *TRAIN_EXPERIMENTS, "report")
CLASS_NAMES = comparison.CLASS_NAMES
MINIMUM_FREE_BYTES = 5_000_000_000
read_json = comparison.read_json
write_json = comparison.write_json
file_hash = comparison.file_hash
fingerprint = comparison.fingerprint
append_row = comparison.append_row
tensor_digest = comparison.tensor_digest
runtime_args = comparison.runtime_args
tracker_probe = comparison.tracker_probe


def expected_config():
    original = comparison.load_config(comparison.CONFIG)
    historical = comparison.campaign.load_config(comparison.HISTORICAL_CONFIG)
    settings = copy.deepcopy(original["settings"])
    settings["retention"] = "round_checkpoints_and_final_rgb_mp4"
    methods = {name: copy.deepcopy(original["method_settings"][name])
               for name in ("flash_grpo", "flow_grpo")}
    methods["rloo"] = {"head_learning_rate": 1e-6, "updates_per_round": 1,
                       "advantage": "leave_one_out_unstandardized",
                       "log_probability": "sum_coordinates", "gradient_clip": None}
    panels = copy.deepcopy(original["panels"])
    panels["endpoint"] = copy.deepcopy(historical["panels"]["endpoint"])
    provenance = copy.deepcopy(original["provenance"])
    provenance["note"] = ("New qualitative reproduction runs, including RLOO. "
                          "The historical evaluation panel was previously inspected. "
                          "Visible failures and tracking scores do not certify 3D geometry.")
    return {
        "format": "rgb_geometry_qualitative_v1", "settings": settings,
        "experiments": list(EXPERIMENTS[:-1]), "method_settings": methods,
        "source_recipes": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (comparison.CONFIG, comparison.HISTORICAL_CONFIG)},
        "provenance": provenance, "panels": panels,
        "retention": {"checkpoint_every_round": True, "save_initial_checkpoint": True,
                      "save_optimizer_and_rng": True, "video_panel": "evaluation",
                      "video_format": "RGB_H264_MP4", "videos_per_variant": 20,
                      "save_training_videos": False, "copy_frozen_models": False},
    }


def load_config(path):
    path = comparison.campaign.previous.safe_path(path, existing=True)
    config = read_json(path)
    if config != expected_config():
        raise ValueError("Study settings changed: use the fixed seven-variant configuration.")
    return config


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--plan", action="store_true")
    result.add_argument("--check-inputs", action="store_true")
    result.add_argument("--check-output", action="store_true", help=argparse.SUPPRESS)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--resume-from", type=Path,
                      help="Continue after a saved round into a fresh output root.")
    mode.add_argument("--checkpoint", type=Path,
                      help="Generate and score the 20 final cases without training.")
    return result


def experiment_spec(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown qualitative experiment: " + str(experiment))
    if experiment.startswith("rloo_"):
        arm = experiment[-1]
        return {"experiment": experiment, "method": "rloo", "arm": arm,
                "policy": "head", "condition": "rgb" if arm == "R" else "semantic",
                "trainable": True, "gpu": True}
    return comparison.experiment_spec(experiment)


def cases(config, panel):
    if panel not in config["panels"]:
        raise ValueError("Unknown study panel.")
    return comparison.campaign.cases(config, panel)


def validate_output(args):
    root = comparison.validate_output(args)
    checkpoint = getattr(args, "resume_from", None) or getattr(args, "checkpoint", None)
    if checkpoint:
        checkpoint = comparison.campaign.previous.safe_path(checkpoint, existing=True)
        original_run = checkpoint.parent
        if checkpoint.parent.name == "checkpoints":
            original_run = checkpoint.parent.parent.parent
        if root == original_run or root in original_run.parents or original_run in root.parents:
            raise ValueError("Use a fresh output root outside the checkpoint's original run.")
    return root


def check_free_space(root):
    ancestor = Path(root)
    while not ancestor.exists():
        ancestor = ancestor.parent
    free_bytes = shutil.disk_usage(ancestor).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError("At least 5 GB of filesystem free space is required; check your quota too.")
    return free_bytes


def source_hashes():
    hashes = comparison.source_hashes()
    for folder in ("src/rgb_geometry_qualitative", "jobs/rgb_geometry_qualitative"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                hashes[str(path.relative_to(ROOT))] = file_hash(path)
    hashes[str(CONFIG.relative_to(ROOT))] = file_hash(CONFIG)
    return hashes


def validate_inputs(args, config):
    validation = {"provenance": config["provenance"], "methods": [],
                  "prompts": [row for panel in config["panels"].values() for row in panel]}
    info = comparison.evaluation.validate_inputs(args, validation)
    info["identity"]["sources"] = source_hashes()
    info["identity"]["config_sha256"] = file_hash(args.config)
    info["identity"]["source_recipes"] = config["source_recipes"]
    return info


def check_submitted_identity(args, info):
    path = args.output_root / "input_check.json"
    if path.exists() and read_json(path).get("identity") != info["identity"]:
        raise ValueError("Source/model/config changed since submission.")


def check_cpu_identity(args):
    path = args.output_root / "input_check.json"
    if path.exists():
        identity = read_json(path)["identity"]
        if (identity["sources"] != source_hashes()
                or identity["config_sha256"] != file_hash(args.config)):
            raise ValueError("Source or configuration changed after submission.")
