"""Historical R/S/LoRA comparison plan and read-only input checks."""

import argparse
from pathlib import Path

from rgb_geometry_campaign import protocol as campaign
from rgb_geometry_evaluation import protocol as evaluation


ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = ROOT
CONFIG = ROOT / "configs/rgb_geometry_rl_comparison_v1.json"
HISTORICAL_CONFIG = ROOT / "configs/rgb_geometry_campaign_v1.json"
METHODS = ("flash_grpo", "awm", "diffusion_nft", "flow_grpo")
ARMS = ("R", "S", "L")
TRAIN_EXPERIMENTS = tuple(method + "_" + arm for method in METHODS for arm in ARMS)
EXPERIMENTS = ("baseline", *TRAIN_EXPERIMENTS, "report")
CLASS_NAMES = campaign.CLASS_NAMES
read_json = campaign.read_json
write_json = campaign.write_json
file_hash = campaign.file_hash
fingerprint = campaign.fingerprint
append_row = campaign.append_row
tensor_digest = campaign.tensor_digest
tracker_probe = campaign.tracker_probe
check_retention = campaign.check_retention

FIXED_SETTINGS = {
    "rounds": 10, "siblings": 4, "seed": 42,
    "train_steps": 50, "eval_steps": 50, "inference_steps": 50,
    "action_indices": [10, 20, 30, 40], "noise_scale": 0.03,
    "height": 480, "width": 832, "frames": 81, "fps": 16,
    "guidance_scale": 5.0, "flow_shift": 3.0,
    "alpha": 0.25, "correction_rms_cap": 0.25,
    "head_learning_rate": 1e-4, "lora_learning_rate": 1e-5,
    "gradient_clip": 1.0, "advantage_epsilon": 1e-4,
    "advantage_standard_deviation": "population",
    "maximum_concurrent_gpus": 4, "retention": "json_csv_only",
    "reward": "expected_class_lifetime",
    "observer": "historical_frozen_semantic_flow_update4600",
    "sampler": "euler_four_gaussian_actions",
    "sibling_initial_noise": "shared_within_prompt",
    "prompt_initial_noise": "fixed_across_rounds",
}
METHOD_SETTINGS = {
    "flow_grpo": {"replay_passes": 2, "replay_steps": 4,
                  "clip_range": 0.001, "reference_coefficient": 0.004},
    "flash_grpo": {"replay_passes": 2, "replay_steps": 1,
                   "replay_choice": "uniform_over_four_actions",
                   "temporal_weight": "inverse_euler_mean_sensitivity_normalized_over_four_actions",
                   "clip_range": 0.001, "reference_coefficient": 0.0},
    "awm": {"training_times": 4, "reference_coefficient": 0.4},
    "diffusion_nft": {"training_times": 4, "reference_coefficient": 0.0001,
                      "mixture_beta": 1.0, "advantage_bound": 5.0,
                      "denominator_floor": 1e-5, "ema_rate": 0.001, "ema_max": 0.5},
}


def experiment_spec(experiment):
    """R and S train equal heads; L trains only the historical Wan adapters."""
    if experiment in ("baseline", "report"):
        return {"experiment": experiment, "method": None, "arm": "B",
                "policy": None, "condition": None, "parameterization": None,
                "conditioning": None, "trainable": False, "gpu": experiment == "baseline"}
    for method in METHODS:
        for arm in ARMS:
            if experiment == method + "_" + arm:
                policy = "lora" if arm == "L" else "head"
                condition = {"R": "rgb", "S": "semantic", "L": "none"}[arm]
                return {"experiment": experiment, "method": method, "arm": arm,
                        "policy": policy, "condition": condition,
                        "parameterization": policy, "conditioning": condition,
                        "trainable": True, "gpu": True}
    raise ValueError("Unknown matched RL experiment: " + str(experiment))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--plan", action="store_true")
    result.add_argument("--check-inputs", action="store_true")
    result.add_argument("--check-output", action="store_true", help=argparse.SUPPRESS)
    return result


argument_parser = parser


def load_config(path):
    """Reject accidental changes to the historical comparison or new objectives."""
    config = read_json(path)
    if config.get("format") != "rgb_geometry_rl_comparison_v1":
        raise ValueError("Unexpected matched RL configuration format.")
    if config.get("settings") != FIXED_SETTINGS:
        raise ValueError("The declared matched generation/training recipe changed.")
    if (config.get("method_settings") != METHOD_SETTINGS
            or config.get("method_order") != list(METHODS)
            or config.get("arm_order") != list(ARMS)):
        raise ValueError("The declared method settings or order changed.")
    historical = campaign.load_config(HISTORICAL_CONFIG)
    reference = config["historical_reference"]
    if (reference.get("config") != str(HISTORICAL_CONFIG.relative_to(ROOT))
            or reference.get("config_sha256") != file_hash(HISTORICAL_CONFIG)
            or reference.get("arms") != {"R": "rl3", "S": "rl4", "L": "rl8"}
            or reference.get("training_seed") != 42 or reference.get("rerun_rloo") is not False):
        raise ValueError("The reference must remain the unchanged seed-42 campaign R/S/LoRA arms.")
    panels = {name: historical["panels"][name] for name in ("rl_train", "sentinel", "evaluation")}
    if config.get("panels") != panels:
        raise ValueError("Prompts, scenes and expected classes must exactly match the historical campaign.")
    provenance = config["provenance"]
    if provenance.get("expected_semantic_files") != historical["provenance"]["expected_semantic_files"]:
        raise ValueError("Use the same frozen semantic-flow observer and codec.")
    if (provenance.get("evaluation_panel_previously_inspected") is not True
            or provenance.get("scene_transfer_verified") is not False):
        raise ValueError("This reused historical panel is not an unseen confirmation test.")
    return config


def cases(config, panel):
    """Reuse exact historical case IDs and all three seed streams."""
    if panel not in ("rl_train", "sentinel", "evaluation"):
        raise ValueError("Unknown comparison panel: " + str(panel))
    return campaign.cases(config, panel)


def runtime_args(args, config):
    """Use the historical observer, with no E4 input or checkpoint requirement."""
    previous = campaign.previous
    defaults = {"semantic_run": previous.DEFAULT_SEMANTIC, "model_dir": previous.DEFAULT_MODEL,
                "yolo_model": previous.DEFAULT_YOLO, "track_python": previous.DEFAULT_TRACK_PYTHON,
                "cotracker_repo": previous.DEFAULT_TRACK_REPO,
                "cotracker_weights": previous.DEFAULT_TRACK_WEIGHTS}
    for name, value in defaults.items():
        setattr(args, name, Path(value))
    return args


def validate_output(args):
    root = campaign.previous.safe_path(args.output_root)
    protected = [ROOT, args.semantic_run, args.model_dir, args.yolo_model.parent,
                 args.cotracker_repo, args.cotracker_weights.parent, args.track_python.parent.parent]
    for path in protected:
        path = campaign.previous.safe_path(path)
        if root == path or root in path.parents or path in root.parents:
            raise ValueError("Comparison output overlaps a protected input: " + str(path))
    return root


def source_hashes():
    identities = campaign.source_hashes()
    for folder in ("src/rgb_geometry_rl_comparison", "jobs/rgb_geometry_rl_comparison"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                identities[str(path.relative_to(ROOT))] = file_hash(path)
    # Reused objective kernels are part of this run's identity too.
    for path in (CONFIG, HISTORICAL_CONFIG, ROOT / "src/rgb_geometry_rl_methods/objectives.py"):
        identities[str(path.relative_to(ROOT))] = file_hash(path)
    return identities


def validate_inputs(args, config):
    validation = {"provenance": config["provenance"], "methods": [],
                  "prompts": [row for rows in config["panels"].values() for row in rows]}
    info = evaluation.validate_inputs(args, validation)
    info["identity"]["sources"] = source_hashes()
    info["identity"]["config_sha256"] = file_hash(args.config)
    info["identity"]["historical_reference"] = config["historical_reference"]
    return info


def check_submitted_identity(args, info):
    path = args.output_root / "input_check.json"
    if path.exists() and read_json(path).get("identity") != info["identity"]:
        raise ValueError("Source/model/config changed since submission; preserve the failed run.")


def check_cpu_identity(args):
    path = args.output_root / "input_check.json"
    if path.exists():
        identity = read_json(path)["identity"]
        if (identity["sources"] != source_hashes()
                or identity["config_sha256"] != file_hash(args.config)):
            raise ValueError("Source or configuration changed after submission.")
