"""Read-only input checks and small JSON/CSV records (no ML imports)."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

from rgb_geometry_evaluation import protocol as evaluation
from rgb_geometry_rl_v2 import protocol as previous

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/rgb_geometry_campaign_v1.json"
SEMANTIC = ("m1", "m2", "m3", "m4", "m5_current", "m5_history", "m6_control", "m6_depth")
RL = tuple("rl" + str(number) for number in range(1, 11)) + ("rl3_repeat", "rl4_repeat")
EXPERIMENTS = SEMANTIC + RL + ("baseline", "choose_fusion", "choose_observer", "report")
CLASS_NAMES = evaluation.CLASS_NAMES
read_json = evaluation.read_json
write_json = evaluation.write_json
file_hash = evaluation.file_hash
fingerprint = evaluation.fingerprint


def append_row(path, row):
    """Append one complete row; nested measurements use JSON inside CSV cells."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as handle:
            fields = next(csv.reader(handle))
        if set(fields) != set(row):
            raise ValueError("CSV columns changed: " + str(path))
    else:
        fields = list(row)
    serialized = {}
    for key, value in row.items():
        serialized[key] = json.dumps(value, allow_nan=False) if isinstance(value, (dict, list, tuple)) else value
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(serialized)
        handle.flush()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    result.add_argument("--config", type=Path, default=CONFIG)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--plan", action="store_true")
    result.add_argument("--check-inputs", action="store_true")
    result.add_argument("--check-output", action="store_true", help=argparse.SUPPRESS)
    return result


def load_config(path):
    config = read_json(path)
    fixed = {"rounds": 10, "siblings": 4, "semantic_updates": 400,
             "semantic_learning_rate": 2e-5, "head_learning_rate": 1e-6,
             "lora_learning_rate": 1e-8, "seed": 42,
             "action_indices": [10, 20, 30, 40], "inference_steps": 50,
             "alpha": .25, "noise_scale": .03, "maximum_concurrent_gpus": 4,
             "retention": "json_csv_only"}
    if config.get("format") != "rgb_geometry_campaign_v1":
        raise ValueError("Unexpected campaign format.")
    if any(config["settings"].get(key) != value for key, value in fixed.items()):
        raise ValueError("The declared campaign recipe changed.")
    counts = {"rl_train": 4, "semantic_train": 8, "semantic_development": 4,
              "semantic_confirmation": 10, "evaluation": 10, "sentinel": 2, "endpoint": 4}
    prompts, scenes = [], []
    for panel, count in counts.items():
        rows = config["panels"][panel]
        if len(rows) != count:
            raise ValueError("Wrong panel size: " + panel)
        for row in rows:
            if not set(row["expected_classes"]) <= set(CLASS_NAMES[1:]):
                raise ValueError("Unknown expected RGB class.")
            prompts.append(row["prompt"])
            scenes.append(row["scene"])
    if len(set(prompts)) != len(prompts) or len(set(scenes)) != len(scenes):
        raise ValueError("Training, development and confirmation panels overlap.")
    if any("car" not in row["expected_classes"] for row in config["panels"]["rl_train"]):
        raise ValueError("Both reward controls require cars in every training prompt.")
    return config


def cases(config, panel):
    offsets = {"rl_train": 310001, "semantic_train": 320001,
               "semantic_development": 330001, "semantic_confirmation": 340001,
               "evaluation": 350001, "sentinel": 360001, "endpoint": 370001}
    repeats = 1 if panel in ("rl_train", "sentinel") else 2
    rows = []
    for scene_index, scene in enumerate(config["panels"][panel]):
        for repeat in range(repeats):
            seed = offsets[panel] + scene_index * 2 + repeat
            rows.append(dict(scene, case_id=f"{panel}_{scene_index:02d}_{repeat}",
                             role=panel, seed=seed, rgb_seed=seed,
                             semantic_seed=seed + 100000, action_seed=seed + 200000))
    return rows


def runtime_args(args, config):
    """Use existing frozen local models; preserve the venv executable symlink."""
    defaults = {"semantic_run": previous.DEFAULT_SEMANTIC, "model_dir": previous.DEFAULT_MODEL,
                "yolo_model": previous.DEFAULT_YOLO, "track_python": previous.DEFAULT_TRACK_PYTHON,
                "cotracker_repo": previous.DEFAULT_TRACK_REPO,
                "cotracker_weights": previous.DEFAULT_TRACK_WEIGHTS}
    for name, value in defaults.items():
        setattr(args, name, Path(value))
    args.e4_checkpoint = Path(config["settings"]["e4_checkpoint"])
    return args


def validate_output(args):
    """Reject protected inputs before creating even a log or failure receipt."""
    root = previous.safe_path(args.output_root)
    protected = [ROOT, args.semantic_run, args.model_dir, args.yolo_model.parent,
                 args.cotracker_repo, args.cotracker_weights.parent,
                 args.track_python.parent.parent, args.e4_checkpoint.parent]
    for path in protected:
        path = previous.safe_path(path)
        if root == path or root in path.parents or path in root.parents:
            raise ValueError('Campaign output overlaps a protected input: ' + str(path))
    return root


def source_hashes():
    hashes = evaluation.source_hashes()
    for folder in ("src/rgb_geometry_campaign", "src/rgb_geometry_four_step",
                   "src/rgb_geometry_lora_memory", "src/rgb_geometry_lora_exploratory",
                   "src/semantic_denoiser_experiments", "jobs/rgb_geometry_campaign"):
        for path in sorted((ROOT / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                hashes[str(path.relative_to(ROOT))] = file_hash(path)
    return hashes


def validate_inputs(args, config):
    # Reuse the exact frozen observer/codec/teacher provenance checks, without
    # reading supplied RGB/semantic training latent files.
    validation = {"provenance": config["provenance"], "methods": [],
                  "prompts": [row for rows in config["panels"].values() for row in rows]}
    info = evaluation.validate_inputs(args, validation)
    e4_path = previous.safe_path(args.e4_checkpoint, existing=True)
    output = previous.safe_path(args.output_root)
    if output == e4_path.parent or e4_path.parent in output.parents or output in e4_path.parents:
        raise ValueError("Campaign output overlaps existing E4 weights.")
    e4_hash = file_hash(args.e4_checkpoint)
    e4_config = read_json(args.e4_checkpoint.parent / "training_config.json")
    e4_complete = read_json(args.e4_checkpoint.parent / "training_complete.json")
    if (e4_config.get("experiment") != "E4" or e4_complete.get("experiment") != "E4"
            or e4_complete.get("updates") != 6400
            or e4_complete.get("endpoint_sha256") != e4_hash
            or e4_complete.get("config_fingerprint") != fingerprint(e4_config)):
        raise ValueError("E4 must be the verified completed 6400-update endpoint.")
    for name in ("frozen_codec.pt", "normalization_stats.json"):
        if e4_config["weight_files"].get(name) != info["identity"]["semantic_files"][name]:
            raise ValueError("E4 and the incumbent observer use different codec coordinates.")
    info["identity"]["e4_sha256"] = e4_hash
    info["identity"]["e4_config_fingerprint"] = fingerprint(e4_config)
    info["identity"]["sources"] = source_hashes()
    info["identity"]["config_sha256"] = file_hash(args.config)
    return info


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_submitted_identity(args, info):
    path = args.output_root / "input_check.json"
    if path.exists():
        recorded = read_json(path)
        if recorded.get("identity") != info["identity"]:
            raise ValueError("Source/model/config changed since submission; preserve the failed run.")


def check_cpu_identity(args):
    """Selectors also use the submitted code/config; they need no model loading."""
    path = args.output_root / "input_check.json"
    if path.exists():
        identity = read_json(path)["identity"]
        if (identity["sources"] != source_hashes()
                or identity["config_sha256"] != file_hash(args.config)):
            raise ValueError("Source or configuration changed after campaign submission.")


def tracker_probe(args):
    environment = dict(os.environ)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [str(args.track_python), "-B", "-c",
               "import sys; sys.path.insert(0,sys.argv[1]); import numpy,torch,cotracker.predictor; "
               "torch.from_numpy(numpy.zeros(1,dtype=numpy.float32)).numpy(); "
               "print('CoTracker imports verified')", str(args.cotracker_repo)]
    subprocess.run(command, check=True, env=environment, timeout=120)


def check_retention(directory):
    unexpected = [str(path) for path in Path(directory).rglob("*")
                  if path.is_file() and path.suffix not in (".json", ".csv")]
    if unexpected:
        raise RuntimeError("Unexpected persistent artifacts: " + repr(unexpected))
