"""Small, standard-library-only plan and input checks for shared evaluation."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

from rgb_geometry_rl_v2 import protocol as previous

REPOSITORY = Path(__file__).resolve().parents[2]
CONFIG = REPOSITORY / "configs/rgb_geometry_evaluation_20_v1.json"
FORMAT = "rgb_geometry_evaluation_20_v1"
CLASS_NAMES = ("background", "person", "car", "bus", "bicycle", "truck",
               "road", "sidewalk", "building", "vegetation", "sky")
METHOD_IDS = ("flow_baseline", "rl_baseline", "flow_rgb_0p1", "flow_sem_0p1",
              "flow_rgb_0p25", "flow_sem_0p25", "flow_rgb_1", "flow_sem_1",
              "rl_r", "rl_s", "rl_lora")
read_json = previous.read_json
file_hash = previous.file_hash
fingerprint = previous.fingerprint


def write_json(path, value):
    """Publish a complete JSON file, including during a resumed evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate saved RGB policies; never train.")
    parser.add_argument("--method", choices=("all", *METHOD_IDS), required=True)
    defaults = {"config": CONFIG, "semantic_run": previous.DEFAULT_SEMANTIC,
                "model_dir": previous.DEFAULT_MODEL, "yolo_model": previous.DEFAULT_YOLO,
                "track_python": previous.DEFAULT_TRACK_PYTHON,
                "cotracker_repo": previous.DEFAULT_TRACK_REPO,
                "cotracker_weights": previous.DEFAULT_TRACK_WEIGHTS,
                "output_root": Path("/home/dhiazr/gt69_scratch2/dhiazgeo/rgb_geometry_evaluation_20")}
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=default)
    for name in ("plan", "check-inputs", "resume"):
        parser.add_argument("--" + name, action="store_true")
    return parser


def load_settings(path):
    settings = read_json(path)
    fixed = {"height": 480, "width": 832, "frames": 81, "fps": 16,
             "inference_steps": 50, "guidance_scale": 5.0,
             "anchor_frames": [0, 16, 32, 48, 64], "points_per_class": 16,
             "car_erosion_radius": 2, "seeds_per_prompt": 2,
             "visible_unknown": "mismatch", "unobserved_query_score": 0.0,
             "missing_expected_class_score": 0.0, "maximum_concurrent_gpus": 2,
             "retained_case_id": "eval_00_0",
             "overall": "mean_of_class_means_over_predeclared_expected_cases",
             "scored_classes": list(CLASS_NAMES[1:]),
             "samplers": "preserve_each_trained_recipe_with_separate_frozen_Wan_controls"}
    if settings.get("format") != FORMAT or any(settings["settings"].get(key) != value for key, value in fixed.items()):
        raise ValueError("This evaluator requires the declared 20-case, five-anchor recipe.")
    methods = settings["methods"]
    if [method["id"] for method in methods] != list(METHOD_IDS):
        raise ValueError("Use the eleven declared methods in their recorded order.")
    for method in methods:
        name = method["id"]
        family = "flow" if name.startswith("flow_") else "rl"
        expected_arm = {"flow_baseline": "A0", "rl_baseline": "B", "rl_r": "R",
                        "rl_s": "S", "rl_lora": "L"}.get(name)
        if family == "flow" and expected_arm is None:
            expected_arm = "A-RGB" if "_rgb_" in name else "A-Sem"
        if method["family"] != family or method["arm"] != expected_arm or method["baseline"] != family + "_baseline":
            raise ValueError("Method identity differs from its trained generation recipe.")
        expected_alpha = 0.25
        if family == "flow":
            expected_alpha = 0.0 if name == "flow_baseline" else {"0p1": 0.1, "0p25": 0.25, "1": 1.0}[name.rsplit("_", 1)[1]]
        if method["alpha"] != expected_alpha:
            raise ValueError("Correction strength differs from the named method.")
        if expected_arm not in ("A0", "B"):
            if len(method.get("checkpoint_sha256", "")) != 64 or not Path(method["checkpoint"]).is_absolute():
                raise ValueError("A trained method needs an absolute checkpoint and recorded SHA256.")
    prompts = settings["prompts"]
    if len(prompts) != 10 or len({row["scene"] for row in prompts}) != 10 or len({row["prompt"] for row in prompts}) != 10:
        raise ValueError("Use ten distinct, predeclared prompts.")
    covered = set()
    for row in prompts:
        classes = row["expected_classes"]
        if not classes or len(set(classes)) != len(classes) or not set(classes).issubset(CLASS_NAMES[1:]):
            raise ValueError("Expected classes must be distinct supported RGB labels.")
        covered.update(classes)
    if covered != set(CLASS_NAMES[1:]):
        raise ValueError("The shared panel must cover all ten scored classes.")
    return settings


def planned_cases(settings):
    cases = []
    for scene_index, scene in enumerate(settings["prompts"]):
        for seed_index in range(2):
            seed = settings["settings"]["base_seed"] + 2 * scene_index + seed_index
            if not isinstance(seed, int) or seed < 0 or seed == 47:
                raise ValueError("Use permitted nonnegative seeds.")
            cases.append(dict(scene, case_id=f"eval_{scene_index:02d}_{seed_index}",
                              seed=seed, semantic_seed=100000 + seed, action_seed=200000 + seed))
    return cases


def source_hashes():
    sources = previous.sources()
    for folder in ("src/rgb_geometry_evaluation", "jobs/rgb_geometry_evaluation"):
        for path in sorted((REPOSITORY / folder).glob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                sources[str(path.relative_to(REPOSITORY))] = file_hash(path)
    return sources


def validate_inputs(args, settings):
    """Read only frozen inputs; inference does not require saved training latents."""
    semantic_root = previous.safe_path(args.semantic_run, existing=True)
    names = ("prepared.json", "training_config.json", "training_complete.json",
             "semantic_best.pt", "frozen_codec.pt", "normalization_stats.json")
    hashes = {name: file_hash(semantic_root / name) for name in names}
    if hashes != settings["provenance"]["expected_semantic_files"]:
        raise ValueError("Frozen semantic inputs differ from the recorded evaluation checkpoints' observer.")
    prepared = read_json(semantic_root / "prepared.json")
    training = read_json(semantic_root / "training_config.json")
    completion = read_json(semantic_root / "training_complete.json")
    if prepared.get("format") != "joint_semantic_mixed_v1":
        raise ValueError("Use the original nine-class joint semantic representation.")
    if training["prepared_sha256"] != hashes["prepared.json"]:
        raise ValueError("Frozen semantic preparation changed after training.")
    if completion["best_update"] != 4600 or completion["best_weights_sha256"] != hashes["semantic_best.pt"]:
        raise ValueError("Use the original update-4600 semantic observer.")
    for name in ("frozen_codec.pt", "normalization_stats.json"):
        if prepared["weight_files"][name] != hashes[name]:
            raise ValueError("Frozen codec or normalization changed.")
    if previous.safe_path(args.model_dir, existing=True) != previous.safe_path(training["model_dir"], existing=True):
        raise ValueError("Wan model directory differs from the trained semantic observer.")
    if file_hash(args.yolo_model) != training["yolo"]["sha256"]:
        raise ValueError("YOLO checkpoint differs from the original judge.")
    specification = importlib.util.spec_from_file_location("eval_inventory", REPOSITORY / "src/datasets/create_walking_pov_100.py")
    inventory_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(inventory_module)
    if inventory_module.model_inventory(args.model_dir) != training["model_inventory"]:
        raise ValueError("Frozen Wan file inventory changed.")
    if not Path(args.track_python).is_file() or not os.access(args.track_python, os.X_OK):
        raise ValueError("CoTracker environment Python is missing or not executable.")
    from cotracker_evaluation.run_cotracker import model_provenance
    tracker = model_provenance(args.cotracker_repo, args.cotracker_weights)
    checkpoint_hashes = {}
    for method in settings["methods"]:
        if "checkpoint" in method and method["checkpoint"] not in checkpoint_hashes:
            path = previous.safe_path(method["checkpoint"], existing=True)
            checkpoint_hashes[method["checkpoint"]] = file_hash(path)
        if "checkpoint" in method and checkpoint_hashes[method["checkpoint"]] != method["checkpoint_sha256"]:
            raise ValueError("Trained checkpoint hash differs: " + method["id"])
    # Reject known exact prompt reuse, without claiming an exhaustive scene audit.
    known = {row["prompt"].strip().casefold() for row in prepared["records"]}
    known.add(training["final_new_prompt"].strip().casefold())
    for preview in training.get("preview_plan", []):
        known.update(row["prompt"].strip().casefold() for row in preview["cases"])
    for filename in ("rgb_geometry_pilot_v1.json", "rgb_geometry_rl_v2.json"):
        old = read_json(REPOSITORY / "configs" / filename)
        for key, rows in old.items():
            if "prompt" in key and isinstance(rows, list):
                known.update(row["prompt"].strip().casefold() for row in rows if isinstance(row, dict) and "prompt" in row)
    if any(row["prompt"].strip().casefold() in known for row in settings["prompts"]):
        raise ValueError("New evaluation prompt duplicates a recorded training/development prompt.")
    root = previous.safe_path(args.output_root)
    protected = [REPOSITORY, semantic_root, args.model_dir, Path(args.yolo_model).parent,
                 args.cotracker_repo, *[Path(path).parent for path in checkpoint_hashes]]
    for path in protected:
        path = previous.safe_path(path)
        if root == path or root in path.parents or path in root.parents:
            raise ValueError("Evaluation output overlaps a protected input: " + str(path))
    sources = source_hashes()
    for name, original_hash in prepared["sources"].items():
        critical = any(part in name for part in ("codec", "phase1_semantic_models", "wan_middle_feature_capture",
                                                 "my_wan_pipeline", "yolo_labelmap"))
        if critical and name in sources and sources[name] != original_hash:
            raise ValueError("Frozen scientific implementation changed: " + name)
    identity = {"semantic_files": hashes, "model_inventory": training["model_inventory"],
                "yolo_sha256": training["yolo"]["sha256"], "cotracker": tracker,
                "checkpoints": checkpoint_hashes, "sources": sources,
                "input_paths": {key: str(getattr(args, key)) for key in
                                ("semantic_run", "model_dir", "yolo_model", "track_python", "cotracker_repo", "cotracker_weights")},
                "scene_transfer_verified": False}
    return {"prepared": prepared, "training_config": training, "completion": completion, "identity": identity}


def make_plan(settings, identity):
    plan = {"format": FORMAT, "settings": settings["settings"], "methods": settings["methods"],
            "cases": planned_cases(settings), "input_identity": identity, "provenance": settings["provenance"]}
    return dict(plan, fingerprint=fingerprint(plan))


def save_shared_plan(root, plan):
    """Two jobs may arrive together: publish once, atomically, then compare."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "evaluation_plan.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=root, suffix=".json.tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_json(path) != plan:
                raise ValueError("Output root belongs to a different evaluation plan.")
    finally:
        temporary.unlink(missing_ok=True)
