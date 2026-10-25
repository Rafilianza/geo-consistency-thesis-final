"""Fixed four-arm cases and file provenance. Importing this module needs only stdlib."""

import copy
import importlib.util
import math
from pathlib import Path
import re

from rgb_geometry_pilot import protocol as historical


class BudgetStop(RuntimeError):
    """The fixed resource allowance cannot cover the next operation."""


REPOSITORY = Path(__file__).resolve().parents[2]
FORMAT = "rgb_geometry_rl_v2"
DEFAULT_CONFIG = REPOSITORY / "configs/rgb_geometry_rl_v2.json"
DEFAULT_SEMANTIC = historical.DEFAULT_SEMANTIC
DEFAULT_MODEL = historical.DEFAULT_MODEL
DEFAULT_YOLO = historical.DEFAULT_YOLO
DEFAULT_TRACK_REPO = historical.DEFAULT_TRACK_REPO
DEFAULT_TRACK_PYTHON = historical.DEFAULT_TRACK_PYTHON
DEFAULT_TRACK_WEIGHTS = historical.DEFAULT_TRACK_WEIGHTS
DEFAULT_WEIGHTS = Path("/home/dhiazr/rgb_geometry_rl_v2")
DEFAULT_OUTPUTS = Path("/home/dhiazr/gt69_scratch2/dhiazgeo/rgb_geometry_rl_v2")
EXPERIMENTS = {
    "preflight_heads": {"kind": "preflight", "arm": "R/S", "cases": 3, "primary_cases": 2, "hours": 6.0, "output_gib": 32.0},
    "preflight_lora": {"kind": "preflight", "arm": "L", "cases": 2, "primary_cases": 1, "hours": 6.0, "output_gib": 32.0},
    "b_baseline": {"kind": "baseline", "arm": "B", "cases": 10, "hours": 6.0, "output_gib": 32.0},
    "r_train": {"kind": "train", "arm": "R", "cases": 80, "hours": 24.0, "output_gib": 128.0},
    "s_train": {"kind": "train", "arm": "S", "cases": 80, "hours": 24.0, "output_gib": 128.0},
    "l_train": {"kind": "train", "arm": "L", "cases": 80, "hours": 24.0, "output_gib": 128.0},
}

# These helpers perform no model execution and retain the established hash rules.
read_json = historical.read_json
write_json = historical.write_json
file_hash = historical.file_hash
fingerprint = historical.fingerprint
safe_path = historical.safe_path
verify_completed = historical.verify_completed
gpu_queue_count = historical.gpu_queue_count
check_gpu_allocation = historical.check_gpu_allocation

PROVENANCE_FILES = (
    "configs/rgb_geometry_pilot_v1.json",
    "configs/datasets/walking_pov_100_v1.json",
    "configs/datasets/walking_pov_mixed_100_v1.json",
    "configs/semantic_denoiser_experiments/confirmation.json",
    "configs/cotracker_evaluation/reference_rgb_20260909.json",
)


def sources():
    """Bind the complete new implementation and every reused scientific source."""
    identities = historical.sources()
    paths = list((REPOSITORY / "src/rgb_geometry_rl_v2").rglob("*.py"))
    paths += list((REPOSITORY / "jobs/rgb_geometry_rl_v2").glob("*.sh"))
    paths += list((REPOSITORY / "jobs/rgb_geometry_rl_v2_recovery").glob("*.sh"))
    paths += [DEFAULT_CONFIG]
    paths += [REPOSITORY / "configs/rgb_geometry_rl_v2_rescore_20260920.json"]
    paths += [REPOSITORY / name for name in (
        "src/datasets/create_walking_pov_100.py",
        "src/rgb_geometry_qualification/controls.py",
        "src/rgb_geometry_qualification/diagnostics.py",
        "src/rgb_geometry_reward_check/controls.py",
    )]
    paths += [REPOSITORY / name for name in PROVENANCE_FILES]
    for path in sorted(paths):
        identities[str(path.relative_to(REPOSITORY))] = file_hash(path)
    return identities


def _recorded_scene_prompts(value):
    """Read nested JSON records without importing any training code."""
    scenes, prompts = set(), set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "scenes" and isinstance(item, list):
                scenes.update(scene.strip().casefold() for scene in item if isinstance(scene, str))
            elif key == "scene" and isinstance(item, str):
                scenes.add(item.strip().casefold())
            elif key in ("prompt", "final_new_prompt", "new_prompt") and isinstance(item, str):
                prompts.add(" ".join(item.split()).casefold())
            child_scenes, child_prompts = _recorded_scene_prompts(item)
            scenes.update(child_scenes)
            prompts.update(child_prompts)
    elif isinstance(value, list):
        for item in value:
            child_scenes, child_prompts = _recorded_scene_prompts(item)
            scenes.update(child_scenes)
            prompts.update(child_prompts)
    return scenes, prompts


def provenance_audit(plan, prepared=None, training_config=None):
    """Reject known reuse, without promoting missing history to a clean holdout."""
    known_scenes, known_prompts = set(), set()
    evidence = {}
    for name in PROVENANCE_FILES:
        path = REPOSITORY / name
        scenes, prompts = _recorded_scene_prompts(read_json(path))
        known_scenes.update(scenes)
        known_prompts.update(prompts)
        evidence[name] = file_hash(path)
    for name, records in (("semantic_prepared", prepared), ("semantic_training_config", training_config)):
        if records is not None:
            scenes, prompts = _recorded_scene_prompts(records)
            known_scenes.update(scenes)
            known_prompts.update(prompts)
            evidence[name] = fingerprint(records)
    for group in ("training_prompts", "sentinel_prompts", "endpoint_prompts"):
        for case in plan[group]:
            prompt = " ".join(case["prompt"].split()).casefold()
            if case["scene"].strip().casefold() in known_scenes or prompt in known_prompts:
                raise ValueError(f"Known scene/prompt reuse in {group}: {case['scene']}")
    return {
        "status": "unverified", "known_record_overlap": False,
        "semantic_runtime_records_checked": prepared is not None and training_config is not None,
        "endpoint_transfer_claim_allowed": False,
        "evidence": evidence,
        "limitation": "Exact scene IDs/prompt texts were checked; aliases, similar scenes and unrecorded inspections remain unresolved.",
    }


def load_plan(path):
    plan = read_json(path)
    fixed_settings = {
        "format": FORMAT, "semantic_update": 4600, "rl_rounds": 4,
        "candidates_per_prompt": 4, "maximum_concurrent_gpus": 3,
        "action_indices": [10, 20], "inference_steps": 50, "guidance_scale": 5.0,
        "alpha": 0.25, "rho": 0.25, "action_noise_scale": 0.03,
        "learning_rate": 1e-6, "head_initialization_seed": 42,
        "lora_initialization_seed": 12000,
        "endpoint_selection": "last_fully_accepted_checkpoint",
    }
    if any(plan.get(key) != value for key, value in fixed_settings.items()):
        raise ValueError("The fixed four-arm recipe settings cannot be changed.")
    all_cases = []
    for group, count in (("training_prompts", 4), ("sentinel_prompts", 2), ("endpoint_prompts", 4)):
        cases = plan.get(group)
        if not isinstance(cases, list) or len(cases) != count:
            raise ValueError(f"Expected {count} fixed {group}.")
        for case in cases:
            if (not isinstance(case.get("prompt"), str) or not case["prompt"].strip()
                    or not isinstance(case.get("scene"), str)
                    or re.fullmatch(r"[a-z][a-z0-9_]*", case["scene"]) is None
                    or case.get("provenance") != "unverified"):
                raise ValueError("Each proposed case needs a scene, prompt and honest unverified provenance.")
        all_cases += cases
    prompts = {" ".join(case["prompt"].split()).casefold() for case in all_cases}
    if len(prompts) != 10 or len({case["scene"] for case in all_cases}) != 10:
        raise ValueError("Training, sentinels and endpoints must have ten distinct scenes/prompts.")
    if plan.get("provenance", {}).get("status") != "unverified" or plan["provenance"].get("endpoint_transfer_claim_allowed") is not False:
        raise ValueError("Incomplete scene history must remain unverified, with transfer claims withheld.")
    provenance_audit(plan)
    return plan


def training_contexts(plan):
    """The same four RGB/semantic initial noises are used in every update."""
    cases = []
    for prompt_index, item in enumerate(plan["training_prompts"]):
        cases.append(dict(item, case_id=f"training_context_{prompt_index}", role="training_context",
                          prompt_index=prompt_index, seed=21001 + prompt_index,
                          semantic_seed=31001 + prompt_index, action_seed=41001 + prompt_index))
    return cases


def rollout_cases(plan, round_index):
    if type(round_index) is not int or not 0 <= round_index < 4:
        raise ValueError("Training round must be 0, 1, 2 or 3.")
    cases = []
    for context in training_contexts(plan):
        for candidate in range(4):
            prompt_index = context["prompt_index"]
            cases.append(dict(context, case_id=f"round_{round_index + 1}_p{prompt_index}_k{candidate}",
                              role="training_rollout", round=round_index, candidate=candidate,
                              action_seed=51000 + 16 * round_index + 4 * prompt_index + candidate))
    return cases


def sentinel_cases(plan, round_index=None):
    if round_index is not None and (type(round_index) is not int or not 0 <= round_index < 4):
        raise ValueError("Sentinel round must be omitted or be 0, 1, 2 or 3.")
    cases = []
    for index, item in enumerate(plan["sentinel_prompts"]):
        prefix = "sentinel" if round_index is None else f"round_{round_index + 1}_sentinel"
        cases.append(dict(item, case_id=f"{prefix}_{index}", sentinel_index=index,
                          role="sentinel", seed=61001 + index, semantic_seed=62001 + index,
                          action_seed=63001 + index))
    return cases


def endpoint_cases(plan):
    cases = []
    for prompt_index, item in enumerate(plan["endpoint_prompts"]):
        for repeat in range(2):
            index = 2 * prompt_index + repeat
            cases.append(dict(item, case_id=f"endpoint_{prompt_index}_{repeat}", role="endpoint",
                              prompt_index=prompt_index, repeat=repeat, seed=71001 + index,
                              semantic_seed=72001 + index, action_seed=73001 + index))
    return cases


def preflight_cases(plan, experiment):
    if experiment not in ("preflight_heads", "preflight_lora"):
        raise ValueError("Use a preflight experiment name.")
    arms = ("B", "R", "S") if experiment == "preflight_heads" else ("B", "L")
    return [dict(training_contexts(plan)[0], case_id=f"preflight_{arm.lower()}", role="preflight",
                 policy_arm=arm, seed=91001, semantic_seed=92001, action_seed=93001) for arm in arms]


def planned_cases(plan, experiment):
    spec = EXPERIMENTS[experiment]
    if spec["kind"] == "preflight":
        return preflight_cases(plan, experiment)
    if spec["kind"] == "baseline":
        return sentinel_cases(plan) + endpoint_cases(plan)
    cases = []
    for round_index in range(4):
        cases += rollout_cases(plan, round_index)
        cases += sentinel_cases(plan, round_index)
    return cases + endpoint_cases(plan)


def require_separate_outputs(outputs, weights, sources_to_protect=()):
    historical_roots = [historical.DEFAULT_OUTPUTS, historical.DEFAULT_WEIGHTS]
    historical.require_separate_outputs(outputs, weights, [REPOSITORY, *historical_roots, *sources_to_protect])


def validate_inputs(args, plan):
    # Reuse checkpoint/codec/model/YOLO/tracker and prepared-latent identity checks.
    # Passing every new case ensures even training and sentinels reject known reuse.
    compatibility_plan = copy.deepcopy(plan)
    compatibility_plan["confirmation_prompts"] = plan["training_prompts"] + plan["sentinel_prompts"] + plan["endpoint_prompts"]
    result = historical.validate_inputs(args, compatibility_plan)
    require_separate_outputs(args.output_root, args.weights_root,
                             [args.config, args.cotracker_repo, args.cotracker_weights, args.track_python])
    specification = importlib.util.spec_from_file_location(
        "rl_v2_model_inventory", REPOSITORY / "src/datasets/create_walking_pov_100.py")
    inventory_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(inventory_module)
    if inventory_module.model_inventory(safe_path(args.model_dir, existing=True)) != result["training_config"]["model_inventory"]:
        raise ValueError("Frozen Wan files differ from the semantic training inventory.")
    audit = provenance_audit(plan, result["prepared"], result["training_config"])
    identity = result["identity"]
    identity.update(format=FORMAT, plan=plan, sources=sources(), provenance=audit,
                    config_sha256=file_hash(args.config), semantic_update=4600)
    result.update(identity=identity, provenance=audit, study_fingerprint=fingerprint(identity))
    return result


def runtime_limits(experiment, max_gpu_hours=None, max_output_gib=None):
    """Overrides can tighten the fixed reservation, never extend it."""
    spec = EXPERIMENTS[experiment]
    limits = {}
    for name, supplied in (("hours", max_gpu_hours), ("output_gib", max_output_gib)):
        value = spec[name] if supplied is None else supplied
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= spec[name]:
            raise ValueError(f"{name} must be finite, positive and no greater than {spec[name]}.")
        limits[name] = float(value)
    return limits


def prerequisite_names(experiment):
    if experiment in ("r_train", "s_train"):
        return ["preflight_heads", "b_baseline"]
    if experiment == "l_train":
        return ["preflight_lora", "b_baseline"]
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown four-arm experiment.")
    return []


def prerequisites(args, spec, study_id):
    receipts = {}
    if spec != EXPERIMENTS[args.experiment]:
        # Resource limits may be tightened by the runner, without changing arm/kind.
        if any(spec.get(key) != EXPERIMENTS[args.experiment][key] for key in ("kind", "arm", "cases")):
            raise ValueError("Prerequisite request differs from the declared experiment.")
    for name in prerequisite_names(args.experiment):
        directory = safe_path(args.output_root) / name
        receipt = verify_completed(directory, study_id)
        result = receipt.get("result", {})
        if name.startswith("preflight") and result.get("status") != "passed":
            raise ValueError(f"New-study integration preflight has not passed: {name}")
        if name == "b_baseline" and (result.get("status") != "complete" or result.get("sentinel_baselines_supported") is not True):
            raise ValueError("New-study B requires supported sentinel baselines and controls.")
        receipts[name] = file_hash(directory / "complete.json")
    return receipts
