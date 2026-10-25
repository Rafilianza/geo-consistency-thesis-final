"""Fixed cases, input identities and small file helpers; no model loading."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY / "configs/rgb_geometry_pilot_v1.json"
DEFAULT_SEMANTIC = Path("/home/dhiazr/joint_semantic_mixed_v1_epoch60")
DEFAULT_MODEL = Path("/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Wan2.1-T2V-1.3B-Diffusers")
DEFAULT_YOLO = Path("/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Yolo26/yolo26x-sem.pt")
DEFAULT_WEIGHTS = Path("/home/dhiazr/rgb_geometry_pilot_v1")
DEFAULT_OUTPUTS = Path("/home/dhiazr/gt69_scratch2/dhiazgeo/rgb_geometry_pilot_v1")
DEFAULT_TRACK_REPO = Path("/home/dhiazr/cotracker3/co-tracker")
DEFAULT_TRACK_PYTHON = Path("/home/dhiazr/cotracker3/envcotrack/bin/python")
DEFAULT_TRACK_WEIGHTS = Path("/home/dhiazr/cotracker3/checkpoints/scaled_offline.pth")
EXPERIMENTS = {
    "a0_baseline": {"kind": "flow_eval", "arm": "A0", "alpha": 0.0, "cases": 12, "hours": 3.0},
    "a_rgb_train": {"kind": "flow_train", "arm": "A-RGB", "cases": 0, "hours": 4.0},
    "a_sem_train": {"kind": "flow_train", "arm": "A-Sem", "cases": 0, "hours": 4.0},
    "b0_baseline": {"kind": "rl_baseline", "arm": "B0", "cases": 24, "hours": 6.0},
    "b_rgb_train": {"kind": "rl_train", "arm": "R", "cases": 73, "hours": 15.0},
    "b_sem_train": {"kind": "rl_train", "arm": "S", "cases": 73, "hours": 15.0},
}
for control, arm in (("rgb", "A-RGB"), ("sem", "A-Sem")):
    for suffix, alpha in (("0p1", 0.1), ("0p25", 0.25), ("1", 1.0)):
        EXPERIMENTS[f"a_{control}_alpha_{suffix}"] = {
            "kind": "flow_eval", "arm": arm, "alpha": alpha,
            "cases": 12 if alpha == 0.25 else 4, "hours": 3.0 if alpha == 0.25 else 2.0,
        }


def safe_path(value, *, existing=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"Use an absolute path: {path}")
    reserved = r"(?:seed[ _=-]?|(?:^|[/_.-])s[ _=-]?)0*47(?:\D|$)"
    if re.search(reserved, str(path), re.I):
        raise ValueError("Reserved data is outside this experiment.")
    resolved = path.resolve(strict=existing)
    if re.search(reserved, str(resolved), re.I):
        raise ValueError("A path resolves into reserved data.")
    return resolved


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sources():
    paths = list((REPOSITORY / "src/rgb_geometry_pilot").glob("*.py"))
    paths += list((REPOSITORY / "jobs/rgb_geometry_pilot").glob("*.sh"))
    names = [
        "src/joint_semantic_mixed/common.py", "src/joint_semantic_mixed/examples.py",
        "src/joint_semantic_mixed/monitoring.py", "src/codec_retraining/data.py",
        "src/codec_retraining/model.py", "src/phase3/semantic_codec.py",
        "src/middle_feature_semantic/phase1_semantic_models.py",
        "src/middle_feature_semantic/reference_rgb_correction.py",
        "src/middle_feature_semantic/wan_middle_feature_capture.py",
        "src/video_generation/my_wan_pipeline.py", "src/video_generation/gen_dataset.py",
        "src/yolo/yolo_labelmap_concepts.py", "src/cotracker_evaluation/tracking_metrics.py",
        "src/cotracker_evaluation/run_cotracker.py", "src/cotracker_evaluation/comparison_io.py",
    ]
    paths += [REPOSITORY / name for name in names]
    return {str(path.relative_to(REPOSITORY)): file_hash(path) for path in sorted(paths)}


def load_plan(path):
    plan = read_json(path)
    if plan.get("format") != "rgb_geometry_pilot_v1" or len(plan["training_prompts"]) != 4 or len(plan["confirmation_prompts"]) != 4:
        raise ValueError("Expected four fixed training and four confirmation prompts.")
    cases = plan["training_prompts"] + plan["confirmation_prompts"]
    texts = [case["prompt"].strip().casefold() for case in cases]
    if len(set(texts)) != 8 or any(not text for text in texts):
        raise ValueError("All fixed prompt texts must be nonempty and distinct.")
    if len({case["scene"] for case in cases}) != 8:
        raise ValueError("Each fixed scene needs a distinct identifier.")
    fixed_settings = {"semantic_update": 4600, "flow_updates": 400,
                      "flow_confirmation_alpha": 0.25, "rl_rounds": 4,
                      "candidates_per_prompt": 4, "maximum_concurrent_gpus": 3}
    if any(plan.get(key) != value for key, value in fixed_settings.items()):
        raise ValueError("The fixed pilot settings cannot be changed inside this recipe.")
    return plan


def training_contexts(plan):
    cases = []
    for round_index in range(4):
        for prompt_index, item in enumerate(plan["training_prompts"]):
            context_index = 4 * round_index + prompt_index
            cases.append({"case_id": f"reference_{context_index:02d}", "prompt": item["prompt"],
                          "scene": item["scene"], "round": round_index, "prompt_index": prompt_index,
                          "seed": 1000 + context_index, "semantic_seed": 4000 + context_index,
                          "action_seed": 2000 + context_index, "role": "independent_reference"})
    return cases


def rollout_cases(plan, round_index):
    if not isinstance(round_index, int) or not 0 <= round_index < 4:
        raise ValueError("Training round must be one of 0, 1, 2, 3.")
    cases = []
    contexts = training_contexts(plan)[4 * round_index:4 * round_index + 4]
    for context in contexts:
        for candidate in range(4):
            cases.append(dict(context, case_id=f"round_{round_index + 1}_p{context['prompt_index']}_k{candidate}",
                              reference_id=context["case_id"], candidate=candidate, role="training_rollout",
                              action_seed=3000 + 16 * round_index + 4 * context["prompt_index"] + candidate))
    return cases


def confirmation_cases(plan, family):
    if family not in ("flow", "rl"):
        raise ValueError("Confirmation family must be flow or rl.")
    cases = []
    for prompt_index, item in enumerate(plan["confirmation_prompts"]):
        for repeat in range(2):
            index = 2 * prompt_index + repeat
            seed = (109, 131)[repeat] if family == "flow" else 6000 + index
            cases.append({"case_id": f"confirmation_{prompt_index}_{repeat}", "prompt": item["prompt"],
                          "scene": item["scene"], "role": "confirmation", "seed": seed,
                          "semantic_seed": 100000 + seed if family == "flow" else 7000 + index,
                          "action_seed": 8000 + index})
    return cases


def development_cases(plan):
    return [dict(item, case_id=f"development_{index}", role="development", seed=103, semantic_seed=100103)
            for index, item in enumerate(plan["training_prompts"])]


def flow_cases(plan, alpha):
    cases = development_cases(plan)
    if alpha in (0.0, 0.25):
        cases += confirmation_cases(plan, "flow")
    return cases


def require_separate_outputs(outputs, weights, sources_to_protect):
    outputs, weights = safe_path(outputs), safe_path(weights)
    if outputs == weights or outputs in weights.parents or weights in outputs.parents:
        raise ValueError("Use separate, non-nested weights and output roots.")
    for destination in (outputs, weights):
        for source in sources_to_protect:
            source = safe_path(source)
            if destination == source or destination in source.parents or source in destination.parents:
                raise ValueError(f"Output overlaps a protected source: {destination}")


def validate_inputs(args, plan):
    semantic_run = safe_path(args.semantic_run, existing=True)
    paths = {name: semantic_run / name for name in (
        "prepared.json", "training_config.json", "training_complete.json", "semantic_best.pt",
        "frozen_codec.pt", "normalization_stats.json")}
    prepared = read_json(paths["prepared.json"])
    config = read_json(paths["training_config.json"])
    completion = read_json(paths["training_complete.json"])
    if prepared.get("format") != "joint_semantic_mixed_v1":
        raise ValueError("Use the prepared nine-class joint semantic run.")
    if config["prepared_sha256"] != file_hash(paths["prepared.json"]):
        raise ValueError("Semantic preparation changed after training.")
    if completion["best_update"] != 4600 or file_hash(paths["semantic_best.pt"]) != completion["best_weights_sha256"]:
        raise ValueError("Use the verified selected update-4600 semantic checkpoint.")
    for name, expected in prepared["weight_files"].items():
        source = safe_path(semantic_run / name, existing=True)
        if source.parent != semantic_run or file_hash(source) != expected:
            raise ValueError("Frozen codec or normalization identity differs.")
    model_path = safe_path(args.model_dir, existing=True)
    if model_path != safe_path(config["model_dir"], existing=True):
        raise ValueError("Use the Wan model used by the selected semantic run.")
    yolo_path = safe_path(args.yolo_model, existing=True)
    if file_hash(yolo_path) != config["yolo"]["sha256"]:
        raise ValueError("Use the same frozen YOLO teacher as semantic training.")
    records = prepared["records"]
    if {split: sum(record["split"] == split for record in records) for split in ("train", "val")} != {"train": 80, "val": 20}:
        raise ValueError("This fixed 400-update pilot uses the accepted 80/20 split.")
    for split in ("train", "val"):
        if [record["clip_id"] for record in records if record["split"] == split] != prepared["accepted"][split]:
            raise ValueError("Accepted clip order/list changed.")
    for record in records:
        if record["seed"] == 47:
            raise ValueError("Reserved seed is not an input to this study.")
        for key in ("rgb_latent", "semantic_latent"):
            source = safe_path(record[key], existing=True)
            if file_hash(source) != record[key + "_sha256"]:
                raise ValueError(f"Changed prepared latent: {record['clip_id']}")
    used_prompts = {record["prompt"].strip().casefold() for record in records}
    used_prompts.add(config["final_new_prompt"].strip().casefold())
    for preview in config.get("preview_plan", []):
        used_prompts.update(case["prompt"].strip().casefold() for case in preview["cases"])
    if any(item["prompt"].strip().casefold() in used_prompts for item in plan["confirmation_prompts"]):
        raise ValueError("Confirmation prompt overlaps semantic development data.")
    require_separate_outputs(args.output_root, args.weights_root,
                             [semantic_run, model_path, yolo_path.parent, Path(prepared["outputs_dir"])]
                             + [Path(record["directory"]).parent.parent for record in records])
    for key in ("track_python", "cotracker_repo", "cotracker_weights"):
        safe_path(getattr(args, key), existing=True)
    if not Path(args.track_python).is_file() or not os.access(args.track_python, os.X_OK):
        raise ValueError("CoTracker needs its existing executable environment Python.")
    for relative in ("hubconf.py", "cotracker/predictor.py"):
        if not (Path(args.cotracker_repo) / relative).is_file():
            raise ValueError(f"CoTracker repository is incomplete: {relative}")
    # Reuse the historical CPU-only source/checkpoint provenance, not Torch Hub.
    from cotracker_evaluation.run_cotracker import model_provenance

    tracker_identity = model_provenance(args.cotracker_repo, args.cotracker_weights)
    source_ids = sources()
    critical = [name for name in source_ids if name in prepared["sources"] and (
        "codec" in name or "phase1_semantic_models" in name or "wan_middle_feature_capture" in name
        or "my_wan_pipeline" in name or "yolo_labelmap" in name)]
    for name in critical:
        if source_ids[name] != prepared["sources"][name]:
            raise ValueError(f"Frozen scientific implementation changed: {name}")
    identity = {"plan": plan, "sources": source_ids,
                "semantic_files": {name: file_hash(path) for name, path in paths.items()},
                "model_inventory": config["model_inventory"], "yolo_sha256": file_hash(yolo_path),
                "cotracker_weights_sha256": file_hash(args.cotracker_weights),
                "cotracker": tracker_identity}
    return {"prepared": prepared, "training_config": config, "completion": completion,
            "identity": identity, "study_fingerprint": fingerprint(identity)}


def verify_completed(directory, study_id):
    directory = safe_path(directory, existing=True)
    receipt = read_json(directory / "complete.json")
    if receipt["study_fingerprint"] != study_id or receipt["status"] != "complete":
        raise ValueError(f"Incompatible or incomplete prerequisite: {directory}")
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("Completion artifacts must be a relative-path hash mapping.")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path != directory / "complete.json"}
    if set(artifacts) != actual:
        raise ValueError(f"Prerequisite artifact inventory changed: {directory}")
    for relative, expected in artifacts.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Receipt artifact must stay inside its directory: {relative}")
        path = safe_path(directory / relative, existing=True)
        if directory not in path.parents or file_hash(path) != expected:
            raise ValueError(f"Prerequisite artifact changed: {relative}")
    weights = receipt.get("external_weights", {})
    if not isinstance(weights, dict):
        raise ValueError("External weights must be an absolute-path hash mapping.")
    for filename, expected in weights.items():
        path = safe_path(filename, existing=True)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"External checkpoint changed: {filename}")
    return receipt


def gpu_queue_count(snapshot):
    total = 0
    for row in snapshot.splitlines():
        if not row.strip():
            continue
        fields = row.split("|")
        if len(fields) != 2 or not fields[0].strip():
            raise ValueError("Cannot interpret the allocated GPU queue snapshot.")
        entries = {}
        for item in fields[1].strip().split(","):
            name, separator, value = item.strip().partition("=")
            if name == "gres/gpu" or name.startswith("gres/gpu:"):
                if not separator or not value.isdigit() or name in entries:
                    raise ValueError("Cannot interpret GPU TRES allocation counts.")
                entries[name] = int(value)
        generic = entries.get("gres/gpu")
        typed = sum(value for key, value in entries.items() if key != "gres/gpu")
        if generic is not None and generic < typed:
            raise ValueError("Generic GPU allocation cannot be below its typed GPU total.")
        total += generic if generic is not None else typed
    return total


def check_gpu_allocation():
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit() or os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_NTASKS", "1") != "1":
        raise RuntimeError("Submit one experiment as one single-GPU job; no arrays.")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SQUEUE_")}
    result = subprocess.run(["squeue", "--all", "--local", "--me", "--states=RUNNING,COMPLETING,CONFIGURING",
                             "--noheader", "--Format=JobID:32|,tres-alloc:256"],
                            check=True, capture_output=True, text=True, timeout=30, env=environment)
    if not any(row.split("|")[0].strip() == job_id for row in result.stdout.splitlines()):
        raise RuntimeError("Current job missing from the allocation check.")
    count = gpu_queue_count(result.stdout)
    if count < 1 or count > 3:
        raise RuntimeError(f"Expected 1–3 GPUs allocated across your running jobs, found {count}.")
    return result.stdout


def check_budget(directory, started, hours, remaining_cases, measured_case_seconds=None):
    if (not isinstance(remaining_cases, int) or remaining_cases < 0
            or not math.isfinite(started) or not math.isfinite(hours) or hours <= 0
            or (measured_case_seconds is not None
                and (not math.isfinite(measured_case_seconds) or measured_case_seconds < 0))):
        raise ValueError("Resource projections require finite positive budgets and nonnegative counts/costs.")
    elapsed = time.monotonic() - started
    seconds_per_case = max(360.0, measured_case_seconds or 0.0)
    if elapsed + 1.25 * remaining_cases * seconds_per_case + 900 > hours * 3600:
        raise RuntimeError("budget_stop: projected work and reporting reserve exceed this job's fixed budget.")
    # Uncompressed RGB/labels and two saved actions give a conservative bound.
    required = 2 * remaining_cases * 220 * 1024**2 + 5 * 1024**3
    ancestor = Path(directory)
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required:
        raise RuntimeError(f"budget_stop: need {required / 1024**3:.1f} GiB free for remaining outputs and reserve.")
