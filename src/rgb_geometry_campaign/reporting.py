"""Standard-library selectors and fixed-panel reports for compact campaign data.

Semantic status schema: status, development_score, trainable_parameters, recipe,
model_digest, comparison_fingerprint. The fingerprint identifies matching data,
metric and development settings, and excludes the candidate architecture.

RGB CSV schema: clips.csv has case_id, role='evaluation', status='complete';
classes.csv has case_id, class_name, score, query_count and optional diagnostics.
Class rows may include all phases; only planned final case IDs are summarized.
"""

import csv
import hashlib
import json
import math
from pathlib import Path


SEMANTIC_JOBS = ("m1", "m2", "m3", "m4", "m5_current", "m5_history", "m6_control", "m6_depth")
RGB_METHODS = ("baseline", "rl1", "rl2", "rl3", "rl4", "rl5", "rl6", "rl7", "rl8", "rl9", "rl10", "rl3_repeat", "rl4_repeat")
CLASS_NAMES = ("person", "car", "bus", "bicycle", "truck", "road", "sidewalk", "building", "vegetation", "sky")
COUNT_FIELDS = ("query_count", "visible_observation_count", "matching_observation_count", "unknown_observation_count",
                "possible_observation_count", "censored_observation_count", "cycle_eligible_count", "cycle_valid_return_count")
DIAGNOSTICS = ("availability", "censoring", "retention_16", "continuous_retention_16", "observable_seconds_mean",
               "cycle_error_pixel_median", "cycle_valid_return_fraction")
LIMITATIONS = [
    "Scores measure tracked class persistence and expected-class support, not geometry accuracy.",
    "Class consistency cannot reliably detect two same-class objects merging or certify appearance and camera movement.",
    "Missing expected classes score zero; technically failed or missing clips are unmeasured and invalidate the fixed-panel overall score.",
    "Training rewards and final class-balanced evaluation are different measurements.",
    "Track points and seeds from the same scene are not independent object or scene samples.",
    "Semantic selection uses development scores only; YOLO agreement is not independent ground-truth segmentation accuracy.",
    "Only numerical artifacts were retained; visual inspection and new metrics require regeneration.",
]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return value


def read_csv(path, required):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError("Missing CSV fields in " + str(path))
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed or truncated CSV: " + str(path))
    return rows


def write_json(path, value):
    # Only new report products are written; historical files are not overwritten.
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows, fields):
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value, name, lower=None, upper=None):
    if isinstance(value, bool) or value is None or value == "":
        raise ValueError("Missing or invalid number: " + name)
    parsed = float(value)
    if not math.isfinite(parsed) or (lower is not None and parsed < lower) or (upper is not None and parsed > upper):
        raise ValueError("Out-of-range number: " + name)
    return parsed


def integer(value, name):
    parsed = number(value, name, lower=0)
    if not parsed.is_integer():
        raise ValueError("Expected an integer: " + name)
    return int(parsed)


def average(values):
    return sum(values) / len(values) if values else None


def semantic_rgb_identity(directory, root, evidence, declared_score):
    """Verify actual generated RGB was identical, not merely seeded identically."""
    training_path = directory / "generated_training.csv"
    agreement_path = directory / "agreement.csv"
    for path in (training_path, agreement_path):
        evidence[str(path.relative_to(root))] = file_hash(path)
    training = read_csv(training_path, ("case_id", "rgb_sha256", "codec_labels_sha256"))
    agreement = read_csv(agreement_path, ("case_id", "panel", "step_index", "score", "rgb_sha256", "codec_labels_sha256"))
    train_hashes = {}
    for row in training:
        if row["case_id"] in train_hashes or not row["rgb_sha256"] or not row["codec_labels_sha256"]:
            raise ValueError("Duplicate training clip or missing generated RGB hash: " + str(directory))
        train_hashes[row["case_id"]] = (row["rgb_sha256"], row["codec_labels_sha256"])
    if len(train_hashes) != 16:
        raise ValueError("Semantic selection requires all sixteen generated training clips.")
    dev_hashes = {}
    scores = []
    for row in agreement:
        if row["panel"] != "semantic_development":
            continue
        step = integer(row["step_index"], "semantic step index")
        key = (row["case_id"], step)
        if (key in dev_hashes or step not in (10, 20, 30, 40)
                or not row["rgb_sha256"] or not row["codec_labels_sha256"]):
            raise ValueError("Duplicate/invalid development step or missing RGB hash: " + str(directory))
        dev_hashes[key] = (row["rgb_sha256"], row["codec_labels_sha256"])
        scores.append(number(row["score"], "development agreement", 0, 1))
    ids = {case_id for case_id, _ in dev_hashes}
    if len(ids) != 8 or len(dev_hashes) != 32:
        raise ValueError("Semantic selection requires eight development clips at all four readout positions.")
    for case_id in ids:
        if len({dev_hashes[(case_id, step)] for step in (10, 20, 30, 40)}) != 1:
            raise ValueError("Readout positions do not refer to the same final generated RGB: " + case_id)
    if not math.isclose(average(scores), declared_score, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Declared semantic development score differs from saved per-clip measurements.")
    return train_hashes, dev_hashes


def select_semantic(output_root, stage, candidate_names, required_names):
    root = Path(output_root)
    candidates = []
    evidence = {}
    generated_identity = None
    for name in required_names:
        path = root / name / "status.json"
        before_hash = file_hash(path)
        status = read_json(path)
        evidence[str(path.relative_to(root))] = before_hash
        if status.get("status") != "complete":
            raise ValueError("Semantic prerequisite is incomplete: " + name)
        score = number(status.get("development_score"), name + ".development_score", 0, 1)
        actual_identity = semantic_rgb_identity(root / name, root, evidence, score)
        if generated_identity is None:
            generated_identity = actual_identity
        elif actual_identity != generated_identity:
            raise ValueError("Semantic methods generated different RGB clips or teacher labels; matched comparison is invalid: " + name)
        if name not in candidate_names:
            continue
        parameters = integer(status.get("trainable_parameters"), name + ".trainable_parameters")
        recipe = status.get("recipe")
        digest = status.get("model_digest")
        fingerprint = status.get("comparison_fingerprint")
        if not isinstance(recipe, dict) or not recipe or not isinstance(digest, str) or not digest:
            raise ValueError("Missing semantic recipe or model identity: " + name)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("Missing semantic comparison fingerprint: " + name)
        candidates.append({
            "experiment": name, "development_score": score, "trainable_parameters": parameters,
            "recipe": recipe, "model_digest": digest, "comparison_fingerprint": fingerprint,
        })
    if len(candidates) != len(candidate_names):
        raise ValueError("A required semantic candidate is missing.")
    fingerprints = {row["comparison_fingerprint"] for row in candidates}
    if len(fingerprints) != 1:
        raise ValueError("Semantic candidates use different development data or measurements.")
    selected = min(candidates, key=lambda row: (-row["development_score"], row["trainable_parameters"], row["experiment"]))
    # Verify a stable snapshot, including prerequisites not eligible for selection.
    if any(file_hash(root / relative) != digest for relative, digest in evidence.items()):
        raise RuntimeError("Semantic status changed during selection.")
    result = {
        "format": "rgb_geometry_campaign_selection_v1", "status": "complete", "stage": stage,
        "chosen_experiment": selected["experiment"], "recipe": selected["recipe"],
        "model_digest": selected["model_digest"], "development_score": selected["development_score"],
        "trainable_parameters": selected["trainable_parameters"],
        "comparison_fingerprint": selected["comparison_fingerprint"],
        "selection_rule": "Maximum development score; exact ties prefer fewer trainable parameters, then experiment ID.",
        "used_final_rgb_evaluation": False, "candidates": candidates, "source_sha256": evidence,
        "same_generated_rgb_verified": True, "same_teacher_labels_verified": True, "verified_training_clips": 16,
        "verified_development_clips": 8, "verified_readout_positions": [10, 20, 30, 40],
        "geometry_certified": False,
    }
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "selection.json", result)
    return result


def choose_fusion(output_root):
    return select_semantic(output_root, "choose_fusion", ("m2", "m3"), ("m2", "m3"))


def choose_observer(output_root):
    return select_semantic(output_root, "choose_observer", ("m2", "m3", "m6_depth"), SEMANTIC_JOBS)


def reward_history(path):
    rewards = read_csv(path, ("round", "reward"))
    by_round = {}
    for row in rewards:
        round_index = integer(row["round"], "round")
        by_round.setdefault(round_index, []).append(number(row["reward"], "reward"))
    return {
        "rollouts": len(rewards), "rounds": [
            {"round": round_index, "rollouts": len(values), "mean_reward": average(values),
             "minimum_reward": min(values), "maximum_reward": max(values)}
            for round_index, values in sorted(by_round.items())
        ],
    }


def summarize_method(root, method, cases, expected_classes):
    directory = root / method
    status_path = directory / "status.json"
    summary = {"method": method, "status": "incomplete", "execution_status": "missing",
               "planned_clips": len(cases), "completed_clips": 0, "overall_class_balanced_score": None,
               "per_class": [], "per_clip": [], "source_sha256": {}, "errors": [], "reward_history": {}}
    status = {}
    if status_path.exists():
        status = read_json(status_path)
        summary["execution_status"] = status.get("status", "unknown")
        summary["source_sha256"][str(status_path.relative_to(root))] = file_hash(status_path)
    for key in ("attempted_updates", "effective_policy_updates", "retained_updates", "quality_warning_count", "elapsed_seconds", "stop_reason", "comparison_fingerprint", "trainable_parameters", "rounds", "seed"):
        summary[key] = status.get(key)
    summary["trained_result"] = None
    effective = status.get("effective_policy_updates")
    if method != "baseline" and effective is not None:
        summary["trained_result"] = integer(effective, "effective_policy_updates") > 0
    for filename in ("failure.json", "launcher_status.json"):
        path = directory / filename
        if path.exists():
            summary[filename[:-5]] = read_json(path)
            summary["source_sha256"][str(path.relative_to(root))] = file_hash(path)
    rewards_path = directory / "rewards.csv"
    if rewards_path.exists():
        summary["reward_history"] = reward_history(rewards_path)
        summary["source_sha256"][str(rewards_path.relative_to(root))] = file_hash(rewards_path)

    clips_path = directory / "clips.csv"
    classes_path = directory / "classes.csv"
    if not clips_path.exists() or not classes_path.exists():
        summary["errors"].append("Missing clips.csv or classes.csv; final evaluation is incomplete.")
        return summary
    clips = read_csv(clips_path, ("case_id", "role", "status"))
    classes = read_csv(classes_path, ("case_id", "class_name", "score", "query_count"))
    for path in (clips_path, classes_path):
        summary["source_sha256"][str(path.relative_to(root))] = file_hash(path)
    expected_ids = {case["case_id"] for case in cases}
    final_clips = {}
    for clip in clips:
        if clip["role"] != "evaluation":
            continue
        case_id = clip["case_id"]
        if case_id not in expected_ids or case_id in final_clips:
            raise ValueError("Unexpected or duplicate final case: " + method + "/" + case_id)
        final_clips[case_id] = clip
    class_rows = {}
    for row in classes:
        case_id = row["case_id"]
        if row.get("role", "evaluation") != "evaluation":
            continue
        if case_id not in expected_ids or row["class_name"] not in expected_classes:
            continue
        key = (case_id, row["class_name"])
        if key in class_rows:
            raise ValueError("Duplicate final class row: " + str(key))
        score = number(row["score"], "score", 0, 1)
        count = integer(row["query_count"], "query_count")
        if count == 0 and score != 0:
            raise ValueError("A missing expected class received nonzero credit: " + str(key))
        class_rows[key] = dict(row, score=score, query_count=count)

    complete_ids = set()
    for case in cases:
        case_id = case["case_id"]
        clip = final_clips.get(case_id, {})
        if clip:
            for field in ("rgb_seed", "semantic_seed", "action_seed"):
                if field in case and integer(clip.get(field), field) != case[field]:
                    raise ValueError("Final clip differs from its planned " + field + ": " + case_id)
            generation_path = directory / "cases" / case_id / "generation.json"
            if not generation_path.is_file():
                raise ValueError("Missing final generation record; planned prompt/scene cannot be verified: " + case_id)
            generation = read_json(generation_path).get("case", {})
            for field in ("case_id", "scene", "prompt", "expected_classes", "rgb_seed", "semantic_seed", "action_seed"):
                if field in case and generation.get(field) != case[field]:
                    raise ValueError("Generation record differs from planned " + field + ": " + case_id)
            summary["source_sha256"][str(generation_path.relative_to(root))] = file_hash(generation_path)
        required = case["expected_classes"]
        complete = clip.get("status") == "complete" and all((case_id, name) in class_rows for name in required)
        row = {"case_id": case_id, "scene": case.get("scene"), "status": "complete" if complete else clip.get("status", "missing"),
               "score": None, "expected_classes": list(required)}
        if clip.get("status") == "complete" and not complete:
            row["status"] = "missing_class_measurement"
        if complete:
            complete_ids.add(case_id)
            row["score"] = average([class_rows[(case_id, name)]["score"] for name in required])
            row["classes"] = {name: class_rows[(case_id, name)]["score"] for name in required}
        summary["per_clip"].append(row)

    for class_name in expected_classes:
        required_ids = [case["case_id"] for case in cases if class_name in case["expected_classes"]]
        rows = [class_rows[(case_id, class_name)] for case_id in required_ids if case_id in complete_ids]
        complete = bool(required_ids) and len(rows) == len(required_ids)
        scores = [row["score"] for row in rows]
        entry = {"class_name": class_name, "expected_clips": len(required_ids), "completed_expected_clips": len(rows),
                 "complete": complete, "score": average(scores) if complete else None,
                 "partial_mean_diagnostic": average(scores), "no_query_clips": sum(row["query_count"] == 0 for row in rows)}
        for field in COUNT_FIELDS:
            values = [integer(row[field], field) for row in rows if row.get(field) not in (None, "")]
            entry[field] = sum(values) if values else None
            entry[field + "_measured_clips"] = len(values)
        for field in DIAGNOSTICS:
            values = [number(row[field], field) for row in rows if row.get(field) not in (None, "")]
            entry["mean_clip_" + field] = average(values)
            entry[field + "_measured_clips"] = len(values)
        summary["per_class"].append(entry)
    summary["completed_clips"] = len(complete_ids)
    complete = len(complete_ids) == len(cases) and all(row["complete"] for row in summary["per_class"])
    summary["evaluation_status"] = "complete" if complete else "incomplete"
    if complete:
        summary["status"] = "complete" if summary["execution_status"] == "complete" else "incomplete"
        summary["overall_class_balanced_score"] = average([row["score"] for row in summary["per_class"]])
    # New campaign sentinels are report-only measurements, not a newly calibrated
    # acceptance test. Their detailed numerical files remain available per case.
    sentinel_rows = [row for row in clips if row["role"] == "sentinel"]
    summary["sentinel_review"] = {
        "status": "measurements_only", "measured_clips": len(sentinel_rows),
        "calibrated_rejection_thresholds": False, "used_to_accept_updates": False,
        "reason": "This campaign does not recalibrate sentinel rejection thresholds; inspect numerical measurements and warnings.",
    }
    return summary


def build_report(output_root, cases, expected_classes=CLASS_NAMES):
    root = Path(output_root)
    expected_classes = tuple(expected_classes)
    if not expected_classes or len(set(expected_classes)) != len(expected_classes):
        raise ValueError("Expected classes must be a nonempty unique list.")
    ids = [case["case_id"] for case in cases]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Final evaluation case IDs must be nonempty and unique.")
    for case in cases:
        classes = case["expected_classes"]
        if not classes or len(set(classes)) != len(classes) or not set(classes) <= set(expected_classes):
            raise ValueError("Invalid fixed expected classes for " + case["case_id"])
    if set(expected_classes) != set().union(*(set(case["expected_classes"]) for case in cases)):
        raise ValueError("Every scored class must have at least one planned expected clip.")
    methods = {}
    for method in RGB_METHODS:
        try:
            methods[method] = summarize_method(root, method, cases, expected_classes)
        except (OSError, ValueError, TypeError, KeyError) as error:
            methods[method] = {"method": method, "status": "invalid", "overall_class_balanced_score": None,
                               "completed_clips": 0, "planned_clips": len(cases), "errors": [str(error)],
                               "per_class": [], "per_clip": [], "source_sha256": {}}
    paired = []
    pairs = [(name, "baseline") for name in RGB_METHODS if name != "baseline"]
    pairs += [("rl2", "rl1"), ("rl4", "rl3"), ("rl5", "rl4"), ("rl7", "rl6"), ("rl4_repeat", "rl3_repeat"),
              ("rl9", "rl4"), ("rl10", "rl4")]
    for candidate, reference in pairs:
        first = methods[candidate]
        second = methods[reference]
        first_cases = {row["case_id"]: row for row in first["per_clip"]}
        second_cases = {row["case_id"]: row for row in second["per_clip"]}
        first_fingerprint = first.get("comparison_fingerprint")
        second_fingerprint = second.get("comparison_fingerprint")
        verified = bool(first_fingerprint) and first_fingerprint == second_fingerprint
        for case_id in ids:
            left = first_cases.get(case_id, {})
            right = second_cases.get(case_id, {})
            measured = left.get("score") is not None and right.get("score") is not None
            paired.append({"candidate": candidate, "reference": reference, "case_id": case_id,
                           "paired": verified and measured, "comparison_verified": verified,
                           "reason": None if verified and measured else "unverified_panel" if not verified else "missing_measurement",
                           "difference": left["score"] - right["score"] if verified and measured else None})
    inventory = {}
    for method in methods.values():
        inventory.update(method["source_sha256"])
    semantic = {}
    selections = {}
    for name in (*SEMANTIC_JOBS, "choose_fusion", "choose_observer"):
        filename = "selection.json" if name.startswith("choose_") else "status.json"
        path = root / name / filename
        target = selections if name.startswith("choose_") else semantic
        if not path.exists():
            target[name] = {"status": "missing"}
            continue
        try:
            target[name] = read_json(path)
            inventory[str(path.relative_to(root))] = file_hash(path)
        except (OSError, ValueError) as error:
            target[name] = {"status": "invalid", "error": str(error)}
    if any(file_hash(root / relative) != digest for relative, digest in inventory.items()):
        raise RuntimeError("Numerical inputs changed during reporting; wait for writers to finish.")
    complete = all(row["status"] == "complete" for row in methods.values())
    complete = complete and all(row.get("status") == "complete" for row in [*semantic.values(), *selections.values()])
    return {"format": "rgb_geometry_campaign_comparison_v1",
            "status": "complete" if complete else "incomplete", "semantic": semantic, "selections": selections,
            "methods": methods, "paired_clip_differences": paired, "expected_classes": list(expected_classes),
            "planned_case_ids": ids, "source_sha256": inventory, "limitations": LIMITATIONS,
            "geometry_certified": False, "geometry_accuracy": None, "ranking": None,
            "score_definition": "Equal mean of fixed class means, each over that class's predefined expected clips.",
            "difference_units": "fractions; multiply by 100 for percentage points"}


def write_report(output_root, cases, expected_classes=CLASS_NAMES):
    result = build_report(output_root, cases, expected_classes)
    directory = Path(output_root) / "report"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "comparison.json", result)
    methods = list(result["methods"].values())
    write_csv(directory / "methods.csv", methods,
              ["method", "status", "execution_status", "evaluation_status", "completed_clips", "planned_clips", "overall_class_balanced_score",
               "attempted_updates", "effective_policy_updates", "retained_updates", "trained_result", "quality_warning_count",
               "trainable_parameters", "rounds", "seed", "elapsed_seconds"])
    classes = []
    for method in methods:
        classes.extend(dict(row, method=method["method"]) for row in method["per_class"])
    write_csv(directory / "classes.csv", classes,
              ["method", "class_name", "score", "complete", "expected_clips", "completed_expected_clips", "no_query_clips",
               "partial_mean_diagnostic", *COUNT_FIELDS, *("mean_clip_" + field for field in DIAGNOSTICS)])
    write_csv(directory / "paired.csv", result["paired_clip_differences"],
              ["candidate", "reference", "case_id", "paired", "comparison_verified", "reason", "difference"])
    semantic = [dict(row, experiment=name) for name, row in result["semantic"].items()]
    frozen_e4 = result["semantic"].get("m1", {}).get("unadapted_e4_reference")
    if isinstance(frozen_e4, dict):
        semantic.append(dict(frozen_e4, experiment="e4_frozen_reference"))
    write_csv(directory / "semantic.csv", semantic,
              ["experiment", "status", "development_score", "confirmation_score", "updates",
               "trainable_parameters", "model_digest", "comparison_fingerprint"])
    return result
