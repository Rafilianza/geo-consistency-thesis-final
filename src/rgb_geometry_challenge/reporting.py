"""Verify the new video panel and prepare separate visual and numerical reviews.

This module reads saved outputs only. It never loads a research model, trains,
generates a video, or changes either source campaign.
"""

import html
import json
import math
import random
import subprocess
from pathlib import Path
from urllib.parse import quote

from rgb_geometry_campaign import reporting as shared
from rgb_geometry_qualitative import artifacts
from rgb_geometry_qualitative import reporting as original_report
from rgb_geometry_qualitative_lora.reporting import LABELS
from . import protocol


REVIEW_FIELDS = (
    "review_id", "case_id", "prompt_adherence", "visible_object_count",
    "requested_speed_observed", "object_shape", "object_deformation_over_time",
    "merging_or_splitting", "scene_structure", "perspective", "occlusion",
    "unexplained_appearance_or_disappearance", "blur_or_texture", "affected_object",
    "first_frame", "last_frame", "observation", "uncertainty", "comparison_to_baseline",
)
LIMITATIONS = [
    "This is a new inference-only challenge panel using existing final checkpoints; no weights are updated.",
    "Difficulty, crowding and speed are prompt requests. Their actual presence must be checked in the videos.",
    "Tracked-class persistence and semantic agreement are proxies, not measurements of 3D geometry accuracy.",
    "Same-class objects can merge while retaining their class labels and receiving high persistence scores.",
    "MP4s are lossy previews; numerical scores use the original decoded RGB pixels before encoding.",
    "Point records permit arithmetic verification, not independent verification that YOLO labels or CoTracker trajectories are correct.",
    "Two generation seeds do not constitute two independent training seeds. Existing methods have different learning rates and update counts.",
    "This panel contains different scenes from the original study. Compare variants within this panel; its overall score is not a controlled before-and-after comparison with the older panel.",
    "Review all cases before selecting examples. Preserve ambiguous cases, failures and ties instead of forcing a winner.",
    "Anonymous review labels hide immediate method names, but filenames and the mapping can reveal them; this is not a formal blinded human study.",
    "Prompt adherence and qualitative observations are initially unreviewed. No automated score in this report establishes visual success.",
]


def close_number(actual, expected, name):
    actual = shared.number(actual, name)
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(name + " differs from its saved point counts.")


def verify_point_scores(directory, case):
    """Recalculate point, class and clip scores without running the judges again."""
    tracking = directory / "tracking"
    metrics = shared.read_json(tracking / "metrics.json")
    points = shared.read_csv(tracking / "points.csv", ("query_index",))
    class_records = metrics.get("per_class", [])
    if ([row.get("class_name") for row in class_records] != list(shared.CLASS_NAMES)
            or metrics.get("anchor_frames") != [0, 16, 32, 48, 64]
            or metrics.get("evaluation") is not True
            or metrics.get("expected_class_names") != case["expected_classes"]):
        raise ValueError("Tracking classes, anchors or evaluation case differ from the plan.")
    grouped = {name: [] for name in shared.CLASS_NAMES}
    query_ids = set()
    for point in points:
        query_id = shared.integer(point.get("query_index"), "query_index")
        class_name = point.get("class_name")
        if query_id in query_ids or class_name not in grouped:
            raise ValueError("Duplicate point index or unsupported point class.")
        query_ids.add(query_id)
        if shared.integer(point.get("class_id"), "class_id") != shared.CLASS_NAMES.index(class_name) + 1:
            raise ValueError("Point class name and class ID disagree.")
        anchor = shared.integer(point.get("anchor_frame"), "anchor_frame")
        if anchor not in (0, 16, 32, 48, 64):
            raise ValueError("Unexpected point anchor frame.")
        possible = shared.integer(point.get("possible_observation_count"), "possible observations")
        visible = shared.integer(point.get("visible_observation_count"), "visible observations")
        matching = shared.integer(point.get("matching_observation_count"), "matching observations")
        unknown = shared.integer(point.get("unknown_observation_count"), "unknown observations")
        if possible != 80 - anchor or not 0 <= matching <= visible <= possible or unknown > visible:
            raise ValueError("Point counts do not exclude the query frame correctly.")
        histogram = [shared.integer(point.get("observed_" + name), "observed " + name)
                     for name in shared.CLASS_NAMES]
        if sum(histogram) + unknown != visible or histogram[shared.CLASS_NAMES.index(class_name)] != matching:
            raise ValueError("Observed-label histogram does not match point counts.")
        close_number(point.get("score"), matching / visible if visible else 0.0, "Point score")
        grouped[class_name].append(point)
    if query_ids != set(range(len(points))):
        raise ValueError("Point indices are not a complete consecutive sequence.")
    scores = {}
    for record in class_records:
        name = record["class_name"]
        selected = grouped[name]
        for anchor in (0, 16, 32, 48, 64):
            if sum(int(point["anchor_frame"]) == anchor for point in selected) > 16:
                raise ValueError("More than sixteen points for a class and anchor.")
        score = sum(float(point["score"]) for point in selected) / len(selected) if selected else 0.0
        close_number(record.get("score"), score, name + " score")
        if record.get("query_count") != len(selected):
            raise ValueError("Class query count differs from point records.")
        for field in ("visible_observation_count", "matching_observation_count", "unknown_observation_count"):
            if record.get(field) != sum(int(point[field]) for point in selected):
                raise ValueError("Class observation count differs from point records: " + field)
        scores[name] = score
    classes_csv = shared.read_csv(tracking / "classes.csv", ("class_name", "score", "query_count"))
    if [row["class_name"] for row in classes_csv] != list(shared.CLASS_NAMES):
        raise ValueError("Tracking classes.csv does not contain the ten ordered classes.")
    for row in classes_csv:
        close_number(row["score"], scores[row["class_name"]], "Tracking CSV class score")
        if int(row["query_count"]) != len(grouped[row["class_name"]]):
            raise ValueError("Tracking CSV query count differs from point records.")
    reward = sum(scores[name] for name in case["expected_classes"]) / len(case["expected_classes"])
    for field in ("broader_reward", "reward", "overall_score"):
        close_number(metrics.get(field), reward, field)
    close_number(metrics.get("car_reward"), scores["car"], "car_reward")
    return {"verified_points": len(points), "clip_reward": reward, "class_scores": scores}


def noise_signature(statistics, case):
    """Compare actual sampled initial states and action noise across variants."""
    if (statistics.get("semantic_steps") != 50 or statistics.get("action_steps") != [10, 20, 30, 40]
            or statistics.get("sampler") != "euler_four_gaussian_actions"
            or statistics.get("guidance_scale") != 5.0):
        raise ValueError("Generation did not use the fixed matched sampler.")
    seeds = {name: case[name + "_seed"] for name in ("rgb", "semantic", "action")}
    if statistics.get("seeds") != seeds:
        raise ValueError("Generation seeds differ from the planned case.")
    initial = statistics.get("initial_latent_sha256", {})
    if set(initial) != {"rgb", "semantic"} or any(len(value) != 64 for value in initial.values()):
        raise ValueError("Missing initial RGB or semantic latent hash.")
    ledger = statistics.get("ledger", [])
    if [row.get("step_index") for row in ledger] != list(range(50)):
        raise ValueError("The generation ledger must contain all fifty steps.")
    actions = [row for row in ledger if row.get("stochastic")]
    if ([row["step_index"] for row in actions] != [10, 20, 30, 40]
            or any(not isinstance(row.get("noise_sha256"), str) or len(row["noise_sha256"]) != 64 for row in actions)):
        raise ValueError("Missing or unexpected action-noise hashes.")
    sigmas = statistics.get("sigmas", [])
    timesteps = statistics.get("timesteps", [])
    if (len(sigmas) != 51 or len(timesteps) != 50
            or not all(math.isfinite(float(value)) for value in sigmas + timesteps)
            or sigmas[-1] != 0 or any(left <= right for left, right in zip(sigmas, sigmas[1:]))):
        raise ValueError("Missing or invalid fifty-step scheduler clock.")
    return {"initial_latents": initial, "action_noise": [row["noise_sha256"] for row in actions],
            "sigmas": sigmas, "timesteps": timesteps}


def verify_arm(root, variant, config, input_identity):
    status = shared.read_json(root / variant / "status.json")
    manifest = shared.read_json(root / variant / "manifest.json")
    if status.get("status") != "complete":
        raise ValueError("Variant has not completed successfully.")
    if (status.get("comparison_fingerprint") != protocol.fingerprint(config)
            or status.get("source_fingerprint") != protocol.fingerprint(input_identity["sources"])):
        raise ValueError("Variant source or panel differs from the submitted campaign.")
    if (status.get("inference_only") is not True or status.get("optimizer_steps") != 0
            or not status.get("initial_policy_digest")
            or status["initial_policy_digest"] != status.get("final_policy_digest")):
        raise ValueError("Inference must preserve the loaded policy and perform zero optimiser updates.")
    specification = protocol.experiment_spec(variant)
    if (manifest.get("config") != config or manifest.get("spec") != specification
            or manifest.get("inference_only") is not True
            or manifest.get("comparison_fingerprint") != protocol.fingerprint(config)
            or manifest.get("source_fingerprint") != protocol.fingerprint(input_identity["sources"])):
        raise ValueError("Variant manifest differs from the fixed inference configuration or sources.")
    for field in ("experiment", "method", "arm", "condition"):
        if status.get(field) != specification[field]:
            raise ValueError("Variant label differs from the planned " + field + ".")
    expected_identity = dict(input_identity, experiment=variant, training_spec=specification)
    if manifest.get("identity") != expected_identity:
        raise ValueError("Variant manifest identity differs from the submitted input identity.")
    origin = input_identity["checkpoint_origins"][variant]
    if (status.get("checkpoint_origin") != origin
            or status["initial_policy_digest"] != origin["policy_digest"]
            or shared.read_json(root / variant / "checkpoint_origin.json") != origin):
        raise ValueError("Loaded policy differs from its original checkpoint provenance.")
    if (status.get("training_rounds") != 0 or status.get("learning_clips") != 0
            or status.get("policy_parameters_frozen") is not True
            or status.get("retained_videos") != len(protocol.cases(config, "evaluation"))):
        raise ValueError("Inference completion record does not match the planned video count and frozen policy.")
    return status


def verify_generation_policy(statistics, specification, origin):
    """Check that the saved policy was applied only at its four registered steps."""
    arm = specification["arm"]
    expected_version = origin.get("parameter_version", 0)
    if (statistics.get("arm") != arm or statistics.get("outer_gain") != 0.25
            or statistics.get("rms_cap") != 0.25
            or statistics.get("parameter_version") != expected_version):
        raise ValueError("Generation arm, correction bound or parameter version differs from its checkpoint.")
    for row in statistics["ledger"]:
        is_action = row["step_index"] in (10, 20, 30, 40)
        if (row.get("adapted") != (is_action and arm != "B")
                or row.get("lora_enabled") != (is_action and arm == "L")
                or row.get("base_conditional_captures") != 1
                or row.get("semantic_advances") != 1):
            raise ValueError("Generation ledger does not preserve the registered adaptation and observer steps.")


def gallery(cases, videos, anonymous=False):
    title = "Challenge videos: anonymous review" if anonymous else "Challenge videos: all ten variants"
    content = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
               "<meta name='viewport' content='width=device-width,initial-scale=1'>",
               "<title>" + title + "</title>",
               "<style>body{font:16px system-ui;max-width:1550px;margin:24px auto;padding:0 20px}"
               ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:14px}"
               "article{border:1px solid #ccc;padding:12px;border-radius:8px}video{width:100%;background:#111}"
               "section{margin:32px 0}p{line-height:1.5}</style><body><h1>" + title + "</h1>",
               "<p>Check prompt adherence, object shape, deformation, merging, scene structure, perspective and occlusion. "
               "Separate initial malformed objects from changes over time, and blur from structural failures. "
               "Record frame ranges, uncertainty and whether ordinary occlusion or leaving the image explains an event.</p>",
               "<p>Use annotation_template.csv. Review the videos before opening numerical scores. Crowding and speed are "
               "requested conditions, not confirmed properties. These observations do not establish 3D geometry accuracy.</p>",
               "<p>Anonymous labels can be decoded from mapping.csv or video URLs; this is not a formal blinded study.</p>"]
    for case in cases:
        content.append("<section><h2>" + html.escape(case["case_id"]) + "</h2><p>"
                       + html.escape(case["prompt"]) + "</p><div class='grid'>")
        rows = [row for row in videos if row["case_id"] == case["case_id"]]
        if anonymous:
            rows.sort(key=lambda row: row["review_id"])
        for row in rows:
            label = row["review_id"] if anonymous else LABELS[row["method"]]
            content.append("<article><h3>" + html.escape(label) + "</h3>")
            if row["status"] == "complete":
                source = html.escape("../" + quote(row["video_path"], safe="/"), quote=True)
                content.append("<video controls preload='none' src='" + source + "'></video>")
            else:
                content.append("<p>Video unavailable or verification incomplete. See video_index.csv.</p>")
            content.append("</article>")
        content.append("</div></section>")
    return "\n".join(content + ["</body></html>\n"])


def write_report(args, config):
    """Check all saved files and write a fresh report, including incomplete runs."""
    root = Path(args.output_root)
    destination = root / "report"
    if destination.exists() and any(path.name != "execution_log.csv" for path in destination.iterdir()):
        raise FileExistsError("Preserving an existing report: " + str(destination))
    cases = protocol.cases(config, "evaluation")
    input_identity = shared.read_json(root / "input_check.json")["identity"]
    variants = protocol.VARIANTS
    methods, videos, point_checks, noise_by_case = {}, [], [], {}
    for variant in variants:
        status, arm_error = {}, ""
        recorded_clips, recorded_rewards = {}, {}
        try:
            status = verify_arm(root, variant, config, input_identity)
            result = shared.summarize_method(root, variant, cases, shared.CLASS_NAMES)
            clip_rows = shared.read_csv(root / variant / "clips.csv", ("case_id", "reward", "car_reward"))
            reward_rows = shared.read_csv(root / variant / "evaluation_rewards.csv", ("case_id", "role", "reward", "car_reward"))
            recorded_clips = {row["case_id"]: row for row in clip_rows}
            recorded_rewards = {row["case_id"]: row for row in reward_rows}
            expected_ids = {case["case_id"] for case in cases}
            if (len(clip_rows) != len(cases) or len(reward_rows) != len(cases)
                    or set(recorded_clips) != expected_ids or set(recorded_rewards) != expected_ids
                    or any(row["role"] != "evaluation" for row in reward_rows)):
                raise ValueError("Expected exactly one evaluation clip and reward row per planned case.")
        except (OSError, ValueError, TypeError, KeyError) as error:
            arm_error = str(error)
            result = {"method": variant, "status": "incomplete", "execution_status": status.get("status", "unknown"),
                      "completed_clips": 0, "overall_class_balanced_score": None, "per_class": [],
                      "per_clip": [], "errors": [arm_error]}
        clip_scores = {row["case_id"]: row for row in result["per_clip"]}
        for case_index, case in enumerate(cases, start=1):
            row = original_report.video_record(root, variant, case, status, config,
                    clip_scores.get(case["case_id"], {}).get("classes", {}))
            video_path = root / row["video_path"]
            row.update(source_id=f"C{case_index:03d}-{variant}", rgb_seed=case["rgb_seed"],
                       semantic_seed=case["semantic_seed"], action_seed=case["action_seed"],
                       prompt=case["prompt"], frames=81, height=480, width=832, fps=16,
                       original_execution_path=str(video_path.absolute()),
                       original_m3_path=str(video_path.absolute()) if str(video_path.absolute()).startswith("/fs04/") else "",
                       clip_reward=None, geometry_certified=False,
                       category=case.get("category"), family=case.get("family"), condition=case.get("condition"),
                       checkpoint_path=input_identity.get("checkpoint_origins", {}).get(variant, {}).get("path"),
                       checkpoint_sha256=input_identity.get("checkpoint_origins", {}).get(variant, {}).get("sha256"))
            if arm_error:
                row.update(status="invalid", error=arm_error)
            if row["status"] == "complete":
                try:
                    directory = root / variant / "cases" / case["case_id"]
                    checks = verify_point_scores(directory, case)
                    close_number(clip_scores[case["case_id"]]["score"], checks["clip_reward"], "Summary clip score")
                    for table in (recorded_clips, recorded_rewards):
                        close_number(table[case["case_id"]]["reward"], checks["clip_reward"], "Recorded clip reward")
                        close_number(table[case["case_id"]]["car_reward"], checks["class_scores"]["car"], "Recorded car reward")
                    statistics = shared.read_json(directory / "generation.json")["statistics"]
                    origin = input_identity["checkpoint_origins"][variant]
                    if (statistics.get("inference_only") is not True or statistics.get("training") is not False
                            or statistics.get("checkpoint_sha256") != origin["sha256"]):
                        raise ValueError("Generation record differs from its frozen checkpoint or inference mode.")
                    signature = noise_signature(statistics, case)
                    verify_generation_policy(statistics, protocol.experiment_spec(variant), origin)
                    known = noise_by_case.setdefault(case["case_id"], signature)
                    if signature != known:
                        raise ValueError("Initial states, action noise or schedules differ across variants.")
                    artifacts._verify_video(video_path, (81, 480, 832, 3), 16)
                    point_checks.append(dict(method=variant, case_id=case["case_id"], **checks))
                    row["clip_reward"] = checks["clip_reward"]
                except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                    row.update(status="invalid", error=str(error))
            videos.append(row)
        arm_videos = [row for row in videos if row["method"] == variant]
        result["verified_videos"] = sum(row["status"] == "complete" for row in arm_videos)
        if result["verified_videos"] != len(cases):
            result["status"] = "incomplete"
            result["overall_class_balanced_score"] = None
        methods[variant] = result

    # Keep observation forms free of reward scores and method names.
    randomizer = random.Random(20261022)
    annotations, mapping = [], []
    for case_index, case in enumerate(cases, start=1):
        selected = [row for row in videos if row["case_id"] == case["case_id"]]
        randomizer.shuffle(selected)
        for position, row in enumerate(selected, start=1):
            row["review_id"] = f"case{case_index:03d}_clip{position:02d}"
            entry = {field: "" for field in REVIEW_FIELDS}
            entry.update(review_id=row["review_id"], case_id=row["case_id"])
            annotations.append(entry)
            mapping.append({key: row[key] for key in ("review_id", "source_id", "case_id", "method", "video_path")})
    complete = (all(result["status"] == "complete" for result in methods.values())
                and len(videos) == 800 and len(point_checks) == 800)
    summary = {"format": "rgb_geometry_challenge_report_v1", "status": "complete" if complete else "incomplete",
               "planned_videos": len(cases) * len(variants), "verified_videos": sum(row["status"] == "complete" for row in videos),
               "video_bytes": sum(row["bytes"] for row in videos if row["status"] == "complete"),
               "matched_inputs_and_action_noise": complete, "geometry_certified": False,
               "inference_only": True, "manual_review_status": "not_started", "methods": methods,
               "videos": videos, "limitations": LIMITATIONS,
               "arithmetic_checked_points": sum(row["verified_points"] for row in point_checks),
               "metric_definition": "Mean of ten class means; each class uses its predeclared expected cases."
               " A clip reward instead averages its own expected classes."}
    destination.mkdir(exist_ok=True)
    shared.write_json(destination / "summary.json", summary)
    shared.write_json(destination / "point_score_checks.json", {"cases": point_checks,
                      "meaning": "Arithmetic agreement only; labels and track correctness are not independently verified."})
    shared.write_csv(destination / "video_index.csv", videos, tuple(videos[0]))
    shared.write_csv(destination / "methods.csv", list(methods.values()),
                     ("method", "status", "execution_status", "completed_clips", "verified_videos", "overall_class_balanced_score"))
    class_rows = [dict(row, method=method) for method, result in methods.items() for row in result["per_class"]]
    shared.write_csv(destination / "per_class.csv", class_rows,
                     ("method", "class_name", "expected_clips", "completed_expected_clips", "complete", "score", "query_count"))
    shared.write_csv(destination / "annotation_template.csv", annotations, REVIEW_FIELDS)
    shared.write_csv(destination / "mapping.csv", mapping, tuple(mapping[0]))
    for name, anonymous in (("video_index.html", False), ("review.html", True)):
        with (destination / name).open("x") as handle:
            handle.write(gallery(cases, videos, anonymous))
    return {"status": summary["status"], "planned_videos": summary["planned_videos"],
            "verified_videos": summary["verified_videos"], "video_bytes": summary["video_bytes"],
            "geometry_certified": False, "inference_only": True,
            "matched_inputs_and_action_noise": complete, "report_directory": str(destination),
            "summary_path": str(destination / "summary.json")}
