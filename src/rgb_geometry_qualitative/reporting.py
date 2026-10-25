"""Build portable video galleries and fresh numerical summaries without ML imports."""

import html
import random
from pathlib import Path
from urllib.parse import quote

from rgb_geometry_campaign import reporting as shared
from . import protocol


METHODS = ("baseline", "rloo_R", "rloo_S", "flow_grpo_R", "flow_grpo_S", "flash_grpo_R", "flash_grpo_S")
LABELS = {"baseline": "Frozen baseline", "rloo_R": "RLOO RGB", "rloo_S": "RLOO semantic",
          "flow_grpo_R": "Flow-GRPO RGB", "flow_grpo_S": "Flow-GRPO semantic",
          "flash_grpo_R": "Flash-GRPO RGB", "flash_grpo_S": "Flash-GRPO semantic"}
REVIEW_SEED = 20261012
ANNOTATION_FIELDS = ("review_id", "case_id", "shape", "object_separation", "layout", "perspective",
                     "occlusion", "motion", "missing_object", "timestamp_seconds", "notes")
LIMITATIONS = [
    "These are new reproduction runs, not recovered outputs from the historical campaigns.",
    "The previously inspected scenes form a retrospective qualitative case study, not an unseen confirmation test.",
    "Tracked-class persistence and segmentation agreement are proxies, not geometry accuracy.",
    "MP4 encoding is lossy; numerical scores use the original RGB pixels before encoding.",
    "Anonymous labels reduce immediate method cues, but video URLs and mapping.csv can reveal methods; this is not a formal blinded study.",
    "Review all cases before choosing illustrative improvements, regressions and ties; missing outputs are not zero scores.",
]


def video_record(root, method, case, status, config, clip_scores):
    """Link the encoded video to its policy, original pixels and scoring receipts."""
    case_id = case["case_id"]
    directory = root / method / "cases" / case_id
    video_path = root / method / "videos" / (case_id + ".mp4")
    receipt_path = video_path.with_suffix(".json")
    row = {"method": method, "case_id": case_id, "scene": case["scene"], "status": "missing",
           "video_path": str(video_path.relative_to(root)), "bytes": 0, "error": "",
           "policy_digest": None, "video_sha256": None, "source_rgb_sha256": None}
    required = [video_path, receipt_path, directory / "generation.json", directory / "rgb_scores.json"]
    required += [directory / "tracking" / name for name in ("complete.json", "metrics.json", "classes.csv", "points.csv")]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        row["error"] = "Missing: " + ", ".join(missing)
        return row
    try:
        receipt = shared.read_json(receipt_path)
        if receipt.get("format") != "rgb_geometry_qualitative_video_v1":
            raise ValueError("Unknown video receipt format.")
        generation = shared.read_json(directory / "generation.json")
        statistics = generation["statistics"]
        policy_digest = status.get("final_policy_digest")
        if not policy_digest or receipt.get("policy_digest") != policy_digest or statistics.get("policy_digest") != policy_digest:
            raise ValueError("Video and generation must use the recorded final policy.")
        if receipt.get("case_id") != case_id or receipt.get("case") != case or generation.get("case") != case:
            raise ValueError("Video or generation differs from its planned case.")
        if not status.get("source_fingerprint") or statistics.get("source_fingerprint") != status["source_fingerprint"]:
            raise ValueError("Generation source differs from the recorded arm source.")
        for field in ("frames", "height", "width", "fps"):
            if receipt.get(field) != config["settings"][field]:
                raise ValueError("Video receipt differs from the planned " + field + ".")
        if receipt.get("video_sha256") != shared.file_hash(video_path) or video_path.stat().st_size == 0:
            raise ValueError("Video is empty or fails its SHA-256 check.")
        original_hash = shared.read_json(directory / "rgb_scores.json").get("rgb_uint8_sha256")
        if not original_hash or receipt.get("source_rgb_sha256") != original_hash:
            raise ValueError("Video source pixels differ from the scored RGB pixels.")
        tracking = directory / "tracking"
        marker = shared.read_json(tracking / "complete.json")
        artifacts = marker.get("artifacts", {})
        if marker.get("status") != "complete" or set(artifacts) != {"metrics.json", "classes.csv", "points.csv"}:
            raise ValueError("Tracking completion receipt is incomplete.")
        for filename, expected_hash in artifacts.items():
            if shared.file_hash(tracking / filename) != expected_hash:
                raise ValueError("Tracking artifact fails its SHA-256 check: " + filename)
        raw_classes = shared.read_json(tracking / "metrics.json").get("per_class", [])
        scores = {entry["class_name"]: entry["score"] for entry in raw_classes}
        if any(scores.get(name) != value for name, value in clip_scores.items()):
            raise ValueError("Summary class scores differ from the raw tracking result.")
        row.update(status="complete", bytes=video_path.stat().st_size, policy_digest=policy_digest,
                   video_sha256=receipt["video_sha256"], source_rgb_sha256=original_hash)
    except (OSError, ValueError, TypeError, KeyError) as error:
        row.update(status="invalid", error=str(error))
    return row


def gallery(cases, videos, review=False):
    """Use relative URLs so copying the whole output folder keeps videos playable."""
    by_case = {case["case_id"]: [] for case in cases}
    for row in videos:
        by_case[row["case_id"]].append(row)
    title = "Anonymous qualitative review" if review else "Qualitative video index"
    content = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
               "<meta name='viewport' content='width=device-width,initial-scale=1'>",
               "<title>" + title + "</title>",
               "<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#17212b}"
               ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}"
               "article{border:1px solid #ccd3db;border-radius:8px;padding:12px}video{width:100%;background:#111}"
               "section{margin:32px 0}h2{font-size:20px}p{line-height:1.5}.warning{color:#934314}</style>",
               "<body><h1>" + title + "</h1>",
               "<p>New reproduction runs on a previously inspected panel. Review RGB videos for visible shape and scene failures; "
               "these observations do not establish geometry accuracy. Numerical scores use original pixels before lossy video encoding.</p>"]
    if review:
        content.append("<p>Record observations in annotation_template.csv using the review ID. For shape, object separation, layout, "
                       "perspective and occlusion use: no visible failure / mild failure / clear failure / cannot assess. "
                       "For motion note near-static / moving / cannot assess; record missing objects separately, with timestamps and notes. "
                       "Watch every case before selecting examples.</p><p>Labels are anonymous; filenames and the separate mapping.csv can reveal methods. "
                       "This is not a formal blinded study.</p>")
    else:
        content.append("<p><a href='review.html'>Anonymous review</a> · <a href='methods.csv'>Fresh numerical summary</a> · "
                       "<a href='annotation_template.csv'>Annotation template</a></p>")
    for case in cases:
        content.append("<section><h2>" + html.escape(case["case_id"]) + "</h2><p>" + html.escape(case["prompt"]) + "</p><div class='grid'>")
        rows = sorted(by_case[case["case_id"]], key=lambda row: row["review_id"]) if review else by_case[case["case_id"]]
        for row in rows:
            label = row["review_id"] if review else LABELS[row["method"]]
            content.append("<article><h3>" + html.escape(label) + "</h3>")
            if row["status"] == "complete":
                url = html.escape("../" + quote(row["video_path"], safe="/"), quote=True)
                content.append("<video controls preload='none' src='" + url + "'></video>")
            else:
                content.append("<p class='warning'>Video " + html.escape(row["status"]) + ". See the report after reviewing available clips.</p>")
            content.append("</article>")
        content.append("</div></section>")
    return "\n".join(content + ["</body></html>\n"])


def write_report(output_root, config):
    """Create fresh report files; allow only the launcher's existing execution log."""
    root = Path(output_root)
    destination = root / "report"
    if destination.exists() and (not destination.is_dir()
                                 or any(path.name != "execution_log.csv" for path in destination.iterdir())):
        raise FileExistsError("Refusing to overwrite an existing report: " + str(destination))
    cases = protocol.cases(config, "evaluation")
    expected_classes = [name for name in shared.CLASS_NAMES if any(name in case["expected_classes"] for case in cases)]
    methods, videos = {}, []
    for method in METHODS:
        status = {}
        try:
            result = shared.summarize_method(root, method, cases, expected_classes)
            status_path = root / method / "status.json"
            if status_path.is_file():
                status = shared.read_json(status_path)
                if status.get("comparison_fingerprint") != protocol.fingerprint(config):
                    raise ValueError("Arm configuration fingerprint differs from the report configuration.")
                specification = protocol.experiment_spec(method)
                for field in ("experiment", "method", "arm", "condition"):
                    if status.get(field) != specification[field]:
                        raise ValueError("Recorded arm differs from planned " + field + ".")
                expected_rounds = 0 if method == "baseline" else config["settings"]["rounds"]
                if status.get("status") == "complete" and status.get("rounds") != expected_rounds:
                    raise ValueError("Completed arm has not run the planned number of training rounds.")
            if result.get("execution_status") != "complete":
                result["overall_class_balanced_score"] = None
        except (OSError, ValueError, TypeError, KeyError) as error:
            result = {"method": method, "status": "invalid", "execution_status": status.get("status", "unknown"),
                      "overall_class_balanced_score": None, "completed_clips": 0, "per_class": [],
                      "per_clip": [], "errors": [str(error)]}
        clip_rows = {row["case_id"]: row for row in result["per_clip"]}
        arm_videos = [video_record(root, method, case, status, config,
                                  clip_rows.get(case["case_id"], {}).get("classes", {})) for case in cases]
        result["verified_videos"] = sum(row["status"] == "complete" for row in arm_videos)
        result["final_policy_digest"] = status.get("final_policy_digest")
        result["source_fingerprint"] = status.get("source_fingerprint")
        if result["verified_videos"] != len(cases):
            result["status"] = "invalid" if any(row["status"] == "invalid" for row in arm_videos) else "incomplete"
            result["overall_class_balanced_score"] = None
        methods[method] = result
        videos.extend(arm_videos)

    # Reproducible per-case permutations keep the visual review separate from method names.
    randomizer = random.Random(REVIEW_SEED)
    mapping, annotations = [], []
    for case_index, case in enumerate(cases, start=1):
        shuffled = [row for row in videos if row["case_id"] == case["case_id"]]
        randomizer.shuffle(shuffled)
        for position, row in enumerate(shuffled, start=1):
            row["review_id"] = f"case{case_index:02d}_clip{position}"
            mapping.append({field: row[field] for field in ("review_id", "case_id", "method", "video_path")})
            annotation = {field: "" for field in ANNOTATION_FIELDS}
            annotation.update(review_id=row["review_id"], case_id=row["case_id"])
            annotations.append(annotation)
    source_ids = {result["source_fingerprint"] for result in methods.values() if result["source_fingerprint"]}
    matched_source = len(source_ids) == 1 and all(result["source_fingerprint"] for result in methods.values())
    complete = all(result["status"] == "complete" for result in methods.values()) and matched_source
    summary = {"format": "rgb_geometry_qualitative_report_v1", "status": "complete" if complete else "incomplete",
               "study_type": "new_reproduction_retrospective_case_study", "geometry_certified": False,
               "planned_videos": len(cases) * len(METHODS), "verified_videos": sum(row["status"] == "complete" for row in videos),
               "video_bytes": sum(row["bytes"] for row in videos), "review_seed": REVIEW_SEED,
               "matched_source_identity": matched_source, "methods": methods, "videos": videos,
               "limitations": LIMITATIONS}
    method_fields = ("method", "status", "execution_status", "completed_clips", "verified_videos",
                     "overall_class_balanced_score", "final_policy_digest")
    class_rows = [dict(row, method=method, method_status=result["status"])
                  for method, result in methods.items() for row in result["per_class"]]
    class_fields = ("method", "method_status", "class_name", "expected_clips", "completed_expected_clips", "complete", "score",
                    "partial_mean_diagnostic", "no_query_clips", "query_count")
    destination.mkdir(exist_ok=True)
    shared.write_json(destination / "summary.json", summary)
    shared.write_csv(destination / "methods.csv", list(methods.values()), method_fields)
    shared.write_csv(destination / "per_class.csv", class_rows, class_fields)
    shared.write_csv(destination / "video_index.csv", videos, tuple(videos[0]))
    shared.write_csv(destination / "mapping.csv", mapping, ("review_id", "case_id", "method", "video_path"))
    shared.write_csv(destination / "annotation_template.csv", annotations, ANNOTATION_FIELDS)
    for filename, anonymous in (("video_index.html", False), ("review.html", True)):
        with (destination / filename).open("x") as handle:
            handle.write(gallery(cases, videos, review=anonymous))
    return {"status": summary["status"], "planned_videos": summary["planned_videos"],
            "verified_videos": summary["verified_videos"], "video_bytes": summary["video_bytes"],
            "matched_source_identity": matched_source, "geometry_certified": False,
            "report_directory": str(destination), "summary_path": str(destination / "summary.json")}
