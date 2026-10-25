"""Join the completed first study and new LoRA results without changing either."""

import html
import os
import random
from pathlib import Path
from urllib.parse import quote

from rgb_geometry_qualitative import reporting as first_stage
from . import protocol


shared = first_stage.shared
METHODS = ("baseline", "rloo_R", "rloo_S", "rloo_L", "flow_grpo_R", "flow_grpo_S",
           "flow_grpo_L", "flash_grpo_R", "flash_grpo_S", "flash_grpo_L")
LORA_METHODS = ("rloo_L", "flow_grpo_L", "flash_grpo_L")
LABELS = dict(first_stage.LABELS, rloo_L="RLOO LoRA", flow_grpo_L="Flow-GRPO LoRA",
              flash_grpo_L="Flash-GRPO LoRA")
LIMITATIONS = first_stage.LIMITATIONS + [
    "This report links two result folders. Keep both folders and their relative locations when copying the gallery.",
    "The original seven variants are read from phase one; only the three LoRA variants are generated in phase two.",
    "The full source fingerprints differ because phase two adds LoRA orchestration; all shared source hashes must match.",
]


def phase_identity(root, config, reference):
    """Check frozen inputs and shared code while preserving distinct phase identities."""
    identity = shared.read_json(root / "input_check.json")["identity"]
    original = reference["input_identity"]
    if identity.get("reference") != reference:
        raise ValueError("LoRA inputs do not refer to the verified first-stage results.")
    if identity.get("config_sha256") != shared.file_hash(root / "config.json"):
        raise ValueError("LoRA input identity differs from its saved configuration file.")
    if shared.read_json(root / "config.json") != config:
        raise ValueError("Saved LoRA configuration differs from the requested report configuration.")
    # Compare every recorded input, including optional fields when present. Only
    # the extension source, its configuration and reference receipt may differ.
    extension_fields = ("sources", "config_sha256", "reference")
    original_inputs = {name: value for name, value in original.items() if name not in extension_fields}
    lora_inputs = {name: value for name, value in identity.items() if name not in extension_fields}
    if lora_inputs != original_inputs:
        raise ValueError("The two phases differ in frozen input identity.")
    original_sources = original.get("sources", {})
    lora_sources = identity.get("sources", {})
    if (not original_sources or not lora_sources
            or any(lora_sources.get(name) != digest for name, digest in original_sources.items())):
        raise ValueError("The two phases differ in shared source code.")
    if protocol.fingerprint(original_sources) != reference["common_source_fingerprint"]:
        raise ValueError("First-stage common source fingerprint is inconsistent.")
    return identity


def gallery(cases, videos, review=False):
    """Video paths are relative to this report, including the phase-one folder."""
    title = "Anonymous qualitative review" if review else "Ten-variant qualitative video index"
    content = ["<!doctype html><html lang='en'><meta charset='utf-8'>",
               "<meta name='viewport' content='width=device-width,initial-scale=1'>",
               "<title>" + title + "</title>",
               "<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;color:#17212b}"
               ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}"
               "article{border:1px solid #ccd3db;border-radius:8px;padding:12px}video{width:100%;background:#111}"
               "section{margin:32px 0}h2{font-size:20px}p{line-height:1.5}.warning{color:#934314}</style>",
               "<body><h1>" + title + "</h1>",
               "<p>New reproduction runs on a previously inspected panel. Visible shape and scene failures, "
               "tracking persistence and segmentation agreement do not establish geometry accuracy. "
               "Numerical scores use original pixels before lossy MP4 encoding.</p>",
               "<p>Videos come from two result folders. Keep both folders in the same relative locations "
               "when copying this gallery; the LoRA folder alone does not contain the first-stage videos.</p>"]
    if review:
        content.append("<p>Use annotation_template.csv and the review IDs. For shape, object separation, layout, "
                       "perspective and occlusion record: no visible failure / mild failure / clear failure / cannot assess. "
                       "For motion record near-static / moving / cannot assess. Record missing objects separately, "
                       "with timestamps and notes. Review every case before selecting examples.</p>"
                       "<p>Anonymous labels reduce immediate method cues, but video URLs and mapping.csv can reveal methods. "
                       "This is a retrospective case study, not a formal blinded study.</p>")
    else:
        content.append("<p><a href='review.html'>Anonymous review</a> · <a href='methods.csv'>Fresh numerical summary</a> · "
                       "<a href='annotation_template.csv'>Annotation template</a></p>")
    for case in cases:
        content.append("<section><h2>" + html.escape(case["case_id"]) + "</h2><p>" + html.escape(case["prompt"]) + "</p><div class='grid'>")
        rows = [row for row in videos if row["case_id"] == case["case_id"]]
        if review:
            rows.sort(key=lambda row: row["review_id"])
        for row in rows:
            label = row["review_id"] if review else LABELS[row["method"]]
            content.append("<article><h3>" + html.escape(label) + "</h3>")
            if row["status"] == "complete":
                url = html.escape(quote(row["video_path"], safe="/"), quote=True)
                content.append("<video controls preload='none' src='" + url + "'></video>")
            else:
                content.append("<p class='warning'>Video " + html.escape(row["status"]) + ". See the report after reviewing available clips.</p>")
            content.append("</article>")
        content.append("</div></section>")
    return "\n".join(content + ["</body></html>\n"])


def write_report(output_root, config, reference_root):
    """Write the combined report only in a fresh phase-two report directory."""
    # Use canonical paths for both roots before computing links between them.
    root = protocol.safe_path(output_root, existing=True)
    reference_root = Path(reference_root)
    destination = root / "report"
    if destination.exists() and (not destination.is_dir()
                                 or any(path.name != "execution_log.csv" for path in destination.iterdir())):
        raise FileExistsError("Refusing to overwrite an existing report: " + str(destination))
    reference = protocol.validate_reference(reference_root)
    reference_root = Path(reference["root"])
    reference_config = protocol.reference_config(config)
    lora_identity = phase_identity(root, config, reference)
    cases = protocol.cases(config, "evaluation")
    if cases != protocol.cases(reference_config, "evaluation"):
        raise ValueError("The phases do not share the same final evaluation cases and seeds.")
    expected_classes = [name for name in shared.CLASS_NAMES if any(name in case["expected_classes"] for case in cases)]
    methods, videos = {}, []
    for method in METHODS:
        is_lora = method in LORA_METHODS
        phase_root = root if is_lora else reference_root
        phase_config = config if is_lora else reference_config
        identity = lora_identity if is_lora else reference["input_identity"]
        phase_sources = protocol.fingerprint(identity["sources"])
        status = {}
        arm_error = None
        try:
            result = shared.summarize_method(phase_root, method, cases, expected_classes)
            status_path = phase_root / method / "status.json"
            if status_path.is_file():
                status = shared.read_json(status_path)
                if status.get("comparison_fingerprint") != protocol.fingerprint(phase_config):
                    raise ValueError("Arm configuration fingerprint differs from its phase configuration.")
                specification = protocol.experiment_spec(method) if is_lora else first_stage.protocol.experiment_spec(method)
                for field in ("experiment", "method", "arm", "condition"):
                    if status.get(field) != specification[field]:
                        raise ValueError("Recorded arm differs from planned " + field + ".")
                expected_rounds = 0 if method == "baseline" else phase_config["settings"]["rounds"]
                if status.get("status") == "complete" and status.get("rounds") != expected_rounds:
                    raise ValueError("Completed arm has not run the planned number of training rounds.")
                if status.get("source_fingerprint") != phase_sources:
                    raise ValueError("Arm source fingerprint differs from its phase input identity.")
                manifest = shared.read_json(phase_root / method / "manifest.json")
                expected_identity = dict(identity, experiment=method, training_spec=specification)
                if manifest.get("identity") != expected_identity:
                    raise ValueError("Arm manifest differs from its phase input identity.")
            if result.get("execution_status") != "complete":
                result["overall_class_balanced_score"] = None
        except (OSError, ValueError, TypeError, KeyError) as error:
            arm_error = str(error)
            result = {"method": method, "status": "invalid", "execution_status": status.get("status", "unknown"),
                      "overall_class_balanced_score": None, "completed_clips": 0, "per_class": [],
                      "per_clip": [], "errors": [arm_error]}
        clip_rows = {row["case_id"]: row for row in result["per_clip"]}
        arm_videos = []
        for case in cases:
            row = first_stage.video_record(phase_root, method, case, status, phase_config,
                                           clip_rows.get(case["case_id"], {}).get("classes", {}))
            row["video_path"] = os.path.relpath(phase_root / row["video_path"], destination)
            row["phase"] = "lora" if is_lora else "baseline_and_heads"
            if arm_error:
                row.update(status="invalid", error=arm_error)
            arm_videos.append(row)
        result["verified_videos"] = sum(row["status"] == "complete" for row in arm_videos)
        result["final_policy_digest"] = status.get("final_policy_digest")
        result["source_fingerprint"] = status.get("source_fingerprint")
        result["phase"] = "lora" if is_lora else "baseline_and_heads"
        if result["verified_videos"] != len(cases):
            result["status"] = "invalid" if any(row["status"] == "invalid" for row in arm_videos) else "incomplete"
            result["overall_class_balanced_score"] = None
        methods[method] = result
        videos.extend(arm_videos)

    randomizer = random.Random(first_stage.REVIEW_SEED)
    mapping, annotations = [], []
    for case_index, case in enumerate(cases, start=1):
        shuffled = [row for row in videos if row["case_id"] == case["case_id"]]
        randomizer.shuffle(shuffled)
        for position, row in enumerate(shuffled, start=1):
            row["review_id"] = f"case{case_index:02d}_clip{position:02d}"
            mapping.append({field: row[field] for field in ("review_id", "case_id", "method", "video_path", "phase")})
            annotation = {field: "" for field in first_stage.ANNOTATION_FIELDS}
            annotation.update(review_id=row["review_id"], case_id=row["case_id"])
            annotations.append(annotation)
    complete = all(result["status"] == "complete" for result in methods.values())
    summary = {"format": "rgb_geometry_qualitative_lora_report_v1", "status": "complete" if complete else "incomplete",
               "study_type": "new_reproduction_retrospective_case_study", "geometry_certified": False,
               "planned_videos": len(cases) * len(METHODS), "verified_videos": sum(row["status"] == "complete" for row in videos),
               "video_bytes": sum(row["bytes"] for row in videos), "review_seed": first_stage.REVIEW_SEED,
               "matched_frozen_inputs_and_common_sources": True,
               "phase_source_fingerprints": {"baseline_and_heads": reference["common_source_fingerprint"],
                                             "lora": protocol.fingerprint(lora_identity["sources"])},
               "reference": reference, "methods": methods, "videos": videos, "limitations": LIMITATIONS}
    method_fields = ("method", "phase", "status", "execution_status", "completed_clips", "verified_videos",
                     "overall_class_balanced_score", "final_policy_digest", "source_fingerprint")
    class_rows = [dict(row, method=method, method_status=result["status"])
                  for method, result in methods.items() for row in result["per_class"]]
    class_fields = ("method", "method_status", "class_name", "expected_clips", "completed_expected_clips", "complete", "score",
                    "partial_mean_diagnostic", "no_query_clips", "query_count")
    destination.mkdir(exist_ok=True)
    shared.write_json(destination / "summary.json", summary)
    shared.write_csv(destination / "methods.csv", list(methods.values()), method_fields)
    shared.write_csv(destination / "per_class.csv", class_rows, class_fields)
    shared.write_csv(destination / "video_index.csv", videos, tuple(videos[0]))
    shared.write_csv(destination / "mapping.csv", mapping, ("review_id", "case_id", "method", "video_path", "phase"))
    shared.write_csv(destination / "annotation_template.csv", annotations, first_stage.ANNOTATION_FIELDS)
    for filename, anonymous in (("video_index.html", False), ("review.html", True)):
        with (destination / filename).open("x") as handle:
            handle.write(gallery(cases, videos, review=anonymous))
    return {"status": summary["status"], "planned_videos": summary["planned_videos"],
            "verified_videos": summary["verified_videos"], "video_bytes": summary["video_bytes"],
            "matched_frozen_inputs_and_common_sources": True, "geometry_certified": False,
            "report_directory": str(destination), "summary_path": str(destination / "summary.json")}
