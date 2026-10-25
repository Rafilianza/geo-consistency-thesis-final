"""Lifetime car continuity and local operating checks for the new RL study.

These are imperfect RGB measurements. Passing a sentinel never certifies
appearance, individual object preservation, or physical geometry.
"""

import math

import numpy as np

from cotracker_evaluation.tracking_metrics import (
    CLASS_NAMES, _check_tracks, _nearest_pixels, _visible_coordinates, select_queries,
)
from rgb_geometry_pilot.reward import validate_labels
from rgb_geometry_qualification.diagnostics import _interior


ANCHORS = (0, 32, 64)
CAR_LABEL = 2
POINTS_PER_CLASS = 16
EROSION_RADIUS = 2
SUPPORT_TOLERANCE = 0.05
SENTINEL_CONTROL_NAMES = ("freeze", "zoomed_still", "zoomed_still_noisy")


def select_interior_queries(labels):
    """Erode only full-resolution car query support; keep other classes intact."""
    labels = validate_labels(labels)
    query_labels = labels.copy()
    for anchor in ANCHORS:
        if anchor >= len(labels):
            continue
        car_mask = labels[anchor] == CAR_LABEL
        interior = _interior(car_mask, EROSION_RADIUS)
        # Unknown removes boundary candidates without creating another class.
        query_labels[anchor, car_mask & ~interior] = 255
    return select_queries(query_labels, ANCHORS, POINTS_PER_CLASS, CLASS_NAMES)


def lifetime_car_reward(labels, tracks, visibility, queries, query_classes):
    """Average per-query car continuity over all later observable frames.

    Visible unknown labels are mismatches. Invisible, nonfinite and out-of-frame
    positions are censored. Zero-observation queries and empty clips receive
    zero credit and remain in the result. Labels are never eroded for scoring.
    """
    labels = validate_labels(labels)
    tracks, visibility, queries = _check_tracks(tracks, visibility, queries)
    frame_count, height, width = labels.shape
    if tracks.shape[0] != frame_count:
        raise ValueError("Labels and tracks must have the same frame count")
    query_classes = np.asarray(query_classes)
    if query_classes.shape != (len(queries),) or not np.issubdtype(query_classes.dtype, np.integer):
        raise ValueError("query_classes must contain one integer class ID per query")
    if not np.isin(query_classes, np.arange(len(CLASS_NAMES))).all():
        raise ValueError("Queries must use the declared eleven source classes")
    if not np.isin(queries[:, 0], ANCHORS).all():
        raise ValueError("Queries must use anchor frames 0, 32 and 64")
    if not _visible_coordinates(queries[:, 1:], np.ones(len(queries), dtype=bool), width, height).all():
        raise ValueError("Query positions must be in bounds")
    query_pixels = _nearest_pixels(queries[:, 1:], width, height)
    source_classes = labels[queries[:, 0].astype(int), query_pixels[:, 1], query_pixels[:, 0]]
    if not np.array_equal(source_classes, query_classes):
        raise ValueError("Query classes must match original label pixels at their anchors")

    visible = _visible_coordinates(tracks, visibility, width, height)
    car_indices = np.flatnonzero(query_classes == CAR_LABEL)
    query_rows = []
    counts_by_anchor = {str(anchor): 0 for anchor in ANCHORS}
    for query_index in car_indices:
        anchor = int(queries[query_index, 0])
        counts_by_anchor[str(anchor)] += 1
        later_frames = np.flatnonzero(visible[anchor + 1:, query_index]) + anchor + 1
        pixels = _nearest_pixels(tracks[later_frames, query_index], width, height)
        observed_labels = labels[later_frames, pixels[:, 1], pixels[:, 0]]
        visible_count = len(later_frames)
        match_count = int(np.count_nonzero(observed_labels == CAR_LABEL))
        remaining_frames = frame_count - anchor - 1
        query_rows.append({
            "query_index": int(query_index), "anchor_frame": anchor,
            "visible_observation_count": visible_count, "matching_observation_count": match_count,
            "unknown_observation_count": int(np.count_nonzero(observed_labels == 255)),
            "possible_observation_count": remaining_frames,
            "censored_observation_count": remaining_frames - visible_count,
            "censoring": 1.0 - visible_count / remaining_frames if remaining_frames else None,
            "reward": match_count / visible_count if visible_count else 0.0,
            "observed": bool(visible_count),
        })

    query_count = len(query_rows)
    if any(count > POINTS_PER_CLASS for count in counts_by_anchor.values()):
        raise ValueError("At most sixteen car queries per anchor are allowed")
    visible_count = sum(row["visible_observation_count"] for row in query_rows)
    unknown_count = sum(row["unknown_observation_count"] for row in query_rows)
    censoring_values = [row["censoring"] for row in query_rows if row["censoring"] is not None]
    supported = query_count > 0 and visible_count > 0
    reason = "measured"
    if not query_count:
        reason = "no_car_queries"
    elif not visible_count:
        reason = "no_later_visible_observations"
    return {
        "reward": float(np.mean([row["reward"] for row in query_rows])) if query_count else 0.0,
        "supported": supported, "support_reason": reason,
        "query_count": query_count, "queries_by_anchor": counts_by_anchor,
        "availability": query_count / (len(ANCHORS) * POINTS_PER_CLASS),
        "censoring": float(np.mean(censoring_values)) if censoring_values else None,
        "visible_observation_count": visible_count,
        "unknown_observation_count": unknown_count,
        "unknown_label_rate": unknown_count / visible_count if visible_count else None,
        "zero_observation_queries": sum(not row["observed"] for row in query_rows),
        "per_query": query_rows,
    }


def sentinel_controls(frames):
    """Yield the three existing deterministic RGB edits; rerun YOLO on each.

    These controls require no semantic input. The unknown placeholder below
    merely satisfies the existing transform API; it is never a scored label.
    """
    from rgb_geometry_reward_check.controls import iter_controls

    frames = np.asarray(frames)
    unknown_labels = np.full(frames.shape[:3], 255, dtype=np.uint8)
    for name, transformed, metadata in iter_controls(frames, unknown_labels):
        if name in SENTINEL_CONTROL_NAMES:
            yield name, transformed, metadata


def _sentinel_values(metrics):
    """Accept a saved full tracker report or its compact sentinel section."""
    return metrics.get("sentinel", metrics)


def _unsupported_reasons(metrics):
    values = _sentinel_values(metrics)
    reasons = list(values.get("unsupported_reasons", []))
    if not values.get("supported", False) and not reasons:
        reasons.append("measurement_marked_unsupported")
    for name in ("availability", "censoring", "gross_displacement", "photo_residual"):
        value = values.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            reasons.append(name + "_unavailable")
        elif value < 0 or (name in ("availability", "censoring") and value > 1):
            reasons.append(name + "_out_of_range")
    return list(dict.fromkeys(reasons))


def qualify_sentinel(baseline, controls):
    """Fix local stop values only when baseline and its controls separate."""
    failures = ["baseline:" + reason for reason in _unsupported_reasons(baseline)]
    for name in SENTINEL_CONTROL_NAMES:
        if name not in controls:
            failures.append(name + ":missing_control")
        else:
            failures.extend(name + ":" + reason for reason in _unsupported_reasons(controls[name]))
    if failures:
        return {"supported": False, "status": "inconclusive", "reason": ";".join(failures), "limits": None}

    baseline = _sentinel_values(baseline)
    freeze = _sentinel_values(controls["freeze"])
    zoom = _sentinel_values(controls["zoomed_still"])
    noisy_zoom = _sentinel_values(controls["zoomed_still_noisy"])
    limits = {
        "gross_displacement_maximum": freeze["gross_displacement"],
        "photo_residual_maximum": max(zoom["photo_residual"], noisy_zoom["photo_residual"]),
        "availability_drop_maximum": SUPPORT_TOLERANCE,
        "censoring_rise_maximum": SUPPORT_TOLERANCE,
    }
    if baseline["gross_displacement"] <= limits["gross_displacement_maximum"]:
        failures.append("baseline_displacement_not_above_freeze")
    if baseline["photo_residual"] <= limits["photo_residual_maximum"]:
        failures.append("baseline_photo_residual_not_above_zoom_controls")
    return {"supported": not failures, "status": "inconclusive" if failures else "passed",
            "reason": ";".join(failures) if failures else "baseline_separates_from_controls",
            "limits": None if failures else limits}


def check_sentinel(candidate, baseline, limits):
    """Check support first, then local freeze/animated-photo stop values."""
    failures = ["baseline:" + reason for reason in _unsupported_reasons(baseline)]
    failures.extend("candidate:" + reason for reason in _unsupported_reasons(candidate))
    required_limits = ("gross_displacement_maximum", "photo_residual_maximum",
                       "availability_drop_maximum", "censoring_rise_maximum")
    if not isinstance(limits, dict):
        failures.append("sentinel_limits_unavailable")
    else:
        for name in required_limits:
            value = limits.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                failures.append(name + "_invalid")
        if limits.get("availability_drop_maximum") != SUPPORT_TOLERANCE or limits.get("censoring_rise_maximum") != SUPPORT_TOLERANCE:
            failures.append("support_tolerance_changed")
    if failures:
        return {"passed": False, "status": "inconclusive", "reason": ";".join(failures)}

    candidate = _sentinel_values(candidate)
    baseline = _sentinel_values(baseline)
    # Comparing levels avoids a rounding error at exactly five percentage points.
    if candidate["availability"] < baseline["availability"] - SUPPORT_TOLERANCE:
        failures.append("query_availability_fell_over_five_percentage_points")
    if candidate["censoring"] > baseline["censoring"] + SUPPORT_TOLERANCE:
        failures.append("censoring_rose_over_five_percentage_points")
    if failures:
        return {"passed": False, "status": "inconclusive", "reason": ";".join(failures)}
    if candidate["gross_displacement"] <= limits["gross_displacement_maximum"]:
        failures.append("displacement_at_or_below_freeze")
    if candidate["photo_residual"] <= limits["photo_residual_maximum"]:
        failures.append("photo_residual_at_or_below_zoom_controls")
    return {"passed": not failures, "status": "shortcut_warning" if failures else "passed",
            "reason": ";".join(failures) if failures else "declared_sentinel_checks_passed"}
