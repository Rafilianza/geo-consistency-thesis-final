"""Shared class-persistence measurements; these are not geometry accuracy.

Only NumPy is required. Unknown visible observations are mismatches, while
invisible or invalid positions are unobserved. Empty support stays explicit.
"""

import numpy as np

from cotracker_evaluation.tracking_metrics import (
    CLASS_NAMES, _check_tracks, _nearest_pixels, _visible_coordinates,
    compute_metrics, select_queries,
)
from rgb_geometry_pilot.reward import BACKGROUND_CLASSES, _segment_motion, validate_labels
from rgb_geometry_qualification.diagnostics import _interior, animated_photo_diagnostic


ANCHORS = (0, 16, 32, 48, 64)
POINTS_PER_CLASS = 16
FPS = 16.0
CAR_LABEL = 2
SCORED_CLASS_IDS = tuple(range(1, len(CLASS_NAMES)))
PHOTO_WINDOWS = tuple((anchor, anchor + 1, anchor + 16) for anchor in ANCHORS)
SETTINGS = {
    "format": "rgb_geometry_evaluation_tracking_v1",
    "anchor_frames": list(ANCHORS), "points_per_class": POINTS_PER_CLASS,
    "fps": FPS, "frame_count": 81, "cycle_horizon": 16,
    "retention_horizons": [16], "scored_class_ids": list(SCORED_CLASS_IDS),
    "class_names": list(CLASS_NAMES), "background": "unsupported; treated as unknown",
    "car_query_erosion": "full-resolution radius-two square; selection only",
    "later_labels": "original hard RGB YOLO labels",
    "visible_unknown_label": "mismatch", "zero_observation_query_score": 0.0,
    "no_query_class_score": 0.0,
    "class_score": "equal mean over selected points, including zero-observation points",
    "photo_version": "one_second_anchor_affine_v1", "photo_sample_stride": 4,
    "photo_windows": [list(window) for window in PHOTO_WINDOWS],
    "geometry_certified": False,
}


def select_evaluation_queries(labels):
    """Add deterministic points each second; background never seeds a query."""
    labels = validate_labels(labels)
    query_labels = labels.copy()
    query_labels[query_labels == 0] = 255
    for anchor in ANCHORS:
        if anchor >= len(labels) - 1:
            continue
        car_mask = labels[anchor] == CAR_LABEL
        interior = _interior(car_mask, radius=2)
        query_labels[anchor, car_mask & ~interior] = 255
    anchors = [anchor for anchor in ANCHORS if anchor < len(labels) - 1]
    return select_queries(query_labels, anchors, POINTS_PER_CLASS, CLASS_NAMES)


def _summarize_points(rows, possible_points):
    visible_count = sum(row["visible_observation_count"] for row in rows)
    matching_count = sum(row["matching_observation_count"] for row in rows)
    unknown_count = sum(row["unknown_observation_count"] for row in rows)
    supported = bool(rows) and visible_count > 0
    reason = "measured" if supported else "no_later_visible_observations"
    if not rows:
        reason = "no_class_queries"
    score = float(np.mean([row["score"] for row in rows])) if rows else 0.0
    summary = {
        "score": score, "persistence": score,
        "supported": supported, "support_reason": reason,
        "query_count": len(rows), "availability": len(rows) / possible_points,
        "visible_observation_count": visible_count,
        "matching_observation_count": matching_count,
        "unknown_observation_count": unknown_count,
        "unknown_label_rate": unknown_count / visible_count if visible_count else None,
        "zero_observation_queries": sum(not row["observed"] for row in rows),
        "possible_observation_count": sum(row["possible_observation_count"] for row in rows),
        "censored_observation_count": sum(row["censored_observation_count"] for row in rows),
        "censoring": float(np.mean([row["censoring"] for row in rows])) if rows else None,
        "observable_seconds_mean": float(np.mean([row["observable_seconds"] for row in rows])) if rows else None,
        "continuous_visible_seconds_mean": float(np.mean([row["continuous_visible_seconds"] for row in rows])) if rows else None,
    }
    for name in ("invisible_observation_count", "nonfinite_observation_count", "out_of_frame_observation_count"):
        summary[name] = sum(row[name] for row in rows)
    for name in ("retention_16", "continuous_retention_16"):
        values = [row[name] for row in rows if row[name] is not None]
        summary[name] = float(np.mean(values)) if values else None
    summary["retention_16_eligible_count"] = sum(row["retention_16"] is not None for row in rows)
    endpoint_count = sum(row["cycle_forward_endpoint"] for row in rows)
    reverse_count = sum(row["cycle_reverse_query"] for row in rows)
    return_count = sum(row["cycle_valid_return"] for row in rows)
    summary.update(
        cycle_eligible_count=sum(row["cycle_eligible"] for row in rows),
        cycle_forward_endpoint_count=endpoint_count,
        cycle_reverse_query_count=reverse_count,
        cycle_valid_return_count=return_count,
        cycle_valid_return_fraction=return_count / endpoint_count if endpoint_count and reverse_count else None,
    )
    for source_name, result_name in (("cycle_error_pixels", "cycle_error_pixel"),
                                     ("cycle_error_diagonal", "cycle_error_diagonal")):
        values = [row[source_name] for row in rows if row[source_name] is not None]
        for suffix, percentile in (("median", 50), ("p90", 90)):
            summary[result_name + "_" + suffix] = float(np.percentile(values, percentile)) if values else None
    return summary


def class_persistence(labels, tracks, visibility, queries, query_classes,
                      reverse_tracks=None, reverse_visibility=None, reverse_query_indices=None):
    """Return eleven source-label rows, with background explicitly unsupported.

    The original query frame is excluded. Every selected point receives equal
    weight, including points with zero usable later observations. Histograms
    retain all later visible labels, so changes do not end a track.
    """
    labels = validate_labels(labels)
    tracks, visibility, queries = _check_tracks(tracks, visibility, queries)
    frame_count, height, width = labels.shape
    if frame_count != SETTINGS["frame_count"] or tracks.shape[0] != frame_count:
        raise ValueError("Shared evaluation requires 81 aligned RGB/label/track frames")
    query_classes = np.asarray(query_classes)
    if (query_classes.shape != (len(queries),) or query_classes.dtype.kind not in "iu"
            or not np.isin(query_classes, SCORED_CLASS_IDS).all()):
        raise ValueError("Queries must use genuine source classes 1 through 10")
    if not np.isin(queries[:, 0], ANCHORS).all():
        raise ValueError("Query anchors must be 0, 16, 32, 48 or 64")

    # This helper validates query coordinates and their original class labels.
    # Reuse only its visibility/cycle diagnostics, not its older persistence.
    diagnostics = compute_metrics(
        labels, tracks, visibility, queries, query_classes,
        reverse_tracks=reverse_tracks, reverse_visibility=reverse_visibility,
        reverse_query_indices=reverse_query_indices, fps=FPS,
        cycle_horizon=16, retention_horizons=(16,))
    observable = _visible_coordinates(tracks, visibility, width, height)
    point_rows = []
    for query_index, query in enumerate(queries):
        class_id = int(query_classes[query_index])
        anchor = int(query[0])
        later_visible = observable[anchor + 1:, query_index]
        frame_indices = np.flatnonzero(later_visible) + anchor + 1
        pixels = _nearest_pixels(tracks[frame_indices, query_index], width, height)
        observed_labels = labels[frame_indices, pixels[:, 1], pixels[:, 0]]
        visible_count = len(frame_indices)
        matching_count = int(np.count_nonzero(observed_labels == class_id))
        unknown_count = int(np.count_nonzero((observed_labels == 255) | (observed_labels == 0)))
        possible_count = frame_count - anchor - 1
        later_positions = tracks[anchor + 1:, query_index]
        later_marked_visible = visibility[anchor + 1:, query_index]
        finite_positions = np.isfinite(later_positions).all(axis=1)
        # Mutually exclusive observation categories, in this order. A false
        # visibility flag means tracker-invisible, not proven physical occlusion.
        invisible_count = int(np.count_nonzero(~later_marked_visible))
        nonfinite_count = int(np.count_nonzero(later_marked_visible & ~finite_positions))
        outside_count = int(np.count_nonzero(later_marked_visible & finite_positions & ~later_visible))
        first_missing = np.flatnonzero(~later_visible)
        continuous_count = int(first_missing[0]) if len(first_missing) else possible_count
        label_counts = {}
        for observed_class in SCORED_CLASS_IDS:
            label_counts[CLASS_NAMES[observed_class]] = int(np.count_nonzero(observed_labels == observed_class))
        label_counts["unknown"] = unknown_count
        row = {
            "query_index": query_index, "class_id": class_id, "class_name": CLASS_NAMES[class_id],
            "anchor_frame": anchor, "query_x": float(query[1]), "query_y": float(query[2]),
            "visible_observation_count": visible_count, "matching_observation_count": matching_count,
            "unknown_observation_count": unknown_count, "observed_label_counts": label_counts,
            "possible_observation_count": possible_count,
            "censored_observation_count": possible_count - visible_count,
            "invisible_observation_count": invisible_count,
            "nonfinite_observation_count": nonfinite_count,
            "out_of_frame_observation_count": outside_count,
            "censoring": 1.0 - visible_count / possible_count,
            "score": matching_count / visible_count if visible_count else 0.0,
            "observed": bool(visible_count), "observable_seconds": visible_count / FPS,
            "possible_seconds": possible_count / FPS,
            "continuous_visible_seconds": continuous_count / FPS,
            "first_observed_frame": int(frame_indices[0]) if visible_count else None,
            "last_observed_frame": int(frame_indices[-1]) if visible_count else None,
        }
        old_row = diagnostics["per_query"][query_index]
        for name in ("retention_16", "continuous_retention_16", "cycle_eligible",
                     "cycle_forward_endpoint", "cycle_reverse_query", "cycle_valid_return",
                     "cycle_error_pixels", "cycle_error_diagonal"):
            row[name] = old_row[name]
        point_rows.append(row)

    per_class = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        rows = [row for row in point_rows if row["class_id"] == class_id]
        result = dict(class_id=class_id, class_name=class_name)
        result.update(_summarize_points(rows, len(ANCHORS) * POINTS_PER_CLASS))
        result["class_pixel_fraction"] = float(np.mean(labels == class_id))
        result["per_query"] = rows
        result["by_anchor"] = {}
        for anchor in ANCHORS:
            anchor_rows = [row for row in rows if row["anchor_frame"] == anchor]
            if len(anchor_rows) > POINTS_PER_CLASS:
                raise ValueError("At most sixteen points per class per anchor are allowed")
            summary = _summarize_points(anchor_rows, POINTS_PER_CLASS)
            summary.update(anchor_frame=anchor, available_seconds=(frame_count - anchor - 1) / FPS)
            result["by_anchor"][str(anchor)] = summary
        old_class = diagnostics["per_class"][class_id]
        for name, value in old_class.items():
            if name.startswith(("retention_", "continuous_retention_", "cycle_", "motion_diagonals_")):
                result[name] = value
        if class_id == 0:
            result.update(score=None, persistence=None, supported=False,
                          support_reason="background_is_unknown")
            for summary in result["by_anchor"].values():
                summary.update(score=None, persistence=None, supported=False,
                               support_reason="background_is_unknown")
        per_class.append(result)
    return per_class


def background_motion(labels, tracks, visibility, queries):
    """Measure one-second background segments independently of car support."""
    labels = validate_labels(labels)
    tracks, visibility, queries = _check_tracks(tracks, visibility, queries)
    if len(labels) != tracks.shape[0] or not np.isin(queries[:, 0], ANCHORS).all():
        raise ValueError("Background tracks must use matching frames and evaluation anchors")
    segment_values = []
    query_indices = []
    for query_index, query in enumerate(queries):
        # This numerical helper accepts arbitrary anchors; its historical
        # wrapper does not, so use it after our own shape/anchor checks.
        value = _segment_motion(labels, tracks, visibility, query_index, int(query[0]))
        if value is not None:
            query_indices.append(query_index)
            segment_values.append(value)
    return {
        "supported": bool(segment_values),
        "support_reason": "measured" if segment_values else "no_supported_background_segments",
        "gross_displacement": float(np.mean(segment_values)) if segment_values else None,
        "query_count": len(queries), "supported_segments": len(segment_values),
        "unsupported_segments": len(queries) - len(segment_values),
        "query_indices": query_indices, "segment_displacements": segment_values,
        "segment_positions": 17, "candidate_local": True,
        "definition": "mean supported 17-position segment net_squared/(path+1e-8), image-diagonal units",
        "diagnostic_only": True, "calibrated": False,
    }


def one_second_photo_diagnostic(frames, tracks, visibility, queries):
    """Use every second's own background points for the next sixteen frames."""
    frames = np.asarray(frames)
    tracks, visibility, queries = _check_tracks(tracks, visibility, queries)
    if len(frames) != 81 or len(tracks) != 81 or not np.isin(queries[:, 0], ANCHORS).all():
        raise ValueError("Photo diagnostic requires 81 frames and five one-second anchors")
    frame_rows = []
    windows = []
    for anchor, first_target, last_target in PHOTO_WINDOWS:
        cohort = queries[:, 0] == anchor
        diagnostic = animated_photo_diagnostic(
            frames[anchor:last_target + 1], tracks[anchor:last_target + 1, cohort],
            visibility[anchor:last_target + 1, cohort], sample_stride=4)
        for row in diagnostic["frames"]:
            frame_rows.append(dict(row, frame_index=anchor + row["frame_index"], anchor_frame=anchor))
        windows.append({
            "anchor_frame": anchor, "first_target_frame": first_target,
            "last_target_frame": last_target, "query_count": int(cohort.sum()),
            "sampled_frames": diagnostic["sampled_frames"], "measured_frames": diagnostic["measured_frames"],
            "support_reason": diagnostic["support_reason"],
        })
    measured = [row for row in frame_rows if row["support_reason"] == "measured"]
    supported = len(measured) == 80
    return {
        "diagnostic_only": True, "calibrated": False,
        "version": SETTINGS["photo_version"], "sample_stride": 4,
        "sampled_frames": len(frame_rows), "measured_frames": len(measured),
        "supported": supported,
        "support_reason": "measured" if supported else "photo_not_supported_on_all_80_later_frames",
        "mean_rgb_mae": float(np.mean([row["rgb_mae"] for row in measured])) if measured else None,
        "mean_brightness_aligned_rgb_mae": float(np.mean([row["brightness_aligned_rgb_mae"] for row in measured])) if measured else None,
        "windows": windows, "frames": frame_rows,
    }
