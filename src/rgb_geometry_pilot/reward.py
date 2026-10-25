"""Automatic car-only reward and deterministic qualification controls.

These are measurements of category continuity, not verified physical geometry.
YOLO labels use the historical eleven classes; joint labels use nine classes.
Unknown pixels remain 255. Model-unscorable samples earn -1; invalid inputs raise.
"""

import math

import numpy as np

from cotracker_evaluation.tracking_metrics import CLASS_NAMES


ANCHORS = (0, 32, 64)
WINDOW = 17
BACKGROUND_CLASSES = (6, 7, 8)
JOINT_CLASS_NAMES = ("other", "car", "bus", "truck", "road", "sidewalk", "building", "vegetation", "sky")
CORRUPTION_NAMES = (
    "freeze", "foreground_only_motion", "all_car_deletion", "separated_region_deletion",
    "one_region_remains", "oversized_foreground", "zoomed_still", "local_deformation",
    "temporary_occlusion", "mild_viewpoint_change",
)


class SetupFailure(ValueError):
    """The fixed qualification recipe cannot be constructed from its inputs."""


def validate_labels(labels, class_count=11):
    labels = np.asarray(labels)
    if labels.dtype != np.uint8 or labels.ndim != 3 or min(labels.shape) < 1:
        raise ValueError("Labels must be uint8 [T,H,W] with nonempty dimensions")
    if not np.all((labels < class_count) | (labels == 255)):
        raise ValueError(f"Expected {class_count}-class labels or unknown 255")
    return labels


def dilation_radius(height, width):
    return max(1, math.ceil(0.01 * min(height, width)))


def _disk(radius):
    coordinates = np.arange(-radius, radius + 1)
    return coordinates[:, None] ** 2 + coordinates[None, :] ** 2 <= radius ** 2


def _components(mask):
    from scipy import ndimage

    regions, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    return regions, count


def content_score(reference_labels, candidate_labels):
    """Worst component recall in fixed 17-frame windows, capped by precision."""
    from scipy import ndimage

    reference_labels = validate_labels(reference_labels)
    candidate_labels = validate_labels(candidate_labels)
    if reference_labels.shape != candidate_labels.shape:
        raise ValueError("Reference and candidate labels must share aligned [T,H,W] coordinates")
    frame_count, height, width = reference_labels.shape
    radius = dilation_radius(height, width)
    disk = _disk(radius)
    frame_recalls = []
    component_counts = []
    supported_precision_pixels = 0
    candidate_pixels = 0
    minimum_component_area = None
    for reference_frame, candidate_frame in zip(reference_labels, candidate_labels):
        reference_car = reference_frame == 2
        candidate_car = candidate_frame == 2
        regions, count = _components(reference_car)
        component_counts.append(int(count))
        if count:
            areas = np.bincount(regions.ravel())[1:]
            covered = ndimage.binary_dilation(candidate_car, structure=disk)
            hits = np.bincount(regions[covered], minlength=count + 1)[1:]
            frame_recalls.append(float(np.min(hits / areas)))
            area = int(areas.min())
            minimum_component_area = area if minimum_component_area is None else min(minimum_component_area, area)
        else:
            frame_recalls.append(None)
        dilated_reference = ndimage.binary_dilation(reference_car, structure=disk)
        supported_precision_pixels += int(np.count_nonzero(candidate_car & dilated_reference))
        candidate_pixels += int(candidate_car.sum())

    # Reference-empty frames alone are excluded. Candidate absence keeps a zero.
    windows = []
    for start in range(0, frame_count, WINDOW):
        supported = [value for value in frame_recalls[start:start + WINDOW] if value is not None]
        windows.append({"start": start, "stop": min(start + WINDOW, frame_count),
                        "reference_present_frames": len(supported),
                        "recall": float(np.mean(supported)) if supported else None})
    supported_windows = [row["recall"] for row in windows if row["recall"] is not None]
    recall = min(supported_windows) if supported_windows else None
    precision = supported_precision_pixels / candidate_pixels if candidate_pixels else 0.0
    return {"C": min(recall, precision) if recall is not None else None,
            "H": recall, "Q": precision, "dilation_radius": radius,
            "frame_worst_component_recall": frame_recalls, "windows": windows,
            "reference_component_counts": component_counts,
            "minimum_reference_component_area": minimum_component_area,
            "reference_car_present_frames": sum(count > 0 for count in component_counts),
            "candidate_car_present_frames": int(np.any(candidate_labels == 2, axis=(1, 2)).sum()),
            "candidate_car_pixels": candidate_pixels,
            "candidate_unknown_fraction": float(np.mean(candidate_labels == 255)),
            "reference_unknown_fraction": float(np.mean(reference_labels == 255))}


def semantic_disagreement(yolo_labels, joint_labels):
    """Pool foreground union over the full clip, including joint-only unknowns."""
    yolo_labels = validate_labels(yolo_labels)
    joint_labels = validate_labels(joint_labels, class_count=9)
    if joint_labels.shape != yolo_labels.shape:
        raise ValueError("Joint and YOLO labels must share aligned [T,H,W] coordinates")
    yolo_car = yolo_labels == 2
    joint_car = joint_labels == 1
    intersection = int(np.count_nonzero(yolo_car & joint_car))
    union = int(np.count_nonzero(yolo_car | joint_car))
    agreement = intersection / union if union else None
    return {"D": 1.0 - agreement if agreement is not None else 1.0,
            "raw_car_iou": agreement, "car_intersection": intersection, "car_union": union,
            "yolo_unknown_fraction": float(np.mean(yolo_labels == 255)),
            "joint_unknown_fraction": float(np.mean(joint_labels == 255))}


def _track_arrays(labels, tracks, visibility, queries):
    labels = validate_labels(labels)
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility)
    queries = np.asarray(queries, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[0] != len(labels) or tracks.shape[2] != 2:
        raise ValueError("Tracks must be [T,N,2] in the labels' frame coordinates")
    if visibility.dtype != np.bool_ or visibility.shape != tracks.shape[:2]:
        raise ValueError("Visibility must be Boolean [T,N]; confidence thresholds cannot change")
    if queries.shape != (tracks.shape[1], 3) or not np.isfinite(queries).all():
        raise ValueError("Queries must be finite [N,3] (frame,x,y)")
    if not np.isin(queries[:, 0], ANCHORS).all():
        raise ValueError("Motion queries must retain historical anchors 0, 32, 64")
    height, width = labels.shape[1:]
    if np.any(queries[:, 0] >= len(labels)) or np.any(queries[:, 1:] < 0):
        raise ValueError("Motion query outside the video")
    if np.any(queries[:, 1] >= width) or np.any(queries[:, 2] >= height):
        raise ValueError("Motion query outside the video")
    return labels, tracks, visibility, queries


def _segment_motion(labels, tracks, visibility, query_index, anchor):
    """None is invalid evidence; valid still segments have a real zero motion."""
    stop = anchor + WINDOW
    if stop > len(labels):
        return None
    positions = tracks[anchor:stop, query_index]
    height, width = labels.shape[1:]
    valid = visibility[anchor:stop, query_index] & np.isfinite(positions).all(axis=1)
    valid &= (positions[:, 0] >= 0) & (positions[:, 0] < width)
    valid &= (positions[:, 1] >= 0) & (positions[:, 1] < height)
    if not valid.all():
        return None
    # Match historical nearest-pixel sampling, including half-pixel ties.
    pixels = np.floor(positions + 0.5).astype(np.int64)
    columns = np.clip(pixels[:, 0], 0, width - 1)
    rows = np.clip(pixels[:, 1], 0, height - 1)
    sampled = labels[np.arange(anchor, stop), rows, columns]
    if not np.isin(sampled, BACKGROUND_CLASSES).all():
        return None
    diagonal = float(np.hypot(height, width))
    net = float(np.linalg.norm(positions[-1] - positions[0])) / diagonal
    path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) / diagonal
    return net ** 2 / (path + 1e-8)


def motion_cohort(labels, tracks, visibility, queries):
    """Freeze eligible reference segments before any candidate is evaluated."""
    labels, tracks, visibility, queries = _track_arrays(labels, tracks, visibility, queries)
    indices, motions = [], []
    for query_index, query in enumerate(queries):
        motion = _segment_motion(labels, tracks, visibility, query_index, int(query[0]))
        if motion is not None:
            indices.append(query_index)
            motions.append(motion)
    return {"query_indices": indices, "reference_motion": motions,
            "reference_mean_motion": float(np.mean(motions)) if motions else None,
            "queries": queries.tolist(), "label_shape": list(labels.shape),
            "segment_positions": WINDOW, "cohort_size": len(indices)}


def motion_score(candidate_labels, candidate_tracks, candidate_visibility, queries, cohort):
    """Measure candidate tracking on the entire frozen reference cohort."""
    labels, tracks, visibility, queries = _track_arrays(
        candidate_labels, candidate_tracks, candidate_visibility, queries)
    if (list(labels.shape) != cohort["label_shape"] or cohort["segment_positions"] != WINDOW
            or not np.array_equal(queries, np.asarray(cohort["queries"]).reshape(-1, 3))):
        raise ValueError("Candidate must use the exact frozen reference query coordinates and dimensions")
    indices = cohort["query_indices"]
    reference_motions = np.asarray(cohort["reference_motion"], dtype=np.float64)
    if (len(indices) != cohort["cohort_size"] or len(indices) != len(reference_motions)
            or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or index < 0 or index >= len(queries) for index in indices)
            or not np.isfinite(reference_motions).all() or np.any(reference_motions < 0)):
        raise ValueError("Corrupt frozen motion cohort")
    reference_mean = float(np.mean(reference_motions)) if len(indices) else None
    if reference_mean != cohort["reference_mean_motion"]:
        raise ValueError("Frozen cohort reference mean changed")
    motions, valid_count = [], 0
    for query_index in indices:
        motion = _segment_motion(labels, tracks, visibility, query_index, int(queries[query_index, 0]))
        valid_count += motion is not None
        motions.append(0.0 if motion is None else motion)
    candidate_mean = float(np.mean(motions)) if motions else None
    ratio = None
    if reference_mean is not None and reference_mean > 0:
        ratio = min(1.0, candidate_mean / reference_mean)
    return {"M": ratio, "candidate_mean_motion": candidate_mean,
            "reference_mean_motion": reference_mean, "candidate_motion": motions,
            "cohort_size": len(indices), "candidate_valid_segments": int(valid_count),
            "candidate_invalid_segments": len(indices) - int(valid_count)}


def score_reward(P, C, M, D):
    """Keep legitimate missing measurements; reject corrupt/nonfinite numbers."""
    values = {"P": P, "C": C, "M": M, "D": D}
    for name, value in values.items():
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be a finite probability or None")
    if D is None:
        raise ValueError("D must use the specified union-empty penalty of one")
    reasons = []
    if P is None:
        reasons.append("persistence_unmeasurable")
    if C is None or C < 0.5:
        reasons.append("content_unmeasurable" if C is None else "content_gate_failed")
    if M is None or M < 0.25:
        reasons.append("motion_unmeasurable" if M is None else "motion_gate_failed")
    passed = not reasons
    return dict(values, R_RGB=float(P) if passed else -1.0,
                R_semantic=float(P * (1 - 0.1 * D)) if passed else -1.0,
                passed=passed, failure_reasons=reasons)


def select_qualification_references(content_reports, motion_reports):
    """First, smallest region, most regions, weakest motion; stable tie breaks."""
    if not content_reports or len(content_reports) != len(motion_reports):
        raise SetupFailure("Qualification requires matching nonempty reference reports")
    if any(report["minimum_reference_component_area"] is None for report in content_reports):
        raise SetupFailure("Every qualification reference must contain a car region")
    if any(report["reference_mean_motion"] is None for report in motion_reports):
        raise SetupFailure("Every qualification reference requires background motion evidence")
    indices = range(len(content_reports))
    selected = [0,
                min(indices, key=lambda index: content_reports[index]["minimum_reference_component_area"]),
                max(indices, key=lambda index: max(content_reports[index]["reference_component_counts"])),
                min(indices, key=lambda index: motion_reports[index]["reference_mean_motion"])]
    return list(dict.fromkeys(selected))


def _ordered_regions(regions, count):
    from scipy import ndimage

    records = []
    areas = np.bincount(regions.ravel(), minlength=count + 1)
    # Bounding boxes and counts avoid scanning the full image for each fragment.
    for region_id, slices in enumerate(ndimage.find_objects(regions), start=1):
        rows, columns = slices
        records.append({"id": region_id, "area": int(areas[region_id]), "left": columns.start,
                        "top": rows.start, "bottom": rows.stop, "right": columns.stop})
    return sorted(records, key=lambda record: (record["left"], record["top"]))


def _fill_color(frame, labels):
    valid_noncar = (labels < len(CLASS_NAMES)) & (labels != 2)
    if not valid_noncar.any():
        raise SetupFailure("Corruption needs valid non-car pixels for its median RGB fill")
    return np.rint(np.median(frame[valid_noncar], axis=0)).astype(np.uint8)


def _sustained(eligible, reference_present):
    for start in range(0, len(reference_present), WINDOW):
        supported = int(reference_present[start:start + WINDOW].sum())
        if supported and int(eligible[start:start + WINDOW].sum()) > supported / 2:
            return True
    return False


def iter_corruptions(frames, reference_labels):
    """Yield ten fixed RGB controls without retaining ten full videos in RAM.

    Rerun YOLO and CoTracker on these RGB outputs to test the P/C/M guards.
    The reference masks only define transformations, never injected scores.
    A transformed RGB control has no jointly sampled semantic trajectory;
    the runner marks semantic agreement unavailable instead of borrowing masks.
    RGB resizing/translation uses bilinear interpolation. No labels are resized.
    """
    from PIL import Image
    from scipy import ndimage

    reference_labels = validate_labels(reference_labels)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.shape != (*reference_labels.shape, 3):
        raise SetupFailure("Corruptions need aligned uint8 RGB [T,H,W,3] and historical labels")
    frame_count, height, width = reference_labels.shape
    if frame_count != 81 or min(height, width) < 4:
        raise SetupFailure("Fixed corruptions require 81 frames and image dimensions at least four")
    car = reference_labels == 2
    reference_present = np.any(car, axis=(1, 2))
    if not reference_present.any():
        raise SetupFailure("Corruption reference has no car regions")
    radius = dilation_radius(height, width)
    separation_disk = _disk(2 * radius)
    descriptions = []
    separated_eligible = np.zeros(frame_count, dtype=bool)
    one_region_eligible = np.zeros(frame_count, dtype=bool)

    # Applicability depends only on the original masks, before any model score.
    for frame_index in range(frame_count):
        regions, count = _components(car[frame_index])
        records = _ordered_regions(regions, count)
        largest = min(records, key=lambda record: (-record["area"], record["left"], record["top"])) if records else None
        separated_id = None
        for record in records if count > 1 else []:
            top = max(0, record["top"] - 2 * radius)
            bottom = min(height, record["bottom"] + 2 * radius)
            left = max(0, record["left"] - 2 * radius)
            right = min(width, record["right"] + 2 * radius)
            local_regions = regions[top:bottom, left:right]
            selected = local_regions == record["id"]
            retained = (local_regions != 0) & ~selected
            # Distance must exceed 2r from ALL retained pixels, not just one car.
            # Pixels beyond this expanded bounding box cannot be within 2r.
            if not np.any(selected & ndimage.binary_dilation(retained, structure=separation_disk)):
                separated_id = record["id"]
                break
        separated_eligible[frame_index] = separated_id is not None
        if largest is not None and count > 1:
            retained = regions == largest["id"]
            near_retained = ndimage.binary_dilation(retained, structure=separation_disk)
            nearby_ids = set(np.unique(regions[near_retained]).tolist())
            for record in records:
                if record["id"] != largest["id"] and record["id"] not in nearby_ids:
                    one_region_eligible[frame_index] = True
                    break
        descriptions.append({"largest": largest, "separated_id": separated_id})
    applicability = {"separated_region_deletion": _sustained(separated_eligible, reference_present),
                     "one_region_remains": _sustained(one_region_eligible, reference_present)}
    # Fail before returning any control if a required fill is unavailable.
    fill_colors = {}
    for frame_index in range(frame_count):
        if reference_present[frame_index] or 36 <= frame_index <= 39:
            fill_colors[frame_index] = _fill_color(frames[frame_index], reference_labels[frame_index])

    for control_index, name in enumerate(CORRUPTION_NAMES):
        applicable = applicability.get(name, True)
        result = {"name": name, "mandatory": control_index < 6, "applicable": bool(applicable),
                  "reason": None if applicable else "no_separated_deletion_in_majority_of_supported_window",
                  "frames": None, "rgb_interpolation": "bilinear", "dilation_radius": radius}
        if name in applicability:
            eligible = separated_eligible if name == "separated_region_deletion" else one_region_eligible
            result["eligible_frames"] = np.flatnonzero(eligible).tolist()
        if not applicable:
            yield result
            continue
        transformed = frames.copy()
        if name == "freeze":
            transformed[:] = frames[0]
        elif name == "foreground_only_motion":
            transformed[:] = frames[0]
            transformed[car] = frames[car]
        elif name in ("all_car_deletion", "separated_region_deletion", "one_region_remains"):
            for frame_index in np.flatnonzero(reference_present):
                regions, _ = _components(car[frame_index])
                description = descriptions[frame_index]
                if name == "all_car_deletion":
                    deleted = car[frame_index]
                elif name == "separated_region_deletion":
                    selected_id = description["separated_id"]
                    deleted = regions == selected_id if selected_id is not None else np.zeros((height, width), dtype=bool)
                else:
                    deleted = car[frame_index] & (regions != description["largest"]["id"])
                transformed[frame_index][deleted] = fill_colors[frame_index]
        elif name == "oversized_foreground":
            first_frame = int(np.flatnonzero(reference_present)[0])
            box = descriptions[first_frame]["largest"]
            crop = frames[first_frame, box["top"]:box["bottom"], box["left"]:box["right"]]
            transformed[:] = np.asarray(Image.fromarray(crop).resize((width, height), Image.Resampling.BILINEAR))
        elif name == "zoomed_still":
            for frame_index, scale in enumerate(np.linspace(1.0, 1.15, frame_count)):
                crop_width, crop_height = max(1, round(width / scale)), max(1, round(height / scale))
                left, top = (width - crop_width) // 2, (height - crop_height) // 2
                crop = frames[0, top:top + crop_height, left:left + crop_width]
                transformed[frame_index] = np.asarray(Image.fromarray(crop).resize((width, height), Image.Resampling.BILINEAR))
        elif name == "local_deformation":
            for frame_index in np.flatnonzero(reference_present):
                box = descriptions[frame_index]["largest"]
                transformed[frame_index, box["top"]:box["bottom"], box["left"]:box["right"]] = frames[
                    frame_index, box["top"]:box["bottom"], box["left"]:box["right"]][:, ::-1]
        elif name == "temporary_occlusion":
            left, right = (3 * width) // 8, (5 * width) // 8
            for frame_index in range(36, 40):
                transformed[frame_index, :, left:right] = fill_colors[frame_index]
        elif name == "mild_viewpoint_change":
            for frame_index, shift in enumerate(np.linspace(0.0, 5.0, frame_count)):
                transformed[frame_index] = ndimage.shift(frames[frame_index], (0, shift, 0),
                                                        order=1, mode="reflect", prefilter=False)
        result["frames"] = transformed
        yield result
