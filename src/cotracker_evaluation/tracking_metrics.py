"""Deterministic queries and descriptive 2D tracking metrics, using NumPy only.

These metrics do not measure verified 3D geometry. Occlusion, segmentation
mistakes, tracking mistakes and legitimate perspective growth remain confounds.
Query-time observations are forced by CoTracker and are excluded from scores.
"""

import numpy as np


CLASS_NAMES = (
    "background", "person", "car", "bus", "bicycle", "truck", "road",
    "sidewalk", "building", "vegetation", "sky",
)


def _check_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 3 or any(size < 1 for size in labels.shape):
        raise ValueError("labels must have nonempty shape [T,H,W]")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integer class IDs")
    return labels


def _check_tracks(tracks, visibility, queries):
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility)
    queries = np.asarray(queries, dtype=np.float64)
    if tracks.ndim != 3 or tracks.shape[2] != 2 or tracks.shape[0] < 1:
        raise ValueError("tracks must have shape [T,N,2]")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("visibility must have shape [T,N]")
    if not np.isin(visibility, (False, True)).all():
        raise ValueError("visibility must be Boolean, not confidence scores")
    if queries.shape != (tracks.shape[1], 3):
        raise ValueError("queries must have shape [N,3] containing (frame,x,y)")
    if not np.isfinite(queries).all():
        raise ValueError("queries must be finite")
    anchors = queries[:, 0]
    if not np.equal(anchors, np.floor(anchors)).all():
        raise ValueError("query frames must be integers")
    if ((anchors < 0) | (anchors >= tracks.shape[0])).any():
        raise ValueError("query frame is outside the video")
    return tracks, visibility.astype(bool, copy=False), queries


def _visible_coordinates(tracks, visibility, width, height):
    if width < 1 or height < 1:
        raise ValueError("video width and height must be positive")
    return (
        visibility & np.isfinite(tracks).all(axis=-1)
        & (tracks[..., 0] >= 0) & (tracks[..., 0] < width)
        & (tracks[..., 1] >= 0) & (tracks[..., 1] < height)
    )


def _nearest_pixels(coordinates, width, height):
    # Pixel centers use integer coordinates. Half-pixel ties round upwards;
    # an in-bounds coordinate at the outer edge uses the nearest border pixel.
    pixels = np.floor(coordinates + 0.5).astype(np.int64)
    pixels[..., 0] = np.clip(pixels[..., 0], 0, width - 1)
    pixels[..., 1] = np.clip(pixels[..., 1], 0, height - 1)
    return pixels


def select_queries(labels, anchors=(0, 32, 64), points_per_class=16,
                   class_names=CLASS_NAMES):
    """Select existing class pixels with deterministic greedy spatial coverage.

    For each anchor and class, start nearest the class-pixel centroid, then
    repeatedly select the pixel farthest from the selected set. Row-major
    order breaks ties. All selected positions are actual integer pixel centers.
    """
    labels = _check_labels(labels)
    if len(class_names) != len(CLASS_NAMES):
        raise ValueError("the declared label space must contain 11 classes")
    if not isinstance(points_per_class, int) or points_per_class < 0:
        raise ValueError("points_per_class must be a nonnegative integer")
    if any(int(anchor) != anchor or anchor < 0 for anchor in anchors):
        raise ValueError("anchor frames must be nonnegative integers")
    queries, query_classes = [], []
    for anchor in sorted(set(int(value) for value in anchors)):
        if anchor >= labels.shape[0]:
            continue
        for class_id in range(len(CLASS_NAMES)):
            pixels = np.argwhere(labels[anchor] == class_id)
            point_count = min(points_per_class, len(pixels))
            if point_count == 0:
                continue
            positions = pixels.astype(np.float64)
            centroid = positions.mean(axis=0)
            next_pixel = int(np.argmin(np.square(positions - centroid).sum(axis=1)))
            nearest_distance = np.full(len(pixels), np.inf)
            for _ in range(point_count):
                row, column = pixels[next_pixel]
                queries.append((anchor, column, row))
                query_classes.append(class_id)
                distance = np.square(positions - positions[next_pixel]).sum(axis=1)
                nearest_distance = np.minimum(nearest_distance, distance)
                next_pixel = int(np.argmax(nearest_distance))
    return (
        np.asarray(queries, dtype=np.float32).reshape(-1, 3),
        np.asarray(query_classes, dtype=np.int64),
    )


def build_reverse_queries(tracks, visibility, queries, width, height, horizon=16):
    """Start independent reversed-video tracking at eligible forward endpoints."""
    tracks, visibility, queries = _check_tracks(tracks, visibility, queries)
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    visible = _visible_coordinates(tracks, visibility, width, height)
    reverse_queries, query_indices = [], []
    frame_count = tracks.shape[0]
    for query_index, query in enumerate(queries):
        endpoint_frame = int(query[0]) + horizon
        if endpoint_frame >= frame_count or not visible[endpoint_frame, query_index]:
            continue
        endpoint_x, endpoint_y = tracks[endpoint_frame, query_index]
        reverse_queries.append((frame_count - 1 - endpoint_frame, endpoint_x, endpoint_y))
        query_indices.append(query_index)
    return (
        np.asarray(reverse_queries, dtype=np.float32).reshape(-1, 3),
        np.asarray(query_indices, dtype=np.int64),
    )


def _mean(values):
    supported = [value for value in values if value is not None]
    return float(np.mean(supported)) if supported else None


def _percentile(values, percentile):
    supported = [value for value in values if value is not None]
    return float(np.percentile(supported, percentile)) if supported else None


def compute_metrics(labels, tracks, visibility, queries, query_classes,
                    reverse_tracks=None, reverse_visibility=None,
                    reverse_query_indices=None, fps=16.0, cycle_horizon=16,
                    retention_horizons=(8, 16)):
    """Return JSON-safe per-query summaries and all 11 per-class rows.

    Only times strictly after each anchor are observations. Consecutive pairs
    require both times after the anchor. Agreement and persistence are averaged
    per query first; motion percentiles summarize per-query median speeds.
    Retention is coordinate visibility, independent of class-label correctness.
    Cycle return fraction uses eligible forward endpoints as its denominator.
    """
    labels = _check_labels(labels)
    tracks, visibility, queries = _check_tracks(tracks, visibility, queries)
    frame_count, height, width = labels.shape
    if tracks.shape[0] != frame_count:
        raise ValueError("labels and tracks must have the same frame count")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(cycle_horizon, int) or cycle_horizon < 1:
        raise ValueError("cycle_horizon must be a positive integer")
    if any(not isinstance(horizon, int) or horizon < 1 for horizon in retention_horizons):
        raise ValueError("retention horizons must be positive integers")
    query_classes = np.asarray(query_classes)
    if query_classes.shape != (len(queries),) or not np.issubdtype(query_classes.dtype, np.integer):
        raise ValueError("query_classes must contain one integer class ID per query")
    if ((query_classes < 0) | (query_classes >= len(CLASS_NAMES))).any():
        raise ValueError("query classes must be between 0 and 10")
    query_positions = queries[:, 1:]
    if not _visible_coordinates(query_positions, np.ones(len(queries), dtype=bool), width, height).all():
        raise ValueError("query coordinates must be in bounds")
    query_pixels = _nearest_pixels(query_positions, width, height)
    source_classes = labels[queries[:, 0].astype(np.int64), query_pixels[:, 1], query_pixels[:, 0]]
    if not np.array_equal(source_classes, query_classes):
        raise ValueError("query classes must match their original label-map pixels")

    visible = _visible_coordinates(tracks, visibility, width, height)
    sampled_classes = np.full(visible.shape, -1, dtype=np.int64)
    frame_indices, point_indices = np.nonzero(visible)
    pixels = _nearest_pixels(tracks[frame_indices, point_indices], width, height)
    sampled_classes[frame_indices, point_indices] = labels[frame_indices, pixels[:, 1], pixels[:, 0]]
    valid_labels = (sampled_classes >= 0) & (sampled_classes < len(CLASS_NAMES))
    observed = visible & valid_labels
    diagonal = float(np.hypot(width, height))

    reverse_supplied = any(value is not None for value in (reverse_tracks, reverse_visibility, reverse_query_indices))
    reverse_columns = {}
    if reverse_supplied:
        if any(value is None for value in (reverse_tracks, reverse_visibility, reverse_query_indices)):
            raise ValueError("reverse tracks, visibility and original query indices must be supplied together")
        reverse_query_indices = np.asarray(reverse_query_indices)
        if reverse_query_indices.ndim != 1 or not np.issubdtype(reverse_query_indices.dtype, np.integer):
            raise ValueError("reverse_query_indices must be an integer vector")
        expected_queries, expected_indices = build_reverse_queries(
            tracks, visibility, queries, width, height, horizon=cycle_horizon,
        )
        expected_columns = dict(zip(expected_indices.tolist(), range(len(expected_indices))))
        if len(set(reverse_query_indices.tolist())) != len(reverse_query_indices):
            raise ValueError("reverse query indices must not be duplicated")
        if any(index not in expected_columns for index in reverse_query_indices.tolist()):
            raise ValueError("reverse query lacks an eligible visible forward endpoint")
        ordered_queries = np.asarray([
            expected_queries[expected_columns[index]] for index in reverse_query_indices.tolist()
        ], dtype=np.float32).reshape(-1, 3)
        reverse_tracks, reverse_visibility, _ = _check_tracks(reverse_tracks, reverse_visibility, ordered_queries)
        if reverse_tracks.shape[0] != frame_count:
            raise ValueError("reverse tracks must use the full reversed video")
        reverse_visible = _visible_coordinates(reverse_tracks, reverse_visibility, width, height)
        reverse_columns = dict(zip(reverse_query_indices.tolist(), range(len(reverse_query_indices))))

    per_query = []
    for query_index, (query, class_id) in enumerate(zip(queries, query_classes)):
        anchor = int(query[0])
        later_observed = observed[anchor + 1:, query_index]
        later_classes = sampled_classes[anchor + 1:, query_index]
        semantic_pairs = later_observed[:-1] & later_observed[1:]
        same_classes = later_classes[:-1] == later_classes[1:]
        later_visible = visible[anchor + 1:, query_index]
        motion_pairs = later_visible[:-1] & later_visible[1:]
        # Subtract only valid pairs, so NaN/Inf observations never enter motion.
        later_tracks = tracks[anchor + 1:, query_index]
        displacement = later_tracks[1:][motion_pairs] - later_tracks[:-1][motion_pairs]
        speeds = np.linalg.norm(displacement, axis=1) * fps / diagonal
        observation_count = int(later_observed.sum())
        pair_count = int(semantic_pairs.sum())
        endpoint_frame = anchor + cycle_horizon
        cycle_eligible = endpoint_frame < frame_count
        forward_endpoint = bool(cycle_eligible and visible[endpoint_frame, query_index])
        row = {
            "query_index": query_index, "class_id": int(class_id), "class_name": CLASS_NAMES[class_id],
            "anchor_frame": anchor, "query_x": float(query[1]), "query_y": float(query[2]),
            "valid_observation_count": observation_count, "valid_pair_count": pair_count,
            "class_agreement": float(same_classes[semantic_pairs].mean()) if pair_count else None,
            "original_class_persistence": float((later_classes[later_observed] == class_id).mean()) if observation_count else None,
            "motion_pair_count": int(motion_pairs.sum()),
            "motion_diagonals_per_second_median": float(np.median(speeds)) if len(speeds) else None,
            "cycle_eligible": bool(cycle_eligible), "cycle_forward_endpoint": forward_endpoint,
            "cycle_reverse_query": query_index in reverse_columns, "cycle_valid_return": False,
            "cycle_error_diagonal": None, "cycle_error_pixels": None,
        }
        if query_index in reverse_columns:
            column = reverse_columns[query_index]
            return_frame = frame_count - 1 - anchor
            if reverse_visible[return_frame, column]:
                distance = float(np.linalg.norm(reverse_tracks[return_frame, column] - query[1:]))
                row["cycle_valid_return"] = True
                row["cycle_error_pixels"] = distance
                row["cycle_error_diagonal"] = distance / diagonal
        for horizon in retention_horizons:
            endpoint = anchor + horizon
            row["retention_%d" % horizon] = None
            row["continuous_retention_%d" % horizon] = None
            if endpoint < frame_count:
                row["retention_%d" % horizon] = float(visible[endpoint, query_index])
                row["continuous_retention_%d" % horizon] = float(visible[anchor + 1:endpoint + 1, query_index].all())
        per_query.append(row)

    per_class = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        rows = [row for row in per_query if row["class_id"] == class_id]
        eligible_count = sum(row["cycle_eligible"] for row in rows)
        endpoint_count = sum(row["cycle_forward_endpoint"] for row in rows)
        return_count = sum(row["cycle_valid_return"] for row in rows)
        result = {
            "class_id": class_id, "class_name": class_name,
            "class_pixel_fraction": float(np.count_nonzero(labels == class_id) / labels.size),
            "query_count": len(rows),
            "valid_pair_count": sum(row["valid_pair_count"] for row in rows),
            "valid_observation_count": sum(row["valid_observation_count"] for row in rows),
            "class_agreement_query_count": sum(row["class_agreement"] is not None for row in rows),
            "original_class_persistence_query_count": sum(row["original_class_persistence"] is not None for row in rows),
            "motion_query_count": sum(row["motion_diagonals_per_second_median"] is not None for row in rows),
            "class_agreement": _mean([row["class_agreement"] for row in rows]),
            "original_class_persistence": _mean([row["original_class_persistence"] for row in rows]),
            "cycle_eligible_count": eligible_count, "cycle_forward_endpoint_count": endpoint_count,
            "cycle_reverse_query_count": sum(row["cycle_reverse_query"] for row in rows),
            "cycle_valid_return_count": return_count,
            "cycle_forward_endpoint_fraction": endpoint_count / eligible_count if eligible_count else None,
            "cycle_valid_return_fraction": return_count / endpoint_count if endpoint_count and reverse_supplied else None,
        }
        for suffix, percentile in (("median", 50), ("p90", 90)):
            result["cycle_error_diagonal_" + suffix] = _percentile([row["cycle_error_diagonal"] for row in rows], percentile)
            result["cycle_error_pixel_" + suffix] = _percentile([row["cycle_error_pixels"] for row in rows], percentile)
            result["motion_diagonals_per_second_" + suffix] = _percentile([row["motion_diagonals_per_second_median"] for row in rows], percentile)
        for horizon in retention_horizons:
            endpoint_key = "retention_%d" % horizon
            continuous_key = "continuous_retention_%d" % horizon
            result[endpoint_key] = _mean([row[endpoint_key] for row in rows])
            result[continuous_key] = _mean([row[continuous_key] for row in rows])
            result[endpoint_key + "_eligible_count"] = sum(row[endpoint_key] is not None for row in rows)
        per_class.append(result)
    return {"per_class": per_class, "per_query": per_query}
