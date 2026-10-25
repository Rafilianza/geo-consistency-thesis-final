"""Provisional RGB diagnostics, deliberately separate from frozen RL rewards.

These measurements have no pass/fail thresholds. Low detail can be legitimate,
and a real camera viewing a flat/distant scene can resemble an animated photo.
Neither measurement alone establishes geometry quality or authorizes training.
"""

import numpy as np


def _frames(value, name):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"{name} must be uint8 RGB with shape [frames, height, width, 3]")
    if value.shape[0] < 1 or min(value.shape[1:3]) < 3:
        raise ValueError(f"{name} needs at least one frame and height/width of three")
    return value


def _indices(frame_count, requested, exclude_first=False):
    if requested is None:
        requested = range(frame_count)
    values = np.asarray(list(requested))
    if values.size == 0:
        return []
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("frame_indices must contain integer indices")
    if np.any(values < 0) or np.any(values >= frame_count) or len(set(values)) != len(values):
        raise ValueError("frame_indices must be unique and within the video")
    return [int(value) for value in values if not exclude_first or value != 0]


def _interior(mask, radius):
    """Square erosion keeps the gradient stencil away from the silhouette."""
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    interior = np.ones_like(mask)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            interior &= padded[row_offset:row_offset + height, column_offset:column_offset + width]
    return interior


def _detail(frame, interior):
    rgb = frame.astype(np.float32) / 255.0
    horizontal = (rgb[1:-1, 2:] - rgb[1:-1, :-2]) / 2
    vertical = (rgb[2:, 1:-1] - rgb[:-2, 1:-1]) / 2
    gradient = np.sqrt(horizontal ** 2 + vertical ** 2).mean(axis=-1)
    gradient_mean = float(gradient[interior[1:-1, 1:-1]].mean())
    # Integer RGB with float64 accumulation gives exactly zero for constant RGB.
    texture = float(frame[interior].std(axis=0).mean() / 255.0)
    return gradient_mean, texture


def appearance_diagnostic(reference_frames, candidate_frames, reference_labels, *,
                          class_id=2, erosion_radius=2, frame_indices=None):
    """Measure retained car-interior detail for aligned saved-control pairs.

    The same eroded *reference* mask is used for both RGB arrays: a candidate
    cannot hide absent detail by ceasing to detect a car. RGB coordinates must
    match, so this is not a comparison between independently generated scenes.
    Ratios for exactly flat reference support are None, not collapse claims.
    Small but nonzero reference detail remains raw and requires calibration.
    """
    reference = _frames(reference_frames, "reference_frames")
    candidate = _frames(candidate_frames, "candidate_frames")
    labels = np.asarray(reference_labels)
    if candidate.shape != reference.shape or labels.shape != reference.shape[:3]:
        raise ValueError("Reference RGB, candidate RGB and reference labels must align exactly")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("reference_labels must contain integer class IDs")
    if not isinstance(erosion_radius, int) or isinstance(erosion_radius, bool) or erosion_radius < 1:
        raise ValueError("erosion_radius must be a positive integer")
    rows = []
    for frame_index in _indices(len(reference), frame_indices):
        mask = labels[frame_index] == class_id
        interior = _interior(mask, erosion_radius)
        row = {"frame_index": frame_index, "reference_class_pixels": int(mask.sum()),
               "interior_pixels": int(interior.sum()), "reference_gradient": None,
               "candidate_gradient": None, "gradient_retention": None,
               "reference_texture_std": None, "candidate_texture_std": None,
               "texture_retention": None, "support_reason": "no_eroded_reference_support"}
        if interior.any():
            original_gradient, original_texture = _detail(reference[frame_index], interior)
            new_gradient, new_texture = _detail(candidate[frame_index], interior)
            row.update(reference_gradient=original_gradient, candidate_gradient=new_gradient,
                       reference_texture_std=original_texture, candidate_texture_std=new_texture,
                       support_reason="measured")
            if original_gradient > 0:
                row["gradient_retention"] = new_gradient / original_gradient
            if original_texture > 0:
                row["texture_retention"] = new_texture / original_texture
            if original_gradient == 0 and original_texture == 0:
                row["support_reason"] = "flat_reference_no_retention_evidence"
        rows.append(row)
    measured = [row for row in rows if row["gradient_retention"] is not None]
    mean_retention = None
    if measured:
        mean_retention = float(np.mean([row["gradient_retention"] for row in measured]))
    return {"diagnostic_only": True, "calibrated": False, "class_id": int(class_id),
            "erosion_radius": erosion_radius, "sampled_frames": len(rows),
            "frames_with_gradient_evidence": len(measured),
            "mean_gradient_retention": mean_retention, "frames": rows}


def _sample_rgb(frame, positions):
    """Bilinear sampling at already validated interior pixel coordinates."""
    left = np.floor(positions[:, 0]).astype(int)
    top = np.floor(positions[:, 1]).astype(int)
    horizontal = (positions[:, 0] - left)[:, None]
    vertical = (positions[:, 1] - top)[:, None]
    rgb = frame.astype(np.float32) / 255.0
    upper = rgb[top, left] * (1 - horizontal) + rgb[top, left + 1] * horizontal
    lower = rgb[top + 1, left] * (1 - horizontal) + rgb[top + 1, left + 1] * horizontal
    return upper * (1 - vertical) + lower * vertical


def animated_photo_diagnostic(frames, bg_tracks=None, bg_visibility=None, *,
                              frame_indices=None, sample_stride=4):
    """Measure how well one affine-warped first frame explains later RGB.

    bg_tracks use the saved CoTracker shape [T,N,2], coordinates (x,y), with
    boolean visibility [T,N]. Fit only points visible and in bounds in both
    frames. All such points enter ordinary least squares: no tuned inlier gate.
    At least six points and full affine rank are required for a redundant fit.
    Reconstruction spans the *whole image* overlap, including moving objects.
    Low residual is a shortcut diagnostic, not proof of an illegitimate video.
    """
    frames = _frames(frames, "frames")
    frame_count, height, width = frames.shape[:3]
    indices = _indices(frame_count, frame_indices, exclude_first=True)
    if not isinstance(sample_stride, int) or isinstance(sample_stride, bool) or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    result = {"diagnostic_only": True, "calibrated": False, "model": "first_frame_affine",
              "sample_stride": sample_stride, "sampled_frames": len(indices),
              "measured_frames": 0, "mean_rgb_mae": None,
              "mean_brightness_aligned_rgb_mae": None, "frames": []}
    if bg_tracks is None and bg_visibility is None:
        result["support_reason"] = "missing_background_tracks"
        return result
    tracks = np.asarray(bg_tracks)
    visibility = np.asarray(bg_visibility)
    if (tracks.ndim != 3 or tracks.shape[0] != frame_count or tracks.shape[-1] != 2
            or tracks.dtype.kind not in "fiu"):
        raise ValueError("bg_tracks must have numeric shape [frames, points, 2]")
    if visibility.dtype != np.bool_ or visibility.shape != tracks.shape[:2]:
        raise ValueError("bg_visibility must be boolean with shape [frames, points]")

    # Estimate the transform in normalized coordinates to avoid pixel-scale conditioning.
    scale = np.array([width - 1, height - 1], dtype=float)
    source = tracks[0]
    source_valid = visibility[0] & np.isfinite(source).all(axis=1)
    source_valid &= ((source >= 0) & (source <= scale)).all(axis=1)
    grid_y, grid_x = np.mgrid[0:height:sample_stride, 0:width:sample_stride]
    grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    target_design = np.column_stack((grid / scale, np.ones(len(grid))))
    for frame_index in indices:
        target = tracks[frame_index]
        valid = source_valid & visibility[frame_index] & np.isfinite(target).all(axis=1)
        valid &= ((target >= 0) & (target <= scale)).all(axis=1)
        row = {"frame_index": frame_index, "visible_points": int(valid.sum()),
               "support_reason": "insufficient_visible_points", "affine_matrix_normalized": None,
               "track_fit_rmse_pixels": None, "source_track_bbox_fraction": None,
               "overlap_pixels": 0, "overlap_fraction": None, "rgb_mae": None,
               "brightness_aligned_rgb_mae": None, "identity_rgb_mae": None}
        if valid.sum() < 6:
            result["frames"].append(row)
            continue
        source_points = source[valid] / scale
        design = np.column_stack((source_points, np.ones(len(source_points))))
        coefficients, _, rank, _ = np.linalg.lstsq(design, target[valid] / scale, rcond=None)
        if rank != 3 or np.linalg.matrix_rank(coefficients[:2]) != 2:
            row["support_reason"] = "degenerate_affine_fit"
            result["frames"].append(row)
            continue
        transform = np.eye(3)
        transform[:2] = coefficients.T
        residual = (design @ coefficients) * scale - target[valid]
        row.update(affine_matrix_normalized=transform.tolist(),
                   track_fit_rmse_pixels=float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
                   source_track_bbox_fraction=float(np.prod(np.ptp(source_points, axis=0))))

        # Exclude newly exposed regions: the first frame cannot reconstruct unseen content.
        mapped = (target_design @ np.linalg.inv(transform).T)[:, :2] * scale
        overlap = np.isfinite(mapped).all(axis=1)
        overlap &= ((mapped >= 0) & (mapped < scale)).all(axis=1)
        row["overlap_pixels"] = int(overlap.sum())
        row["overlap_fraction"] = float(overlap.mean())
        if not overlap.any():
            row["support_reason"] = "no_reconstructable_overlap"
            result["frames"].append(row)
            continue
        reconstructed = _sample_rgb(frames[0], mapped[overlap])
        candidate = frames[frame_index, grid[overlap, 1], grid[overlap, 0]].astype(float) / 255.0
        identity = frames[0, grid[overlap, 1], grid[overlap, 0]].astype(float) / 255.0
        difference = candidate - reconstructed
        brightness_offset = np.median(difference, axis=0)
        row.update(support_reason="measured", rgb_mae=float(np.abs(difference).mean()),
                   brightness_aligned_rgb_mae=float(np.abs(difference - brightness_offset).mean()),
                   identity_rgb_mae=float(np.abs(candidate - identity).mean()))
        result["frames"].append(row)
    measured = [row for row in result["frames"] if row["rgb_mae"] is not None]
    result["measured_frames"] = len(measured)
    result["support_reason"] = "measured" if measured else "no_measurable_frame_pairs"
    if measured:
        result["mean_rgb_mae"] = float(np.mean([row["rgb_mae"] for row in measured]))
        result["mean_brightness_aligned_rgb_mae"] = float(np.mean([
            row["brightness_aligned_rgb_mae"] for row in measured]))
    return result
