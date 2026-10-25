"""Windowed RGB photo diagnostic using saved candidate-local background tracks."""

import numpy as np

from rgb_geometry_qualification.diagnostics import animated_photo_diagnostic


PHOTO_VERSION = "windowed_anchor_affine_v1"
PHOTO_WINDOWS = ((0, 1, 32), (32, 33, 64), (64, 65, 80))
PHOTO_SAMPLE_STRIDE = 4


def windowed_photo_diagnostic(frames, bg_tracks, bg_visibility, bg_queries):
    """Measure every later frame once, relative to its window's own anchor.

    Only background queries created at that anchor enter its affine fits. The
    historical six-point, full-rank, visibility, bounds, and overlap rules are
    reused unchanged. A partial mean is diagnostic only: compact sentinels may
    use it only when ``supported`` is true, meaning all 80 frames were measured.
    """
    frames = np.asarray(frames)
    tracks = np.asarray(bg_tracks)
    visibility = np.asarray(bg_visibility)
    queries = np.asarray(bg_queries)
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != 81
            or frames.shape[-1] != 3 or min(frames.shape[1:3]) < 3):
        raise ValueError("Windowed photo requires 81 uint8 RGB frames with height/width at least three")
    if (queries.ndim != 2 or queries.shape[1] != 3 or queries.dtype.kind not in "fiu"
            or not np.isfinite(queries).all() or not np.isin(queries[:, 0], [0, 32, 64]).all()):
        raise ValueError("bg_queries must have numeric shape [points, 3] and anchors 0, 32, or 64")
    if tracks.shape != (81, len(queries), 2) or tracks.dtype.kind not in "fiu":
        raise ValueError("bg_tracks must have numeric shape [81, points, 2] aligned with bg_queries")
    if visibility.dtype != np.bool_ or visibility.shape != tracks.shape[:2]:
        raise ValueError("bg_visibility must be boolean with shape [81, points]")

    rows = []
    windows = []
    for anchor, first_target, last_target in PHOTO_WINDOWS:
        cohort = queries[:, 0] == anchor
        diagnostic = animated_photo_diagnostic(
            frames[anchor:last_target + 1], tracks[anchor:last_target + 1, cohort],
            visibility[anchor:last_target + 1, cohort], sample_stride=PHOTO_SAMPLE_STRIDE)
        for row in diagnostic["frames"]:
            rows.append(dict(row, frame_index=anchor + row["frame_index"], anchor_frame=anchor))
        windows.append({
            "anchor_frame": anchor, "first_target_frame": first_target,
            "last_target_frame": last_target, "query_count": int(cohort.sum()),
            "sampled_frames": diagnostic["sampled_frames"],
            "measured_frames": diagnostic["measured_frames"],
            "supported": diagnostic["measured_frames"] == diagnostic["sampled_frames"],
            "support_reason": diagnostic["support_reason"],
            "mean_rgb_mae": diagnostic["mean_rgb_mae"],
            "mean_brightness_aligned_rgb_mae": diagnostic["mean_brightness_aligned_rgb_mae"],
        })

    # Weight each measured frame equally; the final window contains 16, not 32.
    measured = [row for row in rows if row["support_reason"] == "measured"]
    mean_rgb_mae = None
    mean_aligned_mae = None
    if measured:
        mean_rgb_mae = float(np.mean([row["rgb_mae"] for row in measured]))
        mean_aligned_mae = float(np.mean([row["brightness_aligned_rgb_mae"] for row in measured]))
    supported = (len(measured) == 80 and [row["frame_index"] for row in rows] == list(range(1, 81))
                 and mean_aligned_mae is not None and np.isfinite(mean_aligned_mae))
    return {
        "diagnostic_only": True, "calibrated": False, "model": "windowed_anchor_affine",
        "version": PHOTO_VERSION, "sample_stride": PHOTO_SAMPLE_STRIDE,
        "sampled_frames": len(rows), "measured_frames": len(measured), "supported": bool(supported),
        "support_reason": "measured" if supported else "windowed_photo_not_supported_on_all_80_later_frames",
        "mean_rgb_mae": mean_rgb_mae, "mean_brightness_aligned_rgb_mae": mean_aligned_mae,
        "windows": windows, "frames": rows,
    }
