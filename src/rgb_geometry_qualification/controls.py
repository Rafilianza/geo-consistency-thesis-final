"""Fixed saved-video stress tests, separate from the historical B0 protocol.

These transformations create RGB videos. Their reference labels define where
to paint, never what YOLO or CoTracker should report afterward. No transformed
video has a newly generated joint semantic trajectory.
"""

import math

import numpy as np
from PIL import Image


CONTROL_NAMES = (
    "unchanged", "moderate_brightness", "rectangle_all_car_removal",
    "gray_silhouette", "freeze", "zoomed_still", "temporary_occlusion",
    "one_car_remains",
)
CAR_LABEL = 2
BRIGHTNESS_FACTOR = 0.9
ZOOM_FINAL_SCALE = 1.15


def _validate_inputs(frames, reference_labels):
    frames = np.asarray(frames)
    reference_labels = np.asarray(reference_labels)
    if reference_labels.dtype != np.uint8 or reference_labels.ndim != 3:
        raise ValueError("Reference labels must be uint8 [81,H,W].")
    if not np.all((reference_labels < 11) | (reference_labels == 255)):
        raise ValueError("Reference labels must use the historical 11 classes or unknown 255.")
    if frames.dtype != np.uint8 or frames.shape != (*reference_labels.shape, 3):
        raise ValueError("RGB must be aligned uint8 [81,H,W,3].")
    if len(frames) != 81 or min(reference_labels.shape[1:]) < 4:
        raise ValueError("Fixed controls require 81 frames and dimensions of at least four pixels.")
    return frames, reference_labels


def _padded_component_boxes(mask, margin):
    """Find eight-connected mask regions without treating them as objects."""
    height, width = mask.shape
    remaining = mask.copy()
    boxes = []
    for pixel_index in np.flatnonzero(mask):
        row, column = divmod(int(pixel_index), width)
        if not remaining[row, column]:
            continue
        remaining[row, column] = False
        pending = [(row, column)]
        top = bottom = row
        left = right = column

        # Visit each region pixel once, including diagonal neighbours.
        while pending:
            row, column = pending.pop()
            top, bottom = min(top, row), max(bottom, row)
            left, right = min(left, column), max(right, column)
            for next_row in range(max(0, row - 1), min(height, row + 2)):
                for next_column in range(max(0, column - 1), min(width, column + 2)):
                    if remaining[next_row, next_column]:
                        remaining[next_row, next_column] = False
                        pending.append((next_row, next_column))
        boxes.append([max(0, top - margin), min(height, bottom + margin + 1),
                      max(0, left - margin), min(width, right + margin + 1)])
    return boxes


def iter_controls(frames, reference_labels):
    """Yield one video at a time without changing either input.

    Each dictionary has name, expectation, applicable, reason, frames and
    metadata. Inapplicable controls have frames=None and a coverage gap; they
    must not count as passed tests. Expectations describe desired scorer
    behavior, not proof that an original generated video has good geometry.

    Paint the union of individually padded component rectangles, keeping small
    regions too. Components are mask regions, not verified separate objects.
    Separate boxes avoid painting a broad bridge between distant regions,
    but still damage some background; this is a stress test, not inpainting.
    Padding is fixed at ceil(1% of the smaller image dimension), minimum one.
    It removes the labelled outline, but cannot guarantee removal of physical
    car parts missing from the original labels.
    """
    frames, reference_labels = _validate_inputs(frames, reference_labels)
    frame_count, height, width = reference_labels.shape
    car_masks = reference_labels == CAR_LABEL
    car_present = np.any(car_masks, axis=(1, 2))
    margin = max(1, math.ceil(0.01 * min(height, width)))

    # Fix erasure locations and colors before seeing any candidate model scores.
    boxes = [[] for _ in range(frame_count)]
    fill_colors = {}
    for frame_index in range(frame_count):
        if car_present[frame_index]:
            boxes[frame_index] = _padded_component_boxes(car_masks[frame_index], margin)
        valid_noncar = (reference_labels[frame_index] < 11) & ~car_masks[frame_index]
        if valid_noncar.any():
            pixels = frames[frame_index][valid_noncar]
            fill_colors[frame_index] = np.rint(np.median(pixels, axis=0)).astype(np.uint8)

    for name in CONTROL_NAMES:
        expectation = "reject"
        if name in ("unchanged", "moderate_brightness"):
            expectation = "accept"
        elif name == "temporary_occlusion":
            expectation = "diagnostic"

        metadata = {"joint_semantic_trajectory": "unavailable", "coverage_gap": False}
        result = {"name": name, "expectation": expectation, "applicable": True,
                  "reason": None, "frames": None, "metadata": metadata}

        # A semantic map cannot distinguish two cars from fragments of one car.
        if name == "one_car_remains":
            result.update(applicable=False, reason="semantic_labels_do_not_verify_distinct_objects")
            metadata.update(coverage_gap=True, required_evidence="independent instance/object evidence")
            yield result
            continue

        fill_frames = []
        if name in ("rectangle_all_car_removal", "gray_silhouette"):
            fill_frames = np.flatnonzero(car_present).tolist()
            if not fill_frames:
                result.update(applicable=False, reason="reference_has_no_car_labels")
        elif name == "temporary_occlusion":
            fill_frames = list(range(36, 40))
        missing_fill_frames = [frame_index for frame_index in fill_frames if frame_index not in fill_colors]
        if missing_fill_frames:
            result.update(applicable=False, reason="no_valid_noncar_pixels_for_median_fill")
            metadata["missing_fill_frames"] = missing_fill_frames
        if not result["applicable"]:
            metadata["coverage_gap"] = True
            yield result
            continue

        transformed = frames.copy()
        if name == "unchanged":
            metadata["note"] = "A copy of the reference; acceptance does not establish good geometry."
        elif name == "moderate_brightness":
            transformed = np.clip(np.rint(frames.astype(np.float32) * BRIGHTNESS_FACTOR), 0, 255).astype(np.uint8)
            metadata.update(brightness_factor=BRIGHTNESS_FACTOR, note="Same motion and shapes; all RGB channels scaled equally.")
        elif name in ("rectangle_all_car_removal", "gray_silhouette"):
            painted_pixels = 0
            for frame_index in fill_frames:
                if name == "rectangle_all_car_removal":
                    erased = np.zeros((height, width), dtype=bool)
                    for top, bottom, left, right in boxes[frame_index]:
                        erased[top:bottom, left:right] = True
                    transformed[frame_index][erased] = fill_colors[frame_index]
                    painted_pixels += int(erased.sum())
                else:
                    transformed[frame_index][car_masks[frame_index]] = fill_colors[frame_index]
                    painted_pixels += int(car_masks[frame_index].sum())
            metadata.update(fill="per-frame median valid noncar RGB; not necessarily neutral gray",
                            reference_car_pixels=int(car_masks.sum()),
                            painted_pixel_fraction=painted_pixels / reference_labels.size)
            if name == "rectangle_all_car_removal":
                metadata.update(margin_pixels=margin, boxes_top_bottom_left_right=boxes,
                                note="Union of padded component rectangles, including small fragments; components are not object instances, and erasure is not inpainting.")
            else:
                metadata["note"] = "Historical mask-only paint preserves the outline; tests appearance damage, not verified deletion."
        elif name == "freeze":
            transformed[:] = frames[0]
            metadata["source_frame"] = 0
        elif name == "zoomed_still":
            for frame_index, scale in enumerate(np.linspace(1.0, ZOOM_FINAL_SCALE, frame_count)):
                crop_width = max(1, round(width / scale))
                crop_height = max(1, round(height / scale))
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                crop = frames[0, top:top + crop_height, left:left + crop_width]
                transformed[frame_index] = np.asarray(
                    Image.fromarray(crop).resize((width, height), Image.Resampling.BILINEAR))
            metadata.update(source_frame=0, final_scale=ZOOM_FINAL_SCALE, rgb_interpolation="bilinear")
        elif name == "temporary_occlusion":
            left, right = (3 * width) // 8, (5 * width) // 8
            for frame_index in fill_frames:
                transformed[frame_index, :, left:right] = fill_colors[frame_index]
            metadata.update(frame_start=36, frame_stop_exclusive=40,
                            note="Synthetic four-frame occluder; diagnose sensitivity without treating ordinary occlusion as bad geometry.")
        result["frames"] = transformed
        yield result
