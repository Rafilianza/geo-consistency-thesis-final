"""Ten fixed saved-video controls; never modify historical transformations."""

import numpy as np

from rgb_geometry_qualification import controls as historical_controls


CONTROL_NAMES = (
    "unchanged", "small_translation", "moderate_brightness",
    "rectangle_all_car_removal", "gray_silhouette", "gray_noisy", "freeze",
    "zoomed_still", "zoomed_still_noisy", "temporary_occlusion",
)
BENIGN_NAMES = (
    "unchanged", "moderate_brightness", "temporary_occlusion", "small_translation",
)
APPEARANCE_NAMES = ("gray_silhouette", "gray_noisy")
PHOTO_NAMES = ("freeze", "zoomed_still", "zoomed_still_noisy")
SETTINGS = {
    "format": "saved_video_reward_check_controls_v1",
    "control_names": list(CONTROL_NAMES),
    "translation_width_divisor": 104,
    "translation_height_divisor": 120,
    "translation_padding": "edge replication",
    "noise_integer_min": -16,
    "noise_integer_max_inclusive": 16,
    "gray_noise_seed": 20260918,
    "zoom_noise_seed": 20260919,
    "zoom_noise_patch_width_fraction": 0.25,
    "zoom_noise_patch_height_fraction": 0.25,
    "occlusion_frame_start": 36,
    "occlusion_frame_stop_exclusive": 40,
    "photo_frame_indices": list(range(1, 81)),
}


def _noisy_video(frames, masks, seed):
    """Independent RGB noise per pixel and frame, restricted to declared masks."""
    transformed = frames.copy()
    generator = np.random.default_rng(seed)
    for frame_index, mask in enumerate(masks):
        pixels = transformed[frame_index][mask].astype(np.int16)
        noise = generator.integers(-16, 17, size=pixels.shape, dtype=np.int16)
        transformed[frame_index][mask] = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    return transformed


def iter_controls(frames, labels):
    """Yield (name, RGB or None, metadata), retaining unsupported controls.

    Seven transformations reuse the old implementation byte for byte. New
    derivatives are streamed beside their source, limiting peak memory. A
    harmless-control expectation is a hypothesis, not a guarantee of passing
    the scorers. These RGB edits have no new joint semantic trajectory.
    """
    for old in historical_controls.iter_controls(frames, labels):
        name = old["name"]
        if name == "one_car_remains":
            continue
        metadata = dict(old["metadata"])
        metadata.update(applicable=old["applicable"], reason=old["reason"],
                        expected="accept" if name in BENIGN_NAMES else "reject")
        transformed = old["frames"]
        yield name, transformed, metadata

        if name == "unchanged":
            height, width = frames.shape[1:3]
            right = max(1, round(width / SETTINGS["translation_width_divisor"]))
            down = max(1, round(height / SETTINGS["translation_height_divisor"]))
            padded = np.pad(frames, ((0, 0), (down, 0), (right, 0), (0, 0)), mode="edge")
            translated = padded[:, :height, :width].copy()
            yield "small_translation", translated, {
                "applicable": True, "reason": None, "expected": "accept",
                "pixels_right": right, "pixels_down": down,
                "padding": "edge replication", "joint_semantic_trajectory": "unavailable",
                "note": "Constant whole-image displacement; not a simulation of 3D viewpoint change.",
            }
        elif name == "gray_silhouette":
            noisy_metadata = dict(metadata)
            noisy_metadata.update(seed=SETTINGS["gray_noise_seed"], noise_range=[-16, 16],
                                  noise_support="original per-frame car-label pixels",
                                  note="Independent local RGB noise added to the painted silhouette.")
            noisy = None
            if transformed is not None:
                masks = labels == historical_controls.CAR_LABEL
                noisy = _noisy_video(transformed, masks, SETTINGS["gray_noise_seed"])
            yield "gray_noisy", noisy, noisy_metadata
        elif name == "zoomed_still":
            height, width = frames.shape[1:3]
            patch_height = max(1, round(height * SETTINGS["zoom_noise_patch_height_fraction"]))
            patch_width = max(1, round(width * SETTINGS["zoom_noise_patch_width_fraction"]))
            top, left = (height - patch_height) // 2, (width - patch_width) // 2
            mask = np.zeros((height, width), dtype=bool)
            mask[top:top + patch_height, left:left + patch_width] = True
            masks = np.broadcast_to(mask, frames.shape[:3])
            noisy = _noisy_video(transformed, masks, SETTINGS["zoom_noise_seed"])
            noisy_metadata = dict(metadata)
            noisy_metadata.update(seed=SETTINGS["zoom_noise_seed"], noise_range=[-16, 16],
                                  noise_box_top_bottom_left_right=[top, top + patch_height, left, left + patch_width],
                                  note="Animated first-frame photograph plus independent noise in a fixed central patch.")
            yield "zoomed_still_noisy", noisy, noisy_metadata
