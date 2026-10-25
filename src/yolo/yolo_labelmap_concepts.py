#!/usr/bin/env python3
"""
Stage 2 (unified YOLO variant): produce a DENSE per-frame segmentation label map
from a frames/ folder, in either INSTANCE or SEMANTIC mode. One switch, one output
contract.

Why this script supersedes yolo_segment_concepts.py + yolo_semantic_concepts.py
-------------------------------------------------------------------------------
The auxiliary-task target for Goal 2/3 (L_seg) wants ONE dense (H, W) integer
label map per frame: every pixel carries a class id, 0 = background. That is a
clean cross-entropy target and it is identical regardless of which teacher made
it. So both modes here emit the SAME thing — a dense (H, W) label map in a stable,
thesis-controlled label space — and you can switch teachers without changing any
downstream code.

  --mode semantic : use a YOLO26 *-sem* model (Cityscapes or ADE20K weights).
                    The model already outputs a dense (H, W) class map; we just
                    remap its native class ids into our concept label space.
  --mode instance : use a YOLO26 *-seg* model (COCO-80 weights). The model
                    outputs one binary mask per detected object; we COMPOSITE
                    those masks into a dense (H, W) label map ourselves
                    (undetected pixels stay 0 = background).

Mode is auto-detected from the model filename ('-sem' -> semantic, '-seg' ->
instance) and can be forced with --mode.

IMPORTANT label-set caveat
--------------------------
Instance (-seg) weights are COCO-80. Semantic (-sem) weights are Cityscapes (~19
classes, street scenes) or ADE20K (~150 classes, indoor/scene parsing) — NOT
COCO-80. A concept only resolves if its name (or an alias) exists in the chosen
model's class list; concepts that don't exist there are skipped with a warning.
Match the weights to your scene: Cityscapes for street, ADE20K for indoor/kitchen.

Output
------
  <run_dir>/labelmaps/00000.npz ... one per frame, key 'label_map' = (H, W) uint16
  <run_dir>/labelmaps/label_space.json ... {id: name} + run metadata

With --save-uncertainty, semantic mode writes a separate directory by default:
  <run_dir>/labelmaps_uncertainty/00000.npz ... label_map plus confidence maps
  <run_dir>/labelmaps_uncertainty/label_space.json ... metadata and array contract

The uncertainty arrays are derived from the native semantic logits using the
installed Ultralytics predictor's exact resize/letterbox-removal path. Unmapped
native classes retain label_map=0 for backward compatibility but valid_mask=0,
so downstream training can ignore them instead of treating them as background.

Optional --legacy-masks ALSO writes the old per-concept binary format
(<run_dir>/sam_concepts/<concept>/<fidx>.npz with obj_ids/masks) so the existing
Stage 3 (sam_overlay_combined_png.py) keeps working during the transition.

Requires: pip install ultralytics   (wan22 env)
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Concept name -> set of lowercased class names that may appear in the model's
# vocabulary (COCO-80 for -seg; Cityscapes / ADE20K for -sem). Matching is by
# name against the actual model.names at runtime, so listing an alias that a
# given model doesn't have is harmless.
CONCEPT_ALIASES = {
    "person":       {"person", "rider"},
    "car":          {"car"},
    "bus":          {"bus"},
    "truck":        {"truck"},
    "bicycle":      {"bicycle", "bike"},
    "motorcycle":   {"motorcycle", "motorbike"},
    "train":        {"train"},
    "dog":          {"dog"},
    "cat":          {"cat"},
    "bench":        {"bench"},
    "backpack":     {"backpack", "bag"},
    "chair":        {"chair", "armchair", "swivel chair"},
    "couch":        {"couch", "sofa"},
    "tv":           {"tv", "tv receiver", "monitor", "screen", "crt screen"},
    "laptop":       {"laptop"},
    "keyboard":     {"keyboard"},
    "dining table": {"dining table", "table", "coffee table"},
    "refrigerator": {"refrigerator", "fridge"},
    "microwave":    {"microwave"},
    "sink":         {"sink"},
    "cup":          {"cup", "glass"},
    "bowl":         {"bowl"},
    "bottle":       {"bottle"},
    "potted plant": {"potted plant", "plant", "flower"},
    # --- semantic "stuff" classes (only exist in -sem vocabularies) ---
    "road":         {"road"},
    "sidewalk":     {"sidewalk", "pavement"},
    "building":     {"building", "edifice"},
    "vegetation":   {"vegetation", "tree"},
    "sky":          {"sky"},
    "wall":         {"wall"},
    "floor":        {"floor", "flooring"},
}

DEFAULT_CONCEPTS = ["person", "car", "bus", "bicycle", "dog", "bench", "backpack"]


def detect_mode(model_name: str) -> str:
    n = model_name.lower()
    if "-sem" in n:
        return "semantic"
    if "-seg" in n:
        return "instance"
    return "instance"  # safest default; override with --mode


def build_label_space(concepts, model_names, all_classes):
    """Return (label_space {id:name}, model_cls_to_thesis {model_cls_id:thesis_id})."""
    if all_classes:
        # Use the model's own vocabulary verbatim. Asymmetric across models
        # (COCO vs Cityscapes vs ADE20K), so don't mix runs of different models.
        label_space = {int(k): str(v) for k, v in model_names.items()}
        mapping = {int(k): int(k) for k in model_names}
        return label_space, mapping

    # Stable concept space: background=0, then concepts in the given order.
    thesis_id = {c: i + 1 for i, c in enumerate(concepts)}
    label_space = {0: "background", **{i + 1: c for i, c in enumerate(concepts)}}

    name_to_cls = {str(v).lower(): int(k) for k, v in model_names.items()}
    mapping = {}
    matched_concepts = set()
    for c in concepts:
        aliases = CONCEPT_ALIASES.get(c, set()) | {c.lower()}
        for nm_l, cls_id in name_to_cls.items():
            if nm_l in aliases:
                mapping[cls_id] = thesis_id[c]
                matched_concepts.add(c)
    for c in concepts:
        if c not in matched_concepts:
            print(f"[yolo] WARNING: concept {c!r} has no matching class in this "
                  f"model's vocabulary — it will be absent from the label map", flush=True)
    return label_space, mapping


def extract_semantic_label_map(r):
    """Pull a dense (H, W) native-class-id map from a *-sem* result.

    The exact attribute name can vary by ultralytics version. We try the known
    candidates and normalise shape. Returns int64 (H, W) or None if not found.
    """
    for attr in ("semantic_mask", "semantic", "seg_map"):
        obj = getattr(r, attr, None)
        if obj is None:
            continue
        data = getattr(obj, "data", obj)
        arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        if arr.ndim == 3 and arr.shape[0] == 1:        # [1, H, W]
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[0] > 1:       # [C, H, W] logits -> argmax
            arr = arr.argmax(0)
        if arr.ndim != 2:
            continue
        return arr.astype(np.int64)
    return None


def remap_native_label_map(native, mapping):
    """Map native semantic ids to thesis ids and flag mapped teacher pixels."""
    label_map = np.zeros_like(native, dtype=np.uint16)
    valid_mask = np.zeros_like(native, dtype=np.uint8)
    for src, dst in mapping.items():
        selected = native == src
        label_map[selected] = dst
        valid_mask[selected] = 1
    return label_map, valid_mask


def group_native_probabilities(probabilities, mapping, num_thesis_classes):
    """Sum native semantic probabilities into the stable thesis classes.

    Native classes absent from ``mapping`` contribute to class 0. This makes
    background/other a real probabilistic target instead of an invalid pixel.
    """
    native_classes = probabilities.shape[1]
    native_to_thesis = torch.zeros(
        native_classes, dtype=torch.long, device=probabilities.device
    )
    for native_id, thesis_id in mapping.items():
        if 0 <= native_id < native_classes:
            native_to_thesis[native_id] = thesis_id
    grouped = torch.zeros(
        (probabilities.shape[0], num_thesis_classes, *probabilities.shape[2:]),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    group_indices = native_to_thesis.view(1, -1, 1, 1).expand_as(probabilities)
    grouped.scatter_add_(1, group_indices, probabilities)
    probability_error = float((grouped.sum(dim=1) - 1.0).abs().max().item())
    if probability_error > 1e-5:
        raise RuntimeError(
            f"grouped thesis probabilities do not sum to one: {probability_error}"
        )
    return grouped


def semantic_uncertainty_forward(
    model,
    frame_path,
    mapping,
    num_thesis_classes,
    uncertainty_version=1,
    baseline_label_map=None,
):
    """Return verified hard labels and native-logit uncertainty for one frame.

    A normal prediction must have run once before this function so that
    ``model.predictor`` is initialised. The implementation intentionally calls
    the installed predictor's own ``postprocess`` and ``ops.scale_masks`` so the
    logit-derived hard map can be checked exactly against the public result.
    """
    image_bgr = cv2.imread(str(frame_path))
    if image_bgr is None:
        raise RuntimeError(f"failed to read frame: {frame_path}")

    predictor = model.predictor
    input_tensor = predictor.preprocess([image_bgr])
    with torch.inference_mode():
        raw_output = predictor.model(input_tensor)

    public_result = predictor.postprocess(raw_output, input_tensor, [image_bgr])[0]
    public_native = extract_semantic_label_map(public_result)
    if public_native is None:
        raise RuntimeError("installed predictor did not return a semantic class map")

    logits = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
    if not torch.is_tensor(logits):
        raise RuntimeError(f"semantic raw output is not a tensor: {type(logits)}")
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise RuntimeError(f"expected semantic logits [1,C,H,W], got {tuple(logits.shape)}")
    if not logits.is_floating_point():
        raise RuntimeError(
            "semantic model returned a baked class map, so confidence is unavailable"
        )
    if logits.shape[1] != len(model.names):
        raise RuntimeError(
            f"logit channels {logits.shape[1]} != model classes {len(model.names)}"
        )

    scores = logits.detach().float()
    if scores.shape[2:] != input_tensor.shape[2:]:
        scores = F.interpolate(scores, input_tensor.shape[2:], mode="bilinear")

    predictor_module = sys.modules.get(type(predictor).__module__)
    predictor_ops = getattr(predictor_module, "ops", None)
    if predictor_ops is None:
        raise RuntimeError("could not access the installed predictor's ops.scale_masks")
    scores = predictor_ops.scale_masks(scores, image_bgr.shape[:2])

    native_tensor = scores[0].argmax(dim=0)
    native = native_tensor.cpu().numpy().astype(np.int64)
    if native.shape != public_native.shape:
        raise RuntimeError(
            f"logit/public shape mismatch: {native.shape} != {public_native.shape}"
        )
    agreement = float(np.mean(native == public_native))
    if agreement < 0.9999:
        raise RuntimeError(
            f"raw logits do not reproduce public semantic map: agreement={agreement:.8f}"
        )

    probabilities = torch.softmax(scores, dim=1)
    top2 = probabilities.topk(2, dim=1).values
    confidence = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / np.log(probabilities.shape[1])

    fresh_label_map, mapped_foreground_mask = remap_native_label_map(native, mapping)
    if uncertainty_version == 1:
        arrays = {
            "label_map": fresh_label_map,
            "teacher_confidence": confidence[0].cpu().numpy().astype(np.float16),
            "teacher_margin": margin[0].cpu().numpy().astype(np.float16),
            "teacher_entropy": entropy[0].cpu().numpy().astype(np.float16),
            "valid_mask": mapped_foreground_mask,
        }
        bounded_float_keys = (
            "teacher_confidence",
            "teacher_margin",
            "teacher_entropy",
        )
    elif uncertainty_version == 2:
        target_label_map = (
            fresh_label_map
            if baseline_label_map is None
            else baseline_label_map.astype(np.uint16, copy=False)
        )
        if target_label_map.shape != fresh_label_map.shape:
            raise RuntimeError(
                f"baseline shape {target_label_map.shape} != "
                f"fresh shape {fresh_label_map.shape}"
            )
        if target_label_map.size and int(target_label_map.max()) >= num_thesis_classes:
            raise RuntimeError(
                f"baseline class id {int(target_label_map.max())} is outside "
                f"the {num_thesis_classes}-class thesis label space"
            )
        grouped = group_native_probabilities(
            probabilities, mapping, num_thesis_classes
        )
        target_tensor = torch.from_numpy(
            target_label_map.astype(np.int64, copy=False)
        ).to(grouped.device).view(1, 1, *target_label_map.shape)
        target_probability = grouped.gather(1, target_tensor)[:, 0]
        alternatives = grouped.clone()
        alternatives.scatter_(1, target_tensor, -1.0)
        strongest_alternative = alternatives.max(dim=1).values
        grouped_margin = target_probability - strongest_alternative
        grouped_entropy = -(
            grouped * grouped.clamp_min(1e-8).log()
        ).sum(dim=1) / np.log(num_thesis_classes)
        agreement_mask = (fresh_label_map == target_label_map).astype(np.uint8)
        numerical_valid_mask = np.ones_like(fresh_label_map, dtype=np.uint8)
        arrays = {
            "label_map": target_label_map.astype(np.uint16, copy=True),
            "teacher_target_probability": (
                target_probability[0].cpu().numpy().astype(np.float16)
            ),
            "teacher_grouped_margin": (
                grouped_margin[0].cpu().numpy().astype(np.float16)
            ),
            "teacher_grouped_entropy": (
                grouped_entropy[0].cpu().numpy().astype(np.float16)
            ),
            "numerical_valid_mask": numerical_valid_mask,
            "agreement_mask": agreement_mask,
            "mapped_foreground_mask": mapped_foreground_mask,
        }
        bounded_float_keys = (
            "teacher_target_probability",
            "teacher_grouped_entropy",
        )
    else:
        raise ValueError(f"unsupported uncertainty version: {uncertainty_version}")
    expected_shape = native.shape
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise RuntimeError(f"{name} shape {array.shape} != {expected_shape}")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contains non-finite values")
    for name in bounded_float_keys:
        array = arrays[name]
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise RuntimeError(
                f"{name} must be in [0,1], got [{array.min()}, {array.max()}]"
            )
    if uncertainty_version == 2:
        grouped_margin_array = arrays["teacher_grouped_margin"]
        if (
            float(grouped_margin_array.min()) < -1.0
            or float(grouped_margin_array.max()) > 1.0
        ):
            raise RuntimeError("teacher_grouped_margin must be in [-1,1]")
    return arrays, agreement, fresh_label_map


def composite_instance_label_map(r, mapping, orig_h, orig_w):
    """Paint per-instance masks into a dense (H, W) thesis-id map.

    Overlap rule: higher-confidence instances win (painted last). Undetected
    pixels stay 0 (background).
    """
    label_map = np.zeros((orig_h, orig_w), dtype=np.uint16)
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return label_map

    masks = r.masks.data.cpu().numpy()                 # [N, h, w] in [0,1]
    if masks.shape[1:] != (orig_h, orig_w):
        masks = np.stack([
            cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            for m in masks
        ])
    masks = masks > 0.5
    cls = r.boxes.cls.cpu().numpy().astype(int)
    conf = r.boxes.conf.cpu().numpy()

    for i in np.argsort(conf):                          # ascending: high conf last
        thesis = mapping.get(int(cls[i]))
        if thesis is None:                              # class not in our concept set
            continue
        label_map[masks[i]] = thesis
    return label_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="Stage 1 output dir (contains frames/ and meta.json)")
    ap.add_argument("--model", default="yolo26x-seg.pt",
                    help="YOLO weights. '-seg' = instance (COCO), '-sem' = semantic "
                         "(Cityscapes/ADE20K). Default: yolo26x-seg.pt")
    ap.add_argument("--mode", choices=["instance", "semantic", "auto"], default="auto",
                    help="Force teacher mode; 'auto' infers from the model filename")
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="Concept list = label space order (ids 1..N). Default: street scene")
    ap.add_argument("--all-classes", action="store_true",
                    help="Use the model's full native vocabulary instead of --concepts "
                         "(label ids = model class ids; do not mix different models)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45,
                    help="NMS IoU (instance mode + --no-end2end only)")
    ap.add_argument("--no-end2end", action="store_true",
                    help="Instance mode: disable NMS-free head, use traditional NMS")
    ap.add_argument("--legacy-masks", action="store_true",
                    help="Also write old per-concept binary .npz (Stage 3 compatibility)")
    ap.add_argument("--save-uncertainty", action="store_true",
                    help="Semantic mode: save confidence, margin, entropy, and valid_mask")
    ap.add_argument("--uncertainty-version", type=int, choices=[1, 2], default=1,
                    help="1=native-class legacy arrays; 2=thesis-grouped arrays")
    ap.add_argument("--output-subdir", default=None,
                    help="Output folder under --run-dir. Default: labelmaps, or "
                         "labelmaps_uncertainty with --save-uncertainty")
    ap.add_argument("--expected-frames", type=int, default=None,
                    help="Fail unless this many input frames and output .npz files exist")
    ap.add_argument("--max-baseline-disagreement", type=float, default=0.001,
                    help="Uncertainty mode: maximum allowed fraction of pixels that "
                         "differ from an existing hard-label baseline (default: 0.001)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    frames_dir = run_dir / "frames"
    assert frames_dir.is_dir(), f"no frames/ in {run_dir}"
    concepts = args.concepts if args.concepts else DEFAULT_CONCEPTS

    mode = detect_mode(args.model) if args.mode == "auto" else args.mode
    if args.save_uncertainty and mode != "semantic":
        ap.error("--save-uncertainty is supported only with --mode semantic")
    if args.uncertainty_version == 2 and not args.save_uncertainty:
        ap.error("--uncertainty-version 2 requires --save-uncertainty")
    if not 0.0 <= args.max_baseline_disagreement <= 1.0:
        ap.error("--max-baseline-disagreement must be in [0,1]")

    output_subdir = args.output_subdir or (
        "labelmaps_uncertainty" if args.save_uncertainty else "labelmaps"
    )
    out_root = run_dir / output_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    legacy_root = run_dir / "sam_concepts"

    from ultralytics import YOLO
    model = YOLO(args.model)

    label_space, mapping = build_label_space(concepts, model.names, args.all_classes)
    inv_label_space = {v: k for k, v in label_space.items()}  # name -> id (for legacy)

    frame_files = sorted(frames_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    n_frames = len(frame_files)
    if not frame_files:
        raise RuntimeError(f"no JPEG frames in {frames_dir}")
    if args.expected_frames is not None and n_frames != args.expected_frames:
        raise RuntimeError(
            f"expected {args.expected_frames} input frames, found {n_frames} in {frames_dir}"
        )
    end2end = not args.no_end2end

    print(f"[yolo] mode     = {mode}", flush=True)
    print(f"[yolo] model    = {args.model}", flush=True)
    print(f"[yolo] frames   = {frames_dir}  ({n_frames} frames)", flush=True)
    print(f"[yolo] device   = {'cuda:' + str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"[yolo] label_space = {label_space}", flush=True)
    print(f"[yolo] output   = {out_root}", flush=True)
    print(f"[yolo] uncertainty = {args.save_uncertainty}", flush=True)
    if mode == "instance":
        print(f"[yolo] end2end  = {end2end} ({'NMS-free' if end2end else 'NMS iou=' + str(args.iou)})", flush=True)

    semantic_warned = False
    uncertainty_agreements = []
    baseline_disagreements = []
    baseline_label_maps_checked = 0
    baseline_disagreement_pixels_marked = 0
    n_written = 0
    for fidx, frame_path in enumerate(frame_files):
        if mode == "semantic":
            if args.save_uncertainty:
                # Initialise the Ultralytics predictor once. All frames, including
                # this first one, are then processed by the verified raw-logit path.
                if getattr(model, "predictor", None) is None:
                    model(str(frame_path), conf=args.conf, verbose=False)
                baseline_path = run_dir / "labelmaps" / f"{fidx:05d}.npz"
                baseline_label_map = None
                if baseline_path.is_file():
                    with np.load(baseline_path) as baseline_data:
                        baseline_label_map = baseline_data["label_map"]
                arrays, agreement, fresh_label_map = semantic_uncertainty_forward(
                    model,
                    frame_path,
                    mapping,
                    len(label_space),
                    uncertainty_version=args.uncertainty_version,
                    baseline_label_map=baseline_label_map,
                )
                label_map = arrays["label_map"]
                uncertainty_agreements.append(agreement)
            else:
                r = model(str(frame_path), conf=args.conf, verbose=False)[0]
                native = extract_semantic_label_map(r)
                if native is None:
                    if not semantic_warned:
                        avail = [a for a in dir(r) if not a.startswith("_")]
                        print(f"[yolo] ERROR: no semantic label map on result. This "
                              f"ultralytics build may name the attribute differently. "
                              f"Available result attributes: {avail}", flush=True)
                        semantic_warned = True
                    continue
                orig_h, orig_w = r.orig_shape
                if native.shape != (orig_h, orig_w):
                    native = cv2.resize(native.astype(np.int32), (orig_w, orig_h),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int64)
                label_map, _ = remap_native_label_map(native, mapping)

        else:  # instance
            predict_kwargs = dict(conf=args.conf, end2end=end2end, verbose=False)
            if not end2end:
                predict_kwargs["iou"] = args.iou
            # restrict to classes we care about (skip when using full vocabulary)
            if not args.all_classes and mapping:
                predict_kwargs["classes"] = sorted(mapping.keys())
            r = model(str(frame_path), **predict_kwargs)[0]
            orig_h, orig_w = r.orig_shape
            label_map = composite_instance_label_map(r, mapping, orig_h, orig_w)

        output_path = out_root / f"{fidx:05d}.npz"
        if args.save_uncertainty:
            baseline_path = run_dir / "labelmaps" / f"{fidx:05d}.npz"
            if baseline_path.is_file():
                with np.load(baseline_path) as baseline_data:
                    baseline_label_map = baseline_data["label_map"]
                if baseline_label_map.shape != label_map.shape:
                    raise RuntimeError(
                        f"{frame_path.name}: baseline shape {baseline_label_map.shape} "
                        f"!= uncertainty shape {label_map.shape}"
                    )
                disagreement_mask = fresh_label_map != baseline_label_map
                disagreement = float(np.mean(disagreement_mask))
                if disagreement > args.max_baseline_disagreement:
                    raise RuntimeError(
                        f"{frame_path.name}: uncertainty label_map differs from "
                        f"existing baseline by {disagreement:.8f} of pixels; maximum "
                        f"allowed is {args.max_baseline_disagreement:.8f}"
                    )
                if disagreement_mask.any():
                    invalidated = int(disagreement_mask.sum())
                    if args.uncertainty_version == 1:
                        arrays["valid_mask"][disagreement_mask] = 0
                    baseline_disagreement_pixels_marked += invalidated
                    print(
                        f"[yolo]   {frame_path.name}: preserving baseline labels and "
                        f"marking {invalidated} disagreement pixels "
                        f"({disagreement:.8f})",
                        flush=True,
                    )
                # Preserve the original hard-label target exactly. Pixels whose
                # fresh logits disagree are excluded through valid_mask above.
                label_map = baseline_label_map.astype(np.uint16, copy=True)
                arrays["label_map"] = label_map
                baseline_disagreements.append(disagreement)
                baseline_label_maps_checked += 1
            np.savez_compressed(output_path, **arrays)
        else:
            np.savez_compressed(output_path, label_map=label_map)
        n_written += 1

        if args.legacy_masks:
            for cid, cname in label_space.items():
                if cid == 0:
                    continue
                binmask = (label_map == cid).astype(np.uint8)
                if binmask.sum() == 0:
                    continue
                cdir = legacy_root / cname.replace(" ", "_")
                cdir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cdir / f"{fidx:05d}.npz",
                                    obj_ids=np.array([0]),
                                    masks=binmask[np.newaxis])

        if (fidx + 1) % 20 == 0 or fidx == n_frames - 1:
            print(f"[yolo]   frame {fidx + 1}/{n_frames}", flush=True)

    output_files = sorted(out_root.glob("[0-9][0-9][0-9][0-9][0-9].npz"))
    if n_written != n_frames or len(output_files) != n_frames:
        raise RuntimeError(
            f"incomplete export: inputs={n_frames}, written={n_written}, "
            f"output_files={len(output_files)}"
        )
    if args.expected_frames is not None and len(output_files) != args.expected_frames:
        raise RuntimeError(
            f"expected {args.expected_frames} outputs, found {len(output_files)}"
        )

    meta = {
        "mode": mode,
        "model": args.model,
        "conf": args.conf,
        "all_classes": args.all_classes,
        "n_frames": n_frames,
        "output_subdir": output_subdir,
        "save_uncertainty": args.save_uncertainty,
        "uncertainty_version": args.uncertainty_version,
        "label_space": {str(k): v for k, v in label_space.items()},
        "note": "label_map is dense (H,W) uint16; pixel value = class id; 0 = background/unmapped",
    }
    if args.save_uncertainty:
        if args.uncertainty_version == 1:
            array_contract = {
                "label_map": "uint16 thesis class id; 0 for background/unmapped",
                "teacher_confidence": "float16 native top-class softmax probability",
                "teacher_margin": "float16 native top-1 minus top-2 probability",
                "teacher_entropy": "float16 normalized native-class entropy",
                "valid_mask": "uint8; 1 only when native class maps to thesis space "
                              "and fresh hard label agrees with an existing baseline",
            }
        else:
            array_contract = {
                "label_map": "uint16 byte-identical baseline thesis class id",
                "teacher_target_probability": "float16 grouped P(stored target), [0,1]",
                "teacher_grouped_margin": "float16 P(target)-max P(other), [-1,1]",
                "teacher_grouped_entropy": "float16 normalized 11-class entropy, [0,1]",
                "numerical_valid_mask": "uint8 finite-value mask; expected all ones",
                "agreement_mask": "uint8 fresh remapped hard label agrees with baseline",
                "mapped_foreground_mask": "uint8 native winner maps to thesis foreground",
            }
        meta.update({
            "array_contract": array_contract,
            "min_reconstruction_agreement": min(uncertainty_agreements),
            "mean_reconstruction_agreement": float(np.mean(uncertainty_agreements)),
            "max_allowed_baseline_disagreement": args.max_baseline_disagreement,
            "baseline_label_maps_checked": baseline_label_maps_checked,
            "max_baseline_disagreement": (
                max(baseline_disagreements) if baseline_disagreements else None
            ),
            "mean_baseline_disagreement": (
                float(np.mean(baseline_disagreements))
                if baseline_disagreements else None
            ),
            "baseline_disagreement_pixels_marked": baseline_disagreement_pixels_marked,
            "baseline_disagreement_action": (
                "valid_mask set to zero"
                if args.uncertainty_version == 1
                else "agreement_mask set to zero; pixel remains a positive-weight target"
            ),
        })
    with open(out_root / "label_space.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[yolo] DONE -> {out_root}", flush=True)
    print(f"[yolo] {json.dumps(meta, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
