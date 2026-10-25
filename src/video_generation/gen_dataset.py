#!/usr/bin/env python3
"""
Segmentation-decoder dataset generator (Goal 2).

Generates Wan 2.1 *street* clips for training the latent->segmentation decoder
f_θ. Design: "different scenes, same objects" — four street sub-scenes, each
prompted to stage the SAME core class set, generated across several seeds so the
decoder cannot win by memorising a fixed street layout (see the mismatched-latent
control in the eval).

Writes ONLY what the decoder dataset needs, into a SEPARATE dataset root:
    final_latent.pt  -> f_θ input   ([1,16,21,60,104] bf16)
    frames/          -> YOLO Stage-2 input + human viewing (00000.jpg..00080.jpg)
    video.mp4        -> human viewing
    meta.json        -> provenance incl. scene / seed / split
No intermediate (capture-step) latents, no legacy masks, no runs.csv row — this
dataset is self-contained under --out-root.

Two modes:
  generate (default): one clip per invocation (this is what the Slurm array calls)
      python gen_dataset.py --scene intersection --seed 42 --split train --out-root <root>
  --build-manifest: scan <root>/clips/*/meta.json -> <root>/dataset.csv
      python gen_dataset.py --build-manifest --out-root <root>
  (build-manifest is light — no torch/GPU — so run it on the login node after the
   generation array finishes.)

Runs in the wan22 env (same as gen_video_for_sam.py). Heavy imports (torch,
diffusers, the pipeline) are deferred into generate() so --build-manifest and
--help work with plain python.
"""
import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path

# ---- paths (M3) ----
MODEL_DIR = "/home/dhiazr/gt69_scratch/dhiazgeo/wanti2v/models/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_OUT_ROOT = "/home/dhiazr/gt69/dhiazgeo/wanti2v/outputs/seg_decoder_ds_v2"

# ---- Wan 2.1 1.3B defaults (480p) — matched to gen_video_for_sam.py ----
HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 81           # 81 = 4*20+1 (Wan temporal VAE factor)
NUM_STEPS = 50
GUIDANCE = 5.0
FLOW_SHIFT = 3.0
FPS = 16

# ---- shared core class set (the "same objects") ----
# Every scene prompt stages these; pass the SAME list to Stage-2 YOLO --concepts
# so the label space (background=0, then 1..N in this order) is identical across
# all clips. All ten are present in yolo_labelmap_concepts.py's CONCEPT_ALIASES
# for the Cityscapes (-sem) vocabulary.
CORE_CONCEPTS = [
    "person", "car", "bus", "bicycle", "truck",
    "road", "sidewalk", "building", "vegetation", "sky",
]

# ---- four street sub-scenes (each stages every CORE_CONCEPT) ----
# Design (v2): SPARSE, well-separated instances (no crowds — crowds are a
# cheap-inconsistency trap that only injects pseudo-GT label noise and leaves no
# headroom for the geometric-consistency metric). Every scene shares TWO cues so
# ego-motion is uniform and reliable: (1) the locomotion clause "smooth
# first-person walk moving steadily forward", and (2) a "world streaming past"
# clause (structure sliding past / receding ahead) — apparent motion that only
# forward camera translation can produce, so Wan is forced to actually walk
# rather than hold a static establishing shot. Result: forward parallax +
# independent object motion at a smooth even pace (small per-frame displacement
# -> stable targets, fair consistency). Different layouts, same object set.
SCENE_PROMPTS = {
    "intersection": (
        "Smooth first-person walk moving steadily forward through a calm "
        "intersection, low buildings drifting past on both sides: a single car "
        "and one bus rolling across ahead, one cyclist, a lone pedestrian, a "
        "parked truck, open road, clean sidewalk, roadside trees, clear sky."
    ),
    "residential": (
        "Smooth first-person walk moving steadily forward down a quiet "
        "residential road, houses sliding past and the sidewalk receding "
        "ahead: a parked car, a bus pulling away, a cyclist passing, a "
        "delivery truck rolling ahead, one pedestrian, garden trees, pale sky."
    ),
    "avenue": (
        "Smooth first-person walk moving steadily forward along an open "
        "avenue, glass facades sliding past and the sidewalk receding ahead: a "
        "couple of cars and one bus on the road, a cyclist gliding past, a "
        "truck ahead, one pedestrian, planted trees, bright sky."
    ),
    "parkside": (
        "Smooth first-person walk moving steadily forward beside a park-side "
        "road, trees drifting past and the road receding ahead: a car and a "
        "bus passing on the left, one cyclist, a distant truck, a single "
        "pedestrian, wide sidewalk, grass verge, buildings behind, clear sky."
    ),
}

NEGATIVE_PROMPT = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
    "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
    "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
    "杂乱的背景,三条腿,背景人很多,倒着走"
)

MANIFEST_FIELDS = ["clip_id", "scene", "seed", "split", "n_frames",
                   "latent_path", "frames_dir", "prompt"]


def generate(scene: str, seed: int, split: str, out_root: Path, overwrite: bool) -> int:
    # Heavy imports live here so --build-manifest / --help stay light.
    import time
    import numpy as np
    import torch
    from diffusers import AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video
    from PIL import Image
    # Import the pipeline from THIS repo's src/ (not the legacy wanti2v copy),
    # so result.final_latent exists. Same anchor as gen_video_for_sam.py.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from my_wan_pipeline import MyWanPipeline

    prompt = SCENE_PROMPTS[scene]
    clip_id = f"{scene}_s{seed}"
    run_dir = out_root / "clips" / clip_id
    frames_dir = run_dir / "frames"
    latent_path = run_dir / "final_latent.pt"

    if latent_path.exists() and not overwrite:
        print(f"[gen] {clip_id}: final_latent.pt exists, skipping (use --overwrite to force)", flush=True)
        return 0

    frames_dir.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "interactive")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[gen] clip_id = {clip_id}  split = {split}", flush=True)
    print(f"[gen] run_dir = {run_dir}", flush=True)
    print(f"[gen] prompt  = {prompt!r}", flush=True)
    print(f"[gen] GPU     = {torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.time()
    print("[gen] loading VAE (fp32) + pipeline (bf16)...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae", torch_dtype=torch.float32)
    pipe = MyWanPipeline.from_pretrained(MODEL_DIR, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)
    pipe.to("cuda")
    print(f"[gen] ready in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print("[gen] generating...", flush=True)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT, width=WIDTH,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE,
        num_inference_steps=NUM_STEPS,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    frames = result.frames[0]
    print(f"[gen] {len(frames)} frames in {time.time()-t1:.1f}s", flush=True)

    # final latent (f_θ input) — the whole reason this dataset exists
    torch.save(result.final_latent, latent_path)
    print(f"[gen] saved final_latent {tuple(result.final_latent.shape)} -> {latent_path}", flush=True)

    # numbered JPEGs (YOLO Stage-2 input) + mp4 (viewing)
    for i, frame in enumerate(frames):
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(
            np.asarray(frame * 255 if np.asarray(frame).max() <= 1.0 else frame).astype(np.uint8)
        )
        img.save(frames_dir / f"{i:05d}.jpg", quality=95)
    mp4_path = run_dir / "video.mp4"
    export_to_video(frames, str(mp4_path), fps=FPS)
    print(f"[gen] wrote {len(frames)} jpgs + mp4", flush=True)

    meta = {
        "clip_id": clip_id, "scene": scene, "seed": seed, "split": split,
        "prompt": prompt, "core_concepts": CORE_CONCEPTS,
        "height": HEIGHT, "width": WIDTH, "num_frames": len(frames),
        "fps": FPS, "num_inference_steps": NUM_STEPS, "guidance_scale": GUIDANCE,
        "flow_shift": FLOW_SHIFT, "frames_dir": str(frames_dir),
        "mp4": str(mp4_path), "final_latent": str(latent_path),
        "latent_shape": list(result.final_latent.shape),
        "job_id": job_id, "timestamp": ts,
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] wrote meta.json", flush=True)
    print(f"[gen] DONE -> {run_dir}", flush=True)
    return 0


def build_manifest(out_root: Path) -> int:
    clips_dir = out_root / "clips"
    if not clips_dir.is_dir():
        print(f"[manifest] no clips/ under {out_root}", flush=True)
        return 1
    rows = []
    for meta_path in sorted(clips_dir.glob("*/meta.json")):
        with open(meta_path) as f:
            m = json.load(f)
        rows.append({
            "clip_id": m["clip_id"], "scene": m["scene"], "seed": m["seed"],
            "split": m["split"], "n_frames": m["num_frames"],
            "latent_path": m["final_latent"], "frames_dir": m["frames_dir"],
            "prompt": m["prompt"],
        })
    manifest = out_root / "dataset.csv"
    with open(manifest, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_train = sum(r["split"] == "train" for r in rows)
    n_val = sum(r["split"] == "val" for r in rows)
    print(f"[manifest] wrote {len(rows)} clips ({n_train} train / {n_val} val) -> {manifest}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", choices=list(SCENE_PROMPTS),
                    help="street sub-scene (generate mode)")
    ap.add_argument("--seed", type=int, help="random seed (generate mode)")
    ap.add_argument("--split", choices=["train", "val"],
                    help="dataset split this clip belongs to (generate mode)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help=f"dataset root (default: {DEFAULT_OUT_ROOT})")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if final_latent.pt already exists")
    ap.add_argument("--build-manifest", action="store_true",
                    help="scan <out-root>/clips/*/meta.json -> dataset.csv, then exit")
    args = ap.parse_args()
    out_root = Path(args.out_root)

    if args.build_manifest:
        return build_manifest(out_root)

    missing = [n for n, v in (("--scene", args.scene), ("--seed", args.seed),
                              ("--split", args.split)) if v is None]
    if missing:
        ap.error(f"generate mode needs {', '.join(missing)}")
    return generate(args.scene, args.seed, args.split, out_root, args.overwrite)


if __name__ == "__main__":
    sys.exit(main())
