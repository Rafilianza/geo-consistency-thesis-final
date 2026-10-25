"""Generate RGB and segmentation together from validation/new text prompts."""

import numpy as np
import torch

from codec_retraining.data import PALETTE
from joint_semantic_mixed.common import (
    CLASS_NAMES, SEMANTIC_SHAPE, check_tensor,
    file_hash, fingerprint, load_prepared, load_wan, read_json, runtime, wan_inventory, write_json,
)
from middle_feature_semantic.phase1_semantic_models import FeatureSemanticFlowModel
from middle_feature_semantic.wan_middle_feature_capture import feature_tokens_to_video_grid


class JointSemanticObserver:
    """Read conditional Wan features, advance semantics, and return no RGB edits."""

    def __init__(self, pipeline, semantic, seed, device):
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        self.pipeline, self.semantic, self.device = pipeline, semantic, device
        self.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        self.scheduler.set_timesteps(50, device=device)
        generator = torch.Generator().manual_seed(100000 + seed)
        self.state = torch.randn(SEMANTIC_SHAPE, generator=generator).to(device)
        self.feature = None
        self.calls = 0
        self.steps = 0

    def capture(self, module, inputs, output):
        self.calls += 1
        # MyWanPipeline calls conditional first, unconditional second at CFG=5.
        if self.calls == 1:
            self.feature = output.detach().clone()
        return None

    @torch.no_grad()
    def step(self, step_idx, t, latents, noise_pred, prompt_embeds):
        if self.calls != 2 or self.feature is None:
            raise RuntimeError("Expected exactly one conditional and one unconditional Wan forward.")
        if float(t) != float(self.scheduler.timesteps[step_idx]):
            raise RuntimeError("RGB and semantic schedules disagree.")
        features = feature_tokens_to_video_grid(self.feature, latents.shape, (1, 2, 2))
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            flow = self.semantic(self.state, features, t.float().reshape(1)).float()
        self.state = self.scheduler.step(flow, t, self.state, return_dict=False)[0]
        check_tensor(self.state, SEMANTIC_SHAPE)
        self.feature, self.calls = None, 0
        self.steps += 1
        # Returning None makes the pipeline use its original RGB latent/flow.
        return None


def example_cases(manifest, new_prompt):
    if new_prompt in {record["prompt"] for record in manifest["records"]}:
        raise ValueError("The new prompt must differ from every training/validation prompt.")
    if not new_prompt.strip():
        raise ValueError("Provide a nonempty new prompt.")
    cases, seen = [], set()
    for record in manifest["records"]:
        if record["split"] == "val" and record["scene"] not in seen:
            cases.append({"case_id": "validation_" + record["clip_id"], "role": "validation_prompt",
                          "prompt": record["prompt"], "seed": record["seed"]})
            seen.add(record["scene"])
    cases.append({"case_id": "new_prompt_s51", "role": "new_prompt_qualitative_example", "prompt": new_prompt, "seed": 51})
    return cases


def generate_rgb(pipeline, case, device, observer=None, embeddings=None, output_type="np"):
    from video_generation.gen_dataset import NEGATIVE_PROMPT
    hook = None
    if observer is not None:
        hook = pipeline.transformer.blocks[15].register_forward_hook(observer.capture)
    prompt_inputs = {"prompt": case["prompt"], "negative_prompt": NEGATIVE_PROMPT}
    if embeddings is not None:
        prompt_inputs = {"prompt_embeds": embeddings[0], "negative_prompt_embeds": embeddings[1]}
    try:
        return pipeline(**prompt_inputs, height=480, width=832,
                        num_frames=81, num_inference_steps=50, guidance_scale=5.0,
                        generator=torch.Generator(device=device).manual_seed(case["seed"]),
                        step_intervention=None if observer is None else observer.step,
                        max_sequence_length=512, verbose=False, output_type=output_type)
    finally:
        if hook is not None:
            hook.remove()


@torch.no_grad()
def semantic_labels(codec, normalized, chunk_frames=9):
    raw = codec.denormalize(normalized.float())
    parts = []
    for start, end, logits in codec.decode_chunks(raw, chunk_frames):
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite segmentation logits.")
        parts.append(logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8))
    return np.concatenate(parts)


def save_videos(directory, frames, labels):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    if labels.ndim != 3 or labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
        raise ValueError("Unexpected semantic label IDs.")
    paths = [directory / name for name in ("generated_rgb.mp4", "generated_semantic.mp4", "rgb_and_semantic.mp4")]
    writers = [imageio.get_writer(str(path), fps=16, codec="libx264", pixelformat="yuv420p", macro_block_size=1) for path in paths]
    try:
        for frame_index, frame in enumerate(frames):
            pixels = np.asarray(frame)
            if pixels.shape != (*labels.shape[1:], 3) or not np.isfinite(pixels).all():
                raise ValueError("Invalid generated RGB frame.")
            if pixels.min() < 0 or pixels.max() > 1:
                raise ValueError("Expected generated RGB pixels in [0, 1].")
            rgb = np.clip(np.rint(pixels * 255), 0, 255).astype(np.uint8)
            colors = PALETTE[labels[frame_index]]
            panel = Image.fromarray(np.concatenate([rgb, colors], axis=1))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 20), fill="black")
            draw.text((8, 4), "Generated RGB", fill="white")
            draw.text((840, 4), "Jointly generated segmentation", fill="white")
            writers[0].append_data(rgb)
            writers[1].append_data(colors)
            writers[2].append_data(np.asarray(panel))
            if frame_index in (0, 40, 80):
                panel.save(directory / f"preview_{frame_index:05d}.jpg", quality=90)
    finally:
        for writer in writers:
            writer.close()
    np.savez_compressed(directory / "predicted_labels.npz", label_map=labels)


def examples(args, device):
    from joint_semantic_mixed import monitoring

    manifest = load_prepared(args)
    completion = read_json(args.weights_dir / "training_complete.json")
    best_path = args.weights_dir / "semantic_best.pt"
    if file_hash(best_path) != completion["best_weights_sha256"]:
        raise ValueError("Selected CNN checkpoint changed.")
    training_config = read_json(args.weights_dir / "training_config.json")
    if str(args.model_dir.resolve()) != training_config["model_dir"] or runtime() != training_config["runtime"]:
        raise ValueError("Use the same Wan path and runtime as CNN training.")
    if wan_inventory(args.model_dir) != training_config["model_inventory"]:
        raise ValueError("Frozen Wan files differ from the CNN training run.")
    teacher_info = monitoring.teacher_provenance(args.yolo_model)
    if teacher_info != training_config["yolo"] or args.new_prompt != training_config["final_new_prompt"]:
        raise ValueError("Keep the configured YOLO teacher and final prompt unchanged.")
    cases = example_cases(manifest, args.new_prompt)
    root = args.outputs_dir / "examples"
    root.mkdir(exist_ok=True)
    config = {"cases": cases, "cnn_sha256": file_hash(best_path), "codec_sha256": manifest["weight_files"]["frozen_codec.pt"],
              "training_config_sha256": file_hash(args.weights_dir / "training_config.json"), "yolo": teacher_info,
              "input": "text prompt and fresh noise only; no reference video or mask conditions generation",
              "sampler": "independent RGB and semantic UniPC schedulers; RGB CFG=5, conditional semantic branch",
              "checkpoint": "best fixed-noise validation loss; may precede the last training update",
              "repeatability": "RGB weights are frozen; identical RGB across generations is not required"}
    config_path = root / "config.json"
    if config_path.exists() and (not args.resume or read_json(config_path) != config):
        raise ValueError("Examples exist. Use --resume unchanged, or a new run.")
    write_json(config_path, config)
    # A completed examples stage can be verified without loading any GPU model.
    if (root / "complete.json").exists():
        receipt = read_json(root / "complete.json")
        if receipt["config_fingerprint"] != fingerprint(config):
            raise ValueError("Final examples belong to different settings.")
        for case in cases:
            directory = root / case["case_id"]
            saved = read_json(directory / "complete.json")
            for name, expected in saved["artifacts"].items():
                if file_hash(directory / name) != expected:
                    raise ValueError(f"Changed final example: {directory / name}")
        print("Final examples already complete and verified.")
        return
    pipeline = load_wan(args.model_dir, device)
    semantic = FeatureSemanticFlowModel().to(device)
    saved = torch.load(best_path, map_location="cpu", weights_only=True)
    if saved["config_fingerprint"] != fingerprint(training_config):
        raise ValueError("CNN checkpoint differs from its training configuration.")
    semantic.load_state_dict(saved["model_state_dict"])
    semantic.requires_grad_(False).eval()
    embeddings = monitoring.cache_prompt_embeddings(pipeline, cases, device)
    context = monitoring.create_context(args, device, embeddings, teacher_info)
    monitoring.evaluate_cases(pipeline, semantic, cases, context, root, saved["update"],
                              saved["update"] / completion["train_clips"], fingerprint(config), keep_videos=True)
    write_json(root / "complete.json", {"cases": cases, "status": "generated_and_scored_requires_visual_review",
                                        "class_names": CLASS_NAMES, "config_fingerprint": fingerprint(config)})
