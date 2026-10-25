"""Train only the semantic CNN with matched-noise RGB/semantic states."""

from pathlib import Path
import time

import torch
import torch.nn.functional as F

from joint_semantic_mixed.common import (
    FORMAT, RGB_SHAPE, SEMANTIC_SHAPE, atomic_torch_save, check_tensor, file_hash,
    fingerprint, load_prepared, load_wan, read_json, runtime, save_history, write_json,
)
from middle_feature_semantic.phase1_semantic_models import FeatureSemanticFlowModel
from middle_feature_semantic.wan_middle_feature_capture import WanMiddleFeatureCapture, feature_tokens_to_video_grid
from joint_semantic_mixed import monitoring


@torch.no_grad()
def capture_feature(transformer, rgb, timestep, embedding):
    with WanMiddleFeatureCapture(transformer, 15) as capture:
        transformer(hidden_states=rgb.to(transformer.dtype), timestep=timestep,
                    encoder_hidden_states=embedding.to(transformer.dtype), return_dict=False)
        if capture.call_count != 1:
            raise ValueError("Expected one conditional Wan feature capture.")
        tokens = capture.feature
    return feature_tokens_to_video_grid(tokens, rgb.shape, transformer.config.patch_size).detach()


def flow_loss(model, pipeline, record, semantic, rgb, embedding, schedule_index, generator, device):
    timestep = pipeline.scheduler.timesteps[schedule_index].float().reshape(1)
    sigma = pipeline.scheduler.sigmas[schedule_index].to(device).float()
    rgb_noise = torch.randn(rgb.shape, generator=generator).to(device)
    semantic_noise = torch.randn(semantic.shape, generator=generator).to(device)
    noisy_rgb = (1 - sigma) * rgb + sigma * rgb_noise
    noisy_semantic = (1 - sigma) * semantic + sigma * semantic_noise
    target = semantic_noise - semantic
    with torch.no_grad():
        feature = capture_feature(pipeline.transformer, noisy_rgb, timestep, embedding)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(noisy_semantic, feature, timestep).float()
    loss = F.mse_loss(prediction, target)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"Nonfinite loss for {record['clip_id']}.")
    return loss


@torch.no_grad()
def validate(model, pipeline, records, latents, embeddings, device):
    model.eval()
    losses = []
    for record_index, record in enumerate(records):
        semantic, rgb = latents[record["clip_id"]]
        for schedule_index in (5, 25, 45):
            # Fixed noise across checkpoints, separate from training randomness.
            generator = torch.Generator().manual_seed(900000 + record_index * 100 + schedule_index)
            loss = flow_loss(model, pipeline, record, semantic.to(device), rgb.to(device),
                             embeddings[record["prompt"]], schedule_index, generator, device)
            losses.append(float(loss))
    return sum(losses) / len(losses)


def checkpoint_payload(model, optimizer, generator, update, order, best_loss, history, config, best_payload):
    return {"model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "generator_state": generator.get_state(),
            "update": update, "order": order, "best_validation_loss": best_loss, "history": history,
            "best_payload": best_payload,
            "config_fingerprint": fingerprint(config)}


def train(args, device):
    manifest = load_prepared(args)
    continuation_path = args.weights_dir / "continuation.json"
    continuation = read_json(continuation_path) if continuation_path.exists() else None
    # Check source model inventory once, without copying model weights.
    import importlib.util
    from joint_semantic_mixed.common import REPOSITORY
    spec = importlib.util.spec_from_file_location("joint_dataset_inventory", REPOSITORY / "src/datasets/create_walking_pov_100.py")
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    # RGB generation settings are preserved in every record's source meta.json.
    for record in manifest["records"]:
        metadata = read_json(Path(record["directory"]) / "meta.json")
        for key, value in dataset.SETTINGS.items():
            if metadata.get(key) != value:
                raise ValueError(f"RGB generation settings changed: {record['clip_id']}")
    model_inventory = dataset.model_inventory(args.model_dir.resolve(strict=True))
    source_plan = read_json(Path(manifest["codec_run"]) / "prepared.json")
    dataset_plan = read_json(Path(source_plan["data_root"]) / "dataset_plan.json")
    if model_inventory != dataset_plan["model_inventory"]:
        raise ValueError("Train with the same frozen Wan model as the dataset generator.")
    records = {split: [record for record in manifest["records"] if record["split"] == split] for split in ("train", "val")}
    preview_plan = monitoring.monitoring_plan(manifest, args.updates, args.preview_every, args.new_prompt)
    teacher_info = monitoring.teacher_provenance(args.yolo_model)
    if teacher_info["sha256"] != dataset_plan["yolo_sha256"]:
        raise ValueError("Use the same frozen YOLO weights as the prepared dataset.")
    config = {"format": FORMAT, "prepared_sha256": file_hash(args.weights_dir / "prepared.json"),
              "model_dir": str(args.model_dir.resolve()), "model_inventory": model_inventory,
              "updates": args.updates, "evaluate_every": args.evaluate_every, "learning_rate": args.learning_rate,
              "seed": 42, "runtime": runtime(), "block_index": 15,
              "loss": "MSE(predicted semantic flow, semantic noise - clean normalized semantic latent)",
              "validation": "all accepted validation clips at fixed schedule indices 5,25,45; teacher-forced flow loss",
              "generation_settings": dataset.SETTINGS, "preview_every": args.preview_every,
              "preview_plan": preview_plan, "yolo": teacher_info, "final_new_prompt": args.new_prompt}
    if continuation is not None:
        from joint_semantic_mixed.continuation import check_parent_settings
        check_parent_settings(continuation["parent_training_config"], config)
        if args.updates != continuation["total_updates"]:
            raise ValueError("Keep the continuation update budget unchanged.")
        config["continuation"] = continuation
    config_path = args.weights_dir / "training_config.json"
    if config_path.exists():
        if read_json(config_path) != config or not args.resume:
            raise ValueError("Training exists. Use --resume with unchanged settings, or a new prepared run.")
        if (args.weights_dir / "training_complete.json").exists():
            complete = read_json(args.weights_dir / "training_complete.json")
            if file_hash(args.weights_dir / "semantic_best.pt") != complete["best_weights_sha256"]:
                raise ValueError("Completed CNN weights changed.")
            print("CNN training already complete and verified.")
            return
    write_json(config_path, config)
    latents = {}
    for record in manifest["records"]:
        semantic = check_tensor(torch.load(record["semantic_latent"], map_location="cpu", weights_only=True), SEMANTIC_SHAPE).float()
        rgb = check_tensor(torch.load(record["rgb_latent"], map_location="cpu", weights_only=True), RGB_SHAPE).float()
        latents[record["clip_id"]] = (semantic, rgb)
    torch.manual_seed(42)
    model = FeatureSemanticFlowModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)
    generator = torch.Generator().manual_seed(42)
    history, order, start, best_loss = [], [], 1, float("inf")
    latest = args.weights_dir / "semantic_latest.pt"
    best = args.weights_dir / "semantic_best.pt"
    if continuation is not None and not latest.exists():
        from joint_semantic_mixed.continuation import initial_checkpoint
        saved = initial_checkpoint(continuation, config)
        atomic_torch_save(best, saved["best_payload"])
        atomic_torch_save(latest, saved)
        save_history(args.outputs_dir, saved["history"])
    if (args.resume or continuation is not None) and latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["config_fingerprint"] != fingerprint(config):
            raise ValueError("Checkpoint settings differ from this training run.")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        generator.set_state(saved["generator_state"])
        start, order = saved["update"] + 1, saved["order"]
        history, best_loss = saved["history"], saved["best_validation_loss"]
        # Keep best/latest consistent if an earlier atomic save was interrupted.
        atomic_torch_save(best, saved["best_payload"])
    pipeline = load_wan(args.model_dir, device)
    embeddings = {}
    for record in manifest["records"]:
        prompt = record["prompt"]
        if prompt not in embeddings:
            if len(pipeline.tokenizer(prompt, truncation=False)["input_ids"]) > 512:
                raise ValueError("Prompt exceeds the text encoder length.")
            with torch.no_grad():
                embedding, _ = pipeline.encode_prompt(prompt=prompt, do_classifier_free_guidance=False,
                                                       num_videos_per_prompt=1, max_sequence_length=512, device=device)
            embeddings[prompt] = embedding.detach()
    preview_embeddings = monitoring.cache_prompt_embeddings(pipeline, preview_plan[0]["cases"], device)
    context = monitoring.create_context(args, device, preview_embeddings, teacher_info)
    # Neither component is used by CNN optimization after text encoding.
    pipeline.text_encoder.to("cpu")
    pipeline.vae.to("cpu")
    torch.cuda.empty_cache()
    pipeline.scheduler.set_timesteps(50, device=device)
    preview_root = args.outputs_dir / "prompt_checks"
    preview_root.mkdir(exist_ok=True)
    write_json(preview_root / "plan.json", {"checks": preview_plan, "yolo": teacher_info,
                                          "config_fingerprint": fingerprint(config)})
    planned_checks = {item["update"]: item["cases"] for item in preview_plan}
    previous_root = None
    previous_update = 0
    if continuation is not None:
        previous_root = Path(continuation["parent_outputs"]) / "prompt_checks"
        previous_update = continuation["start_update"]
        planned_checks = {update: cases for update, cases in planned_checks.items() if update > previous_update}
        monitoring.save_progress(preview_root, preview_plan, previous_root, previous_update)

    def run_prompt_check(update):
        if update not in planned_checks:
            return
        monitoring.evaluate_cases(
            pipeline, model, planned_checks[update], context, preview_root / f"update_{update:06d}",
            update, update / len(records["train"]), fingerprint(config), keep_videos=(update == args.updates),
        )
        monitoring.save_progress(preview_root, preview_plan, previous_root, previous_update)

    # A checkpoint is saved BEFORE its preview. Finish/verify that preview on resume.
    if start > 1:
        run_prompt_check(start - 1)
    if start == 1:
        initial = validate(model, pipeline, records["val"], latents, embeddings, device)
        history = [{"update": 0, "training_loss": None, "validation_loss": initial, "seconds": 0.0}]
        best_loss = initial
        atomic_torch_save(best, {"model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                                 "update": 0, "validation_loss": initial, "config_fingerprint": fingerprint(config)})
        save_history(args.outputs_dir, history)
    interval_losses = []
    # Preserve a continuous elapsed-time axis when resuming/continuing loss history.
    previous_seconds = history[-1]["seconds"] if history else 0.0
    started = time.monotonic()
    for update in range(start, args.updates + 1):
        model.train()
        position = (update - 1) % len(records["train"])
        if position == 0:
            order = torch.randperm(len(records["train"]), generator=generator).tolist()
        record = records["train"][order[position]]
        semantic, rgb = latents[record["clip_id"]]
        schedule_index = int(torch.randint(50, (1,), generator=generator))
        optimizer.zero_grad(set_to_none=True)
        loss = flow_loss(model, pipeline, record, semantic.to(device), rgb.to(device), embeddings[record["prompt"]],
                         schedule_index, generator, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        interval_losses.append(float(loss.detach()))
        if update == 1 or update % 20 == 0:
            print(f"CNN update {update}/{args.updates}: flow_loss={float(loss.detach()):.6f}", flush=True)
        if update % args.evaluate_every == 0 or update in planned_checks or update == args.updates:
            validation_loss = validate(model, pipeline, records["val"], latents, embeddings, device)
            row = {"update": update, "training_loss": sum(interval_losses) / len(interval_losses),
                   "validation_loss": validation_loss, "seconds": previous_seconds + time.monotonic() - started}
            history.append(row)
            interval_losses = []
            if validation_loss < best_loss:
                best_loss = validation_loss
                atomic_torch_save(best, {"model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                                         "update": update, "validation_loss": validation_loss, "config_fingerprint": fingerprint(config)})
            payload = checkpoint_payload(model, optimizer, generator, update, order, best_loss, history, config,
                                         torch.load(best, map_location="cpu", weights_only=True))
            atomic_torch_save(latest, payload)
            save_history(args.outputs_dir, history)
            print(f"Validation update {update}: mean flow loss={validation_loss:.6f}", flush=True)
            optimizer.zero_grad(set_to_none=True)
            run_prompt_check(update)
    saved = torch.load(best, map_location="cpu", weights_only=True)
    if any(parameter.grad is not None or parameter.requires_grad for parameter in pipeline.transformer.parameters()):
        raise RuntimeError("Wan was not frozen during semantic training.")
    write_json(args.weights_dir / "training_complete.json", {
        "updates": args.updates, "best_update": saved["update"], "best_validation_loss": saved["validation_loss"],
        "best_weights_sha256": file_hash(best), "weights": str(best),
        "train_clips": len(records["train"]), "validation_clips": len(records["val"]),
        "epochs": args.updates / len(records["train"]), "prompt_checks": [item["update"] for item in preview_plan],
        "note": "Only semantic CNN trained. Best checkpoint selected by teacher-forced latent-flow MSE, not generated-mask accuracy or RGB geometry.",
    })
