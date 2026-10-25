"""Train small heads or LoRA adapters, retaining round checkpoints and final RGB videos."""

import gc
import json
import math
import time

from rgb_geometry_rl_comparison.experiments import learning_case
from . import artifacts, policy, protocol


def runtime_for(args, config, info):
    import torch
    from rgb_geometry_pilot.runtime import load_runtime

    # RLOO also evaluates eight historical endpoint cases after training.
    all_cases = []
    for panel in config["panels"]:
        all_cases.extend(protocol.cases(config, panel))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = load_runtime(args, all_cases, info)
    runtime["source_fingerprint"] = protocol.fingerprint(info["identity"]["sources"])
    return runtime


def record_case(args, config, runtime, case, trajectory):
    from rgb_geometry_campaign import scoring
    from rgb_geometry_campaign.reporting import COUNT_FIELDS, DIAGNOSTICS

    started = time.monotonic()
    directory = args.output_root / args.experiment / "cases" / case["case_id"]
    directory.mkdir(parents=True, exist_ok=False)
    frames = scoring.decode_rgb(runtime, trajectory["rgb_latent"])
    statistics = dict(trajectory["statistics"])
    statistics["policy_digest"] = policy.digest(runtime)
    statistics["source_fingerprint"] = runtime["source_fingerprint"]
    protocol.write_json(directory / "generation.json", {"case": case, "statistics": statistics})

    # These exact decoded pixels are exported and then passed to the frozen judges.
    # MP4 compression is lossy; the scores below refer to the original RGB pixels.
    if case["role"] == "evaluation":
        video_path = args.output_root / args.experiment / "videos" / (case["case_id"] + ".mp4")
        artifacts.save_video(video_path, frames, case, statistics["policy_digest"], fps=16)
    labeled = scoring.label_frames(runtime, frames)
    scoring.offload_for_scoring(runtime)
    metrics = scoring.track_frames(args, frames, labeled["raw_labels"], directory,
                                  case["expected_classes"], evaluation=case["role"] == "evaluation")
    for name in ("broader_reward", "car_reward"):
        if not math.isfinite(float(metrics[name])):
            raise FloatingPointError("Nonfinite reward: " + name)
    metrics["rgb_sha256"] = labeled["metrics"]["rgb_uint8_sha256"]
    protocol.write_json(directory / "rgb_scores.json", labeled["metrics"])
    if not args.experiment.startswith("rloo_") or case["role"] == "evaluation":
        agreement = scoring.teacher_agreement(runtime, trajectory["semantic_latent"], labeled["codec_labels"])
        agreement.update(measurement="final_historical_flow_observer_vs_final_RGB_teacher",
                         used_for_training=False)
        protocol.write_json(directory / "semantic_agreement.json", agreement)
    protocol.append_row(args.output_root / args.experiment / "clips.csv", {
        "case_id": case["case_id"], "role": case["role"], "status": "complete",
        "reward": metrics["broader_reward"], "car_reward": metrics["car_reward"],
        "seconds": time.monotonic() - started + statistics.get("generation_seconds", 0),
        "rgb_sha256": metrics["rgb_sha256"], "rgb_seed": case["rgb_seed"],
        "semantic_seed": case["semantic_seed"], "action_seed": case["action_seed"],
    })
    for class_row in metrics["per_class"]:
        row = {"case_id": case["case_id"], "role": case["role"],
               "class_name": class_row["class_name"], "score": class_row["score"],
               "support": class_row["query_count"], "query_count": class_row["query_count"]}
        for key in (*COUNT_FIELDS, *DIAGNOSTICS):
            row[key] = class_row.get(key)
        protocol.append_row(args.output_root / args.experiment / "classes.csv", row)
    return metrics


def evaluate(args, config, runtime, cases, method):
    before = policy.digest(runtime)
    for case in cases:
        trajectory = policy.collect(runtime, case, method, training=False)
        record_case(args, config, runtime, case, trajectory)
        del trajectory
        gc.collect()
    if policy.digest(runtime) != before:
        raise RuntimeError("Evaluation changed the trained policy.")


def run_experiment(args, config, runtime, spec, identity):
    started = time.monotonic()
    directory = args.output_root / args.experiment
    method = spec["method"]
    optimizer = policy.initialize(runtime, args, config, spec)
    initial_digest = policy.digest(runtime)
    progress = {"completed_rounds": 0, "optimizer_steps": 0, "learning_clips": 0,
                "retained_updates": 0, "effective_policy_updates": 0,
                "quality_warning_count": 0}
    origin = None
    checkpoint_path = args.resume_from or args.checkpoint
    if checkpoint_path:
        saved = artifacts.restore_checkpoint(checkpoint_path, runtime, optimizer, identity,
                                             restore_training=args.checkpoint is None)
        progress = dict(saved["progress"])
        if progress["completed_rounds"] != saved["completed_round"]:
            raise ValueError("Checkpoint progress and completed round disagree.")
        if args.checkpoint and saved["completed_round"] != 10:
            raise ValueError("The matched final-video study requires the round-10 checkpoint.")
        origin = {"path": str(checkpoint_path.resolve()), "sha256": protocol.file_hash(checkpoint_path),
                  "completed_round": saved["completed_round"],
                  "purpose": "inference_only" if args.checkpoint else "round_boundary_resume"}
        protocol.write_json(directory / "checkpoint_origin.json", origin)

    if spec["arm"] == "B":
        evaluate(args, config, runtime, protocol.cases(config, "sentinel"), method)
    elif args.checkpoint is None:
        # Resumed attempts always write to a fresh directory, leaving previous
        # partial rounds, logs and videos intact in the original attempt.
        if args.resume_from is None:
            artifacts.save_checkpoint(directory / "checkpoints/round_00.pt", runtime, optimizer,
                                      0, identity, progress)
        for round_index in range(progress["completed_rounds"] + 1, 11):
            round_started = time.monotonic()
            collection_digest = policy.digest(runtime)
            groups, rewards = [], []
            for group_index, template in enumerate(protocol.cases(config, "rl_train")):
                group, group_rewards = [], []
                for sibling_index in range(4):
                    case = learning_case(template, round_index, group_index, sibling_index,
                                         config["settings"]["seed"])
                    trajectory = policy.collect(runtime, case, method, training=True)
                    if policy.digest(runtime) != collection_digest:
                        raise RuntimeError("Policy changed during sibling collection.")
                    measurements = record_case(args, config, runtime, case, trajectory)
                    if method != "rloo" and round_index == 1 and group_index == 0 and sibling_index == 0:
                        probe = policy.grpo.technical_probe(runtime, optimizer, trajectory, method, config)
                        if policy.digest(runtime) != collection_digest:
                            raise RuntimeError("Disposable GRPO probe did not restore the policy.")
                        protocol.write_json(directory / "technical_probe.json", probe)
                    reward = float(measurements["broader_reward"])
                    protocol.append_row(directory / "rewards.csv", {
                        "round": round_index, "group": group_index, "sibling": sibling_index,
                        "case_id": case["case_id"], "reward": reward,
                        "rgb_seed": case["rgb_seed"], "semantic_seed": case["semantic_seed"],
                        "action_seed": case["action_seed"], "step_seed": case["step_seed"]})
                    trajectory.pop("rgb_latent", None)
                    trajectory.pop("semantic_latent", None)
                    group.append(trajectory)
                    group_rewards.append(reward)
                groups.append(group)
                rewards.append(group_rewards)
            update = policy.update(runtime, optimizer, groups, rewards, method, round_index, config)
            json.dumps(update, allow_nan=False)
            update.update(round=round_index, policy_digest_before=collection_digest,
                          policy_digest_after=policy.digest(runtime))
            protocol.write_json(directory / "updates" / f"round_{round_index:02d}.json", update)
            progress["learning_clips"] += 16
            progress["optimizer_steps"] += int(update["optimizer_steps"])
            progress["retained_updates"] += int(update["optimizer_steps"]) if update["retained_update"] else 0
            progress["effective_policy_updates"] += int(update["effective_policy_updates"])
            progress["quality_warning_count"] += len(update["warnings"])
            flat_rewards = [reward for group_rewards in rewards for reward in group_rewards]
            protocol.append_row(directory / "training.csv", {
                "round": round_index, "mean_reward": sum(flat_rewards) / 16,
                "optimizer_steps": update["optimizer_steps"],
                "policy_digest": update["policy_digest_after"],
                "seconds": time.monotonic() - round_started})
            del groups, rewards, group, trajectory
            gc.collect()
            sentinels = [dict(case, case_id=f"round_{round_index:02d}_{case['case_id']}")
                         for case in protocol.cases(config, "sentinel")]
            evaluate(args, config, runtime, sentinels, method)
            progress["completed_rounds"] = round_index
            receipt = artifacts.save_checkpoint(directory / "checkpoints" / f"round_{round_index:02d}.pt",
                                                runtime, optimizer, round_index, identity, progress)
            protocol.write_json(directory / "latest_checkpoint.json", receipt)
            protocol.write_json(directory / "progress.json", progress)
        if method == "rloo":
            # Preserve the eight historical RLOO endpoint diagnostics; they do
            # not select weights and their videos are not retained.
            evaluate(args, config, runtime, protocol.cases(config, "endpoint"), method)

    evaluate(args, config, runtime, protocol.cases(config, "evaluation"), method)
    return dict(progress, status="complete", experiment=args.experiment, method=method,
                arm=spec["arm"], condition=spec["condition"], rounds=progress["completed_rounds"],
                initial_policy_digest=initial_digest, final_policy_digest=policy.digest(runtime),
                comparison_fingerprint=protocol.fingerprint(config),
                source_fingerprint=runtime["source_fingerprint"],
                final_evaluation_clips=20, retained_videos=20,
                checkpoint_origin=origin, inference_only=args.checkpoint is not None,
                elapsed_seconds=time.monotonic() - started,
                seed=config["settings"]["seed"],
                trainable_parameters=sum(value.numel() for value in policy.rloo.parameters(runtime)),
                geometry_certified=False, measurement="tracked_class_persistence_not_geometry_accuracy")
