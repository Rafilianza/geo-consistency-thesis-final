"""Campaign-matched training loops; weights and videos remain in memory only."""

import gc
import json
import math
import time

from . import protocol


def manifest_fingerprint(config):
    return protocol.fingerprint(config)


def runtime_for(args, config, info):
    """Load the shared frozen inputs without constructing a historical arm."""
    import torch
    from rgb_geometry_pilot.runtime import load_runtime

    all_cases = []
    for panel in ("rl_train", "sentinel", "evaluation"):
        all_cases.extend(protocol.cases(config, panel))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = load_runtime(args, all_cases, info)
    runtime["source_fingerprint"] = protocol.fingerprint(info["identity"]["sources"])
    return runtime


def learning_case(template, round_index, group_index, sibling_index, seed=42):
    """Reuse historical initial noise; vary only each sibling's action noise.

    Rounds here start at one. The historical campaign loop starts at zero, so
    subtract one when recreating its action seeds. Initial RGB/semantic seeds
    remain the panel template's seeds in every round and every paired arm.
    """
    action_seed = 500001 + seed * 10000 + (round_index - 1) * 16 + group_index * 4 + sibling_index
    return dict(
        template,
        case_id=f"round_{round_index:02d}_group_{group_index}_sibling_{sibling_index}",
        role="learning",
        round_index=round_index,
        group_index=group_index,
        sibling_index=sibling_index,
        template_case_id=template["case_id"],
        action_seed=action_seed,
        step_seed=720_000 + round_index * 100 + group_index,
    )


def record_case(args, config, runtime, case, trajectory, evaluation=False):
    """Reuse historical scoring and label the observer diagnostic explicitly."""
    from rgb_geometry_campaign import scoring
    from rgb_geometry_campaign.reporting import COUNT_FIELDS, DIAGNOSTICS

    started = time.monotonic()
    directory = args.output_root / args.experiment / "cases" / case["case_id"]
    directory.mkdir(parents=True, exist_ok=False)
    frames = scoring.decode_rgb(runtime, trajectory["rgb_latent"])
    labeled = scoring.label_frames(runtime, frames)
    scoring.offload_for_scoring(runtime)
    metrics = scoring.track_frames(
        args, frames, labeled["raw_labels"], directory,
        case["expected_classes"], evaluation=evaluation,
    )
    for name in ("broader_reward", "car_reward"):
        if not math.isfinite(float(metrics[name])):
            raise FloatingPointError("Nonfinite generated-video reward: " + name)
    metrics["rgb_sha256"] = labeled["metrics"]["rgb_uint8_sha256"]
    protocol.write_json(directory / "rgb_scores.json", labeled["metrics"])
    statistics = dict(trajectory["statistics"])
    statistics["source_fingerprint"] = runtime["source_fingerprint"]
    protocol.write_json(directory / "generation.json", {"case": case, "statistics": statistics})

    # This is the terminal historical semantic-flow state. Agreement with the
    # final RGB teacher is a diagnostic, never reward or geometry ground truth.
    if trajectory.get("semantic_latent") is not None:
        agreement = scoring.teacher_agreement(runtime, trajectory["semantic_latent"], labeled["codec_labels"])
        agreement["measurement"] = "final_historical_flow_observer_vs_final_RGB_teacher"
        agreement["used_for_training"] = False
        protocol.write_json(directory / "semantic_agreement.json", agreement)
        protocol.append_row(args.output_root / args.experiment / "agreement.csv", {
            "case_id": case["case_id"], "role": case["role"],
            "score": agreement["fixed_foreground_miou"],
            "measurement": agreement["measurement"], "used_for_training": False,
        })

    protocol.append_row(args.output_root / args.experiment / "clips.csv", {
        "case_id": case["case_id"], "role": case["role"], "status": "complete",
        "reward": metrics["broader_reward"], "car_reward": metrics["car_reward"],
        "seconds": time.monotonic() - started + statistics.get("generation_seconds", 0),
        "rgb_sha256": metrics["rgb_sha256"], "rgb_seed": case["rgb_seed"],
        "semantic_seed": case["semantic_seed"], "action_seed": case["action_seed"],
    })
    for class_row in metrics["per_class"]:
        row = {
            "case_id": case["case_id"], "role": case["role"],
            "class_name": class_row["class_name"], "score": class_row["score"],
            "support": class_row["query_count"], "query_count": class_row["query_count"],
        }
        for key in (*COUNT_FIELDS, *DIAGNOSTICS):
            row[key] = class_row.get(key)
        protocol.append_row(args.output_root / args.experiment / "classes.csv", row)
    return metrics


def evaluate_cases(args, config, runtime, cases, method):
    """Technical failures stop the arm; finite quality measurements do not."""
    from . import policy

    for case in cases:
        trajectory = policy.collect_trajectory(
            runtime, case, method, steps=config["settings"]["eval_steps"], training=False,
        )
        record_case(args, config, runtime, case, trajectory, evaluation=case["role"] == "evaluation")
        del trajectory
        gc.collect()


def run_baseline(args, config, runtime):
    from . import policy

    started = time.monotonic()
    spec = protocol.experiment_spec("baseline")
    policy.initialize(runtime, args, config, spec)
    initial_digest = policy.policy_digest(runtime)
    for panel in ("sentinel", "evaluation"):
        evaluate_cases(args, config, runtime, protocol.cases(config, panel), "baseline")
    final_digest = policy.policy_digest(runtime)
    if initial_digest != final_digest:
        raise RuntimeError("Baseline parameters changed during measurement.")
    return {
        "status": "complete", "experiment": args.experiment, "method": spec["method"],
        "arm": "B", "policy": spec["policy"], "condition": spec["condition"],
        "rounds": 0, "attempted_updates": 0, "optimizer_steps": 0,
        "retained_updates": 0, "effective_policy_updates": 0, "quality_warning_count": 0,
        "learning_clips": 0, "sentinel_clips": 2, "final_evaluation_clips": 20,
        "seed": config["settings"]["seed"], "trainable_parameters": 0,
        "initial_policy_digest": initial_digest, "final_policy_digest": final_digest,
        "comparison_fingerprint": manifest_fingerprint(config),
        "source_fingerprint": runtime["source_fingerprint"],
        "elapsed_seconds": time.monotonic() - started, "geometry_certified": False,
        "measurement": "tracked_class_persistence_not_geometry_accuracy",
    }


def run_training(args, config, runtime, spec):
    """Collect sixteen fresh clips, run the method's updates, then two sentinels."""
    from . import policy

    started = time.monotonic()
    directory = args.output_root / args.experiment
    settings = config["settings"]
    policy.initialize(runtime, args, config, spec)
    optimizer = policy.make_optimizer(runtime, config)
    initial_digest = policy.policy_digest(runtime)
    retained_updates = 0
    retained_rounds = 0
    effective_updates = 0
    effective_rounds = 0
    optimizer_steps = 0
    warning_count = 0
    learning_count = 0

    for round_index in range(1, settings["rounds"] + 1):
        round_started = time.monotonic()
        collection_digest = policy.policy_digest(runtime)
        sampling_digest = None
        groups = []
        rewards = []
        # No optimizer step or policy update is allowed inside this collection.
        for group_index, template in enumerate(protocol.cases(config, "rl_train")):
            group = []
            group_rewards = []
            for sibling_index in range(settings["siblings"]):
                case = learning_case(template, round_index, group_index, sibling_index, settings["seed"])
                trajectory = policy.collect_trajectory(
                    runtime, case, spec["method"], steps=settings["train_steps"], training=True,
                )
                actual_digest = trajectory["statistics"]["policy_digest"]
                if sampling_digest is None:
                    sampling_digest = actual_digest
                elif actual_digest != sampling_digest:
                    raise RuntimeError("The sampling policy changed within a collection round.")
                measurements = record_case(args, config, runtime, case, trajectory)
                if round_index == 1 and group_index == 0 and sibling_index == 0:
                    probe = policy.technical_probe(runtime, optimizer, trajectory, spec["method"], config)
                    if policy.policy_digest(runtime) != collection_digest:
                        raise RuntimeError("The disposable technical probe failed to restore the policy.")
                    probe["disposable_update_restored"] = True
                    probe["scope"] = "one_full_resolution_backward_not_campaign_or_geometry_certification"
                    protocol.write_json(directory / "technical_probe.json", probe)
                reward = float(measurements["broader_reward"])
                protocol.append_row(directory / "rewards.csv", {
                    "round": round_index, "group": group_index, "sibling": sibling_index,
                    "case_id": case["case_id"], "scene": case["scene"],
                    "reward": reward, "broader_reward": reward, "car_reward": measurements["car_reward"],
                    "rgb_seed": case["rgb_seed"], "semantic_seed": case["semantic_seed"],
                    "action_seed": case["action_seed"], "step_seed": case["step_seed"],
                    "collection_policy_digest": sampling_digest, "current_policy_digest": collection_digest,
                })
                group.append(trajectory)
                group_rewards.append(reward)
                learning_count += 1
            groups.append(group)
            rewards.append(group_rewards)
        if policy.policy_digest(runtime) != collection_digest:
            raise RuntimeError("The current policy changed before the round's complete collection.")

        update = policy.update(runtime, optimizer, groups, rewards, spec["method"], round_index, config)
        # Refuse nonfinite or tensor-bearing reports before writing a completion receipt.
        json.dumps(update, allow_nan=False)
        final_digest = policy.policy_digest(runtime)
        update["round"] = round_index
        update["policy_digest_before"] = collection_digest
        update["policy_digest_after"] = final_digest
        update["parameter_digest_changed"] = final_digest != collection_digest
        retained = bool(update["retained_update"])
        effective = bool(update["effective_update"])
        round_steps = int(update["optimizer_steps"])
        planned_steps = 2 if spec["method"] in ("flow_grpo", "flash_grpo") else 1
        expected_steps = 0 if update.get("status") == "tied_rewards_skipped" else planned_steps
        if round_steps != expected_steps:
            raise RuntimeError("Optimizer count differs from the method's declared replay policy.")
        retained_updates += round_steps if retained else 0
        retained_rounds += int(retained)
        effective_updates += int(update["effective_policy_updates"])
        effective_rounds += int(effective)
        optimizer_steps += round_steps
        if update.get("status") == "tied_rewards_skipped":
            update["warnings"].append({"kind": "all_groups_tied", "report_only": True})
        protocol.write_json(directory / "updates" / f"round_{round_index:02d}.json", update)
        for warning in update.get("warnings", []):
            protocol.append_row(directory / "warnings.csv", {
                "round": round_index, "kind": "quality_diagnostic", "detail": warning,
                "training_stopped": False, "update_rejected": False,
            })
            warning_count += 1
        flat_rewards = []
        for group_rewards in rewards:
            flat_rewards.extend(group_rewards)
        protocol.append_row(directory / "training.csv", {
            "round": round_index, "rollouts": len(flat_rewards),
            "mean_reward": sum(flat_rewards) / len(flat_rewards),
            "minimum_reward": min(flat_rewards), "maximum_reward": max(flat_rewards),
            "retained": retained, "effective": effective, "optimizer_steps": round_steps,
            "cumulative_optimizer_steps": optimizer_steps,
            "seconds": time.monotonic() - round_started,
            "policy_digest_before": collection_digest, "policy_digest_after": final_digest,
        })
        del groups, rewards, group, trajectory
        gc.collect()

        # Sentinels run only after the update and cannot change the policy or optimizer.
        sentinel_cases = []
        for template in protocol.cases(config, "sentinel"):
            sentinel_cases.append(dict(template, case_id=f"round_{round_index:02d}_{template['case_id']}",
                                       round_index=round_index))
        evaluate_cases(args, config, runtime, sentinel_cases, spec["method"])
        if policy.policy_digest(runtime) != final_digest:
            raise RuntimeError("Sentinel measurement changed the current policy.")
        protocol.write_json(directory / "progress.json", {
            "completed_rounds": round_index, "learning_clips": learning_count,
            "sentinel_clips": round_index * 2, "optimizer_steps": optimizer_steps,
            "retained_updates": retained_updates, "effective_policy_updates": effective_updates,
            "quality_warning_count": warning_count, "policy_digest": final_digest,
            "retained_rounds": retained_rounds, "effective_rounds": effective_rounds,
        })

    evaluate_cases(args, config, runtime, protocol.cases(config, "evaluation"), spec["method"])
    if policy.policy_digest(runtime) != final_digest:
        raise RuntimeError("Final evaluation changed the current policy.")
    return {
        "status": "complete", "experiment": args.experiment,
        "method": spec["method"], "arm": spec["arm"], "policy": spec["policy"],
        "condition": spec["condition"],
        "rounds": settings["rounds"], "attempted_rounds": settings["rounds"],
        "attempted_updates": settings["rounds"] * planned_steps,
        "optimizer_steps": optimizer_steps, "retained_updates": retained_updates,
        "retained_rounds": retained_rounds, "effective_rounds": effective_rounds,
        "effective_policy_updates": effective_updates, "quality_warning_count": warning_count,
        "learning_clips": learning_count, "sentinel_clips": settings["rounds"] * 2,
        "final_evaluation_clips": 20, "seed": settings["seed"],
        "trainable_parameters": sum(parameter.numel() for parameter in policy.parameters(runtime)),
        "initial_policy_digest": initial_digest, "final_policy_digest": final_digest,
        "comparison_fingerprint": manifest_fingerprint(config),
        "source_fingerprint": runtime["source_fingerprint"],
        "elapsed_seconds": time.monotonic() - started, "geometry_certified": False,
        "measurement": "tracked_class_persistence_not_geometry_accuracy",
        "sentinel_used_for_training": False, "evaluation_policy": "current",
    }
