"""Inference with saved weights: no optimiser, training batches or updates."""

import gc
import time

from . import inputs, protocol


def run_experiment(args, config, info, spec):
    import torch
    from rgb_geometry_campaign import policy as architecture
    from rgb_geometry_pilot.runtime import load_runtime
    from rgb_geometry_qualitative import artifacts, policy
    from rgb_geometry_qualitative.experiments import record_case

    started = time.monotonic()
    cases = protocol.cases(config, "evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = load_runtime(args, cases, info)
    runtime["source_fingerprint"] = protocol.fingerprint(info["identity"]["sources"])

    # Install the historical architecture, then restore its original identity.
    # The new prompt panel has a separate identity; no old guard is bypassed.
    architecture.initialize(runtime, spec["arm"], seed=config["settings"]["seed"])
    runtime["spec"] = dict(spec)
    origin = info["identity"]["checkpoint_origins"][args.experiment]
    if args.experiment != "baseline":
        saved = artifacts.restore_checkpoint(origin["path"], runtime, optimizer=None,
                                             identity=origin["identity"], restore_training=False)
        if saved["completed_round"] != 10 or saved["progress"]["completed_rounds"] != 10:
            raise ValueError("Inference requires a completed round-10 checkpoint.")
        del saved
    loaded_digest = policy.digest(runtime)
    if loaded_digest != origin["policy_digest"]:
        raise ValueError("Loaded weights differ from the original final policy digest.")
    protocol.write_json(args.output_root / args.experiment / "checkpoint_origin.json", origin)

    # Even the adapters and heads are frozen for this evaluation.
    for parameter in architecture.parameters(runtime):
        parameter.requires_grad_(False)
        parameter.grad = None
    for case_index, case in enumerate(cases, start=1):
        inputs.check_free_space(args.output_root, minimum_bytes=1_000_000_000)
        print(f"{args.experiment}: case {case_index}/{len(cases)} {case['case_id']}", flush=True)
        with torch.no_grad():
            trajectory = policy.collect(runtime, case, spec["method"], training=False)
            trajectory["statistics"].update(training=False, inference_only=True,
                                            checkpoint_sha256=origin["sha256"])
            metrics = record_case(args, config, runtime, case, trajectory)
        if policy.digest(runtime) != loaded_digest:
            raise RuntimeError("Inference changed the saved policy weights.")
        protocol.append_row(args.output_root / args.experiment / "evaluation_rewards.csv", {
            "case_id": case["case_id"], "role": "evaluation", "reward": metrics["broader_reward"],
            "car_reward": metrics["car_reward"], "rgb_seed": case["rgb_seed"],
            "semantic_seed": case["semantic_seed"], "action_seed": case["action_seed"]})
        protocol.write_json(args.output_root / args.experiment / "progress.json", {
            "completed_cases": case_index, "planned_cases": len(cases),
            "last_case_id": case["case_id"], "policy_digest": loaded_digest,
            "optimizer_steps": 0, "elapsed_seconds": time.monotonic() - started})
        del trajectory
        gc.collect()

    return {"status": "complete", "experiment": args.experiment, "method": spec["method"],
            "arm": spec["arm"], "condition": spec["condition"],
            "rounds": 0, "training_rounds": 0, "optimizer_steps": 0, "learning_clips": 0,
            "inference_only": True, "policy_parameters_frozen": True,
            "initial_policy_digest": loaded_digest, "final_policy_digest": policy.digest(runtime),
            "checkpoint_origin": origin, "comparison_fingerprint": protocol.fingerprint(config),
            "source_fingerprint": runtime["source_fingerprint"],
            "final_evaluation_clips": len(cases), "retained_videos": len(cases),
            "elapsed_seconds": time.monotonic() - started, "geometry_certified": False,
            "measurement": "tracked_class_persistence_not_geometry_accuracy"}
