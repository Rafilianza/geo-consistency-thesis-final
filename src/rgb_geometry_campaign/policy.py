"""Campaign policies: four-position R/S heads or rank-four Wan LoRA.

B is frozen Wan, R consumes zero semantics, and S consumes the frozen observer's
predicted clean semantic latent. All three inject identical seeded Gaussian
noise at denoising indices 10, 20, 30 and 40 (human steps 11, 21, 31 and 41).
Only new generated videos supply learning rewards. RL policies learn after
four groups of four complete fresh rollouts. No historical policy is modified.
"""

import hashlib
import math
import time

import torch
import torch.nn.functional as F

from joint_semantic_mixed.common import check_tensor
from middle_feature_semantic.reference_rgb_correction import correction_loss
from rgb_geometry_rl_v2 import policy as base_policy
from rgb_geometry_lora_exploratory import policy as report_only_policy


ARMS = ("B", "R", "S", "L")
ACTION_STEPS = (10, 20, 30, 40)
INFERENCE_STEPS = 50
OUTER_GAIN = 0.25
RMS_CAP = 0.25
HEAD_CONFIG = base_policy.HEAD_CONFIG
HEAD_PARAMETER_COUNT = base_policy.HEAD_PARAMETER_COUNT
RGB_SHAPE = base_policy.RGB_SHAPE
SEMANTIC_SHAPE = base_policy.SEMANTIC_SHAPE
historical_policy = base_policy.historical_policy
base_flow_and_feature = base_policy.base_flow_and_feature
guided_flow = base_policy.guided_flow
parameters = base_policy.parameters
state_dict = base_policy.state_dict
load_state_dict = base_policy.load_state_dict
snapshot = base_policy.snapshot
restore = base_policy.restore
frozen_reference_mean = base_policy.frozen_reference_mean
replay_statistics = base_policy.replay_statistics
rloo_advantages = base_policy.rloo_advantages
update_batch = report_only_policy.update_batch
ReplayMismatchError = base_policy.ReplayMismatchError


def initialize(runtime, arm, seed=42):
    base_policy.initialize(runtime, arm)
    if arm in ("R", "S") and seed != 42:
        from middle_feature_semantic.reference_rgb_correction import ReferenceRGBCorrectionHead
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            runtime["head"] = ReferenceRGBCorrectionHead(**HEAD_CONFIG).to(runtime["device"]).eval()
    return parameters(runtime)


def make_optimizer(runtime):
    rate = 1e-8 if runtime["arm"] == "L" else 1e-6
    return torch.optim.Adam(parameters(runtime), lr=rate, betas=(.9,.999), eps=1e-8)


def _check_alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("Correction strength must be a finite number.")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Correction strength must be between zero and one.")
    return float(alpha)


def _tensor_hash(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _check_seeds(seeds):
    if (set(seeds) != {"rgb", "semantic", "action"}
            or any(type(seed) is not int or seed < 0 or seed == 47 for seed in seeds.values())):
        raise ValueError("Use three nonnegative permitted integer seeds.")


def _raw_correction(runtime, state, frozen_flow, q, sigma, timestep, prompt, checkpointed):
    if runtime["arm"] == "L":
        from rgb_geometry_lora_memory.checkpointing import forward_branch
        conditional = forward_branch(runtime, state, timestep, prompt, "cond",
                                     adapted=True, checkpointed=checkpointed)
        unconditional = forward_branch(runtime, state, timestep, prompt, "uncond",
                                       adapted=True, checkpointed=checkpointed)
        return guided_flow(conditional, unconditional) - frozen_flow.detach().float()
    return base_policy._raw_correction(runtime, state, frozen_flow, q, sigma,
                                      timestep, prompt, checkpointed)


@torch.no_grad()
def capture_base(runtime, state, timestep, prompt, blocks):
    from contextlib import ExitStack
    from middle_feature_semantic.wan_middle_feature_capture import (
        WanMiddleFeatureCapture, feature_tokens_to_video_grid)
    transformer = runtime["pipeline"].transformer
    with ExitStack() as stack:
        captures = {block: stack.enter_context(WanMiddleFeatureCapture(transformer, block))
                    for block in blocks}
        conditional = historical_policy._forward_branch(runtime, state, timestep, prompt,
                                                        "cond", adapted=False)
        features = {}
        for block, capture in captures.items():
            if capture.call_count != 1:
                raise RuntimeError("Conditional feature capture must occur exactly once.")
            features[block] = feature_tokens_to_video_grid(
                capture.feature, state.shape, transformer.config.patch_size).detach()
    unconditional = historical_policy._forward_branch(runtime, state, timestep, prompt,
                                                      "uncond", adapted=False)
    return guided_flow(conditional, unconditional).detach(), features


@torch.no_grad()
def generic_features(feature, semantic_shape):
    # Fixed random projection, with no semantic labels or trainable parameters.
    generator = torch.Generator().manual_seed(86103)
    projection = torch.randn(8, feature.shape[1], generator=generator) / math.sqrt(feature.shape[1])
    projected = torch.einsum("oc,bcthw->bothw", projection.to(feature.device), feature.float())
    projected = F.interpolate(projected, size=semantic_shape[-3:], mode="trilinear", align_corners=False)
    return projected / projected.square().mean().sqrt().clamp_min(1e-6)


def bounded_correction(raw_delta, frozen_flow, alpha=OUTER_GAIN):
    """Differentiate through the per-clip cap, then apply the outer gain."""
    alpha = _check_alpha(alpha)
    if raw_delta.shape != frozen_flow.shape or raw_delta.ndim < 2 or raw_delta.numel() == 0:
        raise ValueError("Expected matching nonempty batched correction and flow tensors.")
    with torch.autocast(device_type=raw_delta.device.type, enabled=False):
        raw_delta = raw_delta.float()
        frozen_flow = frozen_flow.detach().float()
        denominator = math.sqrt(raw_delta[0].numel())
        base_rms = torch.linalg.vector_norm(frozen_flow.flatten(1), dim=1) / denominator
        raw_rms = torch.linalg.vector_norm(raw_delta.flatten(1), dim=1) / denominator
        factor = (RMS_CAP * base_rms / raw_rms.clamp_min(1e-12)).clamp(max=1.0)
        broadcast_shape = (raw_delta.shape[0],) + (1,) * (raw_delta.ndim - 1)
        correction = alpha * raw_delta * factor.reshape(broadcast_shape)
        if not bool(torch.isfinite(correction).all() and torch.isfinite(raw_rms).all()
                    and torch.isfinite(base_rms).all()):
            raise FloatingPointError("Nonfinite correction or RMS norm.")
        executed_rms = torch.linalg.vector_norm(correction.flatten(1), dim=1) / denominator
    diagnostics = {"outer_gain": alpha, "rms_cap": RMS_CAP,
                   "raw_rms": float(raw_rms.detach().mean()),
                   "executed_rms": float(executed_rms.detach().mean()),
                   "base_rms": float(base_rms.mean()),
                   "cap_scale": float(factor.detach().mean()),
                   "clip_fraction": float((raw_rms > RMS_CAP * base_rms).float().mean())}
    return correction, diagnostics


def policy_mean(state, frozen_flow, raw_delta, step_size, alpha=OUTER_GAIN):
    """Common FP32 Euler mean, with the same correction arithmetic in replay."""
    if not math.isfinite(float(step_size)) or float(step_size) <= 0:
        raise ValueError("Euler step size must be positive and finite.")
    if state.shape != frozen_flow.shape:
        raise ValueError("State and frozen flow shapes differ.")
    correction, _ = bounded_correction(raw_delta, frozen_flow, alpha)
    with torch.autocast(device_type=state.device.type, enabled=False):
        mean = state.detach().float() - float(step_size) * (frozen_flow.detach().float() + correction)
    if not bool(torch.isfinite(mean).all()):
        raise FloatingPointError("Nonfinite action mean.")
    return mean


def replay_mean(runtime, transition, checkpoint=True, zero_semantics=False):
    """Replay the exact saved pre-action context, without advancing the observer."""
    if transition["arm"] != runtime["arm"] or transition["step_index"] not in ACTION_STEPS:
        raise ValueError("Replay arm or action index does not match the policy.")
    state = transition["state"].detach().to(runtime["device"], dtype=torch.float32)
    frozen_flow = transition["frozen_flow"].detach().to(runtime["device"], dtype=torch.float32)
    q = transition["q"].detach().to(runtime["device"], dtype=torch.float32)
    if zero_semantics:
        q = torch.zeros_like(q)
    delta = _raw_correction(runtime, state, frozen_flow, q, transition["sigma"],
                            transition["timestep"], transition["prompt"], checkpoint)
    return policy_mean(state, frozen_flow, delta, transition["h"], transition["alpha"])


@torch.no_grad()
def collect_trajectory(runtime, prompt, seeds, parameter_version=0, alpha=OUTER_GAIN, save_actions=True,
                       capture_history=False, blocks=(15,), guidance_mode="semantic",
                       drop_semantics=False, donor=None):
    """Generate with four corrections/noise injections and independent RNG streams.

    ``alpha`` is .1 for evaluated dataset heads or .25 for RL collection/replay.
    ``save_actions=False`` skips large CPU action contexts during evaluation;
    all four noise/correction ledger entries and RNG states are still recorded.
    The RGB update is Euler; UniPC supplies the shared clock and advances only
    the frozen semantic observer. No trainable parameters change in this call.
    """
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    started = time.monotonic()
    _check_seeds(seeds)
    alpha = _check_alpha(alpha)
    if runtime["arm"] not in ARMS:
        raise ValueError("Initialize B, R, S or L before generation.")
    if type(parameter_version) is not int or parameter_version < 0:
        raise ValueError("Parameter version must be a nonnegative integer.")
    arm, pipeline = runtime["arm"], runtime["pipeline"]
    device = torch.device(runtime["device"])
    transformer = pipeline.transformer.to(device).eval()
    semantic = runtime["semantic"].to(device).eval().requires_grad_(False)
    if arm in ("R", "S"):
        runtime["head"].to(device=device, dtype=torch.float32).eval()
    pipeline.vae.to("cpu")
    runtime["codec"].to("cpu")
    if getattr(pipeline.config, "expand_timesteps", False):
        raise ValueError("Only scalar Wan T2V timesteps are supported.")
    schedule = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    semantic_schedule = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    for scheduler in (schedule, semantic_schedule):
        scheduler.set_timesteps(INFERENCE_STEPS, device=device)
        if not scheduler.config.use_flow_sigmas or float(scheduler.config.flow_shift) != 3.0:
            raise ValueError("The registered schedule requires flow sigmas and shift three.")
    if (len(schedule.sigmas) != INFERENCE_STEPS + 1
            or not torch.equal(schedule.timesteps, semantic_schedule.timesteps)
            or not torch.equal(schedule.sigmas, semantic_schedule.sigmas)):
        raise ValueError("RGB and semantic scheduler clocks differ.")
    rgb_generator = torch.Generator(device=device).manual_seed(seeds["rgb"])
    action_generator = torch.Generator(device=device).manual_seed(seeds["action"])
    semantic_generator = torch.Generator(device="cpu").manual_seed(seeds["semantic"])
    initial_rng = {"rgb_before": rgb_generator.get_state(), "action_before": action_generator.get_state(),
                   "semantic_before": semantic_generator.get_state()}
    rgb = pipeline.prepare_latents(1, 16, 480, 832, 81, torch.float32, device, rgb_generator, None)
    if tuple(rgb.shape) != RGB_SHAPE:
        raise ValueError("Unexpected RGB latent shape.")
    semantic_state = torch.randn(SEMANTIC_SHAPE, generator=semantic_generator, dtype=torch.float32).to(device)
    initial_hashes = {"rgb": _tensor_hash(rgb), "semantic": _tensor_hash(semantic_state)}
    transitions, ledger = [], []
    history, clean_predictions = {}, {}
    observer = runtime.get("campaign_observer")
    if observer is not None:
        observer.to(device).eval().requires_grad_(False)
        blocks = observer.block_indices
    with historical_policy.adapter_mode(transformer, False):
        for step_index, timestep in enumerate(schedule.timesteps):
            step_size = float(schedule.sigmas[step_index] - schedule.sigmas[step_index + 1])
            sigma = float(schedule.sigmas[step_index])
            is_action = step_index in ACTION_STEPS
            captured_blocks = blocks if is_action else (15,)
            frozen_flow, features = capture_base(runtime, rgb, timestep, prompt, captured_blocks)
            feature = features[15]
            if is_action:
                history[step_index] = {block: value.cpu() for block, value in features.items()}
            # q is computed from the pre-action RGB features and semantic state.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                semantic_flow = semantic(semantic_state.float(), feature, timestep.float().reshape(1)).float()
            q = (semantic_state.float() - sigma * semantic_flow).detach()
            if not bool(torch.isfinite(q).all()):
                raise FloatingPointError("Nonfinite pre-action semantic prediction.")
            if is_action:
                if observer is not None:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        q = observer.forward_history(semantic_state, history, timestep.float().reshape(1), step_index).float()
                clean_predictions[step_index] = q.detach().cpu()
                if guidance_mode == "generic":
                    q = generic_features(feature, q.shape)
                elif donor is not None:
                    q = donor[step_index].to(device).float()
                if drop_semantics:
                    q = torch.zeros_like(q)
            if arm == "R":
                q = torch.zeros_like(q)
            diagnostics = {}
            raw_delta = torch.zeros_like(frozen_flow)
            if is_action:
                raw_delta = _raw_correction(runtime, rgb, frozen_flow, q, sigma, timestep, prompt, False)
                _, diagnostics = bounded_correction(raw_delta, frozen_flow, alpha)
            mean = policy_mean(rgb, frozen_flow, raw_delta, step_size, alpha)
            rgb_next = mean
            noise_hash = None
            if is_action:
                standard_deviation = 0.03 * math.sqrt(step_size)
                noise = torch.randn(rgb.shape, device=device, dtype=torch.float32, generator=action_generator)
                rgb_next = mean + standard_deviation * noise
                noise_hash = _tensor_hash(noise)
                if save_actions:
                    transitions.append({
                        "state": rgb.detach().float().cpu().clone(),
                        "frozen_flow": frozen_flow.detach().float().cpu().clone(),
                        "q": q.detach().float().cpu().clone(),
                        "action": rgb_next.detach().float().cpu().clone(),
                        "old_mean": mean.detach().float().cpu().clone(),
                        "sigma": sigma, "step_index": step_index,
                        "timestep": timestep.detach().cpu().clone(), "h": step_size,
                        "std": standard_deviation, "prompt": prompt, "arm": arm,
                        "parameter_version": parameter_version, "correction": diagnostics, "alpha": alpha,
                    })
            semantic_state = semantic_schedule.step(semantic_flow, timestep, semantic_state, return_dict=False)[0].float()
            rgb = rgb_next.float()
            if not bool(torch.isfinite(rgb).all() and torch.isfinite(semantic_state).all()):
                raise FloatingPointError(f"Nonfinite rollout state at denoising index {step_index}.")
            ledger.append({"step_index": step_index, "h": step_size, "sigma": sigma,
                           "adapted": is_action and arm != "B", "lora_enabled": is_action and arm == "L", "noise_sha256": noise_hash,
                           "stochastic": is_action, "base_conditional_captures": 1,
                           "semantic_advances": 1, **diagnostics})
    if (len(ledger) != 50
            or [item["step_index"] for item in ledger if item["stochastic"]] != list(ACTION_STEPS)
            or (save_actions and [item["step_index"] for item in transitions] != list(ACTION_STEPS))):
        raise RuntimeError("The registered 50-step, four-action schedule did not complete.")
    return {"rgb_latent": rgb.detach().cpu(), "semantic_latent": semantic_state.detach().cpu(),
            "feature_history": history if capture_history else {},
            "clean_predictions": clean_predictions if capture_history else {},
            "rng_states": dict(initial_rng, rgb_after=rgb_generator.get_state(),
                               action_after=action_generator.get_state(), semantic_after=semantic_generator.get_state()),
            "transitions": transitions, "statistics": {
                "seeds": dict(seeds), "arm": arm, "parameter_version": parameter_version,
                "ledger": ledger, "sigmas": schedule.sigmas.cpu().tolist(),
                "timesteps": schedule.timesteps.cpu().tolist(), "semantic_steps": len(ledger),
                "sampler": "euler_four_gaussian_actions", "guidance_scale": 5.0,
                "outer_gain": alpha, "rms_cap": RMS_CAP,
                "action_steps": list(ACTION_STEPS), "initial_latent_sha256": initial_hashes,
                "rgb_latent_sha256": _tensor_hash(rgb), "actions_saved": save_actions,
                "generation_seconds": time.monotonic() - started,
            }}


def update(runtime, optimizer, trajectories, rewards):
    """One fresh 4x4 RLOO update, summing four Gaussian action log-probabilities.

    Report-only quality thresholds never discard a finite update. Invalid saved
    context, stale behavior, nonfinite arithmetic or replay mismatch remain
    technical failures. The runner advances parameter_version after each round.
    """
    if runtime["arm"] not in ("R", "S", "L"):
        raise ValueError("The frozen baseline cannot learn.")
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != {id(parameter) for parameter in parameters(runtime)}:
        raise RuntimeError("Policy parameter identity changed during device transfers.")
    if len(trajectories) != 4 or any(len(group) != 4 for group in trajectories):
        raise ValueError("Exactly four groups of four fresh trajectories are required.")
    advantages = rloo_advantages(rewards)
    if tuple(advantages.shape) != (4, 4):
        raise ValueError("Rewards must match the four-by-four batch.")
    flat_trajectories = [trajectory for group in trajectories for trajectory in group]
    if len({id(trajectory) for trajectory in flat_trajectories}) != 16:
        raise ValueError("Every sibling must be a distinct collected trajectory.")
    batch_contexts = []
    for group in trajectories:
        contexts, action_seeds = [], []
        for trajectory in group:
            if trajectory.get("used_for_update", False):
                raise ValueError("RLOO trajectories cannot be reused.")
            statistics = trajectory["statistics"]
            if (statistics.get("arm") != runtime["arm"]
                    or statistics.get("parameter_version") != runtime.get("parameter_version", 0)
                    or statistics.get("action_steps") != list(ACTION_STEPS)):
                raise ValueError("Trajectory identity or four-action schedule does not match.")
            seeds = statistics["seeds"]
            transitions = trajectory["transitions"]
            if [transition["step_index"] for transition in transitions] != list(ACTION_STEPS):
                raise ValueError("Every video must retain exactly actions 10, 20, 30 and 40.")
            if (set(seeds) != {"rgb", "semantic", "action"}
                    or any(type(seed) is not int or seed < 0 or seed == 47 for seed in seeds.values())):
                raise ValueError("Invalid initial-context or action seed.")
            if any(transition.get("alpha") != OUTER_GAIN for transition in transitions):
                raise ValueError("RL replay requires the collected alpha .25 at all four actions.")
            contexts.append((transitions[0]["prompt"], seeds["rgb"], seeds["semantic"]))
            action_seeds.append(seeds["action"])
            for transition in transitions:
                if (transition["arm"] != runtime["arm"]
                        or transition["parameter_version"] != runtime.get("parameter_version", 0)):
                    raise ValueError("Stale or mismatched behavior-policy identity.")
                if transition["prompt"] != transitions[0]["prompt"]:
                    raise ValueError("An action prompt changed within a trajectory.")
                if (not math.isfinite(float(transition["h"])) or transition["h"] <= 0
                        or not math.isfinite(float(transition["sigma"])) or not 0 <= transition["sigma"] <= 1
                        or not math.isfinite(float(transition["timestep"]))):
                    raise ValueError("Invalid saved scheduler context.")
                if not math.isclose(transition["std"], 0.03 * math.sqrt(transition["h"]), rel_tol=1e-12):
                    raise ValueError("Exploration variance differs from the registered policy.")
                for key in ("state", "action", "old_mean", "frozen_flow", "q"):
                    value = transition[key]
                    if value.dtype != torch.float32 or value.requires_grad or not bool(torch.isfinite(value).all()):
                        raise ValueError("Saved policy context must be detached finite FP32.")
                    expected_shape = SEMANTIC_SHAPE if key == "q" else RGB_SHAPE
                    if tuple(value.shape) != expected_shape:
                        raise ValueError("Saved policy context has the wrong latent shape.")
                if runtime["arm"] == "R" and bool(torch.any(transition["q"] != 0)):
                    raise ValueError("R must save its actually consumed zero semantic input.")
        if len(set(contexts)) != 1 or len(set(action_seeds)) != 4:
            raise ValueError("Siblings must share initial context and use distinct action seeds.")
        batch_contexts.append(contexts[0])
    if len(set(batch_contexts)) != 4:
        raise ValueError("The batch must contain four distinct initial contexts.")
    # Consume the batch even if numerical execution fails; never silently reuse
    # these on-policy outcomes after attempting their one report-only proposal.
    for trajectory in flat_trajectories:
        trajectory["used_for_update"] = True
    result = update_batch(parameters(runtime), optimizer, flat_trajectories, advantages,
                          lambda transition: replay_mean(runtime, transition), denominator=16)
    result["advantages"] = advantages.tolist()
    return result
