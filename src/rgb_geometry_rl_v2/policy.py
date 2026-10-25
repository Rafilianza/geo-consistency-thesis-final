"""Four-arm, two-action RGB policy for the separately versioned proxy study.

Only R/S's fresh correction head or L's LoRA matrices learn. Numerical proposal
acceptance here is provisional: the runner owns sentinels and durable acceptance.
The historical policy and supervised correction arithmetic remain unchanged.
"""

import copy
import math
import random

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from middle_feature_semantic.reference_rgb_correction import ReferenceRGBCorrectionHead
from rgb_geometry_pilot import policy as historical_policy


ARMS = ("B", "R", "S", "L")
ACTION_STEPS = (10, 20)
INFERENCE_STEPS = 50
OUTER_GAIN = 0.25
RMS_CAP = 0.25
HEAD_CONFIG = {"width": 64, "residual_blocks": 3}
HEAD_PARAMETER_COUNT = 736_656
LORA_PARAMETER_COUNT = 737_280
RGB_SHAPE = historical_policy.RGB_SHAPE
SEMANTIC_SHAPE = historical_policy.SEMANTIC_SHAPE
gaussian_log_probability = historical_policy.gaussian_log_probability
rloo_advantages = historical_policy.rloo_advantages
guided_flow = historical_policy.guided_flow
base_flow_and_feature = historical_policy.base_flow_and_feature


class ReplayMismatchError(RuntimeError):
    """The gradient-enabled loss mean differs from the collected action mean."""


def initialize(runtime, arm):
    """Install a fresh policy; discard any old optimizer before switching arms."""
    if arm not in ARMS:
        raise ValueError("Expected B, R, S or L.")
    transformer = runtime["pipeline"].transformer
    # Reusing one loaded backbone for preflight must never retain trained LoRA.
    for name, module in list(transformer.named_modules()):
        if isinstance(module, historical_policy.SelfAttentionLoRA):
            parent_name, attribute = name.rsplit(".", 1)
            setattr(transformer.get_submodule(parent_name), attribute, module.base)
    frozen = [transformer, runtime["semantic"], runtime["codec"]]
    for name in ("vae", "text_encoder"):
        module = getattr(runtime["pipeline"], name, None)
        if module is not None:
            frozen.append(module)
    if "teacher" in runtime:
        frozen.append(runtime["teacher"].model)
    for module in frozen:
        module.requires_grad_(False).eval()
        for parameter in module.parameters():
            parameter.grad = None
    runtime["arm"] = arm
    runtime["parameter_version"] = 0
    runtime.pop("head", None)
    if arm in ("R", "S"):
        # This is flow.py's existing architecture and matched initialization.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            head = ReferenceRGBCorrectionHead(**HEAD_CONFIG)
        runtime["head"] = head.to(device=runtime["device"], dtype=torch.float32).eval()
        if sum(parameter.numel() for parameter in head.parameters()) != HEAD_PARAMETER_COUNT:
            raise ValueError("The registered correction-head parameter count changed.")
    elif arm == "L":
        historical_policy.install_lora(transformer, seed=12000, expected_count=LORA_PARAMETER_COUNT)
    return parameters(runtime)


def parameters(runtime):
    if runtime["arm"] in ("R", "S"):
        return list(runtime["head"].parameters())
    if runtime["arm"] == "L":
        return historical_policy.lora_parameters(runtime["pipeline"].transformer)
    if runtime["arm"] == "B":
        return []
    raise ValueError("Initialize a declared arm first.")


def make_optimizer(runtime):
    return torch.optim.Adam(parameters(runtime), lr=1e-6, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)


def state_dict(runtime):
    """Save only trainable policy tensors, with arm and parameter identity."""
    if runtime["arm"] == "L":
        values = historical_policy.lora_state_dict(runtime["pipeline"].transformer)
    elif runtime["arm"] in ("R", "S"):
        values = {name: value.detach().cpu().clone()
                  for name, value in runtime["head"].state_dict().items()}
    else:
        values = {}
    return {"arm": runtime["arm"], "parameter_version": runtime.get("parameter_version", 0),
            "parameters": values}


def load_state_dict(runtime, saved):
    if saved["arm"] != runtime["arm"]:
        raise ValueError("Policy checkpoint arm does not match the initialized arm.")
    values = saved["parameters"]
    if runtime["arm"] == "L":
        historical_policy.load_lora_state_dict(runtime["pipeline"].transformer, values)
    elif runtime["arm"] in ("R", "S"):
        expected = runtime["head"].state_dict()
        if set(values) != set(expected):
            raise ValueError("Checkpoint must contain exactly the correction-head tensors.")
        for name, value in values.items():
            if (value.shape != expected[name].shape or value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid head checkpoint shape, dtype or values.")
        runtime["head"].load_state_dict(values, strict=True)
    elif values:
        raise ValueError("Baseline checkpoints cannot contain learned parameters.")
    runtime["parameter_version"] = saved["parameter_version"]


def _rng_snapshot():
    numpy_state = np.random.get_state()
    # Primitive keys let accepted checkpoints use torch.load(weights_only=True).
    numpy_saved = {"generator": numpy_state[0], "keys": numpy_state[1].tolist(),
                   "position": numpy_state[2], "has_gaussian": numpy_state[3],
                   "cached_gaussian": numpy_state[4]}
    return {"python": random.getstate(), "numpy": numpy_saved,
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(saved):
    random.setstate(saved["python"])
    numpy_saved = saved["numpy"]
    np.random.set_state((numpy_saved["generator"], np.asarray(numpy_saved["keys"], dtype=np.uint32),
                         numpy_saved["position"], numpy_saved["has_gaussian"], numpy_saved["cached_gaussian"]))
    torch.set_rng_state(saved["torch_cpu"])
    if saved["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(saved["torch_cuda"])


def snapshot(runtime, optimizer):
    return {"policy_state": state_dict(runtime),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None,
            "gradients": [None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                          for parameter in parameters(runtime)],
            "rng": _rng_snapshot(), "parameter_version": runtime.get("parameter_version", 0)}


def restore(runtime, optimizer, saved):
    load_state_dict(runtime, saved["policy_state"])
    if optimizer is not None:
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    for parameter, gradient in zip(parameters(runtime), saved["gradients"]):
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    runtime["parameter_version"] = saved["parameter_version"]
    _restore_rng(saved["rng"])


def bounded_correction(raw_delta, frozen_flow):
    """Differentiate through the per-clip cap, then apply the outer gain."""
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
        correction = OUTER_GAIN * raw_delta * factor.reshape(broadcast_shape)
        if not bool(torch.isfinite(correction).all() and torch.isfinite(raw_rms).all()
                    and torch.isfinite(base_rms).all()):
            raise FloatingPointError("Nonfinite correction or RMS norm.")
        executed_rms = torch.linalg.vector_norm(correction.flatten(1), dim=1) / denominator
    diagnostics = {"outer_gain": OUTER_GAIN, "rms_cap": RMS_CAP,
                   "raw_rms": float(raw_rms.detach().mean()),
                   "executed_rms": float(executed_rms.detach().mean()),
                   "base_rms": float(base_rms.mean()),
                   "cap_scale": float(factor.detach().mean()),
                   "clip_fraction": float((raw_rms > RMS_CAP * base_rms).float().mean())}
    return correction, diagnostics


def policy_mean(state, frozen_flow, raw_delta, step_size):
    """Common FP32 action mean for a head residual or adapted-minus-base flow."""
    if not math.isfinite(float(step_size)) or float(step_size) <= 0:
        raise ValueError("Euler step size must be positive and finite.")
    if state.shape != frozen_flow.shape:
        raise ValueError("State and frozen flow shapes differ.")
    correction, _ = bounded_correction(raw_delta, frozen_flow)
    with torch.autocast(device_type=state.device.type, enabled=False):
        mean = state.detach().float() - float(step_size) * (frozen_flow.detach().float() + correction)
    if not bool(torch.isfinite(mean).all()):
        raise FloatingPointError("Nonfinite action mean.")
    return mean


def _raw_correction(runtime, state, frozen_flow, q, sigma, timestep, prompt, checkpointed):
    arm = runtime["arm"]
    if arm == "B":
        return torch.zeros_like(frozen_flow)
    if arm == "L":
        conditional = historical_policy._forward_branch(
            runtime, state, timestep, prompt, "cond", adapted=True, checkpointed=checkpointed)
        unconditional = historical_policy._forward_branch(
            runtime, state, timestep, prompt, "uncond", adapted=True, checkpointed=checkpointed)
        return guided_flow(conditional, unconditional) - frozen_flow.detach().float()
    if arm not in ("R", "S"):
        raise ValueError("Unknown policy arm.")
    if arm == "R":
        q = torch.zeros_like(q)

    def head_forward(rgb_state, base, semantic_input):
        # Put the context inside the closure so checkpoint backward also uses FP32.
        with torch.autocast(device_type=rgb_state.device.type, enabled=False):
            return runtime["head"](rgb_state.float(), base.float(), semantic_input.float(), sigma)

    if any(parameter.dtype != torch.float32 for parameter in runtime["head"].parameters()):
        raise TypeError("Correction-head parameters must remain FP32.")
    if checkpointed and torch.is_grad_enabled():
        return torch_checkpoint(head_forward, state, frozen_flow, q, use_reentrant=False)
    return head_forward(state, frozen_flow, q)


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
    return policy_mean(state, frozen_flow, delta, transition["h"])


@torch.no_grad()
def collect_trajectory(runtime, prompt, seeds, parameter_version=0):
    """Fifty matched Euler/UniPC steps, with stochastic actions only at 10/20."""
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    if (set(seeds) != {"rgb", "semantic", "action"}
            or any(type(seed) is not int or seed < 0 or seed == 47 for seed in seeds.values())):
        raise ValueError("Use three nonnegative permitted integer seeds.")
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
    semantic_state = torch.randn(SEMANTIC_SHAPE, generator=semantic_generator).to(device)
    transitions, ledger = [], []
    with historical_policy.adapter_mode(transformer, False):
        for step_index, timestep in enumerate(schedule.timesteps):
            step_size = float(schedule.sigmas[step_index] - schedule.sigmas[step_index + 1])
            sigma = float(schedule.sigmas[step_index])
            frozen_flow, feature = base_flow_and_feature(runtime, rgb, timestep, prompt, capture_feature=True)
            # q is computed from the pre-action RGB features and semantic state.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                semantic_flow = semantic(semantic_state.float(), feature, timestep.float().reshape(1)).float()
            q = (semantic_state.float() - sigma * semantic_flow).detach()
            if not bool(torch.isfinite(q).all()):
                raise FloatingPointError("Nonfinite pre-action semantic prediction.")
            if arm == "R":
                q = torch.zeros_like(q)
            is_action = step_index in ACTION_STEPS
            diagnostics = {}
            raw_delta = torch.zeros_like(frozen_flow)
            if is_action:
                raw_delta = _raw_correction(runtime, rgb, frozen_flow, q, sigma, timestep, prompt, False)
                _, diagnostics = bounded_correction(raw_delta, frozen_flow)
            mean = policy_mean(rgb, frozen_flow, raw_delta, step_size)
            rgb_next = mean
            if is_action:
                standard_deviation = 0.03 * math.sqrt(step_size)
                noise = torch.randn(rgb.shape, device=device, dtype=torch.float32, generator=action_generator)
                rgb_next = mean + standard_deviation * noise
                transitions.append({
                    "state": rgb.detach().float().cpu().clone(),
                    "frozen_flow": frozen_flow.detach().float().cpu().clone(),
                    "q": q.detach().float().cpu().clone(),
                    "action": rgb_next.detach().float().cpu().clone(),
                    "old_mean": mean.detach().float().cpu().clone(),
                    "sigma": sigma, "step_index": step_index,
                    "timestep": timestep.detach().cpu().clone(), "h": step_size,
                    "std": standard_deviation, "prompt": prompt, "arm": arm,
                    "parameter_version": parameter_version, "correction": diagnostics,
                })
            semantic_state = semantic_schedule.step(semantic_flow, timestep, semantic_state, return_dict=False)[0].float()
            rgb = rgb_next.float()
            if not bool(torch.isfinite(rgb).all() and torch.isfinite(semantic_state).all()):
                raise FloatingPointError(f"Nonfinite rollout state at denoising index {step_index}.")
            ledger.append({"step_index": step_index, "h": step_size, "sigma": sigma,
                           "adapted": is_action and arm != "B", "lora_enabled": is_action and arm == "L",
                           "stochastic": is_action, "base_conditional_captures": 1,
                           "semantic_advances": 1, **diagnostics})
    if len(ledger) != 50 or [item["step_index"] for item in transitions] != list(ACTION_STEPS):
        raise RuntimeError("The registered 50-step, two-action schedule did not complete.")
    return {"rgb_latent": rgb.detach().cpu(), "semantic_latent": semantic_state.detach().cpu(),
            "rng_states": dict(initial_rng, rgb_after=rgb_generator.get_state(),
                               action_after=action_generator.get_state(), semantic_after=semantic_generator.get_state()),
            "transitions": transitions, "statistics": {
                "seeds": dict(seeds), "arm": arm, "parameter_version": parameter_version,
                "ledger": ledger, "sigmas": schedule.sigmas.cpu().tolist(),
                "timesteps": schedule.timesteps.cpu().tolist(), "semantic_steps": len(ledger),
                "sampler": "euler_two_gaussian_actions", "guidance_scale": 5.0,
                "outer_gain": OUTER_GAIN, "rms_cap": RMS_CAP,
            }}


def frozen_reference_mean(transition, device=None):
    state = transition["state"].detach().to(device=device, dtype=torch.float32)
    flow = transition["frozen_flow"].detach().to(device=device, dtype=torch.float32)
    with torch.autocast(device_type=state.device.type, enabled=False):
        return state - float(transition["h"]) * flow


def _kl_row(mean, reference, standard_deviation):
    difference = mean.detach().double() - reference.detach().to(mean.device, dtype=torch.float64)
    normalized_square = difference.square() / float(standard_deviation) ** 2
    return {"kl": float(0.5 * normalized_square.sum()),
            "normalized_rms": float(normalized_square.mean().sqrt())}


def _summarize_kl(rows):
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise FloatingPointError("Invalid replay KL or action-mean movement.")
    return {"mean_kl": sum(row["kl"] for row in rows) / len(rows),
            "max_kl": max(row["kl"] for row in rows),
            "normalized_rms": math.sqrt(sum(row["normalized_rms"] ** 2 for row in rows) / len(rows)),
            "actions": rows}


@torch.no_grad()
def replay_statistics(transitions, mean_function):
    behavior, reference = [], []
    for transition in transitions:
        mean = mean_function(transition)
        if mean.dtype != torch.float32 or not bool(torch.isfinite(mean).all()):
            raise FloatingPointError("Executed replay mean must be finite FP32.")
        behavior.append(_kl_row(mean, transition["old_mean"], transition["std"]))
        reference.append(_kl_row(mean, frozen_reference_mean(transition, mean.device), transition["std"]))
    return {"behavior": _summarize_kl(behavior), "reference": _summarize_kl(reference)}


def update_batch(trainable_parameters, optimizer, trajectories, advantages, mean_function, *, denominator):
    """One accumulated proposal, with one LR/10 retry only for excessive KL.

    The flexible entry point permits a disposable one-trajectory preflight.
    Production uses update(), which enforces all sixteen fresh trajectories.
    """
    trainable_parameters = list(trainable_parameters)
    advantages = torch.as_tensor(advantages, dtype=torch.float64).detach().flatten()
    if (not trajectories or len(trajectories) != len(advantages)
            or denominator != len(trajectories) or not trainable_parameters):
        raise ValueError("The denominator must include every trajectory, including ties.")
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("Nonfinite advantages.")
    report = {"status": "no_learning_signal", "accepted": False, "numerical_accepted": False,
              "optimizer_steps": 0, "proposals": 0, "backoff": False,
              "learning_rate": optimizer.param_groups[0]["lr"]}
    if not bool(torch.any(advantages != 0)):
        return report
    transitions = [transition for trajectory in trajectories for transition in trajectory["transitions"]]
    saved_parameters = [parameter.detach().cpu().clone() for parameter in trainable_parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = _rng_snapshot()

    def rollback():
        with torch.no_grad():
            for parameter, value in zip(trainable_parameters, saved_parameters):
                parameter.copy_(value.to(parameter.device))
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        _restore_rng(saved_rng)

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    try:
        # Check the actual gradient-enabled checkpointed output immediately
        # before its backward, including zero-advantage siblings' saved states.
        with torch.enable_grad():
            for trajectory, advantage in zip(trajectories, advantages):
                for transition in trajectory["transitions"]:
                    mean = mean_function(transition)
                    old_mean = transition["old_mean"].detach().to(mean.device)
                    if mean.dtype != torch.float32 or not bool(torch.isfinite(mean).all()):
                        raise FloatingPointError("The loss mean must be finite FP32.")
                    if not torch.equal(mean.detach(), old_mean):
                        raise ReplayMismatchError("Gradient-enabled collection/replay mean mismatch; no update.")
                    if float(advantage) != 0:
                        action = transition["action"].detach().to(mean.device, dtype=torch.float32)
                        loss = -(float(advantage) / denominator) * gaussian_log_probability(action, mean, transition["std"])
                        loss.backward()
                        total_loss += float(loss.detach())
                        del action, loss
                    del mean, old_mean
        gradients = [parameter.grad for parameter in trainable_parameters if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("Missing or nonfinite trainable gradients.")
        gradient_norm = math.sqrt(sum(float(gradient.double().square().sum()) for gradient in gradients))
        if not math.isfinite(gradient_norm) or gradient_norm == 0:
            raise FloatingPointError("No finite nonzero policy gradient.")
        before = replay_statistics(transitions, mean_function)
        if before["behavior"]["max_kl"] != 0:
            raise ReplayMismatchError("Inference and gradient-enabled replay disagree; no update.")
        report.update(loss=total_loss, gradient_norm=gradient_norm, unchanged_replay=before)
        for attempt in range(2):
            if attempt == 1:
                rollback()
                for group in optimizer.param_groups:
                    group["lr"] *= 0.1
            optimizer.step()
            report.update(proposals=attempt + 1, backoff=attempt == 1)
            if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable_parameters):
                raise FloatingPointError("Nonfinite proposed policy parameters.")
            for optimizer_values in optimizer.state.values():
                for value in optimizer_values.values():
                    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                        raise FloatingPointError("Nonfinite proposed Adam state.")
            after = replay_statistics(transitions, mean_function)
            report["proposed_replay"] = after
            within_kl = all(after[name]["mean_kl"] <= 10.0 and after[name]["max_kl"] <= 20.0
                            for name in ("behavior", "reference"))
            if within_kl:
                if after["behavior"]["normalized_rms"] == 0:
                    rollback()
                    optimizer.zero_grad(set_to_none=True)
                    report["status"] = "no_effective_policy_change"
                else:
                    report.update(status="numerical_accepted", accepted=True,
                                  numerical_accepted=True, optimizer_steps=1)
                report["learning_rate"] = optimizer.param_groups[0]["lr"]
                return report
        rollback()
        optimizer.zero_grad(set_to_none=True)
        report.update(status="update_rejected", learning_rate=optimizer.param_groups[0]["lr"])
        return report
    except BaseException:
        rollback()
        optimizer.zero_grad(set_to_none=True)
        raise


def update(runtime, optimizer, trajectories, rewards):
    """Strict fresh 4x4 RLOO, summing both full-coordinate Gaussian scores."""
    if runtime["arm"] == "B":
        raise ValueError("The frozen baseline cannot learn.")
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
            seeds = statistics["seeds"]
            transitions = trajectory["transitions"]
            if [transition["step_index"] for transition in transitions] != list(ACTION_STEPS):
                raise ValueError("Every video must retain exactly actions 10 and 20.")
            if (set(seeds) != {"rgb", "semantic", "action"}
                    or any(type(seed) is not int or seed < 0 or seed == 47 for seed in seeds.values())):
                raise ValueError("Invalid initial-context or action seed.")
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
    # Consume the batch even if numerical execution fails; no silent retry with
    # the same collected outcomes is allowed outside the single proposal backoff.
    for trajectory in flat_trajectories:
        trajectory["used_for_update"] = True
    result = update_batch(parameters(runtime), optimizer, flat_trajectories, advantages,
                          lambda transition: replay_mean(runtime, transition), denominator=16)
    result["advantages"] = advantages.tolist()
    return result
