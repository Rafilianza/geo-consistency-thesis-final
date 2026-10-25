"""Two-action Wan LoRA policy and fresh, four-by-four RLOO updates.

RGB uses explicit Euler steps. The frozen semantic observer keeps its own
UniPC scheduler and observes only the adapter-disabled conditional forward.
Checkpoint files should contain ``lora_state_dict``, never the Wan state dict.
"""

from contextlib import contextmanager, nullcontext
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


ACTION_STEPS = (10, 20)
LORA_TARGETS = tuple(
    f"blocks.{block_index}.attn1.to_{projection}"
    for block_index in range(30) for projection in ("q", "v")
)
EXPECTED_TRAINABLE_PARAMETERS = 737_280
RGB_SHAPE = (1, 16, 21, 60, 104)
SEMANTIC_SHAPE = (1, 8, 21, 60, 104)


class SelfAttentionLoRA(nn.Module):
    """Rank-four, scale-one projection; only the two FP32 matrices learn."""

    def __init__(self, base, generator):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Each enumerated self-attention projection must be nn.Linear.")
        self.base = base.requires_grad_(False)
        initial_a = torch.empty(4, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(initial_a, a=math.sqrt(5), generator=generator)
        self.lora_a = nn.Parameter(initial_a.to(base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, 4, device=base.weight.device))
        self.enabled = False

    def forward(self, hidden_states):
        output = self.base(hidden_states)
        if not self.enabled:
            return output
        if self.lora_a.dtype != torch.float32 or self.lora_b.dtype != torch.float32:
            raise TypeError("LoRA parameters must remain FP32, including after device transfers.")
        # Wan remains BF16, while the small adapter matmuls and parameters are FP32.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            low_rank = F.linear(hidden_states.float(), self.lora_a)
            correction = F.linear(low_rank, self.lora_b)
            return (output.float() + correction).to(output.dtype)


def install_lora(transformer, seed=12000, *, expected_count=EXPECTED_TRAINABLE_PARAMETERS):
    """Freeze Wan and modify exactly the 60 predeclared self-Q/V projections."""
    if len(transformer.blocks) != 30:
        raise ValueError("This policy requires exactly 30 Wan blocks.")
    originals = [transformer.get_submodule(name) for name in LORA_TARGETS]
    if any(not isinstance(module, nn.Linear) for module in originals):
        raise ValueError("Install once, on unmodified nn.Linear self-Q/V projections.")
    actual_count = sum(4 * (module.in_features + module.out_features) for module in originals)
    if actual_count != expected_count:
        raise ValueError(f"Expected {expected_count} LoRA parameters; found {actual_count}.")
    transformer.requires_grad_(False).eval()
    # Our outer checkpoint closure controls cache and adapter context on replay.
    if hasattr(transformer, "disable_gradient_checkpointing"):
        transformer.disable_gradient_checkpointing()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    adapters = []
    for name, base in zip(LORA_TARGETS, originals):
        parent_name, attribute = name.rsplit(".", 1)
        adapter = SelfAttentionLoRA(base, generator)
        setattr(transformer.get_submodule(parent_name), attribute, adapter)
        adapters.append(adapter)
    return adapters


def lora_parameters(transformer):
    parameters = []
    for name in LORA_TARGETS:
        module = transformer.get_submodule(name)
        if not isinstance(module, SelfAttentionLoRA):
            raise ValueError("The enumerated LoRA projections have not been installed.")
        parameters.extend((module.lora_a, module.lora_b))
    return parameters


def lora_state_dict(transformer):
    """Copy only 737,280 adapter values to CPU, never the full frozen model."""
    saved = {}
    for name in LORA_TARGETS:
        module = transformer.get_submodule(name)
        for suffix in ("lora_a", "lora_b"):
            saved[f"{name}.{suffix}"] = getattr(module, suffix).detach().float().cpu().clone()
    return saved


@torch.no_grad()
def load_lora_state_dict(transformer, saved):
    expected = {f"{name}.{suffix}" for name in LORA_TARGETS for suffix in ("lora_a", "lora_b")}
    if set(saved) != expected:
        raise ValueError("Checkpoint must contain exactly the enumerated LoRA matrices.")
    for name in LORA_TARGETS:
        module = transformer.get_submodule(name)
        for suffix in ("lora_a", "lora_b"):
            parameter = getattr(module, suffix)
            value = saved[f"{name}.{suffix}"]
            if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Invalid LoRA checkpoint shape, dtype, or values.")
            parameter.copy_(value.to(parameter.device))


def make_optimizer(transformer):
    return torch.optim.Adam(lora_parameters(transformer), lr=1e-6, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.0)


@contextmanager
def adapter_mode(transformer, enabled):
    """Restore every enable flag even on a failed forward or backward replay."""
    adapters = [module for module in transformer.modules() if isinstance(module, SelfAttentionLoRA)]
    if enabled and len(adapters) != 60:
        raise ValueError("An adapted forward needs all 60 LoRA projections installed.")
    previous = [module.enabled for module in adapters]
    try:
        for module in adapters:
            module.enabled = enabled
        yield
    finally:
        for module, was_enabled in zip(adapters, previous):
            module.enabled = was_enabled


def guided_flow(conditional, unconditional):
    """Cast BEFORE guidance, preserving gradients through both CFG branches."""
    return 5.0 * conditional.float() - 4.0 * unconditional.float()


def policy_mean(state, frozen_flow, adapted_flow, step_size):
    """The one shared rollout/replay mean, with a differentiable RMS cap."""
    step_size = float(step_size)
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("Euler step size must be finite and positive.")
    state = state.detach().float()
    frozen_flow = frozen_flow.detach().float()
    if state.shape != frozen_flow.shape or state.shape != adapted_flow.shape:
        raise ValueError("State and flow shapes must match.")
    difference = adapted_flow.float() - frozen_flow
    # vector_norm defines a finite zero derivative; clamp prevents division by zero.
    dimensions = difference.numel()
    frozen_rms = torch.linalg.vector_norm(frozen_flow) / math.sqrt(dimensions)
    difference_rms = torch.linalg.vector_norm(difference) / math.sqrt(dimensions)
    factor = (0.25 * frozen_rms / difference_rms.clamp_min(1e-12)).clamp(max=1.0)
    mean = state - step_size * (frozen_flow + factor * difference)
    if not bool(torch.isfinite(mean).all()):
        raise FloatingPointError("Nonfinite policy mean.")
    return mean


def gaussian_log_probability(action, mean, standard_deviation):
    """Sum all Gaussian coordinates in FP64; saved actions remain detached."""
    standard_deviation = float(standard_deviation)
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("A Gaussian action needs a positive finite standard deviation.")
    if action.shape != mean.shape or action.numel() == 0:
        raise ValueError("Action and mean must have the same nonempty shape.")
    residual = (action.detach().double() - mean.double()) / standard_deviation
    value = -0.5 * (residual.square() + 2 * math.log(standard_deviation) + math.log(2 * math.pi)).sum()
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("Nonfinite Gaussian likelihood.")
    return value


def rloo_advantages(rewards):
    """Leave out each candidate, keeping tied groups exactly zero."""
    values = torch.as_tensor(rewards, dtype=torch.float64).detach()
    if values.ndim != 2 or values.shape[1] != 4 or not torch.isfinite(values).all():
        raise ValueError("Expected finite rewards with four candidates per group.")
    advantages = values - (values.sum(dim=1, keepdim=True) - values) / 3
    tied = (values == values[:, :1]).all(dim=1)
    advantages[tied] = 0
    return advantages


def semantic_advantage_comparison(rgb_rewards, semantic_rewards):
    rgb = rloo_advantages(rgb_rewards).flatten()
    semantic = rloo_advantages(semantic_rewards).flatten()
    denominator = float(torch.dot(rgb, rgb))
    scale = max(0.0, float(torch.dot(rgb, semantic)) / denominator) if denominator else 0.0
    residual = float((semantic - scale * rgb).square().mean().sqrt())
    return {"nonnegative_scale": scale, "residual_rms": residual}


def _forward_branch(runtime, state, timestep, prompt, branch, *, adapted, checkpointed=False):
    transformer = runtime["pipeline"].transformer
    embedding = runtime["embeddings"][prompt][0 if branch == "cond" else 1]
    projection = transformer.get_submodule(LORA_TARGETS[0])
    model_dtype = projection.base.weight.dtype if isinstance(projection, SelfAttentionLoRA) else projection.weight.dtype
    model_input = state.detach().to(dtype=model_dtype)
    time_input = torch.as_tensor(timestep, device=state.device).reshape(1).expand(state.shape[0])
    embedding = embedding.detach().to(device=state.device, dtype=model_dtype)

    def forward(hidden_states, timesteps, prompt_embedding):
        # This closure runs again during checkpoint backward, after callers exited.
        cache = transformer.cache_context(branch) if hasattr(transformer, "cache_context") else nullcontext()
        with adapter_mode(transformer, adapted), cache:
            return transformer(hidden_states=hidden_states, timestep=timesteps,
                               encoder_hidden_states=prompt_embedding, return_dict=False)[0]

    if checkpointed and torch.is_grad_enabled():
        return checkpoint(forward, model_input, time_input, embedding, use_reentrant=False)
    return forward(model_input, time_input, embedding)


@torch.no_grad()
def base_flow_and_feature(runtime, state, timestep, prompt, *, capture_feature):
    """The hook exists only around the one adapter-disabled conditional call."""
    feature = None
    if capture_feature:
        from middle_feature_semantic.wan_middle_feature_capture import WanMiddleFeatureCapture
        from middle_feature_semantic.wan_middle_feature_capture import feature_tokens_to_video_grid

        transformer = runtime["pipeline"].transformer
        with WanMiddleFeatureCapture(transformer, 15) as capture:
            conditional = _forward_branch(runtime, state, timestep, prompt, "cond", adapted=False)
            if capture.call_count != 1:
                raise RuntimeError("The base-conditional observer must capture exactly once.")
            feature = feature_tokens_to_video_grid(capture.feature, state.shape, transformer.config.patch_size)
    else:
        conditional = _forward_branch(runtime, state, timestep, prompt, "cond", adapted=False)
    unconditional = _forward_branch(runtime, state, timestep, prompt, "uncond", adapted=False)
    return guided_flow(conditional, unconditional).detach(), feature


def _adapted_mean(runtime, state, timestep, prompt, step_size, frozen_flow, *, checkpointed):
    conditional = _forward_branch(runtime, state, timestep, prompt, "cond", adapted=True,
                                  checkpointed=checkpointed)
    unconditional = _forward_branch(runtime, state, timestep, prompt, "uncond", adapted=True,
                                    checkpointed=checkpointed)
    return policy_mean(state, frozen_flow, guided_flow(conditional, unconditional), step_size)


def replay_mean(runtime, transition, checkpoint=True):
    """Replay one detached action state; never capture or advance semantics."""
    if transition["step_index"] not in ACTION_STEPS:
        raise ValueError("Only denoising indices 10 and 20 belong to this policy.")
    state = transition["state"].detach().to(runtime["device"], dtype=torch.float32)
    frozen_flow, _ = base_flow_and_feature(runtime, state, transition["timestep"],
                                          transition["prompt"], capture_feature=False)
    return _adapted_mean(runtime, state, transition["timestep"], transition["prompt"],
                         transition["h"], frozen_flow, checkpointed=checkpoint)


@torch.no_grad()
def collect_trajectory(runtime, prompt, seeds, baseline=False):
    """Generate one matched stochastic-Euler RGB / frozen-UniPC semantic pair."""
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipeline = runtime["pipeline"]
    device = torch.device(runtime["device"])
    transformer = pipeline.transformer.to(device).eval()
    semantic = runtime["semantic"].to(device).eval().requires_grad_(False)
    pipeline.vae.to("cpu")
    runtime["codec"].to("cpu")
    if getattr(pipeline.config, "expand_timesteps", False):
        raise ValueError("This policy supports scalar Wan T2V timesteps only.")
    schedule = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    semantic_schedule = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    schedule.set_timesteps(50, device=device)
    semantic_schedule.set_timesteps(50, device=device)
    if not torch.equal(schedule.timesteps, semantic_schedule.timesteps):
        raise ValueError("RGB and semantic timestep clocks differ.")
    if len(schedule.sigmas) != 51:
        raise ValueError("Expected 50 Euler intervals and a terminal sigma.")
    rgb_generator = torch.Generator(device=device).manual_seed(int(seeds["rgb"]))
    action_generator = torch.Generator(device=device).manual_seed(int(seeds["action"]))
    semantic_generator = torch.Generator(device="cpu").manual_seed(int(seeds["semantic"]))
    initial_rng = {"rgb_before": rgb_generator.get_state(), "action_before": action_generator.get_state(),
                   "semantic_before": semantic_generator.get_state()}
    rgb = pipeline.prepare_latents(1, 16, 480, 832, 81, torch.float32, device, rgb_generator, None)
    if tuple(rgb.shape) != RGB_SHAPE:
        raise ValueError("Unexpected Wan RGB latent shape.")
    semantic_state = torch.randn(SEMANTIC_SHAPE, generator=semantic_generator, dtype=torch.float32).to(device)
    transitions, ledger = [], []

    # All arms use this same mean construction and independent action noise.
    with adapter_mode(transformer, False):
        for step_index, timestep in enumerate(schedule.timesteps):
            step_size = float(schedule.sigmas[step_index] - schedule.sigmas[step_index + 1])
            base, feature = base_flow_and_feature(runtime, rgb, timestep, prompt, capture_feature=True)
            if step_index in ACTION_STEPS and not baseline:
                mean = _adapted_mean(runtime, rgb, timestep, prompt, step_size, base, checkpointed=False)
            else:
                mean = policy_mean(rgb, base, base, step_size)
            rgb_next = mean
            if step_index in ACTION_STEPS:
                standard_deviation = 0.03 * math.sqrt(step_size)
                noise = torch.randn(rgb.shape, device=device, dtype=torch.float32, generator=action_generator)
                rgb_next = mean + standard_deviation * noise
                transitions.append({
                    "state": rgb.detach().float().cpu().clone(),
                    "action": rgb_next.detach().float().cpu().clone(),
                    "old_mean": mean.detach().float().cpu().clone(),
                    "step_index": step_index, "timestep": timestep.detach().cpu().clone(), "h": step_size,
                    "std": standard_deviation, "prompt": prompt, "baseline": baseline,
                })
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                semantic_flow = semantic(semantic_state.float(), feature, timestep.float().reshape(1)).float()
            semantic_state = semantic_schedule.step(semantic_flow, timestep, semantic_state, return_dict=False)[0].float()
            rgb = rgb_next.float()
            if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(semantic_state).all()):
                raise FloatingPointError(f"Nonfinite rollout state at denoising step {step_index}.")
            ledger.append({"step_index": step_index, "h": step_size,
                           "adapted": step_index in ACTION_STEPS and not baseline,
                           "stochastic": step_index in ACTION_STEPS, "base_conditional_captures": 1})
    return {"rgb_latent": rgb.detach().cpu(), "semantic_latent": semantic_state.detach().cpu(),
            "rng_states": dict(initial_rng, rgb_after=rgb_generator.get_state(),
                               action_after=action_generator.get_state(), semantic_after=semantic_generator.get_state()),
            "transitions": transitions, "statistics": {
                "seeds": dict(seeds), "baseline": baseline, "ledger": ledger,
                "sigmas": schedule.sigmas.cpu().tolist(), "timesteps": schedule.timesteps.cpu().tolist(),
                "semantic_steps": len(ledger), "sampler": "euler_two_gaussian_actions",
            }}


@torch.no_grad()
def replay_statistics(transitions, mean_function):
    rows = []
    for transition in transitions:
        mean = mean_function(transition)
        old_mean = transition["old_mean"].detach().to(mean.device, dtype=torch.float64)
        change = mean.double() - old_mean
        normalized_square = change.square() / float(transition["std"]) ** 2
        rows.append({"kl": float(0.5 * normalized_square.sum()),
                     "normalized_rms": float(normalized_square.mean().sqrt())})
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values()):
        raise FloatingPointError("Invalid replay KL or mean movement.")
    return {"mean_kl": sum(row["kl"] for row in rows) / len(rows),
            "max_kl": max(row["kl"] for row in rows),
            "normalized_rms": math.sqrt(sum(row["normalized_rms"] ** 2 for row in rows) / len(rows)),
            "actions": rows}


def _parameter_snapshot(parameters):
    return [parameter.detach().cpu().clone() for parameter in parameters]


@torch.no_grad()
def _restore(parameters, values, optimizer, optimizer_state):
    for parameter, value in zip(parameters, values):
        parameter.copy_(value.to(parameter.device))
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))


def accumulated_update(parameters, optimizer, trajectories, advantages, mean_function, *, denominator):
    """Accumulate fresh scores once; restore full Adam state for at most one retry.

    This small numerical entry point also supports real CPU gradient tests.
    Production callers use ``update_policy`` for strict 4x4 batch validation.
    """
    advantages = torch.as_tensor(advantages, dtype=torch.float64).detach().flatten()
    if len(trajectories) != len(advantages) or denominator != len(trajectories):
        raise ValueError("The loss denominator must include every trajectory, including ties.")
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("Nonfinite advantages.")
    if not bool(torch.any(advantages != 0)):
        # Even zero_grad/step would change optimizer-related state: return first.
        return {"status": "no_learning_signal", "accepted": False, "optimizer_steps": 0}
    transitions = [transition for trajectory in trajectories for transition in trajectory["transitions"]]
    before = replay_statistics(transitions, mean_function)
    values = _parameter_snapshot(parameters)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    try:
        for trajectory, advantage in zip(trajectories, advantages):
            if float(advantage) == 0:
                continue
            for transition in trajectory["transitions"]:
                mean = mean_function(transition)
                action = transition["action"].detach().to(mean.device, dtype=torch.float32)
                loss = -(float(advantage) / denominator) * gaussian_log_probability(action, mean, transition["std"])
                loss.backward()
                total_loss += float(loss.detach())
                del mean, action, loss
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("Missing or nonfinite LoRA gradients.")
        gradient_norm = math.sqrt(sum(float(gradient.double().square().sum()) for gradient in gradients))
        for attempt in range(2):
            if attempt == 1:
                _restore(parameters, values, optimizer, optimizer_state)
                for group in optimizer.param_groups:
                    group["lr"] *= 0.1
            optimizer.step()
            after = replay_statistics(transitions, mean_function)
            within_kl = after["mean_kl"] <= 10.0 and after["max_kl"] <= 20.0
            if within_kl:
                moved = after["normalized_rms"] > before["normalized_rms"]
                if not moved:
                    _restore(parameters, values, optimizer, optimizer_state)
                return {"status": "accepted" if moved else "no_effective_policy_change",
                        "accepted": moved, "optimizer_steps": 1 if moved else 0,
                        "proposals": attempt + 1, "backoff": attempt == 1,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "loss": total_loss, "gradient_norm": gradient_norm,
                        "unchanged_replay": before, "proposed_replay": after}
        _restore(parameters, values, optimizer, optimizer_state)
        return {"status": "update_rejected", "accepted": False, "optimizer_steps": 0,
                "proposals": 2, "backoff": True, "learning_rate": optimizer.param_groups[0]["lr"],
                "loss": total_loss, "gradient_norm": gradient_norm,
                "unchanged_replay": before, "proposed_replay": after}
    except BaseException:
        _restore(parameters, values, optimizer, optimizer_state)
        raise


def update_policy(runtime, optimizer, groups, rewards):
    """One fresh 16-video / 32-action update, keeping the denominator fixed."""
    if len(groups) != 4 or any(len(group) != 4 for group in groups):
        raise ValueError("Exactly four groups of four fresh trajectories are required.")
    advantages = rloo_advantages(rewards)
    if tuple(advantages.shape) != (4, 4):
        raise ValueError("Rewards must correspond to the four-by-four trajectory batch.")
    trajectories = [trajectory for group in groups for trajectory in group]
    for trajectory in trajectories:
        if trajectory.get("used_for_update", False):
            raise ValueError("RLOO trajectories cannot be reused for another update.")
        transitions = trajectory["transitions"]
        if [transition["step_index"] for transition in transitions] != list(ACTION_STEPS):
            raise ValueError("Each video must have exactly the two declared actions.")
        for transition in transitions:
            if transition.get("baseline", False):
                raise ValueError("Frozen reference/evaluation actions cannot train the policy.")
            if not math.isclose(transition["std"], 0.03 * math.sqrt(transition["h"]), rel_tol=1e-12):
                raise ValueError("Stored exploration variance differs from the fixed policy.")
            for key in ("state", "action", "old_mean"):
                value = transition[key]
                if value.dtype != torch.float32 or value.requires_grad or not torch.isfinite(value).all():
                    raise ValueError("Stored states, actions, and means must be detached finite FP32.")
    parameters = lora_parameters(runtime["pipeline"].transformer)
    result = accumulated_update(parameters, optimizer, trajectories, advantages,
                                lambda transition: replay_mean(runtime, transition), denominator=16)
    for trajectory in trajectories:
        trajectory["used_for_update"] = True
    result["advantages"] = advantages.tolist()
    return result


def technical_preflight(runtime, trajectory, optimizer):
    """Disposable real replay/backward/observer check, restoring adapters and Adam."""
    transformer = runtime["pipeline"].transformer
    parameters = lora_parameters(transformer)
    values = _parameter_snapshot(parameters)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    previous_gradients = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        transition = trajectory["transitions"][0]
        state = transition["state"].to(runtime["device"])
        _, first_feature = base_flow_and_feature(runtime, state, transition["timestep"], transition["prompt"], capture_feature=True)
        with torch.no_grad():
            for parameter in parameters[1::2]:
                parameter.add_(0.01)
        _, changed_feature = base_flow_and_feature(runtime, state, transition["timestep"], transition["prompt"], capture_feature=True)
        observer_invariant = torch.equal(first_feature, changed_feature)
        del first_feature, changed_feature
        _restore(parameters, values, optimizer, optimizer_state)
        if not observer_invariant:
            raise RuntimeError("Frozen observer depends on loaded adapter weights.")
        result = accumulated_update(parameters, optimizer, [trajectory], [1.0],
                                    lambda saved: replay_mean(runtime, saved), denominator=1)
        b_gradient_nonzero = any(parameter.grad is not None and bool(torch.any(parameter.grad != 0))
                                 for parameter in parameters[1::2])
        if not b_gradient_nonzero:
            raise FloatingPointError("Preflight found no nonzero LoRA-B gradient.")
        result["observer_invariant"] = observer_invariant
        result["lora_b_gradient_nonzero"] = b_gradient_nonzero
        result["disposable_advantage"] = 1.0
        return result
    finally:
        _restore(parameters, values, optimizer, optimizer_state)
        for parameter, gradient in zip(parameters, previous_gradients):
            parameter.grad = gradient
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
