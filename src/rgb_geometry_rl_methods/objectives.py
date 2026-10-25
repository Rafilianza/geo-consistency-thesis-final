"""Small, differentiable objectives for the four published RL method families.

These are independently expressed mathematical kernels, not vendored trainers.
Source attributions (the referenced repositories retain their own licenses):

* Flow-GRPO (MIT), ``yifan123/flow_grpo`` at
  ``879042cf5707f8b90daa98d147d7deac2317c5da``:
  ``flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py`` and
  ``scripts/train_wan2_1.py``.
* Flash-GRPO, ``Shredded-Pork/Flash-GRPO`` at
  ``bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d``:
  ``flow_grpo/diffusers_patch/wan2_1_pipeline_with_logprob2.py`` and
  ``scripts/train_wan2_1_flash_1node.py``.
* AWM (Apache-2.0), ``scxue/advantage_weighted_matching`` at
  ``15f91b8596168bc2399824eea18c0e1d07df7781``:
  ``advantage_weighted_matching/scripts/train_sd3_awm.py``.
* DiffusionNFT (Apache-2.0), ``NVlabs/DiffusionNFT`` at
  ``cbb14f84b8312b620390dfcbe2fab69c1383b104``:
  ``scripts/train_nft_sd3.py``.

The experiment uses group-local population standard deviations, AWM's uniform
matching weight and one-pass surrogate, and NFT beta=1. These choices are
explicit local settings, not reproductions of the papers' benchmark recipes.
Inputs contain a leading batch dimension. No function samples random numbers,
loads models, writes files, or changes a scheduler.
"""

import math

import torch


def _math_tensor(value):
    """Keep float64 references; avoid half-precision probability arithmetic."""
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.float()
    return value


def _example_mean(value):
    """Average latent coordinates without mixing different videos."""
    if value.ndim < 2:
        raise ValueError("Expected a batch dimension and at least one latent dimension.")
    return value.flatten(start_dim=1).mean(dim=1)


def _step_values(sigma, next_sigma, sigma_one):
    sigma = float(sigma)
    next_sigma = float(next_sigma)
    sigma_one = float(sigma_one)
    if not (0.0 <= next_sigma < sigma <= 1.0 and 0.0 < sigma_one <= 1.0):
        raise ValueError("Require 0 <= next_sigma < sigma <= 1 and 0 < sigma_one <= 1.")
    return sigma, sigma - next_sigma, sigma_one * sigma


def advantages(rewards):
    """Compare each video only with siblings from the same prompt.

    ``rewards`` has shape [groups, siblings]. Tied groups return exactly zero;
    no artificial reward difference is introduced.
    """
    reward_tensor = torch.as_tensor(rewards).detach()
    if not reward_tensor.is_floating_point():
        reward_tensor = reward_tensor.float()
    reward_tensor = _math_tensor(reward_tensor)
    if reward_tensor.ndim != 2 or reward_tensor.numel() == 0:
        raise ValueError("Rewards must be a nonempty [groups, siblings] array.")
    if not torch.isfinite(reward_tensor).all():
        raise ValueError("Rewards must be finite.")
    group_mean = reward_tensor.mean(dim=1, keepdim=True)
    group_std = reward_tensor.std(dim=1, unbiased=False, keepdim=True)
    relative_rewards = (reward_tensor - group_mean) / (group_std + 1e-4)
    # A floating-point reduction can round the mean of identical rewards.
    # Exact ties must still produce no preference signal.
    tied_groups = (reward_tensor == reward_tensor[:, :1]).all(dim=1, keepdim=True)
    relative_rewards = torch.where(tied_groups, torch.zeros_like(relative_rewards), relative_rewards)
    return relative_rewards.clamp(-5.0, 5.0)


def wan_transition(state, velocity, sigma, next_sigma, sigma_one):
    """Return the official Wan SDE transition mean and scalar standard deviation.

    ``sigma_one`` is scheduler.sigmas[1], not the largest sigma. The schedule
    must terminate at zero. With h = sigma - next_sigma and g = sigma_one*sigma,
    the action is mean + g*sqrt(h)*noise. The final positive-to-zero transition
    remains stochastic; never call this function at the appended sigma=0.
    """
    sigma, step_size, diffusion = _step_values(sigma, next_sigma, sigma_one)
    if state.shape != velocity.shape:
        raise ValueError("State and velocity must have identical shapes.")
    state = _math_tensor(state.detach())
    velocity = _math_tensor(velocity)
    drift_scale = diffusion ** 2 / (2.0 * sigma)
    drift = velocity + drift_scale * (state + (1.0 - sigma) * velocity)
    mean = state - step_size * drift
    standard_deviation = diffusion * math.sqrt(step_size)
    return mean, standard_deviation


def normal_log_prob(action, mean, std):
    """Return mean coordinate Gaussian log density for each batch item.

    This is dimension-normalized, not the joint log probability (a sum).
    Sampled actions are constants when computing score-function gradients.
    """
    std = float(std)
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("The Gaussian standard deviation must be finite and positive.")
    if action.shape != mean.shape:
        raise ValueError("Action and mean must have identical shapes.")
    mean = _math_tensor(mean)
    action = _math_tensor(action.detach())
    standardized_error = (action - mean) / std
    coordinate_log_density = -0.5 * standardized_error.square()
    coordinate_log_density = coordinate_log_density - math.log(std)
    coordinate_log_density = coordinate_log_density - 0.5 * math.log(2.0 * math.pi)
    return _example_mean(coordinate_log_density)


def gaussian_kl(mean, reference, std):
    """Mean coordinate KL between Gaussians sharing the same fixed variance."""
    std = float(std)
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("The Gaussian standard deviation must be finite and positive.")
    if mean.shape != reference.shape:
        raise ValueError("Policy and reference means must have identical shapes.")
    difference = _math_tensor(mean) - _math_tensor(reference.detach())
    return _example_mean(difference.square()) / (2.0 * std ** 2)


def temporal_weight(sigma, next_sigma, sigma_one):
    """Flash inverse-lambda weight; normalize across the full replay batch outside.

    Computing from actual sigmas avoids the upstream trainer's hard-coded
    twenty-step lookup table. This is positive and independent of model weights.
    """
    sigma, step_size, diffusion = _step_values(sigma, next_sigma, sigma_one)
    root_step = math.sqrt(step_size)
    coefficient = root_step / diffusion
    coefficient += diffusion * root_step * (1.0 - sigma) / (2.0 * sigma)
    return 1.0 / coefficient


def grpo_loss(log_prob, old_log_prob, advantage, kl, clip_range=0.001,
              beta=0.004, weight=1.0):
    """Minimize clipped GRPO loss plus reference KL, with optional Flash weight.

    Keep old log densities fixed through both replay passes. At the initial
    unchanged policy the ratio is one, so clipping does not affect that gradient.
    Flash weights apply to the policy term; the KL term is not time weighted.
    """
    log_prob = _math_tensor(log_prob)
    old_log_prob = torch.as_tensor(old_log_prob, device=log_prob.device,
                                   dtype=log_prob.dtype).detach()
    advantage = torch.as_tensor(advantage, device=log_prob.device,
                                dtype=log_prob.dtype).detach()
    weight = torch.as_tensor(weight, device=log_prob.device,
                             dtype=log_prob.dtype).detach()
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped_loss = -advantage * ratio
    clipped_loss = -advantage * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    policy_loss = weight * torch.maximum(unclipped_loss, clipped_loss)
    return (policy_loss + beta * kl).mean()


def awm_loss(velocity, target, reference, advantage):
    """AWM uniform-weight, one-pass objective with reference MSE coefficient 0.4.

    ell is negative mean squared velocity error, not a trajectory log density.
    exp(ell - stop_gradient(ell)) is numerically one but has a useful gradient:
    positive advantages pull toward the target; negative ones push away.
    This function intentionally does not implement multi-pass old-ell replay.
    """
    if velocity.shape != target.shape or velocity.shape != reference.shape:
        raise ValueError("Velocity, target and reference must have identical shapes.")
    # The pinned AWM trainer calculates its matching surrogate in float64.
    velocity = velocity.double()
    target = target.detach().double()
    reference = reference.detach().double()
    advantage = torch.as_tensor(advantage, device=velocity.device,
                                dtype=velocity.dtype).detach()
    matching_error = _example_mean((velocity - target).square())
    ell = -matching_error
    ratio = torch.exp(ell - ell.detach())
    policy_loss = -advantage * ratio
    reference_loss = _example_mean((velocity - reference).square())
    return (policy_loss + 0.4 * reference_loss).mean()


def nft_loss(velocity, old_velocity, reference, noisy, clean, sigma, advantage):
    """NFT beta=1, advantage bound=5, reference coefficient=1e-4.

    The positive branch predicts velocity; the implicit negative branch predicts
    2*old_velocity-velocity. Each decoded clean-latent error uses its own detached
    mean-absolute-error denominator (at least 1e-5), as in the official trainer.
    ``old_velocity`` is the collection policy, distinct from the frozen reference.
    """
    for value in (old_velocity, reference, noisy, clean):
        if value.shape != velocity.shape:
            raise ValueError("All NFT latent and velocity tensors must have identical shapes.")
    velocity = _math_tensor(velocity)
    old_velocity = _math_tensor(old_velocity.detach())
    reference = _math_tensor(reference.detach())
    noisy = _math_tensor(noisy.detach())
    clean = _math_tensor(clean.detach())
    sigma = torch.as_tensor(sigma, device=velocity.device, dtype=velocity.dtype).detach()
    if sigma.numel() not in (1, velocity.shape[0]):
        raise ValueError("Sigma must be scalar or contain one value per batch item.")
    if not torch.isfinite(sigma).all() or not ((sigma > 0.0) & (sigma <= 1.0)).all():
        raise ValueError("NFT requires finite noise levels in (0, 1].")
    sigma = sigma.reshape(-1, *([1] * (velocity.ndim - 1)))
    advantage = torch.as_tensor(advantage, device=velocity.device,
                                dtype=velocity.dtype).detach().clamp(-5.0, 5.0)
    positive_probability = 0.5 + advantage / 10.0

    # Predict the clean endpoint from the positive and implicit negative flows.
    positive_error = noisy - sigma * velocity - clean
    negative_velocity = 2.0 * old_velocity - velocity
    negative_error = noisy - sigma * negative_velocity - clean
    positive_scale = _example_mean(positive_error.detach().double().abs()).clamp_min(1e-5)
    negative_scale = _example_mean(negative_error.detach().double().abs()).clamp_min(1e-5)
    positive_loss = _example_mean(positive_error.square()) / positive_scale
    negative_loss = _example_mean(negative_error.square()) / negative_scale
    policy_loss = positive_probability * positive_loss
    policy_loss += (1.0 - positive_probability) * negative_loss
    reference_loss = _example_mean((velocity - reference).square())
    return (5.0 * policy_loss + 1e-4 * reference_loss).mean()
