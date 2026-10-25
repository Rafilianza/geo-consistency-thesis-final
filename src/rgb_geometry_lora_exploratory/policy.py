"""Small LoRA steps with scientific thresholds recorded instead of enforced.

The historical sampler, RLOO advantages and summed Gaussian likelihood are
unchanged. A finite Adam update is retained even when its measured KL is high.
Invalid arithmetic or a collection/replay mismatch remains a technical error.
"""

import copy
import math

import torch

from rgb_geometry_rl_v2 import policy as base_policy


LEARNING_RATE = 1e-8
KL_MEAN_LIMIT = 10.0
KL_MAX_LIMIT = 20.0


def make_optimizer(runtime):
    """Use a fixed step size 100 times smaller than the original LoRA attempt."""
    return torch.optim.Adam(base_policy.parameters(runtime), lr=LEARNING_RATE,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)


def update_batch(trainable_parameters, optimizer, trajectories, advantages,
                 mean_function, *, denominator):
    """Keep one finite proposal; report KL, ties and ineffective steps.

The existing v2 update() still validates the sixteen fresh trajectories before
calling this function. A disposable diagnostic may use one trajectory.
"""
    trainable_parameters = list(trainable_parameters)
    advantages = torch.as_tensor(advantages, dtype=torch.float64).detach().flatten()
    if (not trajectories or len(trajectories) != len(advantages)
            or denominator != len(trajectories) or not trainable_parameters):
        raise ValueError("The denominator must include every trajectory, including ties.")
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("Nonfinite advantages.")

    report = {
        "status": "no_learning_signal", "update_applied": False,
        "retained_update": False, "accepted": False, "numerical_accepted": False,
        "parameters_changed": False, "effective_policy_change": False,
        "policy_changed": False, "continue_training": True,
        "optimizer_steps": 0, "proposals": 0, "backoff": False,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "threshold_policy": "report_only", "threshold_violations": [],
        "historical_kl_within_limits": None,
        "historical_kl_limits": {"mean_kl": KL_MEAN_LIMIT, "max_kl": KL_MAX_LIMIT},
        "zero_advantage_trajectories": int((advantages == 0).sum()),
        "scientific_geometry_improvement_established": False,
    }
    # Ties provide no preference. In particular, do not move using old momentum.
    if not bool(torch.any(advantages != 0)):
        return report

    transitions = [transition for trajectory in trajectories
                   for transition in trajectory["transitions"]]
    if not transitions:
        raise ValueError("A policy update needs saved actions.")
    saved_parameters = [parameter.detach().cpu().clone()
                        for parameter in trainable_parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = base_policy._rng_snapshot()

    def rollback():
        with torch.no_grad():
            for parameter, value in zip(trainable_parameters, saved_parameters):
                parameter.copy_(value.to(parameter.device))
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        base_policy._restore_rng(saved_rng)

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    try:
        # Backward one saved action at a time to retain the memory-saving path.
        with torch.enable_grad():
            for trajectory, advantage in zip(trajectories, advantages):
                for transition in trajectory["transitions"]:
                    mean = mean_function(transition)
                    old_mean = transition["old_mean"].detach().to(mean.device)
                    if mean.dtype != torch.float32 or not bool(torch.isfinite(mean).all()):
                        raise FloatingPointError("The loss mean must be finite FP32.")
                    if not torch.equal(mean.detach(), old_mean):
                        raise base_policy.ReplayMismatchError(
                            "Gradient-enabled collection/replay mean mismatch; no update.")
                    if float(advantage) != 0:
                        action = transition["action"].detach().to(mean.device, dtype=torch.float32)
                        loss = -(float(advantage) / denominator) * base_policy.gaussian_log_probability(
                            action, mean, transition["std"])
                        loss.backward()
                        total_loss += float(loss.detach())
                        del action, loss
                    del mean, old_mean

        gradients = [parameter.grad for parameter in trainable_parameters
                     if parameter.grad is not None]
        if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("Missing or nonfinite trainable gradients.")
        gradient_norm = math.sqrt(sum(float(gradient.double().square().sum())
                                      for gradient in gradients))
        if not math.isfinite(gradient_norm) or not math.isfinite(total_loss):
            raise FloatingPointError("Nonfinite policy gradient or accumulated loss.")
        before = base_policy.replay_statistics(transitions, mean_function)
        if before["behavior"]["max_kl"] != 0:
            raise base_policy.ReplayMismatchError("Inference and gradient-enabled replay disagree; no update.")
        report.update(loss=total_loss, gradient_norm=gradient_norm, unchanged_replay=before)
        if gradient_norm == 0:
            report["status"] = "zero_gradient"
            return report

        # Exactly one Adam proposal. No KL backoff, rejection or quality rollback.
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable_parameters):
            raise FloatingPointError("Nonfinite proposed policy parameters.")
        for optimizer_values in optimizer.state.values():
            for value in optimizer_values.values():
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite proposed Adam state.")

        after = base_policy.replay_statistics(transitions, mean_function)
        violations = []
        for reference_name in ("behavior", "reference"):
            for metric, limit in (("mean_kl", KL_MEAN_LIMIT), ("max_kl", KL_MAX_LIMIT)):
                value = after[reference_name][metric]
                if value > limit:
                    violations.append({"metric": f"{reference_name}.{metric}",
                                       "value": value, "limit": limit})
        effective = after["behavior"]["normalized_rms"] > 0
        parameters_changed = any(not torch.equal(parameter.detach().cpu(), previous)
                                 for parameter, previous in zip(trainable_parameters, saved_parameters))
        report.update(
            status="update_retained_with_warnings" if violations else "update_retained",
            update_applied=True, retained_update=True, optimizer_steps=1, proposals=1,
            parameters_changed=parameters_changed, effective_policy_change=effective,
            policy_changed=effective, numerical_accepted=not violations and effective,
            historical_kl_within_limits=not violations, proposed_replay=after,
            threshold_violations=violations,
        )
        if not effective:
            report["status"] = "update_retained_no_effective_policy_change"
        return report
    except BaseException:
        # A technical failure cannot be presented as successful RL learning.
        rollback()
        optimizer.zero_grad(set_to_none=True)
        raise
