"""Dispatch to the unchanged historical RLOO and matched GRPO implementations."""

from rgb_geometry_campaign import policy as rloo
from rgb_geometry_rl_comparison import policy as grpo


def initialize(runtime, args, config, spec):
    if spec["method"] == "rloo":
        rloo.initialize(runtime, spec["arm"], seed=config["settings"]["seed"])
        return rloo.make_optimizer(runtime)
    grpo.initialize(runtime, args, config, spec)
    if spec["arm"] == "B":
        return None
    return grpo.make_optimizer(runtime, config)


def digest(runtime):
    return grpo.policy_digest(runtime)


def collect(runtime, case, method, training):
    if method == "rloo":
        seeds = {name: case[name + "_seed"] for name in ("rgb", "semantic", "action")}
        return rloo.collect_trajectory(runtime, case["prompt"], seeds,
                                      parameter_version=runtime["parameter_version"],
                                      save_actions=training)
    return grpo.collect_trajectory(runtime, case, method or "baseline", steps=50, training=training)


def update(runtime, optimizer, groups, rewards, method, round_index, config):
    if method != "rloo":
        return grpo.update(runtime, optimizer, groups, rewards, method, round_index, config)
    runtime["pipeline"].transformer.to(runtime["device"]).eval()
    if "head" in runtime:
        runtime["head"].to(runtime["device"])
    result = rloo.update(runtime, optimizer, groups, rewards)
    retained = bool(result["retained_update"])
    runtime["parameter_version"] += int(retained)
    # Keep the complete historical receipt and add common reporting fields.
    result["effective_update"] = bool(result["effective_policy_change"])
    result["effective_policy_updates"] = int(result["effective_policy_change"])
    result["optimizer_steps"] = int(retained)
    result["warnings"] = [{"kind": "policy_change", "detail": item, "report_only": True}
                          for item in result["threshold_violations"]]
    return result
