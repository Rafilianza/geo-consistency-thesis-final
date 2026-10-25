"""Replay LoRA with one activation checkpoint per Wan transformer block.

The forward computation is unchanged. During backward, PyTorch reconstructs
one block's temporary tensors at a time, rather than reconstructing all of Wan
inside one outer checkpoint. This uses Diffusers 0.36's native block loop.
"""

from contextlib import contextmanager, nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from rgb_geometry_pilot import policy as historical_policy


@contextmanager
def _branch_context(transformer, branch, adapted):
    """Restore adapter flags and finish the Diffusers cache scope on errors."""
    cache = transformer.cache_context(branch) if hasattr(transformer, "cache_context") else nullcontext()
    with historical_policy.adapter_mode(transformer, adapted):
        cache.__enter__()
        try:
            yield
        finally:
            # Diffusers 0.36 cache_context clears its label after a plain yield,
            # without a finally. Normal closure also handles checkpoint's
            # internal early-stop exception, which is not a model failure.
            cache.__exit__(None, None, None)


@contextmanager
def _block_checkpoints(transformer, branch, adapted):
    """Temporarily select native block checkpoints, preserving caller state."""
    if not callable(getattr(transformer, "enable_gradient_checkpointing", None)):
        raise TypeError("This LoRA replay needs Wan's native block checkpoint API.")

    previous = []
    for module in transformer.modules():
        if hasattr(module, "gradient_checkpointing"):
            previous.append((module, module.gradient_checkpointing,
                             hasattr(module, "_gradient_checkpointing_func"),
                             getattr(module, "_gradient_checkpointing_func", None)))
    if not previous:
        raise TypeError("Wan did not expose a gradient_checkpointing flag.")

    def checkpoint_block(block, *inputs):
        def contexts():
            # The first forward already runs inside the complete branch scope.
            # Backward happens later, after that scope has been restored.
            return nullcontext(), _branch_context(transformer, branch, adapted)

        return checkpoint(block, *inputs, use_reentrant=False, context_fn=contexts)

    try:
        transformer.enable_gradient_checkpointing(gradient_checkpointing_func=checkpoint_block)
        yield
    finally:
        for module, enabled, had_function, function in previous:
            module.gradient_checkpointing = enabled
            if had_function:
                module._gradient_checkpointing_func = function
            elif hasattr(module, "_gradient_checkpointing_func"):
                delattr(module, "_gradient_checkpointing_func")


def forward_branch(runtime, state, timestep, prompt, branch, *, adapted, checkpointed=False):
    """Match the historical branch API, using block checkpoints for replay.

Detached video states are intentional: gradients train LoRA parameters, not
earlier denoising steps. Nonreentrant checkpointing supports those detached
inputs, and both conditional and unconditional branches remain differentiable.
"""
    if branch not in ("cond", "uncond"):
        raise ValueError("Expected the conditional or unconditional Wan branch.")
    transformer = runtime["pipeline"].transformer
    embedding = runtime["embeddings"][prompt][0 if branch == "cond" else 1]
    projection = transformer.get_submodule(historical_policy.LORA_TARGETS[0])
    if isinstance(projection, historical_policy.SelfAttentionLoRA):
        model_dtype = projection.base.weight.dtype
    else:
        model_dtype = projection.weight.dtype
    model_input = state.detach().to(dtype=model_dtype)
    time_input = torch.as_tensor(timestep, device=state.device).reshape(1).expand(state.shape[0])
    embedding = embedding.detach().to(device=state.device, dtype=model_dtype)

    checkpoint_scope = nullcontext()
    if checkpointed and torch.is_grad_enabled():
        checkpoint_scope = _block_checkpoints(transformer, branch, adapted)
    with checkpoint_scope, _branch_context(transformer, branch, adapted):
        return transformer(hidden_states=model_input, timestep=time_input,
                           encoder_hidden_states=embedding, return_dict=False)[0]
