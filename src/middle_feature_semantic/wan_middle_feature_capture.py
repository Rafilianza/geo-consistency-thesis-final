#!/usr/bin/env python3
"""Passively capture one post-block feature from a frozen Wan transformer.

The capture uses a standard PyTorch forward hook. The hook observes the output
of one transformer block, stores a detached view, and returns ``None``. Returning
``None`` is important: it means the hook cannot replace or edit Wan's output.

This module deliberately contains no semantic model and no training code.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn


DEFAULT_CAPTURE_BLOCK_INDEX = 15


def _transformer_blocks(transformer: nn.Module) -> Sequence[nn.Module]:
    """Return Wan's block sequence, failing clearly if the contract changed."""
    blocks = getattr(transformer, "blocks", None)
    if not isinstance(blocks, (nn.ModuleList, list, tuple)):
        raise TypeError("transformer.blocks must be a sequence of PyTorch modules")
    if not blocks:
        raise ValueError("transformer.blocks cannot be empty")
    if not all(isinstance(block, nn.Module) for block in blocks):
        raise TypeError("every item in transformer.blocks must be a PyTorch module")
    return blocks


class WanMiddleFeatureCapture:
    """Context manager for one passive post-block Wan feature hook.

    A capture object can be reused for many forwards. Starting a new capture
    clears the previous feature and resets ``call_count``. Only the latest
    feature is retained, so a long denoising rollout cannot accidentally keep
    every large Wan activation in memory.
    """

    def __init__(
        self,
        transformer: nn.Module,
        block_index: int = DEFAULT_CAPTURE_BLOCK_INDEX,
    ) -> None:
        blocks = _transformer_blocks(transformer)
        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise TypeError("block index must be an integer")
        if block_index < 0 or block_index >= len(blocks):
            raise IndexError(
                f"block index {block_index} is outside [0, {len(blocks) - 1}]"
            )

        self.transformer = transformer
        self.block_index = block_index
        self.target_block = blocks[block_index]
        self._feature: Optional[torch.Tensor] = None
        self._call_count = 0
        self._hook_handle = None

    @property
    def active(self) -> bool:
        """Whether the forward hook is currently installed."""
        return self._hook_handle is not None

    @property
    def call_count(self) -> int:
        """Number of target-block calls observed since ``start``."""
        return self._call_count

    @property
    def feature(self) -> torch.Tensor:
        """Return the latest detached feature captured from the target block."""
        if self._feature is None:
            raise RuntimeError("the target block did not run; no feature was captured")
        return self._feature

    def clear(self) -> None:
        """Release the retained feature and reset the observed call count."""
        self._feature = None
        self._call_count = 0

    def _capture_output(
        self,
        _module: nn.Module,
        _inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("the captured Wan block output must be a tensor")

        self._call_count += 1
        # Clone the one selected feature. This prevents a later in-place
        # operation inside Wan from changing the saved observation.
        self._feature = output.detach().clone()

        # A forward hook may replace a module's output by returning a tensor.
        # This hook intentionally returns None so Wan continues unchanged.
        return None

    def start(self) -> "WanMiddleFeatureCapture":
        """Install the hook after clearing any earlier captured feature."""
        if self.active:
            raise RuntimeError("the feature-capture hook is already active")
        self.clear()
        self._hook_handle = self.target_block.register_forward_hook(
            self._capture_output
        )
        return self

    def stop(self) -> None:
        """Remove only this object's hook; other hooks remain installed."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self) -> "WanMiddleFeatureCapture":
        return self.start()

    def __exit__(self, _exception_type, _exception, _traceback) -> None:
        self.stop()


def _validated_patch_grid(
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Return batch and patch-grid dimensions for a five-dimensional latent."""
    if len(latent_shape) != 5:
        raise ValueError("latent shape must be [batch, channels, time, height, width]")
    if len(patch_size) != 3:
        raise ValueError("patch size must contain time, height, and width")

    batch_size = int(latent_shape[0])
    latent_time, latent_height, latent_width = (
        int(value) for value in latent_shape[2:]
    )
    patch_time, patch_height, patch_width = (int(value) for value in patch_size)

    if batch_size <= 0:
        raise ValueError("latent batch size must be positive")
    if min(latent_time, latent_height, latent_width) <= 0:
        raise ValueError("latent time, height, and width must be positive")
    if min(patch_time, patch_height, patch_width) <= 0:
        raise ValueError("patch-size values must be positive")

    latent_dimensions = (latent_time, latent_height, latent_width)
    patch_dimensions = (patch_time, patch_height, patch_width)
    if any(
        latent_dimension % patch_dimension != 0
        for latent_dimension, patch_dimension in zip(
            latent_dimensions,
            patch_dimensions,
        )
    ):
        raise ValueError("latent dimensions must be divisible by the patch size")

    return (
        batch_size,
        latent_time // patch_time,
        latent_height // patch_height,
        latent_width // patch_width,
    )


def feature_tokens_to_video_grid(
    feature_tokens: torch.Tensor,
    latent_shape: Sequence[int],
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Reshape ``[B, tokens, channels]`` into ``[B, channels, T, H, W]``.

    Wan flattens the patch grid in time-height-width order, with width changing
    fastest. This function performs the exact inverse reshape for a feature
    captured after a transformer block.
    """
    if feature_tokens.ndim != 3:
        raise ValueError("feature tokens must have shape [batch, tokens, channels]")

    batch_size, grid_time, grid_height, grid_width = _validated_patch_grid(
        latent_shape,
        patch_size,
    )
    expected_tokens = grid_time * grid_height * grid_width
    if feature_tokens.shape[0] != batch_size:
        raise ValueError("feature-token and latent batch sizes differ")
    if feature_tokens.shape[1] != expected_tokens:
        raise ValueError(
            f"feature token count is {feature_tokens.shape[1]}; "
            f"expected {expected_tokens}"
        )

    hidden_channels = feature_tokens.shape[2]
    video_grid = feature_tokens.reshape(
        batch_size,
        grid_time,
        grid_height,
        grid_width,
        hidden_channels,
    )
    return video_grid.permute(0, 4, 1, 2, 3).contiguous()


def video_grid_to_feature_tokens(feature_grid: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, channels, T, H, W]`` back to Wan token order."""
    if feature_grid.ndim != 5:
        raise ValueError("feature grid must have shape [batch, channels, T, H, W]")

    tokens_last = feature_grid.permute(0, 2, 3, 4, 1).contiguous()
    return tokens_last.reshape(
        feature_grid.shape[0],
        -1,
        feature_grid.shape[1],
    )
