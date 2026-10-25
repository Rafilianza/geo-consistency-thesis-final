"""Small, reference-guided RGB flow correction with a frozen Wan backbone.

This is a separate feasibility experiment. It does not make the failed semantic
prediction experiments pass or authorize predicted-semantic feedback.

The head learns a residual, not a replacement RGB flow:
    corrected_flow = frozen_wan_flow + correction
Training uses the ordinary RGB flow target ``noise - clean_rgb_latent``. The
same differentiable RMS bound is applied during training and inference.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


RGB_CHANNELS = 16
REFERENCE_CHANNELS = 8
DEFAULT_WIDTH = 64
DEFAULT_RESIDUAL_BLOCKS = 3
RESIDUAL_REGULARIZATION_WEIGHT = 0.01
MAXIMUM_RESIDUAL_RMS_RATIO = 0.25


def _check_volume(value: torch.Tensor, name: str, channels: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 5 or value.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape [batch, {channels}, time, height, width]"
        )
    if min(value.shape) < 1:
        raise ValueError(f"{name} cannot contain an empty dimension")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _check_matching_flow(first: torch.Tensor, second: torch.Tensor) -> None:
    _check_volume(first, "base_flow", RGB_CHANNELS)
    _check_volume(second, "correction_or_target", RGB_CHANNELS)
    if first.shape != second.shape:
        raise ValueError("RGB flow shapes must match")
    if first.device != second.device or first.dtype != second.dtype:
        raise ValueError("RGB flows must share a device and dtype")


def _check_number(value: float, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


class _ResidualBlock(nn.Module):
    """Two small 3D convolutions, with the input added back at the end."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.layers(hidden_states)


class ReferenceRGBCorrectionHead(nn.Module):
    """Predict an RGB-flow residual using a fixed reference semantic latent.

    All four inputs are treated as frozen context. The trainable convolutions
    work at half spatial resolution to limit memory; time is not downsampled.
    A zero-initialized final convolution makes the initial correction zero.
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        residual_blocks: int = DEFAULT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError("width must be an integer")
        if width < 8 or width % 8 != 0:
            raise ValueError("width must be a positive multiple of eight")
        if isinstance(residual_blocks, bool) or not isinstance(residual_blocks, int):
            raise TypeError("residual_blocks must be an integer")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")

        self.width = width
        self.residual_block_count = residual_blocks
        input_channels = RGB_CHANNELS * 2 + REFERENCE_CHANNELS + 1
        self.input_projection = nn.Conv3d(
            input_channels, width, kernel_size=3, padding=1
        )
        self.residual_blocks = nn.ModuleList()
        for _ in range(residual_blocks):
            self.residual_blocks.append(_ResidualBlock(width))
        self.output_projection = nn.Conv3d(width, RGB_CHANNELS, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        noisy_rgb: torch.Tensor,
        base_flow: torch.Tensor,
        reference_latent: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        _check_matching_flow(noisy_rgb, base_flow)
        _check_volume(reference_latent, "reference_latent", REFERENCE_CHANNELS)
        expected_reference_shape = (
            noisy_rgb.shape[0],
            REFERENCE_CHANNELS,
            *noisy_rgb.shape[2:],
        )
        if reference_latent.shape != expected_reference_shape:
            raise ValueError(
                "reference_latent must share the RGB batch and space-time grid"
            )
        if (
            reference_latent.device != noisy_rgb.device
            or reference_latent.dtype != noisy_rgb.dtype
        ):
            raise ValueError(
                "reference_latent and RGB inputs must share a device and dtype"
            )
        if min(noisy_rgb.shape[-2:]) < 2:
            raise ValueError("RGB height and width must be at least two")
        if self.input_projection.weight.device != noisy_rgb.device:
            raise ValueError("correction head and inputs must be on the same device")

        # Detaching context guarantees that this head cannot fine-tune Wan or
        # the semantic codec, even if a caller accidentally enables their graph.
        head_dtype = self.input_projection.weight.dtype
        rgb_context = noisy_rgb.detach().to(dtype=head_dtype)
        flow_context = base_flow.detach().to(dtype=head_dtype)
        reference_context = reference_latent.detach().to(dtype=head_dtype)
        sigma_tensor = torch.as_tensor(sigma, device=noisy_rgb.device, dtype=head_dtype)
        sigma_tensor = sigma_tensor.detach()
        batch_size = noisy_rgb.shape[0]
        if sigma_tensor.ndim == 0:
            sigma_tensor = sigma_tensor.expand(batch_size)
        elif sigma_tensor.shape == (batch_size, 1, 1, 1, 1):
            sigma_tensor = sigma_tensor.reshape(batch_size)
        elif sigma_tensor.shape != (batch_size,):
            raise ValueError("sigma must be scalar, [batch], or [batch, 1, 1, 1, 1]")
        if not bool(torch.isfinite(sigma_tensor).all()):
            raise ValueError("sigma must be finite")
        if bool(((sigma_tensor < 0) | (sigma_tensor > 1)).any()):
            raise ValueError("sigma must be between zero and one")

        # Average neighbouring spatial cells; keep every latent time position.
        rgb_context = F.avg_pool3d(rgb_context, kernel_size=(1, 2, 2))
        flow_context = F.avg_pool3d(flow_context, kernel_size=(1, 2, 2))
        reference_context = F.avg_pool3d(reference_context, kernel_size=(1, 2, 2))
        sigma_map = sigma_tensor.reshape(batch_size, 1, 1, 1, 1)
        sigma_map = sigma_map.expand(batch_size, 1, *rgb_context.shape[2:])
        conditioning = torch.cat(
            (rgb_context, flow_context, reference_context, sigma_map), dim=1
        )
        hidden_states = self.input_projection(conditioning)
        for residual_block in self.residual_blocks:
            hidden_states = residual_block(hidden_states)

        # The 1x1 output layer is the only learned operation on the full grid.
        hidden_states = F.interpolate(
            hidden_states,
            size=noisy_rgb.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        correction = self.output_projection(hidden_states)
        correction = correction.to(dtype=noisy_rgb.dtype)
        if not bool(torch.isfinite(correction).all()):
            raise ValueError("correction head produced non-finite values")
        return correction


def correction_loss(
    predicted_delta: torch.Tensor,
    base_flow: torch.Tensor,
    target_flow: torch.Tensor,
    regularization_weight: float = RESIDUAL_REGULARIZATION_WEIGHT,
) -> torch.Tensor:
    """Train the residual against the ordinary, *RGB* flow-matching target.

    ``target_flow`` must be ``rgb_noise - clean_rgb_latent`` for the actual
    forward-noised training state. No semantic-to-RGB channel addition occurs.
    Pass the RMS-bounded correction used at inference as ``predicted_delta``.
    Base flow and target are detached; gradients reach only predicted_delta.
    """
    _check_matching_flow(base_flow, predicted_delta)
    _check_matching_flow(base_flow, target_flow)
    weight = _check_number(regularization_weight, "regularization_weight")
    corrected_flow = base_flow.detach().float() + predicted_delta.float()
    flow_error = F.mse_loss(corrected_flow, target_flow.detach().float())
    residual_penalty = predicted_delta.float().square().mean()
    loss = flow_error + weight * residual_penalty
    if not bool(torch.isfinite(loss)):
        raise ValueError("RGB correction loss is non-finite")
    return loss


def compose_correction(
    base_flow: torch.Tensor,
    predicted_delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Add a scaled RGB residual after Wan prediction, before the scheduler.

    Returning the original tensor for alpha=0 avoids even a rounding change.
    This exact bypass is used for the baseline replay control.
    """
    _check_matching_flow(base_flow, predicted_delta)
    scale = _check_number(alpha, "alpha")
    if scale == 0.0:
        return base_flow
    corrected_flow = base_flow + scale * predicted_delta
    if not bool(torch.isfinite(corrected_flow).all()):
        raise ValueError("corrected RGB flow contains non-finite values")
    return corrected_flow


def bound_correction_rms(
    predicted_delta: torch.Tensor,
    base_flow: torch.Tensor,
    max_rms_ratio: float = MAXIMUM_RESIDUAL_RMS_RATIO,
) -> tuple[torch.Tensor, dict]:
    """Cap each clip's correction RMS at 25% of base-flow RMS.

    Use this same differentiable bound during training and inference. It is a
    correction magnitude bound, not an experimental success threshold.
    It limits correction magnitude but cannot guarantee good video quality.
    Diagnostics report how often the bound activates (fraction of this batch).
    """
    _check_matching_flow(base_flow, predicted_delta)
    ratio = _check_number(max_rms_ratio, "max_rms_ratio")
    elements_per_clip = predicted_delta[0].numel()
    rms_denominator = math.sqrt(elements_per_clip)
    # vector_norm defines a finite zero gradient at the initial all-zero
    # residual; sqrt(mean(square)) would have an undefined derivative there.
    base_rms = torch.linalg.vector_norm(base_flow.detach().float().flatten(1), dim=1)
    base_rms = base_rms / rms_denominator
    raw_rms = torch.linalg.vector_norm(predicted_delta.float().flatten(1), dim=1)
    raw_rms = raw_rms / rms_denominator
    if not bool(torch.isfinite(base_rms).all() and torch.isfinite(raw_rms).all()):
        raise ValueError("RGB correction RMS computation overflowed")
    maximum_rms = ratio * base_rms
    scales = maximum_rms / raw_rms.clamp_min(1e-12)
    scales = scales.clamp(max=1.0)
    bounded_delta = predicted_delta * scales.to(predicted_delta.dtype).reshape(
        -1, 1, 1, 1, 1
    )
    if not bool(torch.isfinite(bounded_delta).all()):
        raise ValueError("bounded RGB correction contains non-finite values")
    bounded_rms = torch.linalg.vector_norm(bounded_delta.float().flatten(1), dim=1)
    bounded_rms = bounded_rms / rms_denominator
    diagnostics = {
        "max_rms_ratio": ratio,
        "clip_fraction": float((raw_rms > maximum_rms).float().mean().detach().cpu()),
        "mean_base_rms": float(base_rms.mean().detach().cpu()),
        "mean_raw_rms": float(raw_rms.mean().detach().cpu()),
        "mean_bounded_rms": float(bounded_rms.mean().detach().cpu()),
    }
    return bounded_delta, diagnostics
