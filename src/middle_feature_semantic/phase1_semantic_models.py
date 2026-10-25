"""Small semantic models for the Phase 1 controlled experiment.

The Wan transformer is not part of these models.  Wan features must be captured
with the existing frozen feature-capture code and passed in as conditioning.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


SEMANTIC_CHANNELS = 8
WAN_FEATURE_CHANNELS = 1536
RAW_VIDEO_LATENT_CHANNELS = 16
HIDDEN_CHANNELS = 96
RESIDUAL_BLOCK_COUNT = 4


def resize_frames_bilinear(
    video: torch.Tensor,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Resize each video frame independently with bilinear interpolation.

    PyTorch's bilinear interpolation expects four dimensions.  We temporarily
    place time beside the batch dimension, resize, and then restore the video
    layout.  No value from one time step can enter another time step here.
    """

    if video.ndim != 5:
        raise ValueError(
            "video must have shape [batch, channels, time, height, width]"
        )

    output_height, output_width = output_size
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output height and width must be positive")

    batch_size, channels, time_steps, height, width = video.shape

    frames = video.permute(0, 2, 1, 3, 4)
    frames = frames.reshape(batch_size * time_steps, channels, height, width)

    resized_frames = F.interpolate(
        frames,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )

    resized_video = resized_frames.reshape(
        batch_size,
        time_steps,
        channels,
        output_height,
        output_width,
    )
    return resized_video.permute(0, 2, 1, 3, 4).contiguous()


def average_pool_frames(video: torch.Tensor) -> torch.Tensor:
    """Average-pool every frame to half height and width without mixing time."""

    if video.ndim != 5:
        raise ValueError(
            "video must have shape [batch, channels, time, height, width]"
        )

    batch_size, channels, time_steps, height, width = video.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("video height and width must both be even")

    frames = video.permute(0, 2, 1, 3, 4)
    frames = frames.reshape(batch_size * time_steps, channels, height, width)
    pooled_frames = F.avg_pool2d(frames, kernel_size=2, stride=2)

    pooled_height = height // 2
    pooled_width = width // 2
    pooled_video = pooled_frames.reshape(
        batch_size,
        time_steps,
        channels,
        pooled_height,
        pooled_width,
    )
    return pooled_video.permute(0, 2, 1, 3, 4).contiguous()


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_size: int,
) -> torch.Tensor:
    """Convert one non-negative timestep per sample into sine/cosine values."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must have shape [batch]")
    if not torch.isfinite(timesteps).all().item():
        raise ValueError("timesteps must contain only finite values")
    if (timesteps < 0).any().item():
        raise ValueError("timesteps must be non-negative")
    if embedding_size < 2 or embedding_size % 2 != 0:
        raise ValueError("embedding_size must be an even number of at least 2")

    half_size = embedding_size // 2
    frequency_positions = torch.arange(
        half_size,
        device=timesteps.device,
        dtype=torch.float32,
    )
    frequencies = torch.exp(
        -math.log(10_000.0) * frequency_positions / max(half_size - 1, 1)
    )

    angles = timesteps.to(torch.float32).unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class SemanticResidualBlock(nn.Module):
    """Two 3D convolutions with a readable residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first_convolution = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.second_convolution = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.first_convolution(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.second_convolution(hidden_states)
        return residual + hidden_states


class _SemanticPredictionBody(nn.Module):
    """Shared prediction body used by every experimental arm."""

    def __init__(self) -> None:
        super().__init__()

        self.noisy_semantic_stem = nn.Conv3d(
            SEMANTIC_CHANNELS,
            HIDDEN_CHANNELS,
            kernel_size=1,
        )
        self.conditioning_fusion = nn.Conv3d(
            HIDDEN_CHANNELS * 2,
            HIDDEN_CHANNELS,
            kernel_size=1,
        )

        self.timestep_mlp = nn.Sequential(
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
            nn.SiLU(),
            nn.Linear(HIDDEN_CHANNELS, HIDDEN_CHANNELS),
        )

        self.residual_blocks = nn.ModuleList(
            [
                SemanticResidualBlock(HIDDEN_CHANNELS)
                for _ in range(RESIDUAL_BLOCK_COUNT)
            ]
        )

        self.output_head = nn.Conv3d(
            HIDDEN_CHANNELS,
            SEMANTIC_CHANNELS,
            kernel_size=1,
        )

        # A zero head makes the initial iterative prediction exactly zero.
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def _validate_common_inputs(
        self,
        noisy_semantic: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> None:
        if noisy_semantic.ndim != 5:
            raise ValueError(
                "noisy_semantic must have shape "
                "[batch, 8, time, height, width]"
            )
        if noisy_semantic.shape[1] != SEMANTIC_CHANNELS:
            raise ValueError(
                f"noisy_semantic must have {SEMANTIC_CHANNELS} channels"
            )
        if not noisy_semantic.is_floating_point():
            raise ValueError("noisy_semantic must be a floating-point tensor")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [batch]")
        if timesteps.shape[0] != noisy_semantic.shape[0]:
            raise ValueError("timesteps must contain one value per batch sample")

        # This also checks finite and non-negative values.
        sinusoidal_timestep_embedding(timesteps, HIDDEN_CHANNELS)

    def _predict(
        self,
        noisy_semantic: torch.Tensor,
        conditioning: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_common_inputs(noisy_semantic, timesteps)

        if conditioning.shape[:1] != noisy_semantic.shape[:1]:
            raise ValueError("conditioning and noisy_semantic batch sizes must match")
        if conditioning.shape[2:] != noisy_semantic.shape[2:]:
            raise ValueError(
                "conditioning and noisy_semantic time and spatial sizes must match"
            )

        model_dtype = self.noisy_semantic_stem.weight.dtype
        noisy_semantic = noisy_semantic.to(dtype=model_dtype)
        conditioning = conditioning.to(dtype=model_dtype)

        noisy_hidden_states = self.noisy_semantic_stem(noisy_semantic)
        hidden_states = torch.cat([noisy_hidden_states, conditioning], dim=1)
        hidden_states = self.conditioning_fusion(hidden_states)

        model_timesteps = timesteps.to(device=hidden_states.device)
        timestep_embedding = sinusoidal_timestep_embedding(
            model_timesteps,
            HIDDEN_CHANNELS,
        )
        timestep_embedding = timestep_embedding.to(dtype=model_dtype)
        timestep_embedding = self.timestep_mlp(timestep_embedding)
        hidden_states = hidden_states + timestep_embedding[:, :, None, None, None]

        for residual_block in self.residual_blocks:
            hidden_states = residual_block(hidden_states)

        return self.output_head(hidden_states)

    def use_head_only_bootstrap(self) -> None:
        """Freeze everything except the zero-initialized output head."""

        set_head_only_trainable(self)

    def unfreeze_all(self) -> None:
        """Make the complete semantic branch trainable."""

        set_all_trainable(self)

    def parameter_counts(self) -> Dict[str, int]:
        """Return total, trainable, and frozen scalar parameter counts."""

        return parameter_counts(self)


class FeatureSemanticFlowModel(_SemanticPredictionBody):
    """Predict semantic flow from noisy semantics and frozen Wan features."""

    prediction_type = "flow"

    def __init__(self) -> None:
        super().__init__()
        self.feature_normalization = nn.LayerNorm(WAN_FEATURE_CHANNELS)
        self.feature_projection = nn.Linear(
            WAN_FEATURE_CHANNELS,
            HIDDEN_CHANNELS,
        )

    def encode_features(
        self,
        wan_features: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Project every Wan token, then resize each frame independently."""

        if wan_features.ndim != 5:
            raise ValueError(
                "wan_features must have shape "
                "[batch, 1536, time, half_height, half_width]"
            )
        if wan_features.shape[1] != WAN_FEATURE_CHANNELS:
            raise ValueError(
                f"wan_features must have {WAN_FEATURE_CHANNELS} channels"
            )
        if not wan_features.is_floating_point():
            raise ValueError("wan_features must be a floating-point tensor")

        output_height, output_width = output_size
        feature_height = wan_features.shape[3]
        feature_width = wan_features.shape[4]
        if output_height != feature_height * 2:
            raise ValueError("Wan feature height must equal half the output height")
        if output_width != feature_width * 2:
            raise ValueError("Wan feature width must equal half the output width")

        # Detaching here is a second safety boundary around the frozen Wan model.
        wan_features = wan_features.detach()
        wan_features = wan_features.to(dtype=self.feature_projection.weight.dtype)

        feature_tokens = wan_features.permute(0, 2, 3, 4, 1)
        feature_tokens = self.feature_normalization(feature_tokens)
        feature_tokens = self.feature_projection(feature_tokens)
        feature_grid = feature_tokens.permute(0, 4, 1, 2, 3).contiguous()

        return resize_frames_bilinear(feature_grid, output_size)

    def forward(
        self,
        noisy_semantic: torch.Tensor,
        wan_features: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_common_inputs(noisy_semantic, timesteps)

        output_size = (noisy_semantic.shape[3], noisy_semantic.shape[4])
        conditioning = self.encode_features(wan_features, output_size)
        return self._predict(noisy_semantic, conditioning, timesteps)


class FeatureSemanticDirectModel(FeatureSemanticFlowModel):
    """Predict a clean semantic latent directly from frozen Wan features.

    This control deliberately has exactly the same trainable architecture and
    parameter count as the iterative feature model.  Only the training target
    and interpretation of its output are different.
    """

    prediction_type = "clean_latent"

    def forward(
        self,
        semantic_state: torch.Tensor,
        wan_features: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict directly while deliberately ignoring the semantic state.

        Keeping the state argument gives every experimental arm the same call
        shape.  Replacing it with zeros makes this a true direct-decoder control.
        """

        zero_semantic_state = torch.zeros_like(semantic_state)
        return super().forward(zero_semantic_state, wan_features, timesteps)


class RawRGBSemanticFlowModel(_SemanticPredictionBody):
    """Iterative control conditioned on the 16-channel Wan video latent.

    Despite the short experiment name "raw RGB", this model receives Wan's
    16-channel video latent rather than three-channel display pixels.
    """

    prediction_type = "flow"

    def __init__(self) -> None:
        super().__init__()
        self.raw_conditioning_projection = nn.Conv3d(
            RAW_VIDEO_LATENT_CHANNELS,
            HIDDEN_CHANNELS,
            kernel_size=1,
        )

    def encode_raw_video(
        self,
        raw_video_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Pool to the feature grid, project, then resize back frame by frame."""

        if raw_video_latent.ndim != 5:
            raise ValueError(
                "raw_video_latent must have shape "
                "[batch, 16, time, height, width]"
            )
        if raw_video_latent.shape[1] != RAW_VIDEO_LATENT_CHANNELS:
            raise ValueError(
                "raw_video_latent must have "
                f"{RAW_VIDEO_LATENT_CHANNELS} channels"
            )
        if not raw_video_latent.is_floating_point():
            raise ValueError("raw_video_latent must be a floating-point tensor")

        output_size = (raw_video_latent.shape[3], raw_video_latent.shape[4])

        raw_video_latent = raw_video_latent.detach()
        raw_video_latent = raw_video_latent.to(
            dtype=self.raw_conditioning_projection.weight.dtype
        )
        pooled_video = average_pool_frames(raw_video_latent)
        conditioning_grid = self.raw_conditioning_projection(pooled_video)
        return resize_frames_bilinear(conditioning_grid, output_size)

    def forward(
        self,
        noisy_semantic: torch.Tensor,
        raw_video_latent: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_common_inputs(noisy_semantic, timesteps)

        conditioning = self.encode_raw_video(raw_video_latent)
        return self._predict(noisy_semantic, conditioning, timesteps)


def set_head_only_trainable(model: _SemanticPredictionBody) -> None:
    """Prepare the model for the short output-head bootstrap stage."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.output_head.parameters():
        parameter.requires_grad_(True)


def set_all_trainable(model: _SemanticPredictionBody) -> None:
    """Unfreeze the complete semantic model after head bootstrapping."""

    for parameter in model.parameters():
        parameter.requires_grad_(True)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count scalar parameters, optionally counting trainable values only."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (
            parameter for parameter in parameters if parameter.requires_grad
        )
    return sum(parameter.numel() for parameter in parameters)


def parameter_counts(model: nn.Module) -> Dict[str, int]:
    """Return parameter counts suitable for an experiment report."""

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
