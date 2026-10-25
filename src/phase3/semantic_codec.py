#!/usr/bin/env python3
"""Small RGB-blind semantic codec for the native joint MVP.

The production interface is fixed by ``semantic_codec_contract_v1.json``:

* labels and validity: [B, 81, 480, 832]
* raw semantic latent: [B, 8, 21, 60, 104]
* decoded logits: [B, 11, 81, 480, 832]

The encoder sees only integer semantic labels and their validity mask. It never
receives RGB pixels, Wan latents, or text features.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
from torch import nn
import torch.nn.functional as F


NUM_CLASSES = 11
IGNORE_TOKEN_ID = 11
NUM_EMBEDDINGS = 12
IGNORE_VALUE = 255

SOURCE_FRAMES = 81
LATENT_TIMES = 21
SOURCE_HEIGHT = 480
SOURCE_WIDTH = 832
LATENT_CHANNELS = 8
LATENT_HEIGHT = 60
LATENT_WIDTH = 104

EXPECTED_TRAINABLE_PARAMETERS = 43_883
MINIMUM_LATENT_STANDARD_DEVIATION = 1e-6
DEFAULT_DECODE_CHUNK_FRAMES = 9
CONTRACT_VERSION = "semantic_codec_contract_v1"
CONTRACT_PATH = Path(__file__).with_name("semantic_codec_contract_v1.json")


def semantic_codec_contract_sha256() -> str:
    """Return the SHA-256 fingerprint of the frozen codec contract."""
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def source_frame_to_latent_time(frame_index: int) -> int:
    """Return the latent time assigned to one source frame."""
    if not 0 <= frame_index < SOURCE_FRAMES:
        raise ValueError(
            f"frame_index must be between 0 and {SOURCE_FRAMES - 1}"
        )
    if frame_index == 0:
        return 0
    return 1 + (frame_index - 1) // 4


def latent_time_to_source_frames(latent_time: int) -> Tuple[int, int]:
    """Return the inclusive source-frame range for one latent time."""
    if not 0 <= latent_time < LATENT_TIMES:
        raise ValueError(
            f"latent_time must be between 0 and {LATENT_TIMES - 1}"
        )
    if latent_time == 0:
        return 0, 0

    frame_start = 1 + (latent_time - 1) * 4
    return frame_start, frame_start + 3


def prepend_first_frame(features: torch.Tensor) -> torch.Tensor:
    """Prepend three copies of feature frame 0 for causal pack-by-four."""
    if features.ndim != 5:
        raise ValueError("features must have shape [B, C, T, H, W]")
    if features.shape[2] != SOURCE_FRAMES:
        raise ValueError(
            f"features must contain {SOURCE_FRAMES} source frames"
        )

    first_frame = features[:, :, :1]
    prefix = first_frame.expand(-1, -1, 3, -1, -1)
    return torch.cat((prefix, features), dim=2)


def prepare_embedding_tokens(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace every invalid label with the dedicated ignore token."""
    if labels.shape != valid_mask.shape:
        raise ValueError("labels and valid_mask must have the same shape")
    if labels.dtype != torch.int64:
        raise TypeError("labels must use torch.int64")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use torch.bool")

    ignore_tokens = torch.full_like(labels, IGNORE_TOKEN_ID)
    return torch.where(valid_mask, labels, ignore_tokens)


def framewise_resize(
    features: torch.Tensor,
    output_size: Tuple[int, int],
) -> torch.Tensor:
    """Resize H and W independently for every frame without mixing time."""
    if features.ndim != 5:
        raise ValueError("features must have shape [B, C, T, H, W]")

    batch_size, channels, frames, height, width = features.shape
    frame_batch = features.permute(0, 2, 1, 3, 4)
    frame_batch = frame_batch.reshape(batch_size * frames, channels, height, width)
    resized_frames = F.interpolate(
        frame_batch,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )

    output_height, output_width = output_size
    resized = resized_frames.reshape(
        batch_size,
        frames,
        channels,
        output_height,
        output_width,
    )
    return resized.permute(0, 2, 1, 3, 4).contiguous()


class FrameGroupNorm(nn.Module):
    """Apply GroupNorm to each video frame independently."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=epsilon,
            affine=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, C, T, H, W]")

        batch_size, channels, frames, height, width = features.shape
        frame_batch = features.permute(0, 2, 1, 3, 4)
        frame_batch = frame_batch.reshape(
            batch_size * frames,
            channels,
            height,
            width,
        )
        normalized_frames = self.group_norm(frame_batch)

        normalized = normalized_frames.reshape(
            batch_size,
            frames,
            channels,
            height,
            width,
        )
        return normalized.permute(0, 2, 1, 3, 4).contiguous()


class FrameConvBlock(nn.Module):
    """One Conv3d followed by framewise GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.normalization = FrameGroupNorm(
            num_groups=4,
            num_channels=out_channels,
        )
        self.activation = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.convolution(features)
        features = self.normalization(features)
        return self.activation(features)


class SemanticEncoder(nn.Module):
    """Encode integer semantic maps into an eight-channel raw latent."""

    def __init__(
        self,
        input_height: int = SOURCE_HEIGHT,
        input_width: int = SOURCE_WIDTH,
        allow_test_shape: bool = False,
    ) -> None:
        super().__init__()
        _validate_configured_size(
            input_height,
            input_width,
            allow_test_shape,
        )
        self.input_height = input_height
        self.input_width = input_width
        self.latent_height = input_height // 8
        self.latent_width = input_width // 8

        self.class_embedding = nn.Embedding(
            num_embeddings=NUM_EMBEDDINGS,
            embedding_dim=LATENT_CHANNELS,
        )
        self.enc_down_1 = FrameConvBlock(
            8,
            16,
            kernel_size=(4, 3, 3),
            stride=(4, 2, 2),
            padding=(0, 1, 1),
        )
        self.enc_down_2 = FrameConvBlock(
            16,
            24,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.enc_down_3 = FrameConvBlock(
            24,
            32,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )
        self.enc_refine = FrameConvBlock(
            32,
            32,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.latent_projection = nn.Conv3d(
            32,
            LATENT_CHANNELS,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(labels, valid_mask)

    def encode(
        self,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode labels and validity; no RGB-shaped input is accepted."""
        self._validate_inputs(labels, valid_mask)
        embedding_tokens = prepare_embedding_tokens(labels, valid_mask)

        embedded = self.class_embedding(embedding_tokens)
        features = embedded.permute(0, 4, 1, 2, 3).contiguous()
        features = prepend_first_frame(features)

        features = self.enc_down_1(features)
        features = self.enc_down_2(features)
        features = self.enc_down_3(features)
        features = self.enc_refine(features)
        return self.latent_projection(features)

    def _validate_inputs(
        self,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if labels.ndim == 5 and (
            labels.shape[1] == 3 or labels.shape[-1] == 3
        ):
            raise ValueError(
                "RGB-shaped input is forbidden; encode expects [B, T, H, W]"
            )
        if labels.ndim != 4:
            raise ValueError("labels must have shape [B, T, H, W]")
        if valid_mask.ndim != 4:
            raise ValueError("valid_mask must have shape [B, T, H, W]")
        if labels.shape != valid_mask.shape:
            raise ValueError("labels and valid_mask must have the same shape")
        if labels.dtype != torch.int64:
            raise TypeError("labels must use torch.int64")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must use torch.bool")

        expected_shape = (
            labels.shape[0],
            SOURCE_FRAMES,
            self.input_height,
            self.input_width,
        )
        if tuple(labels.shape) != expected_shape:
            raise ValueError(
                f"labels must have shape {expected_shape}, got {tuple(labels.shape)}"
            )

        # Meta tensors carry only shapes, so value checks are tested separately.
        if labels.is_meta:
            return

        valid_labels = labels.masked_select(valid_mask)
        if valid_labels.numel() == 0:
            return
        minimum_label = int(valid_labels.min().item())
        maximum_label = int(valid_labels.max().item())
        if minimum_label < 0 or maximum_label >= NUM_CLASSES:
            raise ValueError(
                "valid labels must contain class IDs from 0 through 10"
            )


class SemanticDecoder(nn.Module):
    """Decode an eight-channel semantic latent into 11-class logits."""

    def __init__(
        self,
        output_height: int = SOURCE_HEIGHT,
        output_width: int = SOURCE_WIDTH,
        allow_test_shape: bool = False,
    ) -> None:
        super().__init__()
        _validate_configured_size(
            output_height,
            output_width,
            allow_test_shape,
        )
        self.output_height = output_height
        self.output_width = output_width
        self.latent_height = output_height // 8
        self.latent_width = output_width // 8

        self.dec_refine = FrameConvBlock(
            8,
            32,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.phase_projection = nn.Conv3d(
            32,
            128,
            kernel_size=1,
            bias=True,
        )
        self.dec_up_1 = FrameConvBlock(
            32,
            24,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.dec_up_2 = FrameConvBlock(
            24,
            16,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.dec_up_3 = FrameConvBlock(
            16,
            12,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )
        self.logit_projection = nn.Conv3d(
            12,
            NUM_CLASSES,
            kernel_size=1,
            bias=True,
        )

    def forward(self, semantic_latent: torch.Tensor) -> torch.Tensor:
        return self.decode(semantic_latent)

    def decode(self, semantic_latent: torch.Tensor) -> torch.Tensor:
        """Decode all 81 frames using the canonical memory-bounded path."""
        logits_chunks = [
            logits_chunk
            for _, _, logits_chunk in self.decode_chunks(
                semantic_latent,
                chunk_frames=DEFAULT_DECODE_CHUNK_FRAMES,
            )
        ]
        return torch.cat(logits_chunks, dim=2)

    def decode_chunks(
        self,
        semantic_latent: torch.Tensor,
        chunk_frames: int = DEFAULT_DECODE_CHUNK_FRAMES,
    ) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """Yield decoded time chunks where ``frame_end`` is exclusive."""
        self._validate_input(semantic_latent)
        if not isinstance(chunk_frames, int) or chunk_frames <= 0:
            raise ValueError("chunk_frames must be a positive integer")

        expanded_features = self._expand_temporal_phases(semantic_latent)
        for frame_start in range(0, SOURCE_FRAMES, chunk_frames):
            frame_end = min(frame_start + chunk_frames, SOURCE_FRAMES)
            feature_chunk = expanded_features[:, :, frame_start:frame_end]
            logits_chunk = self._decode_expanded_features(feature_chunk)
            yield frame_start, frame_end, logits_chunk

    def _expand_temporal_phases(
        self,
        semantic_latent: torch.Tensor,
    ) -> torch.Tensor:
        features = self.dec_refine(semantic_latent)
        phase_features = self.phase_projection(features)

        batch_size, _, latent_times, height, width = phase_features.shape
        phase_features = phase_features.reshape(
            batch_size,
            32,
            4,
            latent_times,
            height,
            width,
        )
        phase_features = phase_features.permute(0, 1, 3, 2, 4, 5)
        expanded = phase_features.reshape(
            batch_size,
            32,
            latent_times * 4,
            height,
            width,
        )

        # Latent time 0 represents only source frame 0. Its other phases are
        # deliberately discarded before spatial decoding.
        return torch.cat((expanded[:, :, :1], expanded[:, :, 4:]), dim=2)

    def _decode_expanded_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        first_size = (self.output_height // 4, self.output_width // 4)
        second_size = (self.output_height // 2, self.output_width // 2)
        final_size = (self.output_height, self.output_width)

        features = framewise_resize(features, first_size)
        features = self.dec_up_1(features)
        features = framewise_resize(features, second_size)
        features = self.dec_up_2(features)
        features = framewise_resize(features, final_size)
        features = self.dec_up_3(features)
        return self.logit_projection(features)

    def _validate_input(self, semantic_latent: torch.Tensor) -> None:
        if semantic_latent.ndim != 5:
            raise ValueError(
                "semantic_latent must have shape [B, 8, 21, H, W]"
            )
        if not torch.is_floating_point(semantic_latent):
            raise TypeError("semantic_latent must use a floating-point dtype")

        expected_shape = (
            semantic_latent.shape[0],
            LATENT_CHANNELS,
            LATENT_TIMES,
            self.latent_height,
            self.latent_width,
        )
        if tuple(semantic_latent.shape) != expected_shape:
            raise ValueError(
                "semantic_latent must have shape "
                f"{expected_shape}, got {tuple(semantic_latent.shape)}"
            )


class SemanticCodec(nn.Module):
    """Combined encoder, decoder, and latent-normalization state."""

    def __init__(
        self,
        input_height: int = SOURCE_HEIGHT,
        input_width: int = SOURCE_WIDTH,
        allow_test_shape: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SemanticEncoder(
            input_height=input_height,
            input_width=input_width,
            allow_test_shape=allow_test_shape,
        )
        self.decoder = SemanticDecoder(
            output_height=input_height,
            output_width=input_width,
            allow_test_shape=allow_test_shape,
        )

        self.register_buffer(
            "latent_channel_mean",
            torch.zeros(LATENT_CHANNELS),
        )
        self.register_buffer(
            "latent_channel_std",
            torch.ones(LATENT_CHANNELS),
        )
        self.register_buffer(
            "normalization_fitted",
            torch.tensor(False, dtype=torch.bool),
        )

    def encode(
        self,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unnormalized semantic latent."""
        return self.encoder.encode(labels, valid_mask)

    def decode(self, semantic_latent: torch.Tensor) -> torch.Tensor:
        """Decode one raw semantic latent to class logits."""
        return self.decoder.decode(semantic_latent)

    def decode_chunks(
        self,
        semantic_latent: torch.Tensor,
        chunk_frames: int = DEFAULT_DECODE_CHUNK_FRAMES,
    ) -> Iterator[Tuple[int, int, torch.Tensor]]:
        """Yield memory-bounded decoded chunks in source-frame order."""
        return self.decoder.decode_chunks(semantic_latent, chunk_frames)

    def set_normalization_statistics(
        self,
        channel_mean: torch.Tensor,
        channel_std: torch.Tensor,
    ) -> None:
        """Store frozen training-split statistics for all eight channels."""
        expected_shape = (LATENT_CHANNELS,)
        if tuple(channel_mean.shape) != expected_shape:
            raise ValueError(f"channel_mean must have shape {expected_shape}")
        if tuple(channel_std.shape) != expected_shape:
            raise ValueError(f"channel_std must have shape {expected_shape}")
        if not torch.is_floating_point(channel_mean):
            raise TypeError("channel_mean must use a floating-point dtype")
        if not torch.is_floating_point(channel_std):
            raise TypeError("channel_std must use a floating-point dtype")
        if not bool(torch.isfinite(channel_mean).all()):
            raise ValueError("channel_mean must contain only finite values")
        if not bool(torch.isfinite(channel_std).all()):
            raise ValueError("channel_std must contain only finite values")
        if bool((channel_std < MINIMUM_LATENT_STANDARD_DEVIATION).any()):
            raise ValueError(
                "every channel_std value must be at least "
                f"{MINIMUM_LATENT_STANDARD_DEVIATION}"
            )

        self.latent_channel_mean.copy_(
            channel_mean.to(
                device=self.latent_channel_mean.device,
                dtype=self.latent_channel_mean.dtype,
            )
        )
        self.latent_channel_std.copy_(
            channel_std.to(
                device=self.latent_channel_std.device,
                dtype=self.latent_channel_std.dtype,
            )
        )
        self.normalization_fitted.fill_(True)

    def normalize(self, raw_semantic_latent: torch.Tensor) -> torch.Tensor:
        """Normalize a raw latent with frozen training-split statistics."""
        self._require_normalization_statistics()
        self.decoder._validate_input(raw_semantic_latent)
        channel_mean = self.latent_channel_mean.view(1, -1, 1, 1, 1)
        channel_std = self.latent_channel_std.view(1, -1, 1, 1, 1)
        return (raw_semantic_latent - channel_mean) / channel_std

    def denormalize(
        self,
        normalized_semantic_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return a normalized latent to the codec's raw latent space."""
        self._require_normalization_statistics()
        self.decoder._validate_input(normalized_semantic_latent)
        channel_mean = self.latent_channel_mean.view(1, -1, 1, 1, 1)
        channel_std = self.latent_channel_std.view(1, -1, 1, 1, 1)
        return normalized_semantic_latent * channel_std + channel_mean

    def _require_normalization_statistics(self) -> None:
        if not bool(self.normalization_fitted):
            raise RuntimeError(
                "latent normalization statistics have not been fitted"
            )


def median_frequency_class_weights(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    maximum_weight: float = 20.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute training-only class counts and median-frequency weights."""
    _validate_loss_targets(labels, valid_mask)
    if maximum_weight <= 0:
        raise ValueError("maximum_weight must be positive")

    valid_labels = labels.masked_select(valid_mask)
    counts = torch.bincount(valid_labels, minlength=NUM_CLASSES)
    counts = counts.to(dtype=torch.float64)

    weights = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=labels.device)
    present_classes = counts > 0
    if bool(present_classes.any()):
        frequencies = counts[present_classes] / counts[present_classes].sum()
        median_frequency = torch.median(frequencies)
        weights[present_classes] = median_frequency / frequencies
        weights.clamp_(max=maximum_weight)

    return counts, weights.to(dtype=torch.float32)


def foreground_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Average soft Dice loss over valid foreground classes present."""
    _validate_loss_inputs(logits, labels, valid_mask)
    probabilities = torch.softmax(logits.float(), dim=1)
    valid_float = valid_mask.to(dtype=torch.float32)

    class_losses = []
    for class_id in range(1, NUM_CLASSES):
        class_target = (labels == class_id) & valid_mask
        if not bool(class_target.any()):
            continue

        class_probability = probabilities[:, class_id]
        class_target_float = class_target.to(dtype=torch.float32)
        valid_probability = class_probability * valid_float
        intersection = (valid_probability * class_target_float).sum()
        denominator = valid_probability.sum() + class_target_float.sum()
        class_dice = (2.0 * intersection + epsilon) / (
            denominator + epsilon
        )
        class_losses.append(1.0 - class_dice)

    if not class_losses:
        return probabilities.sum() * 0.0
    return torch.stack(class_losses).mean()


def semantic_reconstruction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    class_weights: torch.Tensor,
    dice_coefficient: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Compute valid-only weighted cross-entropy plus foreground Dice."""
    _validate_loss_inputs(logits, labels, valid_mask)
    if tuple(class_weights.shape) != (NUM_CLASSES,):
        raise ValueError(f"class_weights must have shape ({NUM_CLASSES},)")
    if not torch.is_floating_point(class_weights):
        raise TypeError("class_weights must use a floating-point dtype")
    if not bool(torch.isfinite(class_weights).all()):
        raise ValueError("class_weights must contain only finite values")
    if bool((class_weights < 0).any()):
        raise ValueError("class_weights cannot contain negative values")
    if dice_coefficient < 0:
        raise ValueError("dice_coefficient cannot be negative")

    float_logits = logits.float()
    float_weights = class_weights.to(
        device=logits.device,
        dtype=torch.float32,
    )
    targets_with_ignore = torch.where(
        valid_mask,
        labels,
        torch.full_like(labels, IGNORE_VALUE),
    )
    per_pixel_cross_entropy = F.cross_entropy(
        float_logits,
        targets_with_ignore,
        weight=float_weights,
        ignore_index=IGNORE_VALUE,
        reduction="none",
    )

    selected_weights = float_weights[labels.masked_select(valid_mask)]
    weight_sum = selected_weights.sum()
    if bool(weight_sum > 0):
        cross_entropy = per_pixel_cross_entropy.sum() / weight_sum
    else:
        cross_entropy = float_logits.sum() * 0.0

    dice = foreground_dice_loss(
        float_logits,
        labels,
        valid_mask,
    )
    total = cross_entropy + dice_coefficient * dice
    return {
        "loss": total,
        "cross_entropy": cross_entropy,
        "foreground_dice": dice,
    }


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the number of parameters that receive gradients."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _validate_configured_size(
    height: int,
    width: int,
    allow_test_shape: bool,
) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError("height and width must both be divisible by 8")
    if not allow_test_shape and (height, width) != (
        SOURCE_HEIGHT,
        SOURCE_WIDTH,
    ):
        raise ValueError(
            "production codec size is fixed at 480x832; "
            "allow_test_shape is only for lightweight unit tests"
        )


def _validate_loss_targets(
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if labels.ndim != 4:
        raise ValueError("labels must have shape [B, T, H, W]")
    if labels.shape != valid_mask.shape:
        raise ValueError("labels and valid_mask must have the same shape")
    if labels.dtype != torch.int64:
        raise TypeError("labels must use torch.int64")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must use torch.bool")

    valid_labels = labels.masked_select(valid_mask)
    if valid_labels.numel() == 0:
        return
    minimum_label = int(valid_labels.min().item())
    maximum_label = int(valid_labels.max().item())
    if minimum_label < 0 or maximum_label >= NUM_CLASSES:
        raise ValueError("valid labels must contain class IDs from 0 through 10")


def _validate_loss_inputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    _validate_loss_targets(labels, valid_mask)
    if logits.ndim != 5:
        raise ValueError("logits must have shape [B, 11, T, H, W]")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must use a floating-point dtype")

    expected_shape = (
        labels.shape[0],
        NUM_CLASSES,
        labels.shape[1],
        labels.shape[2],
        labels.shape[3],
    )
    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"logits must have shape {expected_shape}, got {tuple(logits.shape)}"
        )
