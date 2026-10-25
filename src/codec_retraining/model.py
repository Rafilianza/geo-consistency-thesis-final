"""A separate nine-class codec using the original spatial/temporal architecture."""

import torch
from torch import nn

from phase3.semantic_codec import SemanticCodec, SemanticEncoder, prepend_first_frame


class LargeClassEncoder(SemanticEncoder):
    def __init__(self, input_height=480, input_width=832, allow_test_shape=False):
        super().__init__(input_height, input_width, allow_test_shape)
        self.class_embedding = nn.Embedding(10, 8)  # Nine classes plus ignore token 9.

    def encode(self, labels, valid_mask):
        self._validate_inputs(labels, valid_mask)
        if bool((labels[valid_mask] >= 9).any()):
            raise ValueError("Valid reduced-class labels must be 0 through 8.")
        tokens = torch.where(valid_mask, labels, torch.full_like(labels, 9))
        features = self.class_embedding(tokens).permute(0, 4, 1, 2, 3).contiguous()
        features = prepend_first_frame(features)
        features = self.enc_down_1(features)
        features = self.enc_down_2(features)
        features = self.enc_down_3(features)
        features = self.enc_refine(features)
        return self.latent_projection(features)


class LargeClassCodec(SemanticCodec):
    def __init__(self, input_height=480, input_width=832, allow_test_shape=False):
        super().__init__(input_height, input_width, allow_test_shape)
        self.encoder = LargeClassEncoder(input_height, input_width, allow_test_shape)
        self.decoder.logit_projection = nn.Conv3d(12, 9, kernel_size=1)
