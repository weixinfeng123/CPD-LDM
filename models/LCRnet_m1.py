"""M1 ablation: low-light single-view latent encoder-decoder.

This model is the single-view baseline used to determine whether the gains of
the full CPD-LDM come from the proposed Zero-DCE dual-view/curve-prior path or
merely from the underlying encoder-decoder.

Compared with :mod:`models.LCRnet_new`, M1 removes:

* the Zero-DCE network and its pre-enhanced image;
* the 24-channel curve maps and curve-prior pyramid;
* curve-guided cross gating;
* curve-conditioned detail/denoising refinement.

It keeps the same image feature pyramid, shared multi-scale latent encoder,
three-channel H/8 latent bottleneck, gated decoder skips and reconstruction
decoder.  Its public forward outputs match the keys consumed by
``train_LCR.py`` and the stage-II diffusion code.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LCRnet import (
    ChannelUp,
    FeaturePyramid,
    ResidualBlock,
    UpSamplingBlock,
)
from models.LCRnet_new import (
    BoundedScalarGates,
    FeatureLevels,
    SharedMultiScaleLatentEncoder,
    _group_count,
)


class LowLightSingleViewReconNet(nn.Module):
    """Encode and decode using only the original low-light view."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels should be at least 4")

        # Shared by the low-light condition and normal-light target branches.
        self.image_pyramid = FeaturePyramid(base_channels)
        self.shared_latent_encoder = SharedMultiScaleLatentEncoder(base_channels)

        # Keep the full model's decoder topology unchanged for a fair baseline.
        self.channel_up = ChannelUp(base_channels)
        self.skip_norm_8 = nn.GroupNorm(
            _group_count(base_channels * 4),
            base_channels * 4,
        )
        self.skip_norm_4 = nn.GroupNorm(
            _group_count(base_channels * 2),
            base_channels * 2,
        )
        self.skip_norm_2 = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
        )
        self.decoder_skip_gates = BoundedScalarGates(
            count=3,
            initial_value=0.1,
        )

        self.block_up0 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.block_up1 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.up0 = UpSamplingBlock(base_channels * 4, base_channels * 2)

        self.block_up2 = ResidualBlock(base_channels * 2, base_channels * 2)
        self.block_up3 = ResidualBlock(base_channels * 2, base_channels * 2)
        self.up1 = UpSamplingBlock(base_channels * 2, base_channels)

        self.block_up4 = ResidualBlock(base_channels, base_channels)
        self.block_up5 = ResidualBlock(base_channels, base_channels)
        self.up2 = UpSamplingBlock(base_channels, base_channels)

        self.output_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels, 3, 1),
        )

    def encode_plain(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, FeatureLevels]:
        """Encode an image and return its latent and multi-scale features."""
        levels = self.image_pyramid(image)
        latent = self.shared_latent_encoder(levels)
        return latent, levels

    def encode_condition(
        self,
        low_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, FeatureLevels]:
        """Encode the original low-light image without any auxiliary view."""
        return self.encode_plain(low_image)

    @staticmethod
    def _resize_like(
        source: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def decode(
        self,
        pred_feature: torch.Tensor,
        condition_levels: FeatureLevels,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode a latent using skips extracted only from the low-light view."""
        level_2, level_4, level_8 = condition_levels
        skip_gates = self.decoder_skip_gates()

        pred_feature = self.channel_up(pred_feature)
        pred_feature = self._resize_like(pred_feature, level_8)

        feature = self.block_up1(
            self.block_up0(pred_feature)
            + skip_gates[2] * self.skip_norm_8(level_8)
        )
        feature = self.up0(feature)
        feature = self._resize_like(feature, level_4)

        feature = self.block_up3(
            self.block_up2(feature)
            + skip_gates[1] * self.skip_norm_4(level_4)
        )
        feature = self.up1(feature)
        feature = self._resize_like(feature, level_2)

        feature = self.block_up5(
            self.block_up4(feature)
            + skip_gates[0] * self.skip_norm_2(level_2)
        )
        feature = self.up2(feature)

        if feature.shape[-2:] != output_size:
            feature = F.interpolate(
                feature,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.output_head(feature)


class LowLightSingleViewLCR(nn.Module):
    """Public M1 model with the same training/inference contract as LCRnet_new."""

    ablation_name = "M1_low_light_single_view"
    latent_channels = 3

    def __init__(
        self,
        base_channels: int = 64,
        clamp_dce_output: bool = True,
    ) -> None:
        super().__init__()
        # Accepted only for constructor compatibility with the full model.
        del clamp_dce_output
        self.recon_net = LowLightSingleViewReconNet(base_channels)

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                "images must be a 4D tensor [B,C,H,W], "
                f"got {tuple(images.shape)}"
            )
        if images.shape[1] not in (3, 6):
            raise ValueError(
                f"images must have 3 or 6 channels, got {images.shape[1]}"
            )

    def forward(
        self,
        images: torch.Tensor,
        pred_fea: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, object]:
        self._validate_images(images)
        low = images[:, :3]
        high = images[:, 3:6] if images.shape[1] == 6 else None

        condition_feature, condition_levels = (
            self.recon_net.encode_condition(low)
        )
        output: Dict[str, object] = {
            "cond_fea": condition_feature,
        }

        if return_diagnostics:
            # Empty lists make the absence of curve-guided mechanisms explicit
            # while preserving compatibility with diagnostic consumers.
            output["fusion_gates"] = []
            output["frequency_reliabilities"] = []
            output["single_view_levels"] = condition_levels

        if pred_fea is None:
            if high is None:
                # Low-only inference for the M1 baseline: directly decode the
                # condition latent without a diffusion-predicted target latent.
                condition_image = self.recon_net.decode(
                    condition_feature,
                    condition_levels,
                    output_size=low.shape[-2:],
                )
                output["pred_img"] = condition_image
                output["pred_cond_img"] = condition_image
                return output

            target_feature, _ = self.recon_net.encode_plain(high)
            output["gt_fea"] = target_feature
            output["high"] = high

            # Target-latent reconstruction trains the common decoder.
            output["pred_img"] = self.recon_net.decode(
                target_feature,
                condition_levels,
                output_size=low.shape[-2:],
            )
            # Direct condition reconstruction measures the M1 baseline.
            output["pred_cond_img"] = self.recon_net.decode(
                condition_feature,
                condition_levels,
                output_size=low.shape[-2:],
            )
        else:
            if pred_fea.ndim != 4 or pred_fea.shape[1] != self.latent_channels:
                raise ValueError(
                    "pred_fea must have shape [B,3,H/8,W/8], "
                    f"got {tuple(pred_fea.shape)}"
                )
            if pred_fea.shape[0] != images.shape[0]:
                raise ValueError(
                    "pred_fea and images must have the same batch size"
                )

            output["pred_img"] = self.recon_net.decode(
                pred_fea,
                condition_levels,
                output_size=low.shape[-2:],
            )

        return output


# Compatibility aliases for existing import styles.
CurvePriorGuidedLCR = LowLightSingleViewLCR
CurvePriorGuidedLCRNew = LowLightSingleViewLCR
LCR = LowLightSingleViewLCR


__all__ = [
    "LowLightSingleViewReconNet",
    "LowLightSingleViewLCR",
    "CurvePriorGuidedLCR",
    "CurvePriorGuidedLCRNew",
    "LCR",
]
