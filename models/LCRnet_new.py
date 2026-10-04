"""Shared-topology Curve-Prior Guided LCR network.

This variant keeps the Zero-DCE, CP-GDM condition fusion and guided decoder
from :mod:`models.LCRnet`, but changes the latent encoding topology:

    condition fused pyramid -> shared multi-scale latent encoder -> cond_fea
    normal-light pyramid    -> shared multi-scale latent encoder -> gt_fea

Both domains therefore pass through the same H/2 -> H/4 -> H/8 refinement and
the same 3-channel projection.  The two latent distributions are not forced to
be identical, but their semantics are learned through the same encoder.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.LCRnet import (
    ChannelUp,
    CurvePriorGuidedDecouplingBlock,
    CurvePriorPyramid,
    EnhanceNetNoPool,
    FeaturePyramid,
    ResidualBlock,
    UpSamplingBlock,
)


FeatureLevels = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _group_count(channels: int, maximum_groups: int = 32) -> int:
    """Choose the largest valid GroupNorm group count up to the limit."""
    for groups in range(min(maximum_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class StableChannelDown(nn.Module):
    """Project H/8 features to a bounded latent without FP16 saturation.

    GroupNorm prevents the projection logits from growing with the absolute
    activation scale.  The final clamp is deliberately broad: it only guards
    extreme values that could overflow downstream half-precision operations.
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        hidden_channels = base_channels * 2
        self.conv1 = nn.Conv2d(base_channels * 4, hidden_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(_group_count(hidden_channels), hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, base_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(_group_count(base_channels), base_channels)
        self.conv3 = nn.Conv2d(base_channels, 3, 3, 1, 1)
        self.logit_norm = nn.GroupNorm(1, 3)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.activation(self.norm1(self.conv1(feature)))
        feature = self.activation(self.norm2(self.conv2(feature)))
        logits = self.logit_norm(self.conv3(feature))
        return torch.sigmoid(logits.clamp(min=-12.0, max=12.0))


class BoundedScalarGates(nn.Module):
    """Learnable scalar gates constrained to (0, 1)."""

    def __init__(self, count: int, initial_value: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < initial_value < 1.0:
            raise ValueError("initial_value must be strictly between 0 and 1")
        initial_logit = math.log(initial_value / (1.0 - initial_value))
        self.logits = nn.Parameter(torch.full((count,), initial_logit))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)


class SharedMultiScaleLatentEncoder(nn.Module):
    """Map either condition or target pyramids into one shared latent space.

    Unlike the original implementation, this encoder does not project only
    the deepest feature.  It applies the same top-down-to-deep processing to
    all three scales for both domains, so ``cond_fea`` and ``gt_fea`` have the
    same encoding topology as well as the same shape and numeric range.
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.base_channels = base_channels

        self.refine_level_2 = ResidualBlock(base_channels, base_channels)
        self.norm_level_2 = nn.GroupNorm(
            _group_count(base_channels),
            base_channels,
        )
        self.down_level_2 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.refine_level_4 = ResidualBlock(base_channels * 2, base_channels * 2)
        self.norm_level_4 = nn.GroupNorm(
            _group_count(base_channels * 2),
            base_channels * 2,
        )
        self.norm_from_level_2 = nn.GroupNorm(
            _group_count(base_channels * 2),
            base_channels * 2,
        )
        self.merge_level_4 = ResidualBlock(base_channels * 2, base_channels * 2)
        self.norm_merged_level_4 = nn.GroupNorm(
            _group_count(base_channels * 2),
            base_channels * 2,
        )
        self.down_level_4 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.refine_level_8 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.norm_level_8 = nn.GroupNorm(
            _group_count(base_channels * 4),
            base_channels * 4,
        )
        self.norm_from_level_4 = nn.GroupNorm(
            _group_count(base_channels * 4),
            base_channels * 4,
        )
        self.merge_level_8 = ResidualBlock(base_channels * 4, base_channels * 4)
        self.norm_merged_level_8 = nn.GroupNorm(
            _group_count(base_channels * 4),
            base_channels * 4,
        )

        # Cross-scale information starts weak and can grow only within (0, 1).
        self.cross_scale_gates = BoundedScalarGates(count=2, initial_value=0.1)
        self.project_to_latent = StableChannelDown(base_channels)

    @staticmethod
    def _resize_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(
            source,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def _validate_levels(self, levels: FeatureLevels) -> None:
        if len(levels) != 3:
            raise ValueError(f"levels must contain 3 tensors, got {len(levels)}")

        expected_channels = (
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 4,
        )
        for index, (feature, channels) in enumerate(zip(levels, expected_channels)):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(
                    f"levels[{index}] must be [B,{channels},H,W], "
                    f"got {tuple(feature.shape)}"
                )

    def forward(self, levels: FeatureLevels) -> torch.Tensor:
        self._validate_levels(levels)
        level_2, level_4, level_8 = levels

        fusion_gates = self.cross_scale_gates()
        feature_2 = self.norm_level_2(self.refine_level_2(level_2))

        feature_4 = self.norm_level_4(self.refine_level_4(level_4))
        from_level_2 = self._resize_like(self.down_level_2(feature_2), feature_4)
        from_level_2 = self.norm_from_level_2(from_level_2)
        feature_4 = self.norm_merged_level_4(
            self.merge_level_4(feature_4 + fusion_gates[0] * from_level_2)
        )

        feature_8 = self.norm_level_8(self.refine_level_8(level_8))
        from_level_4 = self._resize_like(self.down_level_4(feature_4), feature_8)
        from_level_4 = self.norm_from_level_4(from_level_4)
        feature_8 = self.norm_merged_level_8(
            self.merge_level_8(feature_8 + fusion_gates[1] * from_level_4)
        )

        return self.project_to_latent(feature_8)


class SharedTopologyCurveGuidedReconNet(nn.Module):
    """CP-GDM encoder and decoder with one latent encoder for both domains."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels should be at least 4")

        # The image pyramid is shared by low, DCE-enhanced and normal images.
        self.image_pyramid = FeaturePyramid(base_channels)
        self.curve_pyramid = CurvePriorPyramid(base_channels)
        self.guided_blocks = nn.ModuleList(
            [
                CurvePriorGuidedDecouplingBlock(base_channels),
                CurvePriorGuidedDecouplingBlock(base_channels * 2),
                CurvePriorGuidedDecouplingBlock(base_channels * 4),
            ]
        )

        # Crucial change: both fused condition levels and normal-light levels
        # enter this exact same multi-scale encoder and projection head.
        self.shared_latent_encoder = SharedMultiScaleLatentEncoder(base_channels)

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
        # Guided skips are useful for detail, but starting at 0.1 prevents the
        # decoder from solving the task while ignoring the latent bottleneck.
        self.decoder_skip_gates = BoundedScalarGates(count=3, initial_value=0.1)
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

    def encode_plain(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a normal-light image through the shared latent topology."""
        target_levels = self.image_pyramid(image)
        return self.shared_latent_encoder(target_levels)

    def encode_guided(
        self,
        low_image: torch.Tensor,
        enhanced_image: torch.Tensor,
        curve_maps: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Fuse low-light conditions, then use the shared latent topology."""
        low_levels = self.image_pyramid(low_image)
        enhanced_levels = self.image_pyramid(enhanced_image)
        curve_levels = self.curve_pyramid(curve_maps)

        fused_levels: List[torch.Tensor] = []
        fusion_gates: List[torch.Tensor] = []
        frequency_reliabilities: List[torch.Tensor] = []

        for block, low_feature, enhanced_feature, curve_prior in zip(
            self.guided_blocks,
            low_levels,
            enhanced_levels,
            curve_levels,
        ):
            fused, gate, reliability = block(
                low_feature,
                enhanced_feature,
                curve_prior,
            )
            fused_levels.append(fused)
            fusion_gates.append(gate)
            frequency_reliabilities.append(reliability)

        guided_levels: FeatureLevels = (
            fused_levels[0],
            fused_levels[1],
            fused_levels[2],
        )
        condition_feature = self.shared_latent_encoder(guided_levels)

        if return_diagnostics:
            diagnostics = {
                "fusion_gates": fusion_gates,
                "frequency_reliabilities": frequency_reliabilities,
            }
            return condition_feature, guided_levels, diagnostics
        return condition_feature, guided_levels

    @staticmethod
    def _resize_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
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
        guided_levels: FeatureLevels,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Decode a latent with condition-derived multi-scale skip features."""
        level_2, level_4, level_8 = guided_levels
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


class CurvePriorGuidedLCR(nn.Module):
    """LCR variant whose condition and target use one latent encoder topology.

    Stage 1:
        ``images=[low, high]`` and ``pred_fea=None`` produce ``cond_fea``,
        ``gt_fea``, ``pred_cond_img`` and ``pred_img``.

    Stage 2/inference:
        a diffusion-produced ``pred_fea`` is decoded with condition-derived
        guided levels.  Public output keys match ``models.LCRnet``.
    """

    def __init__(self, base_channels: int = 64, clamp_dce_output: bool = True) -> None:
        super().__init__()
        self.dce = EnhanceNetNoPool()
        self.recon_net = SharedTopologyCurveGuidedReconNet(base_channels)
        self.clamp_dce_output = clamp_dce_output

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must be a 4D tensor [B,C,H,W], got {tuple(images.shape)}"
            )
        if images.shape[1] not in (3, 6):
            raise ValueError(f"images must have 3 or 6 channels, got {images.shape[1]}")

    def forward(
        self,
        images: torch.Tensor,
        pred_fea: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Dict[str, object]:
        self._validate_images(images)
        low = images[:, :3]
        high = images[:, 3:6] if images.shape[1] == 6 else None

        enhanced_mid, enhanced_image, curve_maps = self.dce(low)
        if self.clamp_dce_output:
            enhanced_mid = enhanced_mid.clamp(0.0, 1.0)
            enhanced_image = enhanced_image.clamp(0.0, 1.0)

        guided_result = self.recon_net.encode_guided(
            low,
            enhanced_image,
            curve_maps,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            condition_feature, guided_levels, diagnostics = guided_result
        else:
            condition_feature, guided_levels = guided_result

        output: Dict[str, object] = {
            "enhance_image_mid": enhanced_mid,
            "enhance_image": enhanced_image,
            "curve_maps": curve_maps,
            "cond_fea": condition_feature,
        }
        if return_diagnostics:
            output.update(diagnostics)

        if pred_fea is None:
            if high is None:
                raise ValueError(
                    "stage1 requires 6-channel images=[low, high] when "
                    "pred_fea is None"
                )

            target_feature = self.recon_net.encode_plain(high)
            output["gt_fea"] = target_feature
            output["high"] = high
            output["pred_img"] = self.recon_net.decode(
                target_feature,
                guided_levels,
                output_size=low.shape[-2:],
            )
            output["pred_cond_img"] = self.recon_net.decode(
                condition_feature,
                guided_levels,
                output_size=low.shape[-2:],
            )
        else:
            if pred_fea.ndim != 4 or pred_fea.shape[1] != 3:
                raise ValueError(
                    "pred_fea must have shape [B,3,H/8,W/8], "
                    f"got {tuple(pred_fea.shape)}"
                )
            if pred_fea.shape[0] != images.shape[0]:
                raise ValueError("pred_fea and images must have the same batch size")

            output["pred_img"] = self.recon_net.decode(
                pred_fea,
                guided_levels,
                output_size=low.shape[-2:],
            )

        return output


# Keep both import styles compatible with existing training/diffusion scripts.
CurvePriorGuidedLCRNew = CurvePriorGuidedLCR
LCR = CurvePriorGuidedLCR


__all__ = [
    "SharedMultiScaleLatentEncoder",
    "SharedTopologyCurveGuidedReconNet",
    "CurvePriorGuidedLCR",
    "CurvePriorGuidedLCRNew",
    "LCR",
]
