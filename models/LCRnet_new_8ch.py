"""Eight-channel latent variant of :mod:`models.LCRnet_new`.

This is an isolated capacity ablation.  Relative to ``LCRnet_new.py``, only
the latent bottleneck width changes::

    cond_fea: [B, 3, H/8, W/8] -> [B, 8, H/8, W/8]
    gt_fea:   [B, 3, H/8, W/8] -> [B, 8, H/8, W/8]

The Zero-DCE preprocessing, curve-prior fusion, shared multi-scale encoding
topology, gated decoder skips and RGB image output are unchanged.  The decoder
input projection is widened accordingly so a stage-2 model must provide an
eight-channel ``pred_fea``.

Existing three-channel LCR and diffusion checkpoints are not shape-compatible
with this model's latent projection and decoder input layers.1
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.LCRnet_new import (
    SharedMultiScaleLatentEncoder,
    SharedTopologyCurveGuidedReconNet,
    _group_count,
)


LATENT_CHANNELS = 8


class StableChannelDown8(nn.Module):
    """Project the shared H/8 feature map to an eight-channel latent."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        hidden_channels = base_channels * 2
        self.conv1 = nn.Conv2d(
            base_channels * 4,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm1 = nn.GroupNorm(
            _group_count(hidden_channels), hidden_channels
        )
        self.conv2 = nn.Conv2d(
            hidden_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.GroupNorm(
            _group_count(base_channels), base_channels
        )
        self.conv3 = nn.Conv2d(
            base_channels,
            LATENT_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # One group normalizes the complete latent vector while retaining
        # channel-specific values.  This mirrors the stable 3-channel head.
        self.logit_norm = nn.GroupNorm(1, LATENT_CHANNELS)
        self.activation = nn.LeakyReLU(
            negative_slope=0.2, inplace=True
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        feature = self.activation(self.norm1(self.conv1(feature)))
        feature = self.activation(self.norm2(self.conv2(feature)))
        logits = self.logit_norm(self.conv3(feature))
        return torch.sigmoid(logits.clamp(min=-12.0, max=12.0))


class ChannelUp8(nn.Module):
    """Expand an eight-channel latent back to the decoder's 4C width."""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                LATENT_CHANNELS,
                base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                "decoder latent must have shape "
                f"[B,{LATENT_CHANNELS},H/8,W/8], got {tuple(feature.shape)}"
            )
        return self.body(feature)


class SharedMultiScaleLatentEncoder8(SharedMultiScaleLatentEncoder):
    """The original shared encoder with an eight-channel projection head."""

    def __init__(self, base_channels: int) -> None:
        super().__init__(base_channels)
        self.project_to_latent = StableChannelDown8(base_channels)


class SharedTopologyCurveGuidedReconNet8(
    SharedTopologyCurveGuidedReconNet
):
    """Shared-topology reconstructor whose latent bottleneck has width eight."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__(base_channels)
        # These are the only two architecture substitutions relative to the
        # 3-channel network.  All fusion and decoder body modules are inherited.
        self.shared_latent_encoder = SharedMultiScaleLatentEncoder8(
            base_channels
        )
        self.channel_up = ChannelUp8(base_channels)


class CurvePriorGuidedLCR8(nn.Module):
    """LCRnet_new with eight-channel condition and target latent features."""

    latent_channels = LATENT_CHANNELS

    def __init__(
        self,
        base_channels: int = 64,
        clamp_dce_output: bool = True,
    ) -> None:
        super().__init__()
        # Instantiate Zero-DCE through the original module type so existing
        # trusted Zero-DCE checkpoints retain exactly the same parameter keys.
        from models.LCRnet import EnhanceNetNoPool

        self.dce = EnhanceNetNoPool()
        self.recon_net = SharedTopologyCurveGuidedReconNet8(base_channels)
        self.clamp_dce_output = bool(clamp_dce_output)

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
            if pred_fea.ndim != 4 or pred_fea.shape[1] != LATENT_CHANNELS:
                raise ValueError(
                    "pred_fea must have shape "
                    f"[B,{LATENT_CHANNELS},H/8,W/8], "
                    f"got {tuple(pred_fea.shape)}"
                )
            if pred_fea.shape[0] != images.shape[0]:
                raise ValueError(
                    "pred_fea and images must have the same batch size"
                )

            output["pred_img"] = self.recon_net.decode(
                pred_fea,
                guided_levels,
                output_size=low.shape[-2:],
            )

        return output


# Compatibility aliases for the existing project import conventions.
CurvePriorGuidedLCRNew8 = CurvePriorGuidedLCR8
CurvePriorGuidedLCR = CurvePriorGuidedLCR8
LCR = CurvePriorGuidedLCR8


__all__ = [
    "LATENT_CHANNELS",
    "StableChannelDown8",
    "ChannelUp8",
    "SharedMultiScaleLatentEncoder8",
    "SharedTopologyCurveGuidedReconNet8",
    "CurvePriorGuidedLCR8",
    "CurvePriorGuidedLCRNew8",
    "CurvePriorGuidedLCR",
    "LCR",
]
