"""M2 ablation: ordinary low/enhanced dual-view feature fusion.

M2 sits between the low-light-only M1 baseline and the complete curve-prior
guided model in :mod:`models.LCRnet_new`.  It keeps the frozen Zero-DCE output
as a second image view, but deliberately does not feed the 24-channel curve
maps into the reconstruction network.  At each image-pyramid scale, low-light
and enhanced features are fused by ordinary concatenation and convolution.

Removed from the full model:

* curve-prior pyramid;
* curve-guided cross gating;
* curve-confidence detail/denoising refinement.

Retained for a controlled comparison:

* the shared image feature pyramid;
* the shared multi-scale latent encoder and three-channel H/8 latent;
* the gated reconstruction decoder;
* the public output contract consumed by ``train_LCR.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.LCRnet import EnhanceNetNoPool, ResidualBlock
from models.LCRnet_m1 import LowLightSingleViewReconNet
from models.LCRnet_new import FeatureLevels


class OrdinaryDualViewFusionBlock(nn.Module):
    """Fuse two same-scale views using concatenation and convolution only."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ResidualBlock(channels, channels),
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
    ) -> torch.Tensor:
        if low_feature.shape != enhanced_feature.shape:
            raise ValueError(
                "low_feature and enhanced_feature must have the same shape, "
                f"got {tuple(low_feature.shape)} and "
                f"{tuple(enhanced_feature.shape)}"
            )
        return self.fusion(torch.cat((low_feature, enhanced_feature), dim=1))


class OrdinaryDualViewReconNet(LowLightSingleViewReconNet):
    """Use ordinary multi-scale fusion for low and Zero-DCE image views."""

    def __init__(self, base_channels: int = 64) -> None:
        # This preserves exactly the same pyramid, latent encoder, and decoder
        # topology as M1 and the full model.
        super().__init__(base_channels=base_channels)
        self.fusion_blocks = nn.ModuleList(
            [
                OrdinaryDualViewFusionBlock(base_channels),
                OrdinaryDualViewFusionBlock(base_channels * 2),
                OrdinaryDualViewFusionBlock(base_channels * 4),
            ]
        )

    def encode_condition(
        self,
        low_image: torch.Tensor,
        enhanced_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, FeatureLevels]:
        """Encode two image views without curve-prior guidance."""
        low_levels = self.image_pyramid(low_image)
        enhanced_levels = self.image_pyramid(enhanced_image)

        fused_list: List[torch.Tensor] = []
        for block, low_feature, enhanced_feature in zip(
            self.fusion_blocks,
            low_levels,
            enhanced_levels,
        ):
            fused_list.append(block(low_feature, enhanced_feature))

        fused_levels: FeatureLevels = (
            fused_list[0],
            fused_list[1],
            fused_list[2],
        )
        condition_feature = self.shared_latent_encoder(fused_levels)
        return condition_feature, fused_levels


class OrdinaryDualViewLCR(nn.Module):
    """Public M2 model with the same training contract as LCRnet_new."""

    ablation_name = "M2_ordinary_dual_view_fusion"
    latent_channels = 3

    def __init__(
        self,
        base_channels: int = 64,
        clamp_dce_output: bool = True,
    ) -> None:
        super().__init__()
        self.dce = EnhanceNetNoPool()
        self.recon_net = OrdinaryDualViewReconNet(base_channels)
        self.clamp_dce_output = clamp_dce_output

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

        # curve_maps are produced inside Zero-DCE to form the enhanced image,
        # but are intentionally not passed to the M2 reconstruction network.
        condition_feature, fused_levels = self.recon_net.encode_condition(
            low,
            enhanced_image,
        )
        output: Dict[str, object] = {
            "enhance_image_mid": enhanced_mid,
            "enhance_image": enhanced_image,
            "curve_maps": curve_maps,
            "cond_fea": condition_feature,
        }

        if return_diagnostics:
            output["fusion_gates"] = []
            output["frequency_reliabilities"] = []
            output["ordinary_fused_levels"] = fused_levels

        if pred_fea is None:
            if high is None:
                # Directly decode the M2 condition latent for low-only Stage-I
                # evaluation without invoking a diffusion model.
                condition_image = self.recon_net.decode(
                    condition_feature,
                    fused_levels,
                    output_size=low.shape[-2:],
                )
                output["pred_img"] = condition_image
                output["pred_cond_img"] = condition_image
                return output

            target_feature, _ = self.recon_net.encode_plain(high)
            output["gt_fea"] = target_feature
            output["high"] = high
            output["pred_img"] = self.recon_net.decode(
                target_feature,
                fused_levels,
                output_size=low.shape[-2:],
            )
            output["pred_cond_img"] = self.recon_net.decode(
                condition_feature,
                fused_levels,
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
                fused_levels,
                output_size=low.shape[-2:],
            )

        return output


# Compatibility aliases for scripts that import the full model's class names.
CurvePriorGuidedLCR = OrdinaryDualViewLCR
CurvePriorGuidedLCRNew = OrdinaryDualViewLCR
LCR = OrdinaryDualViewLCR


__all__ = [
    "OrdinaryDualViewFusionBlock",
    "OrdinaryDualViewReconNet",
    "OrdinaryDualViewLCR",
    "CurvePriorGuidedLCR",
    "CurvePriorGuidedLCRNew",
    "LCR",
]
