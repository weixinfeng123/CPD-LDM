"""M3 ablation: curve-prior-guided dual-view fusion without refinement.

M3 retains the original low-light view, the frozen Zero-DCE enhanced view,
the 24-channel curve-prior pyramid, and curve-guided cross gating at all three
feature scales.  It removes only ``CurveConfidenceFrequencyRefinement``, the
detail-preservation/denoising module that follows cross gating in the complete
model.

This creates the controlled comparison required to measure the refinement
module independently::

    M3   = curve-prior dual-view cross gating
    Full = M3 + curve-conditioned detail-denoising refinement
1
The latent encoder, decoder, output dictionary, and Stage-I/Stage-II calling
contract remain identical to :mod:`models.LCRnet_new`.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from models.LCRnet import EnhanceNetNoPool
from models.LCRnet_new import (
    CurvePriorGuidedLCR as FullCurvePriorGuidedLCR,
    FeatureLevels,
    SharedTopologyCurveGuidedReconNet,
)


class CurvePriorDualViewReconNet(SharedTopologyCurveGuidedReconNet):
    """Curve-guided cross gating with no frequency-refinement stage."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__(base_channels=base_channels)

        # Preserve the already initialized cross-gating modules from the full
        # decoupling blocks and discard only their frequency-refinement parts.
        # This also keeps common-module initialization aligned with the full
        # model when both experiments use the same random seed.
        full_guided_blocks = self.guided_blocks
        self.guided_blocks = nn.ModuleList(
            [block.cross_gating for block in full_guided_blocks]
        )

    def encode_guided(
        self,
        low_image: torch.Tensor,
        enhanced_image: torch.Tensor,
        curve_maps: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """Fuse both image views using the curve prior and cross gating only."""
        low_levels = self.image_pyramid(low_image)
        enhanced_levels = self.image_pyramid(enhanced_image)
        curve_levels = self.curve_pyramid(curve_maps)

        fused_levels: List[torch.Tensor] = []
        fusion_gates: List[torch.Tensor] = []

        for block, low_feature, enhanced_feature, curve_prior in zip(
            self.guided_blocks,
            low_levels,
            enhanced_levels,
            curve_levels,
        ):
            fused, gate = block(
                low_feature,
                enhanced_feature,
                curve_prior,
            )
            fused_levels.append(fused)
            fusion_gates.append(gate)

        guided_levels: FeatureLevels = (
            fused_levels[0],
            fused_levels[1],
            fused_levels[2],
        )
        condition_feature = self.shared_latent_encoder(guided_levels)

        if return_diagnostics:
            diagnostics = {
                "fusion_gates": fusion_gates,
                # An empty list explicitly records that M3 has no
                # detail-denoising reliability estimator.
                "frequency_reliabilities": [],
            }
            return condition_feature, guided_levels, diagnostics
        return condition_feature, guided_levels


class CurvePriorDualViewLCR(FullCurvePriorGuidedLCR):
    """Public M3 model with the full model's training/inference interface."""

    ablation_name = "M3_curve_prior_dual_view_fusion"
    latent_channels = 3

    def __init__(
        self,
        base_channels: int = 64,
        clamp_dce_output: bool = True,
    ) -> None:
        # Initialize directly rather than constructing and replacing an entire
        # full reconstruction network.  The inherited forward method remains
        # valid because CurvePriorDualViewReconNet exposes the same interface.
        nn.Module.__init__(self)
        self.dce = EnhanceNetNoPool()
        self.recon_net = CurvePriorDualViewReconNet(base_channels)
        self.clamp_dce_output = clamp_dce_output


# Compatibility aliases for existing import styles.
CurvePriorGuidedLCR = CurvePriorDualViewLCR
CurvePriorGuidedLCRNew = CurvePriorDualViewLCR
LCR = CurvePriorDualViewLCR


__all__ = [
    "CurvePriorDualViewReconNet",
    "CurvePriorDualViewLCR",
    "CurvePriorGuidedLCR",
    "CurvePriorGuidedLCRNew",
    "LCR",
]
