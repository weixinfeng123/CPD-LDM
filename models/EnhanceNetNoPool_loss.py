"""Supervised losses for :mod:`models.EnhanceNetNoPool`.

The final enhanced image is the main optimization target.  The image after
the first four curve iterations receives weaker auxiliary supervision, while
SSIM, image gradients and curve-map total variation preserve structure and
discourage spatially noisy enhancement curves.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """A smooth, outlier-tolerant alternative to pixel-wise L1 loss."""

    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.epsilon_squared = float(epsilon) ** 2

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((prediction - target).square() + self.epsilon_squared).mean()


class SSIMLoss(nn.Module):
    """Differentiable RGB SSIM loss with a fixed Gaussian window."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")
        coordinates = torch.arange(window_size, dtype=torch.float32)
        coordinates = coordinates - window_size // 2
        gaussian = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        gaussian = gaussian / gaussian.sum()
        window = gaussian[:, None] * gaussian[None, :]
        self.register_buffer("window", window[None, None], persistent=False)
        self.window_size = window_size
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"SSIM inputs must have the same shape, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.ndim != 4:
            raise ValueError("SSIM inputs must be BCHW tensors")

        channels = prediction.shape[1]
        window = self.window.to(dtype=prediction.dtype).expand(channels, 1, -1, -1)
        padding = self.window_size // 2

        mean_prediction = F.conv2d(
            prediction, window, padding=padding, groups=channels
        )
        mean_target = F.conv2d(target, window, padding=padding, groups=channels)
        mean_prediction_sq = mean_prediction.square()
        mean_target_sq = mean_target.square()
        mean_product = mean_prediction * mean_target

        variance_prediction = F.conv2d(
            prediction.square(), window, padding=padding, groups=channels
        ) - mean_prediction_sq
        variance_target = F.conv2d(
            target.square(), window, padding=padding, groups=channels
        ) - mean_target_sq
        covariance = F.conv2d(
            prediction * target, window, padding=padding, groups=channels
        ) - mean_product

        numerator = (2.0 * mean_product + self.c1) * (2.0 * covariance + self.c2)
        denominator = (
            (mean_prediction_sq + mean_target_sq + self.c1)
            * (variance_prediction + variance_target + self.c2)
        )
        ssim = numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)
        return (1.0 - ssim.mean()).clamp_min(0.0)


def image_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match horizontal and vertical finite differences."""

    prediction_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    prediction_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(prediction_x, target_x) + F.l1_loss(prediction_y, target_y)


def curve_total_variation(curve_maps: torch.Tensor) -> torch.Tensor:
    """Spatial smoothness regularizer for the 24 predicted curve channels."""

    horizontal = curve_maps[:, :, :, 1:] - curve_maps[:, :, :, :-1]
    vertical = curve_maps[:, :, 1:, :] - curve_maps[:, :, :-1, :]
    return horizontal.abs().mean() + vertical.abs().mean()


def range_penalty(image: torch.Tensor) -> torch.Tensor:
    """Penalize values outside [0, 1] without clamping training gradients."""

    return (F.relu(-image) + F.relu(image - 1.0)).mean()


class SupervisedEnhanceNetLoss(nn.Module):
    """Composite paired-image objective for supervised Zero-DCE training.

    Defaults place most weight on robust final-image reconstruction.  No clamp
    is applied to network outputs inside the loss; clamping is reserved for
    metrics and image export so that out-of-range predictions keep gradients.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        ssim_weight: float = 0.2,
        gradient_weight: float = 0.1,
        color_weight: float = 0.05,
        intermediate_weight: float = 0.25,
        curve_smoothness_weight: float = 0.01,
        range_weight: float = 0.05,
        charbonnier_epsilon: float = 1e-3,
    ) -> None:
        super().__init__()
        weights = {
            "reconstruction_weight": reconstruction_weight,
            "ssim_weight": ssim_weight,
            "gradient_weight": gradient_weight,
            "color_weight": color_weight,
            "intermediate_weight": intermediate_weight,
            "curve_smoothness_weight": curve_smoothness_weight,
            "range_weight": range_weight,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        self.reconstruction_weight = float(reconstruction_weight)
        self.ssim_weight = float(ssim_weight)
        self.gradient_weight = float(gradient_weight)
        self.color_weight = float(color_weight)
        self.intermediate_weight = float(intermediate_weight)
        self.curve_smoothness_weight = float(curve_smoothness_weight)
        self.range_weight = float(range_weight)
        self.reconstruction = CharbonnierLoss(charbonnier_epsilon)
        self.ssim = SSIMLoss()

    def forward(
        self,
        enhanced_mid: torch.Tensor,
        enhanced: torch.Tensor,
        curve_maps: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if enhanced.shape != target.shape or enhanced_mid.shape != target.shape:
            raise ValueError(
                "enhanced_mid, enhanced and target must have the same BCHW shape; "
                f"got {tuple(enhanced_mid.shape)}, {tuple(enhanced.shape)}, "
                f"and {tuple(target.shape)}"
            )
        if curve_maps.ndim != 4 or curve_maps.shape[0] != target.shape[0]:
            raise ValueError("curve_maps must be a BCHW tensor with the same batch size")

        reconstruction = self.reconstruction(enhanced, target)
        ssim = self.ssim(enhanced, target)
        gradient = image_gradient_loss(enhanced, target)
        color = F.l1_loss(
            enhanced.mean(dim=(-2, -1)),
            target.mean(dim=(-2, -1)),
        )
        intermediate = self.reconstruction(enhanced_mid, target)
        curve_smoothness = curve_total_variation(curve_maps)
        output_range = range_penalty(enhanced) + 0.5 * range_penalty(enhanced_mid)

        total = (
            self.reconstruction_weight * reconstruction
            + self.ssim_weight * ssim
            + self.gradient_weight * gradient
            + self.color_weight * color
            + self.intermediate_weight * intermediate
            + self.curve_smoothness_weight * curve_smoothness
            + self.range_weight * output_range
        )

        if not return_components:
            return total
        components = {
            "total": total,
            "reconstruction": reconstruction,
            "ssim": ssim,
            "gradient": gradient,
            "color": color,
            "intermediate": intermediate,
            "curve_smoothness": curve_smoothness,
            "range": output_range,
        }
        return total, components


__all__ = [
    "CharbonnierLoss",
    "SSIMLoss",
    "SupervisedEnhanceNetLoss",
    "curve_total_variation",
    "image_gradient_loss",
    "range_penalty",
]
