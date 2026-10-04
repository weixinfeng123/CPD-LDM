from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhanceNetNoPool(nn.Module):
    """Zero-DCE 预增强网络。

    与常见 Zero-DCE 实现相比，这里明确保留 curve_maps。它包含 8 组 RGB 曲线参数，
    共 24 个通道，是后续 CP-GDM 判断各区域曝光修正强度的核心先验。
    """

    def __init__(self, feature_channels: int = 32) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, feature_channels, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(feature_channels, feature_channels, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(feature_channels * 2, feature_channels, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(feature_channels * 2, feature_channels, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(feature_channels * 2, 24, 3, 1, 1, bias=True)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat((x3, x4), dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat((x2, x5), dim=1)))

        # tanh 把每组曲线参数限制到 [-1, 1]。
        curve_maps = torch.tanh(self.e_conv7(torch.cat((x1, x6), dim=1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.chunk(
            curve_maps,
            chunks=8,
            dim=1,
        )

        # 手动执行 8 次 Zero-DCE 曲线映射：x <- x + A * (x^2 - x)。
        enhanced = image + r1 * (image.square() - image)
        enhanced = enhanced + r2 * (enhanced.square() - enhanced)
        enhanced = enhanced + r3 * (enhanced.square() - enhanced)
        enhanced_mid = enhanced + r4 * (enhanced.square() - enhanced)

        # enhanced_mid 是前 4 次曲线映射产生的中间增强图。
        enhanced = enhanced_mid + r5 * (enhanced_mid.square() - enhanced_mid)
        enhanced = enhanced + r6 * (enhanced.square() - enhanced)
        enhanced = enhanced + r7 * (enhanced.square() - enhanced)
        enhanced = enhanced + r8 * (enhanced.square() - enhanced)

        return enhanced_mid, enhanced, curve_maps

