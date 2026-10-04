"""
Curve-Prior Guided Dual-view Gated Decoupling Network (CP-GDM).

这是一个完整、独立的 PyTorch 模型文件，不依赖 decom.py 或 model.py。

核心思路：
1. Zero-DCE 同时输出中间增强图、最终增强图和 24 通道曲线参数图；
2. 原始低光图与 Zero-DCE 增强图形成两个互补视图；
3. 曲线参数图作为显式曝光先验，在 /2、H/4、H/8 三个尺度上控制双视图融合；
4. 使用曲线置信度频率净化代替普通 Self-Attention，抑制增强噪声并保留结构细节；
5. 融合后的潜在特征可作为扩散模型的条件，扩散预测结果再由本文件中的解码器重建。

输入约定：
- stage1（pred_fea=None）：images=[low, high]，形状为 [B, 6, H, W]；
- stage2（pred_fea!=None）：images 可以是 [B, 3, H, W] 或 [B, 6, H, W]，
  pred_fea 的形状应为 [B, 3, ceil(H/8), ceil(W/8)]。

注意：若 low/high 是非配对数据，不应把 gt_fea 当作逐样本回归目标直接监督条件扩散模型，
需要在训练脚本中使用分布学习或内容一致性约束。这里保留 gt_fea 接口，是为了兼容原 LCR 的两阶段流程。
"""

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


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，用较低计算量保留局部空间建模能力。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    """基础残差块；允许输入和输出通道不同。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class UpSamplingBlock(nn.Module):
    """使用转置卷积进行 2 倍上采样。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.up(x))


class ChannelDown(nn.Module):
    """把 H/8 尺度的 4C 通道特征压缩为 3 通道潜在表示。"""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.body(x))


class ChannelUp(nn.Module):
    """把扩散模型的 3 通道潜在表示恢复为 4C 通道。"""

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FeaturePyramid(nn.Module):
    """共享权重的三尺度图像编码器。

    输出：
        level_2: [B, C,  H/2, W/2]
        level_4: [B, 2C, H/4, W/4]
        level_8: [B, 4C, H/8, W/8]
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 5, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, 5, 1, 2),
        )
        self.block0 = ResidualBlock(base_channels, base_channels)
        self.down0 = nn.Conv2d(base_channels, base_channels, 3, 2, 1)

        self.block1 = ResidualBlock(base_channels, base_channels * 2)
        self.down1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 2, 1)

        self.block2 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.down2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 2, 1)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level_2 = self.down0(self.block0(self.stem(image)))
        level_4 = self.down1(self.block1(level_2))
        level_8 = self.down2(self.block2(level_4))
        return level_2, level_4, level_8


class CurvePriorPyramid(nn.Module):
    """把 Zero-DCE 的 24 通道曲线图编码到与图像金字塔相同的三个尺度。

    这里没有把 curve_maps 简单拼到 RGB 图像上，而是保留独立先验分支，
    便于门控网络显式学习“曝光调整强度 -> 特征可信度”的关系。
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(24, base_channels, 5, 1, 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            DepthwiseSeparableConv(base_channels, base_channels, kernel_size=3),
        )
        self.block0 = ResidualBlock(base_channels, base_channels)
        self.down0 = nn.Conv2d(base_channels, base_channels, 3, 2, 1)

        self.block1 = ResidualBlock(base_channels, base_channels * 2)
        self.down1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 2, 1)

        self.block2 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.down2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 2, 1)

    def forward(self, curves: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prior_2 = self.down0(self.block0(self.stem(curves)))
        prior_4 = self.down1(self.block1(prior_2))
        prior_8 = self.down2(self.block2(prior_4))
        return prior_2, prior_4, prior_8


class CurveGuidedCrossGating(nn.Module):
    """曲线先验引导的双视图交叉门控，用于替代 Cross-Attention。

    输入：原始低光特征 F_low、Zero-DCE 增强特征 F_enh、曲线先验 P。

        G = sigmoid(Conv([F_low, F_enh, F_enh-F_low, P]))
        F_fuse = F_low + G * (F_enh-F_low)

    G 接近 0 时保留原图特征，G 接近 1 时采用增强图特征。与普通注意力相比，
    它直接回答“这个位置的预增强结果是否可信”，并保持 O(HWC) 的计算规模。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
            nn.Sigmoid(),
        )

        # 曲线先验进一步产生仿射参数，使融合结果随曝光状态动态调整。
        self.curve_affine = nn.Conv2d(channels, channels * 2, 1)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        low_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
        curve_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        difference = enhanced_feature - low_feature
        gate = self.gate(torch.cat((low_feature, enhanced_feature, difference, curve_prior), dim=1))
        fused = low_feature + gate * difference

        gamma, beta = torch.chunk(self.curve_affine(curve_prior), chunks=2, dim=1)
        # tanh 限制调制幅度，降低训练初期因曲线先验不稳定造成的特征爆炸风险。
        fused = fused * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        fused = fused + self.refine_scale * self.refine(fused)
        return fused, gate


class CurveConfidenceFrequencyRefinement(nn.Module):
    """曲线置信度频率净化，用于替代 Self-Attention。

    Zero-DCE 能揭示暗区细节，也可能放大暗区噪声。因此这里先用局部低通将特征近似拆成
    低频照明和高频细节，再由曲线先验、双视图差异、高频幅值共同预测可靠性 Q：

        F = F_low_freq + Q * F_detail + (1-Q) * F_denoised

    Q 大：优先保留结构细节；Q 小：优先走噪声抑制分支。
    这对应原 Self-Attention 的内容提取职责，但对低光噪声问题具有更明确的物理含义。
    """

    def __init__(self, channels: int, low_pass_kernel: int = 5) -> None:
        super().__init__()
        if low_pass_kernel % 2 == 0:
            raise ValueError("low_pass_kernel must be odd")
        self.low_pass_kernel = low_pass_kernel

        self.reliability = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
            nn.Sigmoid(),
        )

        self.detail_branch = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            DepthwiseSeparableConv(channels, channels, kernel_size=3),
        )
        self.denoise_branch = nn.Sequential(
            # 膨胀卷积扩大感受野，利用周围结构区分真实边缘和孤立噪点。
            DepthwiseSeparableConv(channels, channels, kernel_size=3, dilation=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        fused_feature: torch.Tensor,
        low_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
        curve_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        padding = self.low_pass_kernel // 2
        low_frequency = F.avg_pool2d(
            fused_feature,
            kernel_size=self.low_pass_kernel,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        high_frequency = fused_feature - low_frequency
        view_difference = (enhanced_feature - low_feature).abs()

        reliability = self.reliability(
            torch.cat((curve_prior, view_difference, high_frequency.abs()), dim=1)
        )

        detail_candidate = high_frequency + self.detail_branch(high_frequency)
        denoised_candidate = self.denoise_branch(high_frequency)
        refined_feature = (
            low_frequency
            + reliability * detail_candidate
            + (1.0 - reliability) * denoised_candidate
        )

        # 残差形式使模块可以从接近恒等映射开始学习，训练通常比直接替换稳定。
        delta = self.output_projection(refined_feature - fused_feature)
        output = fused_feature + self.refine_scale * delta
        return output, reliability


class CurvePriorGuidedDecouplingBlock(nn.Module):
    """CP-GDM 的单尺度实现：交叉门控 + 频率净化。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cross_gating = CurveGuidedCrossGating(channels)
        self.frequency_refinement = CurveConfidenceFrequencyRefinement(channels)

    def forward(
        self,
        low_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
        curve_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, fusion_gate = self.cross_gating(low_feature, enhanced_feature, curve_prior)
        refined, frequency_reliability = self.frequency_refinement(
            fused,
            low_feature,
            enhanced_feature,
            curve_prior,
        )
        return refined, fusion_gate, frequency_reliability


class CurveGuidedReconNet(nn.Module):
    """包含 CP-GDM 编码和多尺度条件解码的 LCR 主体。"""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("base_channels should be at least 4")

        self.image_pyramid = FeaturePyramid(base_channels)
        self.curve_pyramid = CurvePriorPyramid(base_channels)
        self.guided_blocks = nn.ModuleList(
            [
                CurvePriorGuidedDecouplingBlock(base_channels),
                CurvePriorGuidedDecouplingBlock(base_channels * 2),
                CurvePriorGuidedDecouplingBlock(base_channels * 4),
            ]
        )

        self.channel_down = ChannelDown(base_channels)
        self.channel_up = ChannelUp(base_channels)

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
        """普通编码接口，主要用于把正常光图像编码为扩散训练目标。"""
        _, _, level_8 = self.image_pyramid(image)
        return self.channel_down(level_8)

    def encode_guided(
        self,
        low_image: torch.Tensor,
        enhanced_image: torch.Tensor,
        curve_maps: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """使用原图、增强图和曲线图生成内容保持的条件潜变量及三尺度跳连。"""
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
            fused, gate, reliability = block(low_feature, enhanced_feature, curve_prior)
            fused_levels.append(fused)
            fusion_gates.append(gate)
            frequency_reliabilities.append(reliability)

        condition_feature = self.channel_down(fused_levels[2])
        levels_tuple = (fused_levels[0], fused_levels[1], fused_levels[2])

        if return_diagnostics:
            diagnostics = {
                "fusion_gates": fusion_gates,
                "frequency_reliabilities": frequency_reliabilities,
            }
            return condition_feature, levels_tuple, diagnostics
        return condition_feature, levels_tuple

    @staticmethod
    def _resize_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """处理奇数尺寸输入导致的转置卷积与跳连相差 1 像素的问题。"""
        if source.shape[-2:] == reference.shape[-2:]:
            return source
        return F.interpolate(source, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def decode(
        self,
        pred_feature: torch.Tensor,
        guided_levels: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        """将扩散输出的 3 通道潜变量与 CP-GDM 多尺度特征共同解码。"""
        level_2, level_4, level_8 = guided_levels
        pred_feature = self.channel_up(pred_feature)
        pred_feature = self._resize_like(pred_feature, level_8)

        feature = self.block_up1(self.block_up0(pred_feature) + level_8)
        feature = self.up0(feature)
        feature = self._resize_like(feature, level_4)

        feature = self.block_up3(self.block_up2(feature) + level_4)
        feature = self.up1(feature)
        feature = self._resize_like(feature, level_2)

        feature = self.block_up5(self.block_up4(feature) + level_2)
        feature = self.up2(feature)
        if feature.shape[-2:] != output_size:
            feature = F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)

        return self.output_head(feature)


class CurvePriorGuidedLCR(nn.Module):
    """完整的 Zero-DCE + CP-GDM + LCR 模型。

    建议训练阶段：
    - 使用 output["cond_fea"] 作为扩散条件；
    - 成对数据可使用 output["gt_fea"] 作为扩散目标；
    - 使用 output["pred_img"] 监督 gt_fea -> high 的目标重建；
    - 使用 output["pred_cond_img"] 监督 cond_fea -> high 的条件重建；
    - 对 cond_fea 和停止梯度的 gt_fea 施加潜变量对齐损失。

    推理阶段：
    - 扩散模型根据 cond_fea 生成 pred_fea；
    - 再次调用 forward(images, pred_fea) 得到最终 pred_img。
    """

    def __init__(self, base_channels: int = 64, clamp_dce_output: bool = True) -> None:
        super().__init__()
        self.dce = EnhanceNetNoPool()
        self.recon_net = CurveGuidedReconNet(base_channels)
        self.clamp_dce_output = clamp_dce_output

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(f"images must be a 4D tensor [B,C,H,W], got {tuple(images.shape)}")
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

        # 修正原 model.py 中把 DCE 三元组整体传给 Conv2d 的接口问题。
        enhanced_mid, enhanced_image, curve_maps = self.dce(low)
        if self.clamp_dce_output:
            # 输入为 [0,1] 时通常不会越界；clamp 可防止训练早期极端曲线造成数值异常。
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
            # 可视化这些门控图能够直接验证新模块是否在暗区/过曝区做出合理选择。
            output.update(diagnostics)

        if pred_fea is None:
            if high is None:
                raise ValueError(
                    "stage1 requires 6-channel images=[low, high] when pred_fea is None"
                )
            target_feature = self.recon_net.encode_plain(high)
            output["gt_fea"] = target_feature
            output["high"] = high

            # 用目标潜变量闭环训练解码器，使其学会 latent -> normal-light image。
            output["pred_img"] = self.recon_net.decode(
                target_feature,
                guided_levels,
                output_size=low.shape[-2:],
            )

            # 新增模块生成的 cond_fea 必须能够在不依赖真实 gt_fea 的情况下
            # 重建正常光图像。第一阶段同时输出条件重建，使训练脚本可以直接
            # 对 cond_fea -> high 施加图像监督，并避免重复执行 DCE 和编码器。
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


# 兼容希望继续使用 LCR 作为类名的训练脚本。
LCR = CurvePriorGuidedLCR
