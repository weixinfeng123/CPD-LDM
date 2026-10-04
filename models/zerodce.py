import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import pytorch_colors as colors
import numpy as np
import torch
import torch.nn as nn
import warnings
import os
import math
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)  # 将所有的 r 分量拼接在一起，形成一个多通道的输出张量
        return enhance_image_1,enhance_image,r




# 根据通道数自动选择合适的 GroupNorm 分组数。
# 图像增强/重建训练里 batch size 往往较小，GroupNorm 通常比 BatchNorm 更稳定。
def _norm2d(channels):
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)

# 深度可分离卷积模块：降低参数量和计算量，同时保留卷积提取特征的能力。
class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable 3x3 convolution followed by a pointwise projection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # depthwise 卷积：每个输入通道单独做 3x3 空间卷积，参数量较低。
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        # pointwise 卷积：用 1x1 卷积完成通道混合和通道数变换。
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = _norm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 输入: [B, in_channels, H, W]
        # 输出: [B, out_channels, H/stride, W/stride]
        return self.act(self.norm(self.pointwise(self.depthwise(x))))

# 通道注意力模块：让网络自动判断哪些通道更重要。例如低光增强中，有些通道可能更关注亮度，有些通道更关注边缘、颜色或纹理。
class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            # 全局平均池化得到每个通道的全局响应。
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 学习每个通道的重要性权重，再乘回原特征。
        return x * self.net(x)

# 残差注意力块：提取局部纹理和细节，同时通过残差连接保留原始信息。残差结构还能缓解深层网络训练不稳定的问题。
class ResidualAttentionBlock(nn.Module):
    """Residual feature block with lightweight channel attention."""

    def __init__(self, channels):
        super().__init__()
        # 残差分支负责提取局部纹理和通道注意力，主分支保留原始特征。
        self.body = nn.Sequential(
            DepthwiseSeparableConv(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _norm2d(channels),
            ChannelAttention(channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        # 残差连接可以缓解深层网络训练不稳定，并保留细节信息。
        return self.act(x + self.body(x))

# 下采样模块：降低分辨率，扩大感受野，提取更高级的语义和结构信息。
class DownBlock(nn.Module):
    """Downsample by 2 and refine features."""

    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        # 第一个卷积 stride=2 完成下采样，后续残差注意力块继续提炼特征。
        layers = [DepthwiseSeparableConv(in_channels, out_channels, stride=2)]
        layers += [ResidualAttentionBlock(out_channels) for _ in range(num_blocks)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 上采样模块：逐步恢复图像分辨率，同时利用 skip 特征补充细节信息。
class UpBlock(nn.Module):
    """Upsample by 2, fuse a skip feature, and refine."""

    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=2):
        super().__init__()
        # PixelShuffle 上采样：先扩展到 4 倍通道，再重排为空间 2 倍分辨率。
        # 相比转置卷积，它通常更不容易产生棋盘格伪影。
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # skip 特征来自输入图像的编码金字塔，提供纹理和结构细节。
        self.skip_proj = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        # 可学习 skip 权重，初始较小，避免训练初期 skip 特征压过条件特征。
        self.skip_gain = nn.Parameter(torch.tensor(0.1))
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        self.refine = nn.Sequential(*[ResidualAttentionBlock(out_channels) for _ in range(num_blocks)])

    def forward(self, x, skip):
        x = self.up(x)
        # 如果输入尺寸不是 8 的整数倍，上采样后可能出现 1 像素差异，这里对齐到 skip 尺寸。
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_gain * self.skip_proj(skip)
        # 拼接条件解码特征和输入图像 skip 特征，再用 1x1 卷积融合。
        x = self.fuse(torch.cat([x, skip], dim=1))
        return self.refine(x)

# 这是最后一级上采样模块：不再拼接 skip 特征，避免尺寸对齐错误。
class FinalUpBlock(nn.Module):
    """Upsample the H/2 feature map back to full image resolution."""

    def __init__(self, channels, num_blocks=2):
        super().__init__()
        # 最后一级只负责从 H/2 恢复到 H，不再拼接 skip，避免尺寸被错误对齐回低分辨率。
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            *[ResidualAttentionBlock(channels) for _ in range(num_blocks)],
        )

    def forward(self, x, size):
        x = self.net(x)
        # 保证输出严格恢复到输入图像的原始 H、W。
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x

# 特征金字塔模块:它会从输入 RGB 图像中提取一下三个尺度的特征，同时保留浅层细节和深层语义信息。
# fea2: [B, C, H/2, W/2]
# fea4: [B, 2C, H/4, W/4]
# fea8: [B, 4C, H/8, W/8]
class FeaturePyramid(nn.Module):
    """Extract features at H/2, H/4, and H/8."""

    def __init__(self, channels=64):
        super().__init__()
        # stem 在原分辨率提取浅层纹理特征。
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=5, padding=2, bias=False),
            _norm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualAttentionBlock(channels),
        )
        # 三层金字塔输出:
        # fea2: [B, C, H/2, W/2]
        # fea4: [B, 2C, H/4, W/4]
        # fea8: [B, 4C, H/8, W/8]
        self.down2 = DownBlock(channels, channels)
        self.down4 = DownBlock(channels, channels * 2)
        self.down8 = DownBlock(channels * 2, channels * 4)

    def forward(self, x):
        x = self.stem(x)
        fea2 = self.down2(x)
        fea4 = self.down4(fea2)
        fea8 = self.down8(fea4)
        return fea2, fea4, fea8

# 把 H/8 的高维特征压缩成 3 通道条件特征：[B, 4C, H/8, W/8] → [B, 3, H/8, W/8]
class FeatureEncoder(nn.Module):
    """Compress the H/8 feature map to a 3-channel condition feature."""

    def __init__(self, channels=64):
        super().__init__()
        # 将高维瓶颈特征压缩为 3 通道条件特征。
        # 输出分辨率保持 H/8、W/8，用 sigmoid 约束到 [0, 1]。
        self.net = nn.Sequential(
            ResidualAttentionBlock(channels * 4),
            nn.Conv2d(channels * 4, channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# 外部预测的条件特征转换成解码器可以使用的高维瓶颈特征。
class ConditionProjector(nn.Module):
    """Expand a 3-channel condition feature back to bottleneck channels."""

    def __init__(self, channels=64):
        super().__init__()
        # 解码前将 3 通道条件特征升维回瓶颈通道数 [B, 4C, H/8, W/8]。
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=3, padding=1),
            ResidualAttentionBlock(channels * 4),
        )

    def forward(self, x):
        return self.net(x)

# 瓶颈层的多尺度上下文模块：扩大感受野，融合不同尺度的信息。
class ContextBlock(nn.Module):
    """Multi-scale bottleneck context using dilated convolutions."""

    def __init__(self, channels):
        super().__init__()
        branch_channels = channels // 4
        # 多分支空洞卷积在 H/8 的瓶颈层扩大感受野。
        # dilation=1/2/4 分别关注不同尺度的上下文信息。
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, branch_channels, kernel_size=1),
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=1, dilation=1),
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(channels, branch_channels, kernel_size=3, padding=4, dilation=4),
            ]
        )
        self.fuse = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(branch_channels * 4, channels, kernel_size=1),
            ChannelAttention(channels),
        )

    def forward(self, x):
        # 各尺度上下文拼接后融合，再残差加回原瓶颈特征。
        context = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.fuse(context)


class NewReconNet(nn.Module):
    """
    Conditional image reconstruction network.

    Compatible call patterns:
      - encode3(img): [B,3,H,W] -> [B,3,H/8,W/8]
      - forward(x, pred_fea=None): x=[B,6,H,W], returns low/high condition features
      - forward(x, pred_fea=fea): x can be [B,3,H,W] or [B,6,H,W], returns [B,3,H,W]
    """

    def __init__(self, channels=64):
        super().__init__()
        # 编码器：从 RGB 图像中提取 H/2、H/4、H/8 三个尺度的特征。
        self.pyramid = FeaturePyramid(channels)
        # 条件编码头：把 H/8 高维特征压缩成 3 通道条件特征。
        self.encoder = FeatureEncoder(channels)
        # 条件解码头：把 3 通道条件特征恢复成高维瓶颈特征。
        self.condition_proj = ConditionProjector(channels)

        bottleneck_channels = channels * 4
        # 瓶颈增强：融合多尺度上下文，提升全局亮度、颜色和结构建模能力。
        self.bottleneck = nn.Sequential(
            ContextBlock(bottleneck_channels),
            ResidualAttentionBlock(bottleneck_channels),
        )

        # 解码路径:
        # H/8 -> H/4，融合 input_fea4
        # H/4 -> H/2，融合 input_fea2
        # H/2 -> H，输出最终 RGB 图像
        self.up4 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.up2 = UpBlock(channels * 2, channels, channels)
        self.up1 = FinalUpBlock(channels)
        # 输出头使用 sigmoid，把增强图像限制在 [0, 1]。
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def encode3(self, img3):
        # 输入 img3: [B, 3, H, W]
        # 输出条件特征: [B, 3, H/8, W/8]
        _, _, fea8 = self.pyramid(img3)
        return self.encoder(fea8)

    def forward(self, x, pred_fea=None):
        if pred_fea is None:
            # 训练/对齐模式：
            # x 默认是 [B, 6, H, W]，前三通道是 low，后三通道是 high。
            # 分别编码 low/high，返回两者的 H/8 条件特征，可用于特征监督。
            low_fea = self.encode3(x[:, :3, ...])
            high_fea = self.encode3(x[:, 3:, ...])
            return low_fea, high_fea

        else:
            # 重建模式：
            # x 可以是 [B, 3, H, W]，也可以是 [B, 6, H, W]；这里只使用前三通道作为基础图像。
            # pred_fea 是外部给定的条件特征，形状应为 [B, 3, H/8, W/8]。
            base = x[:, :3, ...]
            # 从基础图像提取多尺度 skip 特征，用来恢复纹理、边缘和局部结构。
            input_fea2, input_fea4, input_fea8 = self.pyramid(base)
            # 将低维条件特征升维到瓶颈空间。
            cond = self.condition_proj(pred_fea)

            # 条件瓶颈特征与输入图像的 H/8 特征相加，使重建同时受目标条件和原图结构约束。
            bottleneck = self.bottleneck(cond + input_fea8)
            out = self.up4(bottleneck, input_fea4)
            out = self.up2(out, input_fea2)
            out = self.up1(out, base.shape[-2:])
        return self.out(out)


# Drop-in alias for code that imports "ReconNet" from this file.
ReconNet = NewReconNet


class DCE(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.ReconNet = ReconNet(channels)
        self.dce = enhance_net_nopool()

    def forward(self, images, pred_fea=None):
        """
        images: [B,6,H,W] = low(前三) + high(后三)
        pred_fea:
          - None: stage1，自己产生 pred_fea（编码得到）并解码，便于训练闭环
          - not None: stage2，pred_fea 来自扩散模型，直接解码
        """
        output = {}

        low = images[:, :3]
        high = images[:, 3:]

        # 初步增强（可在 stage2 冻结）
        enhanced_image_1,enhanced_image,A = self.dce(low)  # [B,3,H,W]

        enhanced_image = enhanced_image.clamp(0, 1)

        output["enhanced_image_1"] = enhanced_image_1
        output["enhanced_image"] = enhanced_image
        output["A"] = A


        if pred_fea is None:

            cond_fea = self.ReconNet.encode3(enhanced_image)  # [B,3,H/8,W/8]
            gt_fea = self.ReconNet.encode3(high)  # [B,3,H/8,W/8]
            output["cond_fea"] = cond_fea
            output["gt_fea"] = gt_fea


            # pred_img = self.ReconNet(enhance_image, pred_fea=gt_fea)
            pred_img = self.ReconNet(enhanced_image, pred_fea=gt_fea)
            output["pred_img"] = pred_img
            output["high"] = high

        else:

            pred_img = self.ReconNet(enhanced_image, pred_fea=pred_fea)
            output["pred_img"] = pred_img
            output["high"] = high

        return output
