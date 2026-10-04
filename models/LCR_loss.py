import torch
import torch.nn as nn
import torch.nn.functional as F

# 提供最基本、最稳定的监督信号：整体亮度、颜色、内容结构要对齐到 high。
# 避免 L2 那种“过度平滑”，也比纯 L1 更稳定（对噪点/局部错误不那么敏感）。
# =========================
# 1) Charbonnier Loss (稳健 L1)
# =========================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# 强化细节：减少“糊”“边缘发虚”“纹理丢失”。
# 抑制一些增强网络常见问题：边缘 halo、局部过曝/欠曝造成的边缘断裂。
# 在你这种 decoder 容易“只抄输入纹理”的情况下，它能让输出的边缘更像 high 的边缘，而不是 enhanced 的边缘。
# =========================
# 2) Gradient Loss (边缘/细节)
# =========================
def _img_gradients(x: torch.Tensor):
    # x: [B,C,H,W]
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx, dy


class GradientLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx, pred_dy = _img_gradients(pred)
        tgt_dx, tgt_dy = _img_gradients(target)
        loss = (pred_dx - tgt_dx).abs().mean() + (pred_dy - tgt_dy).abs().mean()
        if self.reduction == "sum":
            # 这里 sum 的定义：把 mean 换成 sum（如果你想严格 sum）
            pred_dx, pred_dy = _img_gradients(pred)
            tgt_dx, tgt_dy = _img_gradients(target)
            loss = (pred_dx - tgt_dx).abs().sum() + (pred_dy - tgt_dy).abs().sum()
        return loss



# 解决“整体偏色”（例如偏黄/偏绿）以及“对比度不对”的问题。
# 防止网络只学到局部细节，却把整幅图的色调/亮度拉偏。
# 对低光增强很有用：很多模型会出现“亮了但颜色怪”的现象，这个 loss 专门管整体色彩分布。
# =========================
# 3) Color Consistency Loss (颜色：均值+方差)
# =========================
class ColorConsistencyLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # [B,C,H,W] -> [B,C]
        mp = pred.mean(dim=(2, 3))
        mt = target.mean(dim=(2, 3))
        sp = pred.var(dim=(2, 3), unbiased=False).add(self.eps).sqrt()
        st = target.var(dim=(2, 3), unbiased=False).add(self.eps).sqrt()

        loss = (mp - mt).abs() + (sp - st).abs()
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# 提升“看起来像”的程度：更自然的纹理、更合理的局部结构。
# 缓解纯像素损失导致的“平滑/塑料感”。
# 在你这种“希望恢复正常光观感”的任务里，VGG loss 往往能显著改善主观质量（但权重通常要小一点，避免产生伪纹理）。
# =========================
# 4) VGG Perceptual Loss (感知损失)
# pred/target: [B,3,H,W] in [0,1]
# =========================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(3, 8, 15, 22), layer_weights=None):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.layers = list(layers)

        if layer_weights is None:
            self.layer_weights = {l: 1.0 for l in self.layers}
        else:
            assert len(layer_weights) == len(self.layers)
            self.layer_weights = {l: float(layer_weights[i]) for i, l in enumerate(self.layers)}

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or pred.size(1) != 3:
            raise ValueError(f"pred must be [B,3,H,W], got {tuple(pred.shape)}")
        if target.ndim != 4 or target.size(1) != 3:
            raise ValueError(f"target must be [B,3,H,W], got {tuple(target.shape)}")

        # ImageNet normalize
        pred_n = (pred - self.mean) / self.std
        tgt_n  = (target - self.mean) / self.std

        loss = pred.new_tensor(0.0)
        x, y = pred_n, tgt_n
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layer_weights:
                loss = loss + self.layer_weights[i] * F.l1_loss(x, y)
            # 最后一个指定层之后无需继续执行 VGG，减少不必要的计算量。
            if i >= max(self.layers):
                break
        return loss


# =========================
# 5) SSIM Loss（亮度、对比度和结构）
# pred/target: [B,C,H,W]，建议数值范围为 [0,1]
# =========================
class SSIMLoss(nn.Module):
    """可微分的单尺度 SSIM 损失，不依赖额外第三方库。"""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd number")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if data_range <= 0:
            raise ValueError("data_range must be positive")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")

        coordinates = torch.arange(window_size, dtype=torch.float32)
        coordinates = coordinates - window_size // 2
        gaussian_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]

        self.register_buffer(
            "window",
            gaussian_2d.view(1, 1, window_size, window_size),
        )
        self.window_size = window_size
        self.data_range = data_range
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must have the same shape, got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}"
            )
        if pred.ndim != 4:
            raise ValueError(f"pred and target must be [B,C,H,W], got {tuple(pred.shape)}")
        if pred.shape[-2] < self.window_size or pred.shape[-1] < self.window_size:
            raise ValueError(
                f"image height and width must be at least {self.window_size}, "
                f"got {tuple(pred.shape[-2:])}"
            )

        channels = pred.shape[1]
        window = self.window.to(dtype=pred.dtype).expand(channels, 1, -1, -1)

        # 反射填充比零填充更适合图像边界，避免边缘区域出现人为黑边统计。
        padding = self.window_size // 2
        pred_pad = F.pad(pred, (padding, padding, padding, padding), mode="reflect")
        target_pad = F.pad(target, (padding, padding, padding, padding), mode="reflect")

        mu_pred = F.conv2d(pred_pad, window, groups=channels)
        mu_target = F.conv2d(target_pad, window, groups=channels)

        mu_pred_sq = mu_pred.square()
        mu_target_sq = mu_target.square()
        mu_cross = mu_pred * mu_target

        sigma_pred = (
            F.conv2d(pred_pad.square(), window, groups=channels) - mu_pred_sq
        ).clamp_min(0.0)
        sigma_target = (
            F.conv2d(target_pad.square(), window, groups=channels) - mu_target_sq
        ).clamp_min(0.0)
        sigma_cross = (
            F.conv2d(pred_pad * target_pad, window, groups=channels) - mu_cross
        )

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
        denominator = (
            (mu_pred_sq + mu_target_sq + c1)
            * (sigma_pred + sigma_target + c2)
        )
        ssim_map = numerator / denominator.clamp_min(torch.finfo(pred.dtype).eps)
        loss_map = 1.0 - ssim_map

        if self.reduction == "mean":
            return loss_map.mean()
        if self.reduction == "sum":
            return loss_map.sum()
        return loss_map


# =========================
# 6) 单个图像域的重建损失
# Charbonnier 保证像素重建，SSIM 保证结构，Gradient 保证边缘和纹理。
# =========================
class ImageReconstructionLoss(nn.Module):
    def __init__(
        self,
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 0.2,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        self.charbonnier_weight = float(charbonnier_weight)
        self.ssim_weight = float(ssim_weight)
        self.gradient_weight = float(gradient_weight)

        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss(data_range=1.0)
        self.gradient = GradientLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ):
        loss_charbonnier = self.charbonnier(pred, target)
        loss_ssim = self.ssim(pred, target)
        loss_gradient = self.gradient(pred, target)

        total = (
            self.charbonnier_weight * loss_charbonnier
            + self.ssim_weight * loss_ssim
            + self.gradient_weight * loss_gradient
        )

        if return_components:
            return total, {
                "charbonnier": loss_charbonnier,
                "ssim": loss_ssim,
                "gradient": loss_gradient,
            }
        return total


# =========================
# 7) 与 model_curve_gdm.py 对应的双域总损失
#
# 目标域：gt_fea -> decoder -> 正常光图 high
# 条件域：cond_fea -> decoder -> 正常光图 high
# 潜变量域：cond_fea -> stop_gradient(gt_fea)
#
# 新增的曲线门控与频率净化模块改变了 cond_fea 的生成分布，因此第一阶段
# 必须显式约束条件特征保留正常光重建信息，并与扩散目标位于可对齐的空间。
# =========================
def latent_batch_variance_loss(
    feature: torch.Tensor,
    target_std: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize latent dimensions that become constant across images.

    For a normal training batch, every channel/spatial latent dimension is
    compared across batch samples. A spatial fallback keeps the loss usable
    for an accidental batch size of one.
    """
    if feature.ndim != 4:
        raise ValueError(f"feature must be [B,C,H,W], got {tuple(feature.shape)}")
    if target_std <= 0.0:
        raise ValueError("target_std must be positive")

    if feature.shape[0] > 1:
        variance = feature.var(dim=0, unbiased=False)
    else:
        variance = feature.flatten(start_dim=2).var(dim=2, unbiased=False)
    standard_deviation = torch.sqrt(variance + eps)
    return F.relu(target_std - standard_deviation).mean()


class DualDomainLCRLoss(nn.Module):
    def __init__(
        self,
        target_weight: float = 1.0,
        condition_weight: float = 0.5,
        latent_alignment_weight: float = 0.1,
        latent_variance_weight: float = 0.1,
        latent_variance_target_std: float = 0.05,
        color_weight: float = 0.05,
        perceptual_weight: float = 0.01,
    ):
        super().__init__()
        self.target_weight = float(target_weight)
        self.condition_weight = float(condition_weight)
        self.latent_alignment_weight = float(latent_alignment_weight)
        self.latent_variance_weight = float(latent_variance_weight)
        self.latent_variance_target_std = float(latent_variance_target_std)
        self.color_weight = float(color_weight)
        self.perceptual_weight = float(perceptual_weight)

        self.target_reconstruction = ImageReconstructionLoss()
        self.condition_reconstruction = ImageReconstructionLoss()
        self.color = ColorConsistencyLoss()

        # VGG 只用于正常光目标分支；权重为 0 时不加载 VGG 参数。
        self.perceptual = (
            VGGPerceptualLoss()
            if self.perceptual_weight > 0.0
            else None
        )

    def forward(
        self,
        pred_high: torch.Tensor,
        target_high: torch.Tensor,
        pred_condition: torch.Tensor,
        target_condition: torch.Tensor,
        condition_feature: torch.Tensor,
        target_feature: torch.Tensor,
        return_components: bool = False,
    ):
        """
        Args:
            pred_high: gt_fea 经解码器重建的正常光图。
            target_high: 配对正常光真值图。
            pred_condition: cond_fea 经解码器重建的条件图。
            target_condition: 条件分支的正常光监督目标，通常与 target_high 相同。
            condition_feature: 曲线先验引导分支生成的 cond_fea。
            target_feature: 正常光图像编码得到的 gt_fea；作为对齐目标时停止梯度。
        """
        target_total, target_parts = self.target_reconstruction(
            pred_high,
            target_high,
            return_components=True,
        )
        condition_total, condition_parts = self.condition_reconstruction(
            pred_condition,
            target_condition,
            return_components=True,
        )

        if condition_feature.shape != target_feature.shape:
            raise ValueError(
                "condition_feature and target_feature must have the same shape, got "
                f"{tuple(condition_feature.shape)} and {tuple(target_feature.shape)}"
            )
        # 只让条件分支追随正常光目标特征，避免对齐项通过 target_feature
        # 反向传播形成两端共同漂移、最终收敛到无信息常量的捷径。
        loss_latent_alignment = F.l1_loss(
            condition_feature,
            target_feature.detach(),
        )

        # Alignment alone admits a constant-latent shortcut. This VICReg-like
        # term requires different images to retain variation in both domains.
        loss_condition_variance = latent_batch_variance_loss(
            condition_feature,
            target_std=self.latent_variance_target_std,
        )
        loss_target_variance = latent_batch_variance_loss(
            target_feature,
            target_std=self.latent_variance_target_std,
        )
        loss_latent_variance = 0.5 * (
            loss_condition_variance + loss_target_variance
        )

        loss_color = self.color(pred_high, target_high)
        loss_perceptual = pred_high.new_tensor(0.0)
        if self.perceptual is not None:
            loss_perceptual = self.perceptual(pred_high, target_high)

        total = (
            self.target_weight * target_total
            + self.condition_weight * condition_total
            + self.latent_alignment_weight * loss_latent_alignment
            + self.latent_variance_weight * loss_latent_variance
            + self.color_weight * loss_color
            + self.perceptual_weight * loss_perceptual
        )

        if return_components:
            components = {
                "total": total,
                "target_reconstruction": target_total,
                "condition_reconstruction": condition_total,
                "latent_alignment": loss_latent_alignment,
                "latent_variance": loss_latent_variance,
                "condition_variance": loss_condition_variance,
                "target_variance": loss_target_variance,
                "target_charbonnier": target_parts["charbonnier"],
                "target_ssim": target_parts["ssim"],
                "target_gradient": target_parts["gradient"],
                "condition_charbonnier": condition_parts["charbonnier"],
                "condition_ssim": condition_parts["ssim"],
                "condition_gradient": condition_parts["gradient"],
                "color": loss_color,
                "perceptual": loss_perceptual,
            }
            return total, components
        return total


