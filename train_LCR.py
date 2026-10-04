"""Train the non-DCE parts of model_curve_gdm.py with paired images.

Expected directory structure:

data/
    train/
        low/
        high/
    val/
        low/
        high/

Low/high images are paired by their relative path without the file extension.
For example, train/low/scene/001.png matches train/high/scene/001.jpg.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.LCR_loss import DualDomainLCRLoss
from models.LCRnet_m1 import LowLightSingleViewLCR
from models.LCRnet_m2 import OrdinaryDualViewLCR
from models.LCRnet_m3 import CurvePriorDualViewLCR
from models.LCRnet_new import CurvePriorGuidedLCR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised training for the non-DCE parts of CurvePriorGuidedLCR."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/v1"))
    parser.add_argument(
        "--dce_weights",
        type=Path,
        default=Path("checkpoints/Epoch059_PSNR_17.7073.pth"),
        help="State dict of the separately trained EnhanceNetNoPool.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints_curve_gdm"))
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument(
        "--model_variant",
        choices=("full", "m1", "m2", "m3"),
        default="full",
        help=(
            "'full' uses Zero-DCE curve-prior dual-view fusion; "
            "'m1' uses only the original low-light view; "
            "'m2' uses ordinary low/enhanced dual-view fusion without "
            "curve-prior guidance; "
            "'m3' uses curve-prior dual-view cross gating without "
            "detail-denoising refinement."
        ),
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")

    # DualDomainLCRLoss weights.
    parser.add_argument("--target_weight", type=float, default=1.0)
    parser.add_argument("--condition_weight", type=float, default=0.5)
    parser.add_argument("--latent_alignment_weight", type=float, default=0.00)
    parser.add_argument("--latent_variance_weight", type=float, default=0.1)
    parser.add_argument("--latent_variance_target_std", type=float, default=0.05)
    parser.add_argument("--color_weight", type=float, default=0.05)
    parser.add_argument("--perceptual_weight", type=float, default=0.01)

    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Run validation and print PSNR every N epochs.",
    )
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--keep_top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_key(path: Path, root: Path) -> str:
    """Use a case-insensitive relative path without suffix as the pair key."""
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/").lower()


def index_images(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    result: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = image_key(path, root)
        if key in result:
            raise RuntimeError(
                f"Duplicate image key '{key}' in {root}: {result[key]} and {path}"
            )
        result[key] = path
    return result


def build_pairs(low_root: Path, high_root: Path) -> List[Tuple[Path, Path]]:
    low_images = index_images(low_root)
    high_images = index_images(high_root)
    common_keys = sorted(set(low_images) & set(high_images))

    missing_high = sorted(set(low_images) - set(high_images))
    missing_low = sorted(set(high_images) - set(low_images))
    if missing_high or missing_low:
        messages = []
        if missing_high:
            messages.append(
                f"{len(missing_high)} low images have no high pair, e.g. {missing_high[:3]}"
            )
        if missing_low:
            messages.append(
                f"{len(missing_low)} high images have no low pair, e.g. {missing_low[:3]}"
            )
        raise RuntimeError("; ".join(messages))

    if not common_keys:
        raise RuntimeError(f"No paired images found in {low_root} and {high_root}")
    return [(low_images[key], high_images[key]) for key in common_keys]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PairedImageDataset(Dataset):
    def __init__(
        self,
        low_root: Path,
        high_root: Path,
        training: bool,
        crop_size: int,
    ) -> None:
        self.pairs = build_pairs(low_root, high_root)
        self.training = training
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.pairs)

    def _paired_crop_and_augment(
        self,
        low: Image.Image,
        high: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        if low.size != high.size:
            raise ValueError(
                f"Paired images must have the same size, got {low.size} and {high.size}"
            )

        width, height = low.size
        if min(width, height) < self.crop_size:
            scale = self.crop_size / min(width, height)
            new_size = (math.ceil(width * scale), math.ceil(height * scale))
            low = low.resize(new_size, Image.Resampling.BICUBIC)
            high = high.resize(new_size, Image.Resampling.BICUBIC)
            width, height = new_size

        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        box = (left, top, left + self.crop_size, top + self.crop_size)
        low = low.crop(box)
        high = high.crop(box)

        if random.random() < 0.5:
            low = low.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            low = low.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            low = low.transpose(Image.Transpose.ROTATE_90)
            high = high.transpose(Image.Transpose.ROTATE_90)
        return low, high

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_path, high_path = self.pairs[index]
        with Image.open(low_path) as image:
            low = image.convert("RGB")
        with Image.open(high_path) as image:
            high = image.convert("RGB")

        if low.size != high.size:
            raise ValueError(
                f"Size mismatch for pair {low_path} and {high_path}: "
                f"{low.size} versus {high.size}"
            )
        if self.training:
            low, high = self._paired_crop_and_augment(low, high)
        return pil_to_tensor(low), pil_to_tensor(high)


def unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    """Accept a plain DCE state dict or common checkpoint wrappers/prefixes."""
    if not isinstance(checkpoint, dict):
        raise TypeError("DCE checkpoint must be a state-dict-like mapping")

    for key in ("state_dict", "model_state_dict", "model", "dce"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    state_dict = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        if clean_key.startswith("dce."):
            clean_key = clean_key[len("dce.") :]
        state_dict[clean_key] = value
    return state_dict


def load_and_freeze_dce(model: CurvePriorGuidedLCR, weight_path: Path) -> None:
    if not weight_path.is_file():
        raise FileNotFoundError(f"Zero-DCE weights do not exist: {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)
    model.dce.load_state_dict(state_dict, strict=True)
    model.dce.requires_grad_(False)
    model.dce.eval()


def trainable_parameters(model: torch.nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = (pred - target).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = PairedImageDataset(
        args.data_root / "train" / "low",
        args.data_root / "train" / "high",
        training=True,
        crop_size=args.crop_size,
    )
    val_dataset = PairedImageDataset(
        args.data_root / "val" / "low",
        args.data_root / "val" / "high",
        training=False,
        crop_size=args.crop_size,
    )

    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False,
    )
    # Validation images may have different resolutions, so batch size 1 is safest.
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: CurvePriorGuidedLCR,
    criterion: DualDomainLCRLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_clip: float,
    log_interval: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    # The full model owns a frozen DCE. M1 deliberately has no DCE module.
    if hasattr(model, "dce"):
        model.dce.eval()
    criterion.eval()

    totals: Dict[str, float] = {}
    sample_count = 0
    consecutive_nonfinite_batches = 0
    start_time = time.time()

    for batch_index, (low, high) in enumerate(loader, start=1):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        paired = torch.cat((low, high), dim=1)
        batch_size = low.shape[0]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(paired)
            pred_high = outputs["pred_img"]
            pred_condition = outputs["pred_cond_img"]
            condition_feature = outputs["cond_fea"]
            target_feature = outputs["gt_fea"]
            del outputs

        # Keep numerically sensitive reconstruction statistics and VGG
        # features in FP32 even when the model forward uses AMP.
        with torch.autocast(device_type=device.type, enabled=False):
            loss, components = criterion(
                pred_high=pred_high.float(),
                target_high=high,
                pred_condition=pred_condition.float(),
                target_condition=high,
                condition_feature=condition_feature.float(),
                target_feature=target_feature.float(),
                return_components=True,
            )

        if not bool(torch.isfinite(loss)):
            consecutive_nonfinite_batches += 1
            print(
                f"WARNING: skipped non-finite loss at epoch {epoch}, "
                f"batch {batch_index} ({consecutive_nonfinite_batches} consecutive)."
            )
            optimizer.zero_grad(set_to_none=True)
            if consecutive_nonfinite_batches >= 3:
                raise FloatingPointError(
                    "Three consecutive non-finite losses were detected; "
                    "training stopped before corrupting the checkpoint."
                )
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model),
            grad_clip,
        )
        if not bool(torch.isfinite(grad_norm)):
            consecutive_nonfinite_batches += 1
            print(
                f"WARNING: skipped non-finite gradients at epoch {epoch}, "
                f"batch {batch_index} ({consecutive_nonfinite_batches} consecutive)."
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            if consecutive_nonfinite_batches >= 3:
                raise FloatingPointError(
                    "Three consecutive non-finite gradient batches were detected; "
                    "training stopped before corrupting the checkpoint."
                )
            continue

        scaler.step(optimizer)
        scaler.update()
        consecutive_nonfinite_batches = 0

        sample_count += batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(loader):
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:03d}/{total_epochs:03d}] "
                f"Batch [{batch_index:04d}/{len(loader):04d}] "
                f"Loss={float(loss.detach()):.6f} "
                f"Target={float(components['target_reconstruction'].detach()):.6f} "
                f"Condition={float(components['condition_reconstruction'].detach()):.6f} "
                f"Align={float(components['latent_alignment'].detach()):.6f} "
                f"Var={float(components['latent_variance'].detach()):.6f} "
                f"LR={optimizer.param_groups[0]['lr']:.2e} "
                f"Time={elapsed:.1f}s"
            )

    if sample_count == 0:
        raise FloatingPointError("No finite batches were completed in this epoch")
    return {name: value / sample_count for name, value in totals.items()}


@torch.inference_mode()
def validate(
    model: CurvePriorGuidedLCR,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    high_psnr_sum = 0.0
    condition_psnr_sum = 0.0
    image_count = 0

    for low, high in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        paired = torch.cat((low, high), dim=1)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(paired)
            pred_high = outputs["pred_img"]
            pred_condition = outputs["pred_cond_img"]

        if not bool(torch.isfinite(pred_high).all()) or not bool(
            torch.isfinite(pred_condition).all()
        ):
            raise FloatingPointError(
                "Validation produced non-finite predictions; checkpoint saving "
                "was stopped to avoid preserving corrupted weights."
            )

        high_psnr_sum += float(psnr_per_image(pred_high, high).sum())
        condition_psnr_sum += float(
            psnr_per_image(pred_condition, high).sum()
        )
        image_count += low.shape[0]

    return high_psnr_sum / image_count, condition_psnr_sum / image_count


def checkpoint_payload(
    model: CurvePriorGuidedLCR,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    high_psnr: Optional[float],
    condition_psnr: Optional[float],
    args: argparse.Namespace,
    top_high_records: List[Dict],
    top_condition_records: List[Dict],
) -> Dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "high_psnr": high_psnr,
        "condition_psnr": condition_psnr,
        # Keep the old key so checkpoints remain readable by earlier scripts.
        "top_records": top_high_records,
        "top_high_records": top_high_records,
        "top_condition_records": top_condition_records,
        "args": vars(args),
    }


def _high_record_value(record: Dict) -> float:
    """Read both the new field and the legacy ``psnr`` field."""
    if "high_psnr" in record:
        return float(record["high_psnr"])
    return float(record["psnr"])


def _condition_record_value(record: Dict) -> float:
    return float(record["condition_psnr"])


def update_dual_top_k(
    payload: Dict,
    output_dir: Path,
    epoch: int,
    high_psnr: float,
    condition_psnr: float,
    top_high_records: List[Dict],
    top_condition_records: List[Dict],
    keep_top_k: int,
) -> Tuple[List[Dict], List[Dict]]:
    checkpoint_name = (
        f"top_epoch_{epoch:04d}_high_{high_psnr:.4f}"
        f"_condition_{condition_psnr:.4f}.pth"
    )
    checkpoint_path = output_dir / checkpoint_name

    # Capture the old union before replacing a repeated epoch, so a stale file
    # can also be removed if the same epoch is evaluated again.
    previous_paths = {
        str(item["path"])
        for item in list(top_high_records) + list(top_condition_records)
        if item.get("path")
    }
    candidate = {
        "epoch": epoch,
        "high_psnr": float(high_psnr),
        "condition_psnr": float(condition_psnr),
        "path": checkpoint_name,
    }

    # Remove a prior entry for this epoch before inserting the current result.
    high_records = [
        dict(item)
        for item in top_high_records
        if int(item.get("epoch", -1)) != epoch
    ]
    condition_records = [
        dict(item)
        for item in top_condition_records
        if int(item.get("epoch", -1)) != epoch
    ]
    high_records.append(candidate.copy())
    condition_records.append(candidate.copy())

    high_records.sort(key=_high_record_value, reverse=True)
    condition_records.sort(key=_condition_record_value, reverse=True)
    high_records = high_records[:keep_top_k]
    condition_records = condition_records[:keep_top_k]

    # A checkpoint shared by both rankings is written only once and is removed
    # only after it leaves both rankings.
    kept_paths = {
        str(item["path"])
        for item in high_records + condition_records
        if item.get("path")
    }
    if checkpoint_name in kept_paths:
        payload["top_records"] = high_records
        payload["top_high_records"] = high_records
        payload["top_condition_records"] = condition_records
        torch.save(payload, checkpoint_path)

    for removed_name in previous_paths - kept_paths:
        removed_path = output_dir / removed_name
        if removed_path.is_file():
            removed_path.unlink()

    (output_dir / "top5_high.json").write_text(
        json.dumps(high_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Preserve the former manifest name for consumers that expect it.
    (output_dir / "top5.json").write_text(
        json.dumps(high_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "top5_condition.json").write_text(
        json.dumps(condition_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return high_records, condition_records


def main() -> None:
    args = parse_args()
    if args.val_interval <= 0:
        raise ValueError("val_interval must be positive")
    if args.keep_top_k <= 0:
        raise ValueError("keep_top_k must be positive")
    if args.val_batch_size != 1:
        print(
            "Warning: val_batch_size > 1 requires all validation images to have the same size."
        )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}; AMP: {use_amp}")

    train_loader, val_loader = make_loaders(args)
    print(f"Training pairs: {len(train_loader.dataset)}")
    print(f"Validation pairs: {len(val_loader.dataset)}")

    model_classes = {
        "full": CurvePriorGuidedLCR,
        "m1": LowLightSingleViewLCR,
        "m2": OrdinaryDualViewLCR,
        "m3": CurvePriorDualViewLCR,
    }
    model_class = model_classes[args.model_variant]
    model = model_class(base_channels=args.base_channels).to(device)
    start_epoch = 1
    top_high_records: List[Dict] = []
    top_condition_records: List[Dict] = []
    resume_checkpoint = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        top_high_records = list(
            resume_checkpoint.get(
                "top_high_records",
                resume_checkpoint.get("top_records", []),
            )
        )
        top_condition_records = list(
            resume_checkpoint.get("top_condition_records", [])
        )
        if hasattr(model, "dce"):
            model.dce.requires_grad_(False)
            model.dce.eval()
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")
    else:
        if args.model_variant in ("full", "m2", "m3"):
            load_and_freeze_dce(model, args.dce_weights)
            print(f"Loaded and froze Zero-DCE: {args.dce_weights}")
            if args.model_variant == "m2":
                print(
                    "Using M2 ordinary low/enhanced dual-view fusion; "
                    "curve maps are not used by the reconstruction network."
                )
            elif args.model_variant == "m3":
                print(
                    "Using M3 curve-prior dual-view cross gating; "
                    "detail-denoising refinement is disabled."
                )
        else:
            print(
                "Using M1 low-light single-view baseline; "
                "Zero-DCE weights are not loaded."
            )

    criterion = DualDomainLCRLoss(
        target_weight=args.target_weight,
        condition_weight=args.condition_weight,
        latent_alignment_weight=args.latent_alignment_weight,
        latent_variance_weight=args.latent_variance_weight,
        latent_variance_target_std=args.latent_variance_target_std,
        color_weight=args.color_weight,
        perceptual_weight=args.perceptual_weight,
    ).to(device)
    criterion.eval()

    # Zero-DCE is frozen before constructing the optimizer, so its parameters
    # are not included in optimizer parameter groups or optimizer checkpoints.
    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n========== Epoch [{epoch:03d}/{args.epochs:03d}] ==========")
        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            use_amp=use_amp,
        )
        scheduler.step()
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] finished: "
            f"Loss={train_metrics['total']:.6f}, "
            f"Target={train_metrics['target_reconstruction']:.6f}, "
            f"Condition={train_metrics['condition_reconstruction']:.6f}, "
            f"Align={train_metrics['latent_alignment']:.6f}, "
            f"Var={train_metrics['latent_variance']:.6f}"
        )

        high_psnr: Optional[float] = None
        condition_psnr: Optional[float] = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            high_psnr, condition_psnr = validate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
            )
            print(
                f"Validation Epoch [{epoch:03d}/{args.epochs:03d}]: "
                f"High-PSNR={high_psnr:.4f} dB, "
                f"Condition-PSNR={condition_psnr:.4f} dB"
            )

            candidate_payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                high_psnr,
                condition_psnr,
                args,
                top_high_records,
                top_condition_records,
            )
            top_high_records, top_condition_records = update_dual_top_k(
                payload=candidate_payload,
                output_dir=args.output_dir,
                epoch=epoch,
                high_psnr=high_psnr,
                condition_psnr=condition_psnr,
                top_high_records=top_high_records,
                top_condition_records=top_condition_records,
                keep_top_k=args.keep_top_k,
            )
            print(
                f"Current High-PSNR Top-{args.keep_top_k}: "
                + ", ".join(
                    f"epoch {item['epoch']}={_high_record_value(item):.4f}dB"
                    for item in top_high_records
                )
            )
            print(
                f"Current Condition-PSNR Top-{args.keep_top_k}: "
                + ", ".join(
                    f"epoch {item['epoch']}="
                    f"{_condition_record_value(item):.4f}dB"
                    for item in top_condition_records
                )
            )

        latest_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            high_psnr,
            condition_psnr,
            args,
            top_high_records,
            top_condition_records,
        )
        torch.save(latest_payload, args.output_dir / "latest.pth")


if __name__ == "__main__":
    main()
