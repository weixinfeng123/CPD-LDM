"""Supervised training for EnhanceNetNoPool on paired LOLv1 images.

Expected data layout::

    data/
        train/
            low/
            high/
        val/
            low/
            high/

Low/high images are paired by their case-insensitive relative paths without
file extensions.  The best plain state dict produced by this script can be
passed directly to ``train_LCR.py --dce_weights``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.EnhanceNetNoPool import EnhanceNetNoPool
from models.EnhanceNetNoPool_loss import SSIMLoss, SupervisedEnhanceNetLoss


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised EnhanceNetNoPool training on paired LOLv1 images."
    )
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("checkpoints_dce_supervised"),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler and scaler from a training checkpoint.",
    )
    parser.add_argument(
        "--init_weights",
        type=Path,
        default=None,
        help="Optional plain/wrapped DCE weights used only for initialization.",
    )

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--feature_channels", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")

    # SupervisedEnhanceNetLoss weights.
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.2)
    parser.add_argument("--gradient_weight", type=float, default=0.1)
    parser.add_argument("--color_weight", type=float, default=0.05)
    parser.add_argument("--intermediate_weight", type=float, default=0.25)
    parser.add_argument("--curve_smoothness_weight", type=float, default=0.01)
    parser.add_argument("--range_weight", type=float, default=0.05)

    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--keep_top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "epochs",
        "batch_size",
        "crop_size",
        "val_interval",
        "save_interval",
        "log_interval",
        "keep_top_k",
        "feature_channels",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.lr <= 0 or args.min_lr < 0:
        raise ValueError("Learning rates must satisfy lr > 0 and min_lr >= 0")
    if args.grad_clip <= 0:
        raise ValueError("--grad_clip must be positive")
    if args.resume is not None and args.init_weights is not None:
        raise ValueError("--resume and --init_weights cannot be used together")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def image_key(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/").lower()


def index_images(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")

    images: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = image_key(path, root)
        if key in images:
            raise RuntimeError(
                f"Duplicate image key '{key}' in {root}: {images[key]} and {path}"
            )
        images[key] = path
    return images


def build_pairs(low_root: Path, high_root: Path) -> List[Tuple[Path, Path]]:
    low_images = index_images(low_root)
    high_images = index_images(high_root)
    low_keys = set(low_images)
    high_keys = set(high_images)

    missing_high = sorted(low_keys - high_keys)
    missing_low = sorted(high_keys - low_keys)
    if missing_high or missing_low:
        messages = []
        if missing_high:
            messages.append(
                f"{len(missing_high)} low images have no high pair, e.g. "
                f"{missing_high[:3]}"
            )
        if missing_low:
            messages.append(
                f"{len(missing_low)} high images have no low pair, e.g. "
                f"{missing_low[:3]}"
            )
        raise RuntimeError("; ".join(messages))

    common_keys = sorted(low_keys & high_keys)
    if not common_keys:
        raise RuntimeError(f"No paired images found in {low_root} and {high_root}")
    return [(low_images[key], high_images[key]) for key in common_keys]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class LOLPairedDataset(Dataset):
    def __init__(
        self,
        low_root: Path,
        high_root: Path,
        training: bool,
        crop_size: int,
    ) -> None:
        self.pairs = build_pairs(low_root, high_root)
        self.training = bool(training)
        self.crop_size = int(crop_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def paired_crop_and_augment(
        self,
        low: Image.Image,
        high: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        width, height = low.size
        if min(width, height) < self.crop_size:
            scale = self.crop_size / min(width, height)
            resized = (math.ceil(width * scale), math.ceil(height * scale))
            low = low.resize(resized, Image.Resampling.BICUBIC)
            high = high.resize(resized, Image.Resampling.BICUBIC)
            width, height = resized

        left = random.randint(0, width - self.crop_size)
        top = random.randint(0, height - self.crop_size)
        crop_box = (left, top, left + self.crop_size, top + self.crop_size)
        low = low.crop(crop_box)
        high = high.crop(crop_box)

        if random.random() < 0.5:
            low = low.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            low = low.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rotations = random.randint(0, 3)
        for _ in range(rotations):
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
                f"Paired image size mismatch: {low_path}={low.size}, "
                f"{high_path}={high.size}"
            )
        if self.training:
            low, high = self.paired_crop_and_augment(low, high)
        return pil_to_tensor(low), pil_to_tensor(high)


def make_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = LOLPairedDataset(
        args.data_root / "train" / "low",
        args.data_root / "train" / "high",
        training=True,
        crop_size=args.crop_size,
    )
    val_dataset = LOLPairedDataset(
        args.data_root / "val" / "low",
        args.data_root / "val" / "high",
        training=False,
        crop_size=args.crop_size,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
        **common,
    )
    # LOLv1 validation images are evaluated at original resolution.  A batch
    # size of one also supports validation sets with mixed image dimensions.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def find_model_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state-dict-like mapping")

    candidate: Mapping = checkpoint
    for key in ("model_state_dict", "state_dict", "model", "dce"):
        value = candidate.get(key)
        if isinstance(value, Mapping):
            candidate = value
            break

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        for prefix in ("module.", "dce."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    if not cleaned:
        raise KeyError("No model tensors were found in the checkpoint")
    return cleaned


def atomic_torch_save(payload: object, path: Path) -> None:
    """Write to a temporary sibling and atomically replace the destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def psnr_per_image(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = (prediction - target).square().flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def train_one_epoch(
    model: EnhanceNetNoPool,
    criterion: SupervisedEnhanceNetLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_clip: float,
    log_interval: int,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    sample_count = 0
    epoch_start = time.time()
    consecutive_nonfinite = 0

    for batch_index, (low, high) in enumerate(loader, start=1):
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        batch_size = low.shape[0]
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            enhanced_mid, enhanced, curve_maps = model(low)

        # SSIM and small residual statistics are more stable in FP32.
        with torch.autocast(device_type=device.type, enabled=False):
            loss, components = criterion(
                enhanced_mid.float(),
                enhanced.float(),
                curve_maps.float(),
                high.float(),
                return_components=True,
            )

        if not bool(torch.isfinite(loss)):
            consecutive_nonfinite += 1
            print(
                f"WARNING: skipped non-finite loss at epoch {epoch}, "
                f"batch {batch_index}."
            )
            if consecutive_nonfinite >= 3:
                raise FloatingPointError("Three consecutive non-finite losses detected")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            consecutive_nonfinite += 1
            if consecutive_nonfinite >= 3:
                raise FloatingPointError("Three consecutive non-finite gradients detected")
            continue

        scaler.step(optimizer)
        scaler.update()
        consecutive_nonfinite = 0

        sample_count += batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(loader):
            print(
                f"Epoch [{epoch:03d}/{total_epochs:03d}] "
                f"Batch [{batch_index:04d}/{len(loader):04d}] "
                f"Loss={float(loss.detach()):.6f} "
                f"Recon={float(components['reconstruction']):.6f} "
                f"SSIM={float(components['ssim']):.6f} "
                f"Grad={float(components['gradient']):.6f} "
                f"Mid={float(components['intermediate']):.6f} "
                f"CurveTV={float(components['curve_smoothness']):.6f} "
                f"LR={optimizer.param_groups[0]['lr']:.2e} "
                f"Time={time.time() - epoch_start:.1f}s"
            )

    if sample_count == 0:
        raise FloatingPointError("No finite training batches completed in this epoch")
    return {name: total / sample_count for name, total in totals.items()}


@torch.inference_mode()
def validate(
    model: EnhanceNetNoPool,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    ssim_loss = SSIMLoss().to(device)
    psnr_sum = 0.0
    ssim_sum = 0.0
    image_count = 0

    for low, high in loader:
        low = low.to(device, non_blocking=True)
        high = high.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            _, enhanced, _ = model(low)

        if not bool(torch.isfinite(enhanced).all()):
            raise FloatingPointError("Validation produced non-finite enhanced images")

        enhanced_metric = enhanced.float().clamp(0.0, 1.0)
        psnr_sum += float(psnr_per_image(enhanced_metric, high).sum())
        ssim_value = 1.0 - ssim_loss(enhanced_metric, high.float())
        ssim_sum += float(ssim_value) * low.shape[0]
        image_count += low.shape[0]

    if image_count == 0:
        raise RuntimeError("Validation loader is empty")
    return psnr_sum / image_count, ssim_sum / image_count


def checkpoint_payload(
    model: EnhanceNetNoPool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_psnr: float,
    val_psnr: Optional[float],
    val_ssim: Optional[float],
    top_records: List[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_psnr": best_psnr,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "top_records": top_records,
        "args": vars(args),
    }


def update_top_k(
    payload: Dict[str, object],
    output_dir: Path,
    epoch: int,
    psnr: float,
    records: List[Dict[str, object]],
    keep_top_k: int,
) -> List[Dict[str, object]]:
    filename = f"top_epoch_{epoch:04d}_psnr_{psnr:.4f}.pth"
    updated = list(records)
    updated.append({"epoch": epoch, "psnr": psnr, "path": filename})
    updated.sort(key=lambda item: float(item["psnr"]), reverse=True)
    kept = updated[:keep_top_k]
    removed = updated[keep_top_k:]

    if any(item["path"] == filename for item in kept):
        payload["top_records"] = kept
        atomic_torch_save(payload, output_dir / filename)

    for item in removed:
        path = output_dir / str(item["path"])
        if path.is_file():
            path.unlink()
    atomic_write_json(kept, output_dir / "top_k.json")
    return kept


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"Device: {device}; AMP: {use_amp}")

    train_loader, val_loader = make_loaders(args)
    print(f"Training pairs: {len(train_loader.dataset)}")
    print(f"Validation pairs: {len(val_loader.dataset)}")

    model = EnhanceNetNoPool(feature_channels=args.feature_channels).to(device)
    criterion = SupervisedEnhanceNetLoss(
        reconstruction_weight=args.reconstruction_weight,
        ssim_weight=args.ssim_weight,
        gradient_weight=args.gradient_weight,
        color_weight=args.color_weight,
        intermediate_weight=args.intermediate_weight,
        curve_smoothness_weight=args.curve_smoothness_weight,
        range_weight=args.range_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_psnr = float("-inf")
    top_records: List[Dict[str, object]] = []

    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(find_model_state_dict(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_psnr = float(checkpoint.get("best_psnr", float("-inf")))
        top_records = list(checkpoint.get("top_records", []))
        print(f"Resumed from epoch {start_epoch - 1}: {args.resume}")
    elif args.init_weights is not None:
        if not args.init_weights.is_file():
            raise FileNotFoundError(f"Initial weights do not exist: {args.init_weights}")
        checkpoint = torch.load(args.init_weights, map_location="cpu", weights_only=False)
        model.load_state_dict(find_model_state_dict(checkpoint), strict=True)
        print(f"Initialized model from: {args.init_weights}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n========== Epoch [{epoch:03d}/{args.epochs:03d}] ==========")
        metrics = train_one_epoch(
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
            f"Loss={metrics['total']:.6f}, "
            f"Recon={metrics['reconstruction']:.6f}, "
            f"SSIM={metrics['ssim']:.6f}, "
            f"Grad={metrics['gradient']:.6f}, "
            f"Mid={metrics['intermediate']:.6f}, "
            f"CurveTV={metrics['curve_smoothness']:.6f}"
        )

        val_psnr: Optional[float] = None
        val_ssim: Optional[float] = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            val_psnr, val_ssim = validate(model, val_loader, device, use_amp)
            is_best = val_psnr > best_psnr
            if is_best:
                best_psnr = val_psnr
            print(
                f"Validation Epoch [{epoch:03d}/{args.epochs:03d}]: "
                f"PSNR={val_psnr:.4f} dB, SSIM={val_ssim:.5f}, "
                f"Best={best_psnr:.4f} dB"
            )

            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_psnr,
                val_psnr,
                val_ssim,
                top_records,
                args,
            )
            top_records = update_top_k(
                payload,
                args.output_dir,
                epoch,
                val_psnr,
                top_records,
                args.keep_top_k,
            )

            if is_best:
                payload["top_records"] = top_records
                atomic_torch_save(payload, args.output_dir / "best_checkpoint.pth")
                # Plain state dict: directly compatible with train_LCR.py.
                atomic_torch_save(model.state_dict(), args.output_dir / "best_weights.pth")
                print(
                    f"Saved new best DCE weights: "
                    f"{(args.output_dir / 'best_weights.pth').resolve()}"
                )
            print(
                "Current Top-K: "
                + ", ".join(
                    f"epoch {item['epoch']}={float(item['psnr']):.4f}dB"
                    for item in top_records
                )
            )

        latest = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_psnr,
            val_psnr,
            val_ssim,
            top_records,
            args,
        )
        atomic_torch_save(latest, args.output_dir / "latest.pth")
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            atomic_torch_save(
                latest,
                args.output_dir / f"checkpoint_epoch_{epoch:04d}.pth",
            )


if __name__ == "__main__":
    main()
