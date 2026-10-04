"""Evaluate Stage-I LCR ablation models M1--M4 on paired images.

The checkpoints produced by ``train_LCR.py`` contain two Stage-I outputs when
the input is a low/normal-light pair:

``pred_img``
    Oracle reconstruction decoded from the normal-light target latent.

``pred_cond_img``
    Actual Stage-I enhancement decoded only from the low-light condition
    latent.  This is the output evaluated by this script.

Expected dataset layout by default::

    DATA_ROOT/
        test/                 # configurable with --split
            low/
            high/

M4 is the complete first-stage model named ``full`` by ``train_LCR.py``.
Per-image metrics are written to CSV and aggregate means to JSON.  The actual
low-light enhancement and the normal-light latent reconstruction are saved as
separate PNG files; this script never joins them into comparison strips.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.LCR_loss import SSIMLoss
from models.LCRnet_m1 import LowLightSingleViewLCR
from models.LCRnet_m2 import OrdinaryDualViewLCR
from models.LCRnet_m3 import CurvePriorDualViewLCR
from models.LCRnet_new import CurvePriorGuidedLCR


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODEL_CLASSES = {
    "m1": LowLightSingleViewLCR,
    "m2": OrdinaryDualViewLCR,
    "m3": CurvePriorDualViewLCR,
    # train_LCR.py calls the complete first stage "full"; the paper ablation
    # sequence calls the same model M4.
    "m4": CurvePriorGuidedLCR,
}
TRAIN_VARIANT_NAMES = {"m1": "m1", "m2": "m2", "m3": "m3", "m4": "full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an M1/M2/M3/M4 Stage-I checkpoint and save per-image "
            "PSNR, SSIM and LPIPS metrics."
        )
    )
    parser.add_argument("--variant", required=True, choices=tuple(MODEL_CLASSES))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split containing low/ and high/ folders (default: test).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results_lcr_stage1"))
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help=(
            "Model base width. By default it is read from checkpoint args, "
            "falling back to 64 for older checkpoints."
        ),
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA FP16 autocast for model inference. Metrics remain FP32.",
    )
    parser.add_argument(
        "--lpips_net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="LPIPS backbone (default: alex).",
    )
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow_variant_mismatch",
        action="store_true",
        help="Allow loading when checkpoint metadata names another variant.",
    )
    return parser.parse_args()


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
            raise RuntimeError(f"Duplicate image key '{key}' in {root}")
        images[key] = path
    return images


def build_pairs(low_root: Path, high_root: Path) -> List[Tuple[str, Path, Path]]:
    low_images = index_images(low_root)
    high_images = index_images(high_root)
    missing_high = sorted(set(low_images) - set(high_images))
    missing_low = sorted(set(high_images) - set(low_images))
    if missing_high or missing_low:
        details = []
        if missing_high:
            details.append(
                f"{len(missing_high)} low images have no high pair, e.g. {missing_high[:3]}"
            )
        if missing_low:
            details.append(
                f"{len(missing_low)} high images have no low pair, e.g. {missing_low[:3]}"
            )
        raise RuntimeError("; ".join(details))
    keys = sorted(low_images)
    if not keys:
        raise RuntimeError(f"No paired images found in {low_root} and {high_root}")
    return [(key, low_images[key], high_images[key]) for key in keys]


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class PairedEvaluationDataset(Dataset):
    def __init__(self, low_root: Path, high_root: Path) -> None:
        self.low_root = low_root
        self.pairs = build_pairs(low_root, high_root)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        key, low_path, high_path = self.pairs[index]
        with Image.open(low_path) as image:
            low = image.convert("RGB")
        with Image.open(high_path) as image:
            high = image.convert("RGB")
        if low.size != high.size:
            raise ValueError(
                f"Paired image size mismatch for {key}: {low.size} vs {high.size}"
            )
        relative_path = low_path.relative_to(self.low_root).with_suffix(".png")
        return pil_to_tensor(low), pil_to_tensor(high), str(relative_path)


def get_checkpoint_arg(checkpoint: Mapping, name: str, default=None):
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, Mapping):
        return saved_args.get(name, default)
    if saved_args is not None:
        return getattr(saved_args, name, default)
    return default


def extract_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state-dict-like mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key[len("module.") :] if key.startswith("module.") else key
        state_dict[clean_key] = value
    if not state_dict:
        raise RuntimeError("No tensor state dict was found in the checkpoint")
    return state_dict


def make_lpips(device: torch.device, network: str):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS is required. Install it in the active environment with: "
            "pip install lpips"
        ) from exc
    metric = lpips.LPIPS(net=network).to(device).eval()
    metric.requires_grad_(False)
    return metric


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    return np.rint(array * 255.0).astype(np.uint8)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(tensor)).save(path)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def mean_metric(rows: Iterable[Mapping[str, object]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    return float(np.mean(values))


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    device = select_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp is supported only with CUDA")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    expected_train_variant = TRAIN_VARIANT_NAMES[args.variant]
    saved_variant = get_checkpoint_arg(checkpoint, "model_variant")
    if (
        saved_variant is not None
        and str(saved_variant).lower() != expected_train_variant
        and not args.allow_variant_mismatch
    ):
        raise ValueError(
            f"Checkpoint metadata says model_variant={saved_variant!r}, but "
            f"--variant {args.variant} expects {expected_train_variant!r}. "
            "Choose the matching variant or pass --allow_variant_mismatch "
            "only if the metadata is known to be wrong."
        )

    base_channels = args.base_channels
    if base_channels is None:
        base_channels = int(get_checkpoint_arg(checkpoint, "base_channels", 64))

    model = MODEL_CLASSES[args.variant](base_channels=base_channels).to(device)
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    split_root = args.data_root / args.split
    low_root = split_root / "low"
    high_root = split_root / "high"
    dataset = PairedEvaluationDataset(low_root, high_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    lpips_metric = make_lpips(device, args.lpips_net)
    ssim_loss = SSIMLoss(data_range=1.0, reduction="mean").to(device).eval()

    run_dir = args.output_dir / args.variant
    low_enhanced_dir = run_dir / "low_light_enhanced"
    normal_reconstruction_dir = run_dir / "normal_light_reconstructed"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    print(
        f"Evaluating {args.variant.upper()} on {len(dataset)} pairs; "
        f"device={device}, checkpoint={args.checkpoint}"
    )

    with torch.inference_mode():
        for index, (low, high, relative_paths) in enumerate(loader, start=1):
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)
            paired = torch.cat((low, high), dim=1)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp,
            ):
                output = model(paired)
                if "pred_cond_img" not in output or "pred_img" not in output:
                    raise KeyError(
                        "Paired Stage-I evaluation requires both 'pred_cond_img' "
                        "and 'pred_img' in the model output."
                    )
                low_enhancement = output["pred_cond_img"]
                normal_reconstruction = output["pred_img"]

            low_enhancement = low_enhancement.float().clamp(0.0, 1.0)
            normal_reconstruction = normal_reconstruction.float().clamp(0.0, 1.0)
            target = high.float().clamp(0.0, 1.0)

            condition_psnr = float(psnr(low_enhancement, target).item())
            condition_ssim = float(1.0 - ssim_loss(low_enhancement, target).item())
            condition_lpips = float(
                lpips_metric(low_enhancement * 2.0 - 1.0, target * 2.0 - 1.0)
                .flatten()
                .mean()
                .item()
            )
            target_psnr = float(psnr(normal_reconstruction, target).item())
            target_ssim = float(
                1.0 - ssim_loss(normal_reconstruction, target).item()
            )
            target_lpips = float(
                lpips_metric(
                    normal_reconstruction * 2.0 - 1.0,
                    target * 2.0 - 1.0,
                )
                .flatten()
                .mean()
                .item()
            )

            relative_path = Path(relative_paths[0])
            save_tensor_image(low_enhancement[0], low_enhanced_dir / relative_path)
            save_tensor_image(
                normal_reconstruction[0],
                normal_reconstruction_dir / relative_path,
            )

            row = {
                "image": relative_path.as_posix(),
                # Main paper metrics: enhancement reconstructed from the
                # low-light condition latent.
                "psnr": condition_psnr,
                "ssim": condition_ssim,
                "lpips": condition_lpips,
                # Diagnostic upper branch: normal-light target latent decoded
                # by the same guided decoder.
                "normal_reconstruction_psnr": target_psnr,
                "normal_reconstruction_ssim": target_ssim,
                "normal_reconstruction_lpips": target_lpips,
            }
            rows.append(row)
            print(
                f"[{index:04d}/{len(dataset):04d}] {row['image']} "
                f"Low-enhanced: PSNR={condition_psnr:.4f} "
                f"SSIM={condition_ssim:.4f} LPIPS={condition_lpips:.4f}; "
                f"Normal-reconstructed: PSNR={target_psnr:.4f} "
                f"SSIM={target_ssim:.4f} LPIPS={target_lpips:.4f}"
            )

    averages = {
        "psnr": mean_metric(rows, "psnr"),
        "ssim": mean_metric(rows, "ssim"),
        "lpips": mean_metric(rows, "lpips"),
    }
    normal_reconstruction_averages = {
        "psnr": mean_metric(rows, "normal_reconstruction_psnr"),
        "ssim": mean_metric(rows, "normal_reconstruction_ssim"),
        "lpips": mean_metric(rows, "normal_reconstruction_lpips"),
    }
    csv_path = run_dir / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image",
                "psnr",
                "ssim",
                "lpips",
                "normal_reconstruction_psnr",
                "normal_reconstruction_ssim",
                "normal_reconstruction_lpips",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "variant": args.variant,
        "train_variant_name": expected_train_variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": (
            checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
        ),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "num_images": len(rows),
        "base_channels": base_channels,
        "lpips_backbone": args.lpips_net,
        "amp": bool(args.amp),
        "main_metric_branch": "pred_cond_img (low-light enhancement)",
        "averages": averages,
        "normal_reconstruction_averages": normal_reconstruction_averages,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nOverall averages: low-light enhancement (main paper metrics)")
    print(f"PSNR : {averages['psnr']:.4f} dB")
    print(f"SSIM : {averages['ssim']:.4f}")
    print(f"LPIPS: {averages['lpips']:.4f}")
    print("\nOverall averages: normal-light target reconstruction (diagnostic)")
    print(f"PSNR : {normal_reconstruction_averages['psnr']:.4f} dB")
    print(f"SSIM : {normal_reconstruction_averages['ssim']:.4f}")
    print(f"LPIPS: {normal_reconstruction_averages['lpips']:.4f}")
    if args.save_comparison:
        print(
            "Note: --save_comparison is retained only for command-line "
            "compatibility and is ignored; no joined images were created."
        )
    print(f"Low-light enhanced images : {low_enhanced_dir}")
    print(f"Normal-light reconstructions: {normal_reconstruction_dir}")
    print(f"Per-image CSV  : {csv_path}")
    print(f"Summary JSON   : {summary_path}")


if __name__ == "__main__":
    main()
