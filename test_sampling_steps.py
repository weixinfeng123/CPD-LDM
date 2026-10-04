"""Evaluate diffusion sampling steps with multiple fixed random seeds.

The script reuses one trained checkpoint and evaluates each requested sampling
step count with three (or more) fixed seeds.  It saves enhanced images,
per-image metrics, per-run summaries, and an aggregate mean/std table for
PSNR, SSIM, LPIPS, and inference time.

Example:

    python test_sampling_steps.py \1
        --config configs/supervised.yml \
        --resume ckpt/stage2/model_best_psnr.pth.tar \
        --sampling-steps 5 10 20 50 \
        --seeds 230 3407 2024 \
        --output-dir results/sampling_steps_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import utils
from datasets.dataset import AllWeatherDataset
from models import DenoisingDiffusion
from models.LCR_loss import SSIMLoss


DEFAULT_STEPS = (5, 10, 20, 50)
DEFAULT_SEEDS = (230, 3407, 2024)


def dict_to_namespace(value):
    if not isinstance(value, dict):
        return value
    namespace = argparse.Namespace()
    for key, item in value.items():
        setattr(namespace, key, dict_to_namespace(item))
    return namespace


def resolve_config_path(value: str) -> Path:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()

    under_configs = Path("configs") / direct
    if under_configs.is_file():
        return under_configs.resolve()

    raise FileNotFoundError(
        f"Configuration file does not exist: {value} "
        f"(also tried {under_configs})"
    )


def load_config(path: Path, device: torch.device):
    with path.open("r", encoding="utf-8") as handle:
        config = dict_to_namespace(yaml.safe_load(handle))
    config.device = device
    # The experiment requires one reproducible noise realization per image.
    config.sampling.batch_size = 1
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one diffusion checkpoint at multiple sampling step counts "
            "and aggregate PSNR/SSIM/LPIPS over fixed random seeds."
        )
    )
    parser.add_argument("--config", default="configs/supervised.yml")
    parser.add_argument(
        "--resume",
        required=True,
        help="Stage-II checkpoint used for every sampling-step/seed run.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sampling_steps_ablation",
        help="Root directory for images, CSV files, and JSON summaries.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="Sampling step counts to test (default: 5 10 20 50).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Fixed random seeds (default: 230 3407 2024).",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=("alex", "vgg", "squeeze"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Untimed warm-up inferences before each sampling-step setting.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Apply checkpoint EMA weights before evaluation.",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-images",
        dest="save_images",
        action="store_true",
        help="Save every enhanced image (default).",
    )
    save_group.add_argument(
        "--no-save-images",
        dest="save_images",
        action="store_false",
        help="Save only metric files and summaries.",
    )
    parser.set_defaults(save_images=True)

    args = parser.parse_args()
    if not args.sampling_steps:
        parser.error("--sampling-steps must contain at least one value")
    if not args.seeds:
        parser.error("--seeds must contain at least one value")
    if len(set(args.sampling_steps)) != len(args.sampling_steps):
        parser.error("--sampling-steps contains duplicate values")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicate values")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not Path(args.resume).expanduser().is_file():
        parser.error(f"--resume checkpoint does not exist: {args.resume}")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_validation_loader(config) -> DataLoader:
    filelist = f"{config.data.val_dataset}_val.txt"
    dataset = AllWeatherDataset(
        config.data.data_dir,
        patch_size=config.data.patch_size,
        filelist=filelist,
        train=False,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def pad_to_multiple(image: torch.Tensor, multiple: int = 64) -> Tuple[torch.Tensor, int, int]:
    _, _, height, width = image.shape
    padded_height = math.ceil(height / multiple) * multiple
    padded_width = math.ceil(width / multiple) * multiple
    pad_height = padded_height - height
    pad_width = padded_width - width
    if pad_height == 0 and pad_width == 0:
        return image, height, width

    mode = "reflect" if pad_height < height and pad_width < width else "replicate"
    padded = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
    return padded, height, width


def prepare_batch(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, List[str]]:
    images, names = batch
    if images.ndim != 4 or images.shape[1] < 6:
        raise ValueError(
            "Paired evaluation expects at least six channels [low, high], "
            f"got {tuple(images.shape)}"
        )
    if images.shape[0] != 1:
        raise ValueError(
            "Sampling-step evaluation requires batch size 1 for per-image seeds, "
            f"got batch size {images.shape[0]}"
        )

    low = images[:, :3].to(device, non_blocking=True)
    target = images[:, 3:6].to(device, non_blocking=True)
    low, height, width = pad_to_multiple(low, multiple=64)
    model_input = torch.cat((low, low), dim=1)
    return model_input, target, height, width, [str(name) for name in names]


class MetricComputer:
    def __init__(self, device: torch.device, lpips_net: str) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is required. Install it with: pip install lpips"
            ) from exc

        self.lpips_net = lpips_net
        self.lpips = lpips.LPIPS(net=lpips_net).to(device).eval()
        self.lpips.requires_grad_(False)
        self.ssim_loss = SSIMLoss(
            data_range=1.0,
            reduction="none",
        ).to(device).eval()

    @torch.inference_mode()
    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        prediction = prediction.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes differ: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )

        mse = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        ssim = 1.0 - self.ssim_loss(
            prediction,
            target,
        ).flatten(start_dim=1).mean(dim=1)
        perceptual = self.lpips(
            prediction * 2.0 - 1.0,
            target * 2.0 - 1.0,
        ).flatten(start_dim=1).mean(dim=1)
        return {
            "psnr": float(psnr[0].item()),
            "ssim": float(ssim[0].item()),
            "lpips": float(perceptual[0].item()),
        }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


@torch.inference_mode()
def warm_up(
    model: torch.nn.Module,
    first_batch,
    device: torch.device,
    count: int,
    step_count: int,
) -> None:
    if count == 0:
        return
    model_input, _, _, _, _ = prepare_batch(first_batch, device)
    for index in range(count):
        set_seed(900_000 + step_count * 100 + index)
        model(model_input)
    synchronize(device)


@torch.inference_mode()
def evaluate_one_seed(
    model: torch.nn.Module,
    loader: DataLoader,
    metric_computer: MetricComputer,
    device: torch.device,
    output_dir: Path,
    sampling_steps: int,
    actual_sampling_steps: int,
    seed: int,
    save_images: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    image_dir = output_dir / "images"
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    for image_index, batch in enumerate(loader):
        model_input, target, height, width, image_names = prepare_batch(batch, device)

        # Reset for every image so corresponding images receive the same initial
        # noise across different sampling-step settings.
        image_seed = seed + image_index
        set_seed(image_seed)
        synchronize(device)
        start = time.perf_counter()
        prediction = model(model_input)["pred_x"][:, :, :height, :width]
        synchronize(device)
        elapsed = time.perf_counter() - start

        prediction = prediction.clamp(0.0, 1.0)
        metrics = metric_computer.compute(prediction, target)
        image_name = image_names[0]
        row: Dict[str, object] = {
            "sampling_steps": sampling_steps,
            "actual_sampling_steps": actual_sampling_steps,
            "seed": seed,
            "image_seed": image_seed,
            "image": image_name,
            "psnr": metrics["psnr"],
            "ssim": metrics["ssim"],
            "lpips": metrics["lpips"],
            "inference_time_seconds": elapsed,
        }
        rows.append(row)

        if save_images:
            safe_name = Path(image_name).name
            utils.logging.save_image(prediction, str(image_dir / safe_name))

        print(
            f"steps={sampling_steps:3d} seed={seed:5d} "
            f"[{image_index + 1:03d}/{len(loader):03d}] {image_name} | "
            f"PSNR={metrics['psnr']:.4f} SSIM={metrics['ssim']:.4f} "
            f"LPIPS={metrics['lpips']:.4f} time={elapsed:.4f}s"
        )

    summary: Dict[str, object] = {
        "sampling_steps": sampling_steps,
        "actual_sampling_steps": actual_sampling_steps,
        "seed": seed,
        "num_images": len(rows),
        "average_psnr": float(np.mean([float(row["psnr"]) for row in rows])),
        "average_ssim": float(np.mean([float(row["ssim"]) for row in rows])),
        "average_lpips": float(np.mean([float(row["lpips"]) for row in rows])),
        "average_inference_time_seconds": float(
            np.mean([float(row["inference_time_seconds"]) for row in rows])
        ),
    }
    return rows, summary


def aggregate_runs(run_summaries: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = {}
    for summary in run_summaries:
        steps = int(summary["sampling_steps"])
        grouped.setdefault(steps, []).append(summary)

    aggregate_rows: List[Dict[str, object]] = []
    for steps in sorted(grouped):
        rows = grouped[steps]

        def values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

        psnr = values("average_psnr")
        ssim = values("average_ssim")
        perceptual = values("average_lpips")
        runtime = values("average_inference_time_seconds")
        aggregate_rows.append(
            {
                "sampling_steps": steps,
                "actual_sampling_steps": int(rows[0]["actual_sampling_steps"]),
                "num_seeds": len(rows),
                "mean_psnr": float(psnr.mean()),
                "std_psnr_across_seeds": float(psnr.std(ddof=0)),
                "mean_ssim": float(ssim.mean()),
                "std_ssim_across_seeds": float(ssim.std(ddof=0)),
                "mean_lpips": float(perceptual.mean()),
                "std_lpips_across_seeds": float(perceptual.std(ddof=0)),
                "mean_inference_time_seconds": float(runtime.mean()),
                "std_inference_time_across_seeds": float(runtime.std(ddof=0)),
            }
        )
    return aggregate_rows


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    checkpoint_path = Path(args.resume).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = load_config(config_path, device)
    total_timesteps = int(config.diffusion.num_diffusion_timesteps)
    for steps in args.sampling_steps:
        if steps <= 0 or steps > total_timesteps:
            raise ValueError(
                f"Sampling steps must be in [1, {total_timesteps}], got {steps}"
            )

    loader = build_validation_loader(config)
    if len(loader) == 0:
        raise RuntimeError(f"No validation images for {config.data.val_dataset}")

    # DenoisingDiffusion expects these evaluation-time attributes.
    args.mode = "evaluation"
    args.image_folder = str(output_root)
    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(str(checkpoint_path), ema=args.use_ema)
    model = diffusion.model.eval()
    metric_computer = MetricComputer(device, args.lpips_net)

    experiment_metadata = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "dataset": str(config.data.val_dataset),
        "num_diffusion_timesteps": total_timesteps,
        "sampling_steps": list(args.sampling_steps),
        "seeds": list(args.seeds),
        "lpips_backbone": args.lpips_net,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "warmup_inferences": args.warmup,
        "save_images": args.save_images,
        "seed_policy": "image_seed = run_seed + zero_based_image_index",
        "metric_aggregation": "mean of per-image metrics, followed by mean/std across seeds",
    }
    write_json(output_root / "experiment_metadata.json", experiment_metadata)

    first_batch = next(iter(loader))
    run_summaries: List[Dict[str, object]] = []
    for sampling_steps in args.sampling_steps:
        config.diffusion.num_sampling_timesteps = sampling_steps
        skip = total_timesteps // sampling_steps
        actual_sampling_steps = len(range(0, total_timesteps, skip))
        if actual_sampling_steps != sampling_steps:
            print(
                f"[warning] requested {sampling_steps} steps but the current "
                f"integer-skip sampler executes {actual_sampling_steps} steps"
            )

        print(f"\nWarm-up for sampling_steps={sampling_steps} ...")
        warm_up(model, first_batch, device, args.warmup, sampling_steps)

        for seed in args.seeds:
            run_dir = (
                output_root
                / str(config.data.val_dataset)
                / f"steps_{sampling_steps:03d}"
                / f"seed_{seed}"
            )
            print(
                f"\n========== steps={sampling_steps}, seed={seed} =========="
            )
            rows, summary = evaluate_one_seed(
                model=model,
                loader=loader,
                metric_computer=metric_computer,
                device=device,
                output_dir=run_dir,
                sampling_steps=sampling_steps,
                actual_sampling_steps=actual_sampling_steps,
                seed=seed,
                save_images=args.save_images,
            )
            summary.update(
                {
                    "dataset": str(config.data.val_dataset),
                    "lpips_backbone": args.lpips_net,
                    "checkpoint": str(checkpoint_path),
                }
            )
            write_csv(run_dir / "per_image_metrics.csv", rows)
            write_json(run_dir / "run_summary.json", summary)
            run_summaries.append(summary)

            print(
                "Run average: "
                f"PSNR={summary['average_psnr']:.4f} dB, "
                f"SSIM={summary['average_ssim']:.4f}, "
                f"LPIPS={summary['average_lpips']:.4f}, "
                f"time={summary['average_inference_time_seconds']:.4f}s/image"
            )

    write_csv(output_root / "all_seed_runs.csv", run_summaries)
    aggregate_rows = aggregate_runs(run_summaries)
    write_csv(output_root / "sampling_steps_summary.csv", aggregate_rows)
    write_json(output_root / "sampling_steps_summary.json", aggregate_rows)

    print("\n================ Aggregate over seeds ================")
    for row in aggregate_rows:
        print(
            f"steps={int(row['sampling_steps']):3d} | "
            f"PSNR={row['mean_psnr']:.4f}+-{row['std_psnr_across_seeds']:.4f} | "
            f"SSIM={row['mean_ssim']:.4f}+-{row['std_ssim_across_seeds']:.4f} | "
            f"LPIPS={row['mean_lpips']:.4f}+-{row['std_lpips_across_seeds']:.4f} | "
            f"time={row['mean_inference_time_seconds']:.4f}s/image"
        )
    print(f"\nSaved aggregate CSV : {output_root / 'sampling_steps_summary.csv'}")
    print(f"Saved aggregate JSON: {output_root / 'sampling_steps_summary.json'}")


if __name__ == "__main__":
    main()
