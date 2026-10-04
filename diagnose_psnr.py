"""Diagnose the three reconstruction stages of LCR + latent diffusion.

Fill the four DEFAULT_* paths below, or pass the corresponding command-line
arguments.  The data list must contain one paired sample per line:

    /path/to/low.png /path/to/high.png

The script reports:
  1. Oracle PSNR:    decode(gt_fea), where gt_fea is encoded from high.
  2. Condition PSNR: decode(cond_fea), without using the diffusion model.
  3. Diffusion PSNR: decode(diffusion(cond_fea)).

All three branches use the same padding, cropping and RGB PSNR implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from models.LCRnet_new import LCR
from models.unet import DiffusionUNet


# ---------------------------------------------------------------------------
# You may fill these paths directly. Command-line arguments override them.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = r"configs/supervised.yml"
DEFAULT_DATA_LIST = r"data/LOL-v1_val.txt"              # e.g. /root/data/LOL_val.txt
DEFAULT_DATA_ROOT = r""              # optional, for relative paths in the list
DEFAULT_LCR_CHECKPOINT = r"checkpoints/v2-s_high_31.5132_condition_24.8246.pth"          # stage-1 checkpoint
DEFAULT_DIFFUSION_CHECKPOINT = r"ckpt/stage2/model_best_psnr.pth.tar"    # stage-2 checkpoint
DEFAULT_OUTPUT_CSV = r"results/psnr_diagnostics.csv"


def dict_to_namespace(value):
    if not isinstance(value, dict):
        return value
    namespace = argparse.Namespace()
    for key, item in value.items():
        setattr(namespace, key, dict_to_namespace(item))
    return namespace


def load_config(path: str, device: torch.device):
    with open(path, "r", encoding="utf-8") as handle:
        config = dict_to_namespace(yaml.safe_load(handle))
    config.device = device
    return config


def torch_load_trusted(path: str):
    """Load a user-owned training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def find_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint)}")

    for key in ("model", "model_state_dict", "state_dict", "net"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate

    if checkpoint and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        return checkpoint

    raise KeyError(
        "Could not find a model state_dict. Available checkpoint keys: "
        f"{list(checkpoint.keys())}"
    )


def strip_leading_module(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def extract_lcr_state(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {strip_leading_module(key): value for key, value in state_dict.items()}

    for prefix in ("LCR.", "model.LCR."):
        selected = {
            key[len(prefix):]: value
            for key, value in cleaned.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected
    return dict(cleaned)


def extract_unet_state(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {strip_leading_module(key): value for key, value in state_dict.items()}

    for prefix in ("Unet.", "model.Unet.", "UNet.", "model.UNet."):
        selected = {
            key[len(prefix):]: value
            for key, value in cleaned.items()
            if key.startswith(prefix)
        }
        if selected:
            return selected

    # A checkpoint containing only DiffusionUNet parameters has keys such as
    # conv_in.weight, temb.dense.0.weight, down.0.block.0.conv1.weight, etc.
    if "conv_in.weight" in cleaned:
        return dict(cleaned)

    sample = list(cleaned.keys())[:10]
    raise KeyError(
        "Could not extract DiffusionUNet parameters from the stage-2 checkpoint. "
        f"First keys: {sample}"
    )


def load_models(args, config, device: torch.device):
    lcr = LCR().to(device)
    lcr_checkpoint = torch_load_trusted(args.lcr_checkpoint)
    lcr_state = extract_lcr_state(find_state_dict(lcr_checkpoint))
    lcr.load_state_dict(lcr_state, strict=True)

    unet = DiffusionUNet(config).to(device)
    diffusion_checkpoint = torch_load_trusted(args.diffusion_checkpoint)

    if args.use_ema:
        ema_state = diffusion_checkpoint.get("ema_helper")
        if not isinstance(ema_state, Mapping):
            raise KeyError("--use-ema was requested, but checkpoint has no ema_helper mapping")
        unet_state = extract_unet_state(ema_state)
    else:
        unet_state = extract_unet_state(find_state_dict(diffusion_checkpoint))

    unet.load_state_dict(unet_state, strict=True)
    lcr.eval()
    unet.eval()
    return lcr, unet


def get_beta_schedule(config) -> torch.Tensor:
    schedule = config.diffusion.beta_schedule
    beta_start = config.diffusion.beta_start
    beta_end = config.diffusion.beta_end
    total = config.diffusion.num_diffusion_timesteps

    if schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, total) ** 2
    elif schedule == "linear":
        betas = np.linspace(beta_start, beta_end, total)
    elif schedule == "const":
        betas = beta_end * np.ones(total)
    elif schedule == "jsd":
        betas = 1.0 / np.linspace(total, 1, total)
    elif schedule == "sigmoid":
        values = np.linspace(-6, 6, total)
        values = 1.0 / (np.exp(-values) + 1.0)
        betas = values * (beta_end - beta_start) + beta_start
    else:
        raise ValueError(f"Unsupported beta schedule: {schedule}")

    return torch.from_numpy(betas).float()


def compute_alpha(beta: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    beta = torch.cat((torch.zeros(1, device=beta.device), beta), dim=0)
    return (1.0 - beta).cumprod(dim=0).index_select(
        0, timesteps + 1
    ).view(-1, 1, 1, 1)



@torch.no_grad()
def sample_latent(
    unet: DiffusionUNet,
    condition_norm: torch.Tensor,
    beta: torch.Tensor,
    total_timesteps: int,
    sampling_timesteps: int,
    eta: float,
) -> torch.Tensor:
    if sampling_timesteps <= 0 or sampling_timesteps > total_timesteps:
        raise ValueError(
            "num_sampling_timesteps must be in [1, num_diffusion_timesteps]"
        )

    skip = total_timesteps // sampling_timesteps
    sequence = list(range(0, total_timesteps, skip))
    next_sequence = [-1] + sequence[:-1]

    batch, channels, height, width = condition_norm.shape
    current = torch.randn(
        batch,
        channels,
        height,
        width,
        device=condition_norm.device,
    )

    for current_step, next_step in zip(
        reversed(sequence), reversed(next_sequence)
    ):
        timestep = torch.full(
            (batch,), current_step, device=current.device, dtype=torch.long
        )
        next_timestep = torch.full(
            (batch,), next_step, device=current.device, dtype=torch.long
        )
        alpha = compute_alpha(beta, timestep)
        alpha_next = compute_alpha(beta, next_timestep)

        noise = unet(
            torch.cat((condition_norm, current), dim=1),
            timestep.float(),
        )
        predicted_x0 = (
            current - noise * torch.sqrt(1.0 - alpha)
        ) / torch.sqrt(alpha)

        coefficient1 = eta * torch.sqrt(
            (1.0 - alpha / alpha_next)
            * (1.0 - alpha_next)
            / (1.0 - alpha)
        )
        coefficient2 = torch.sqrt(
            torch.clamp(1.0 - alpha_next - coefficient1.square(), min=0.0)
        )
        current = (
            torch.sqrt(alpha_next) * predicted_x0
            + coefficient1 * torch.randn_like(current)
            + coefficient2 * noise
        )

    return current


def read_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resolve_image_path(value: str, data_root: Optional[Path], list_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    # Existing dataset lists in this project normally store paths relative to
    # the project working directory (for example ``data/val/low/151.png``),
    # while other lists may be relative to the list file or to --data-root.
    # Try all supported conventions and use the first path that exists.
    candidates = []
    if data_root is not None:
        candidates.append(data_root / path)
        # Avoid data/data/... when data_root already points to "data" and the
        # list entry itself also begins with "data/".
        if path.parts and path.parts[0] == data_root.name:
            candidates.append(data_root.parent / path)
    candidates.extend((Path.cwd() / path, list_dir / path))

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)
        if resolved.is_file():
            return resolved

    attempted = "\n  - ".join(str(item) for item in unique_candidates)
    raise FileNotFoundError(
        f"Image path from data list does not exist: {value}\n"
        f"Tried:\n  - {attempted}"
    )


def iter_pairs(data_list: str, data_root: str) -> Iterable[Tuple[Path, Path]]:
    list_path = Path(data_list).expanduser().resolve()
    root = Path(data_root).expanduser().resolve() if data_root else None

    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"{list_path}:{line_number}: expected 'low_path high_path'"
                )
            yield (
                resolve_image_path(parts[0], root, list_path.parent),
                resolve_image_path(parts[1], root, list_path.parent),
            )


def pad_to_multiple(image: torch.Tensor, multiple: int = 64):
    _, _, height, width = image.shape
    padded_height = math.ceil(height / multiple) * multiple
    padded_width = math.ceil(width / multiple) * multiple
    pad_height = padded_height - height
    pad_width = padded_width - width

    if pad_height == 0 and pad_width == 0:
        return image, (height, width)

    # Reflection padding requires each padding amount to be smaller than the
    # corresponding input dimension. Test images in LOL satisfy this condition.
    mode = "reflect" if pad_height < height and pad_width < width else "replicate"
    return F.pad(image, (0, pad_width, 0, pad_height), mode=mode), (height, width)


def image_psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = F.mse_loss(prediction, target).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-10))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure Oracle, Condition and Diffusion PSNR on paired images"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-list", default=DEFAULT_DATA_LIST)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--lcr-checkpoint", default=DEFAULT_LCR_CHECKPOINT)
    parser.add_argument("--diffusion-checkpoint", default=DEFAULT_DIFFUSION_CHECKPOINT)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=230)
    parser.add_argument(
        "--num-diffusion-samples",
        type=int,
        default=1,
        help="Independent diffusion samples per image; mean PSNR is reported",
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load DiffusionUNet weights from checkpoint['ema_helper']",
    )
    args = parser.parse_args()

    required = {
        "--data-list": args.data_list,
        "--lcr-checkpoint": args.lcr_checkpoint,
        "--diffusion-checkpoint": args.diffusion_checkpoint,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(
            "Fill the DEFAULT_* paths or provide required arguments: "
            + ", ".join(missing)
        )
    if args.num_diffusion_samples < 1:
        parser.error("--num-diffusion-samples must be at least 1")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    config = load_config(args.config, device)
    lcr, unet = load_models(args, config, device)
    beta = get_beta_schedule(config).to(device)

    pairs = list(iter_pairs(args.data_list, args.data_root))
    if not pairs:
        raise RuntimeError(f"No image pairs found in {args.data_list}")

    rows = []
    oracle_scores = []
    condition_scores = []
    diffusion_scores = []
    condition_latent_l1_scores = []
    diffusion_latent_l1_scores = []

    with torch.inference_mode():
        for index, (low_path, high_path) in enumerate(pairs):
            low = read_rgb_tensor(low_path).unsqueeze(0)
            high = read_rgb_tensor(high_path).unsqueeze(0)
            if low.shape != high.shape:
                raise ValueError(
                    f"Low/high shape mismatch for {low_path} and {high_path}: "
                    f"{tuple(low.shape)} vs {tuple(high.shape)}"
                )

            original_high = high.to(device)
            low_padded, (height, width) = pad_to_multiple(low)
            high_padded, _ = pad_to_multiple(high)
            low_padded = low_padded.to(device)
            high_padded = high_padded.to(device)
            paired_input = torch.cat((low_padded, high_padded), dim=1)

            stage1 = lcr(paired_input, pred_fea=None)
            condition_feature = stage1["cond_fea"]
            target_feature = stage1["gt_fea"]

            oracle_image = stage1["pred_img"][:, :, :height, :width]
            condition_image = lcr(
                paired_input, pred_fea=condition_feature
            )["pred_img"][:, :, :height, :width]

            oracle_score = image_psnr(oracle_image, original_high)
            condition_score = image_psnr(condition_image, original_high)
            condition_latent_l1 = F.l1_loss(
                condition_feature, target_feature
            ).item()

            per_sample_psnr = []
            per_sample_latent_l1 = []
            condition_norm = 2.0 * condition_feature - 1.0

            for sample_index in range(args.num_diffusion_samples):
                # Reproducible but independent noise for every image/sample pair.
                set_seed(args.seed + index * args.num_diffusion_samples + sample_index)
                predicted_norm = sample_latent(
                    unet,
                    condition_norm,
                    beta,
                    config.diffusion.num_diffusion_timesteps,
                    config.diffusion.num_sampling_timesteps,
                    args.eta,
                )
                predicted_feature = torch.clamp(
                    (predicted_norm + 1.0) / 2.0, 0.0, 1.0
                )
                diffusion_image = lcr(
                    paired_input, pred_fea=predicted_feature
                )["pred_img"][:, :, :height, :width]
                per_sample_psnr.append(
                    image_psnr(diffusion_image, original_high)
                )
                per_sample_latent_l1.append(
                    F.l1_loss(predicted_feature, target_feature).item()
                )

            diffusion_score = float(np.mean(per_sample_psnr))
            diffusion_score_std = float(np.std(per_sample_psnr))
            diffusion_latent_l1 = float(np.mean(per_sample_latent_l1))

            oracle_scores.append(oracle_score)
            condition_scores.append(condition_score)
            diffusion_scores.append(diffusion_score)
            condition_latent_l1_scores.append(condition_latent_l1)
            diffusion_latent_l1_scores.append(diffusion_latent_l1)

            row = {
                "low_path": str(low_path),
                "high_path": str(high_path),
                "height": height,
                "width": width,
                "oracle_psnr": oracle_score,
                "condition_psnr": condition_score,
                "diffusion_psnr_mean": diffusion_score,
                "diffusion_psnr_std": diffusion_score_std,
                "condition_gt_latent_l1": condition_latent_l1,
                "diffusion_gt_latent_l1": diffusion_latent_l1,
            }
            rows.append(row)
            print(
                f"[{index + 1:03d}/{len(pairs):03d}] {low_path.name} | "
                f"oracle={oracle_score:.4f} "
                f"condition={condition_score:.4f} "
                f"diffusion={diffusion_score:.4f}±{diffusion_score_std:.4f} | "
                f"latent L1: cond={condition_latent_l1:.5f} "
                f"diff={diffusion_latent_l1:.5f}"
            )

    output_path = Path(args.output_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n========== Diagnostic summary ==========")
    print(f"images:                   {len(rows)}")
    print(f"Oracle PSNR:              {np.mean(oracle_scores):.4f} dB")
    print(f"Condition PSNR:           {np.mean(condition_scores):.4f} dB")
    print(f"Diffusion PSNR:           {np.mean(diffusion_scores):.4f} dB")
    print(f"Condition -> GT latent L1:{np.mean(condition_latent_l1_scores):.6f}")
    print(f"Diffusion -> GT latent L1:{np.mean(diffusion_latent_l1_scores):.6f}")
    print(f"CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
