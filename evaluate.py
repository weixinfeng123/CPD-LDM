import argparse
import csv
import json
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from models.LCR_loss import SSIMLoss


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models')
    parser.add_argument("--config", default='supervised.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='evaluation', help='training or evaluation')
    # parser.add_argument('--resume', default='ckpt/stage2/model_epoch_0200.pth.tar', type=str,
    #                     help='Path for the diffusion mls
    #                     odel checkpoint to load for evaluation')
    parser.add_argument('--resume', default='ckpt/stage2/V2/5scc_psnr25.0888.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--lpips_net', default='alex', choices=['alex', 'vgg', 'squeeze'],
                        help='Backbone used by LPIPS (default: alex)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class ImageQualityEvaluator:
    """Accumulate paired PSNR, SSIM and LPIPS values during restoration."""

    def __init__(self, device, lpips_net='alex'):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is required for evaluation. Install it with: pip install lpips"
            ) from exc

        self.device = device
        self.lpips_net = lpips_net
        self.lpips = lpips.LPIPS(net=lpips_net).to(device).eval()
        self.lpips.requires_grad_(False)
        self.ssim_loss = SSIMLoss(data_range=1.0, reduction='none').to(device).eval()
        self.rows = []

    @staticmethod
    def _psnr(prediction, target):
        mse = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
        return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))

    @torch.inference_mode()
    def update(self, prediction, target, image_names):
        prediction = prediction.detach().float().clamp(0.0, 1.0)
        target = target.detach().float().clamp(0.0, 1.0)
        if prediction.shape != target.shape:
            raise ValueError(
                'Prediction and reference must have the same shape, got '
                f'{tuple(prediction.shape)} and {tuple(target.shape)}'
            )

        psnr_values = self._psnr(prediction, target)
        ssim_values = 1.0 - self.ssim_loss(prediction, target).flatten(start_dim=1).mean(dim=1)
        lpips_values = self.lpips(
            prediction * 2.0 - 1.0,
            target * 2.0 - 1.0,
        ).flatten(start_dim=1).mean(dim=1)

        if isinstance(image_names, str):
            image_names = [image_names]
        if len(image_names) != prediction.shape[0]:
            raise ValueError(
                f'Got {len(image_names)} names for a batch of {prediction.shape[0]} images'
            )

        batch_rows = []
        for index, image_name in enumerate(image_names):
            row = {
                'image': str(image_name),
                'psnr': float(psnr_values[index].item()),
                'ssim': float(ssim_values[index].item()),
                'lpips': float(lpips_values[index].item()),
            }
            self.rows.append(row)
            batch_rows.append(row)
        return batch_rows

    def save(self, output_dir):
        if not self.rows:
            raise RuntimeError('No images were evaluated, so metrics cannot be saved.')

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'per_image_metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=['image', 'psnr', 'ssim', 'lpips'])
            writer.writeheader()
            writer.writerows(self.rows)

        averages = {
            metric: float(np.mean([row[metric] for row in self.rows]))
            for metric in ('psnr', 'ssim', 'lpips')
        }
        summary = {
            'num_images': len(self.rows),
            'lpips_backbone': self.lpips_net,
            'averaging': 'arithmetic mean of per-image metrics',
            'average_psnr': averages['psnr'],
            'average_ssim': averages['ssim'],
            'average_lpips': averages['lpips'],
        }
        json_path = os.path.join(output_dir, 'metrics_summary.json')
        with open(json_path, 'w', encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print('\nOverall metrics (mean over all images)')
        print('PSNR : {:.4f} dB'.format(averages['psnr']))
        print('SSIM : {:.4f}'.format(averages['ssim']))
        print('LPIPS: {:.4f}'.format(averages['lpips']))
        print('Per-image metrics: {}'.format(csv_path))
        print('Metrics summary  : {}'.format(json_path))
        return summary


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    metric_evaluator = ImageQualityEvaluator(device, lpips_net=args.lpips_net)
    model.restore(val_loader, metric_evaluator=metric_evaluator)


if __name__ == '__main__':
    main()
