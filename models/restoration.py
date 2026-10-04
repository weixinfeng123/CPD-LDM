import torch
import numpy as np
import utils
import os
import time
import torch.nn.functional as F


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def restore(self, val_loader, metric_evaluator=None):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):

                if x.ndim != 4 or x.shape[1] < 6:
                    raise ValueError(
                        "Paired evaluation expects x to contain low/high RGB images "
                        f"as at least 6 channels, but got shape {tuple(x.shape)}"
                    )
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                target = x[:, 3:6, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_64 = int(64 * np.ceil(h / 64.0))
                img_w_64 = int(64 * np.ceil(w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

                t1 = time.time()
                pred_x = self.diffusion.model(torch.cat((x_cond, x_cond), dim=1))["pred_x"][:, :, :h, :w]
                t2 = time.time()

                for batch_index, image_name in enumerate(y):
                    utils.logging.save_image(
                        pred_x[batch_index:batch_index + 1],
                        os.path.join(image_folder, f"{image_name}"),
                    )

                if metric_evaluator is not None:
                    batch_rows = metric_evaluator.update(pred_x, target, list(y))
                    for row in batch_rows:
                        print(
                            f"processing image {row['image']}, time={t2 - t1:.4f}s, "
                            f"PSNR={row['psnr']:.4f} dB, "
                            f"SSIM={row['ssim']:.4f}, LPIPS={row['lpips']:.4f}"
                        )
                else:
                    print(f"processing batch {i + 1}, time={t2 - t1:.4f}s")

        if metric_evaluator is not None:
            return metric_evaluator.save(image_folder)


