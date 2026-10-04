"""Experiment A: correctness-fixed, independent stage-2 diffusion baseline.

This file is self-contained with respect to the stage-2 trainer: it does not
import models.ddmdce. Model architecture, diffusion sampling, and loss weights
remain identical to the baseline so PSNR/SSIM comparisons stay meaningful.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils
from models.unet import DiffusionUNet
from models.LCRnet_new import LCR
from torchvision import models


# 指数滑动平均（EMA）参数管理器。
# 训练时维护一份平滑权重，供后续验证或推理使用。
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# 构造扩散过程的噪声调度表 beta_t。
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()

        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DiffusionUNet(config)
        # 训练模式：加载第一阶段已经训练好的 LCR 权重。
        if self.args.mode == 'training':
            self.LCR = self.load_stage1(LCR(), 'checkpoints')
        # 推理模式：先创建 LCR，随后由二阶段 checkpoint 一并恢复其权重。
        else:
            self.LCR = LCR()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    # 加载第一阶段 LCR 权重；二阶段训练时会冻结这部分参数。
    @staticmethod
    def load_stage1(model, model_dir):
        checkpoint = utils.logging.load_checkpoint(os.path.join(model_dir, 'v2-s_high_31.5132_condition_24.8246.pth'), 'cuda')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        return model

    # 使用跳步反向扩散，从随机噪声逐步生成目标特征 x0。
    def sample_training(self, x_cond, b, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.config.diffusion.num_sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            # print(f"Max value of x0_t: {x0_t.max().item()}")
            # print(f"Min value of x0_t: {x0_t.min().item()}")

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))

        # return xs[-1]
        return xs

    def forward(self, inputs):

        data_dict = {}

        b = self.betas.to(inputs.device)

        # 训练分支：构造带噪目标并训练 UNet 预测噪声。
        if self.training:
            # 第一阶段只负责生成条件特征和目标特征，不计算梯度。
            with torch.no_grad():
                output = self.LCR(inputs, pred_fea=None)
                # dce_enhance_img = output["enhanced_image"]
                dce_enhance_fea = output["cond_fea"]
                gt_fea = output["gt_fea"]


            # 从这里开始训练扩散 UNet，需要保留梯度。
            data_dict["gt_fea_before"] = gt_fea
            dce_enhance_norm = utils.data_transform(dce_enhance_fea)
            high_input_norm = utils.data_transform(gt_fea)
            # data_dict["gt_fea_before"] = high_input_norm

            t = torch.randint(low=0, high=self.num_timesteps,
                              size=(dce_enhance_norm.shape[0] // 2 + 1,),
                              device=inputs.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:dce_enhance_norm.shape[0]]

            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            e = torch.randn_like(dce_enhance_norm)


            x = high_input_norm * a.sqrt() + e * (1.0 - a).sqrt()
            noise_output = self.Unet(torch.cat([dce_enhance_norm, x], dim=1), t.float())

            pred_fea = self.sample_training(dce_enhance_norm,b)[-1]

            data_dict["pred_fea_before"] = pred_fea

            pred_fea = utils.inverse_data_transform(pred_fea)

            data_dict["noise_output"] = noise_output
            data_dict["e"] = e

            data_dict["pred_fea"] = pred_fea
            data_dict["gt_fea"] = gt_fea

            # 使用扩散模型生成的特征重建增强图像。LCR 的参数虽然被冻结，
            # 但这里不能使用 no_grad，以便图像损失反向传播到扩散 UNet。
            # image_output = self.LCR(inputs, pred_fea=pred_fea)
            # data_dict["pred_img"] = image_output["pred_img"]
            # data_dict["high"] = inputs[:, 3:, :, :]


        else:
            output = self.LCR(inputs, pred_fea=None)

            dce_enhance_fea = output["cond_fea"]

            dce_enhance_norm = utils.data_transform(dce_enhance_fea)
            pred_fea_all = self.sample_training(dce_enhance_norm, b)

            pred_fea = pred_fea_all[-1]
            pred_fea = utils.inverse_data_transform(pred_fea)


            pred_x = self.LCR(inputs, pred_fea=pred_fea)["pred_img"]
            data_dict["pred_x"] = pred_x

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = Net(args, config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        # Pixel-space reconstruction loss weights. Existing configs keep the
        # defaults; experiments may override them under the training section.

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)

        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        # restore step/epoch
        self.step = checkpoint.get('step', 0)
        self.start_epoch = checkpoint.get('epoch', 0)

        # restore optimizer/ema if present (optimizer must be created before calling)
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as ex:
                print(f"[warn] optimizer state not loaded: {ex}")
        if 'ema_helper' in checkpoint:
            try:
                self.ema_helper.load_state_dict(checkpoint['ema_helper'])
            except Exception as ex:
                print(f"[warn] ema state not loaded: {ex}")

        if ema:
            self.ema_helper.ema(self.model)

        print(f"=> loaded checkpoint {load_path} step {self.step} epoch {self.start_epoch}")

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        for name, param in self.model.named_parameters():
            if "LCR" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # 仅使用 requires_grad=True 的扩散模型参数重建优化器。
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = utils.optimize.get_optimizer(self.config, trainable_params)

        # resume AFTER optimizer is created (so optimizer/ema/step can be restored)
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        best_val_loss = float('inf')  # 保留的验证损失变量，当前最佳模型按 PSNR 保存。

        best_psnr = -float('inf')

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                self.model.train()

                if isinstance(self.model, torch.nn.DataParallel):
                    self.model.module.LCR.eval()
                else:
                    self.model.LCR.eval()

                self.step += 1

                x = x.to(self.device)

                output = self.model(x)

                # noise_loss, scc_loss, img_l1, img_mse = self.ddm_losses(output)
                noise_loss, scc_loss = self.ddm_losses(output)
                scc_loss = 1 * scc_loss
                # 像素 L1 保证整体重建，MSE 对较大的像素误差施加更强惩罚。
                # image_loss = (1 * img_l1
                #               + 0.5 * img_mse)
                # scc_loss = 3*scc_loss

                loss = noise_loss+ scc_loss

                # -------------------------------------------------------------------------------

                data_time += time.time() - data_start

                if self.step % 10 == 0:
                    # print("step:{}, noise_loss:{:.5f} "
                    #       "img_l1:{:.5f} img_mse:{:.5f} time:{:.5f} ".
                    #       format(self.step, noise_loss.item(),
                    #               img_l1.item(), img_mse.item(),
                    #              data_time / (i + 1)))

                    print("step:{}, noise_loss:{:.5f} "
                          "scc_loss:{:.5f}  time:{:.5f} ".
                          format(self.step, noise_loss.item(),
                                 scc_loss.item(), data_time / (i + 1)))


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

            print("ddmdce_B_4scc_v2_s")

            self.model.eval()
            val_psnr = self.validate_psnr(val_loader)
            print("[VAL PSNR] epoch:{} psnr:{:.4f} best:{:.4f}".format(epoch + 1, val_psnr, best_psnr))
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_ckpt_path = os.path.join(self.config.data.ckpt_dir, "model_best_psnr")
                print("[SAVE BEST PSNR CKPT PATH] {} psnr:{:.4f}".format(
                    os.path.abspath(best_ckpt_path), best_psnr), flush=True)
                utils.logging.save_checkpoint(
                    self._checkpoint_state(epoch + 1, best_psnr=best_psnr),
                    filename=best_ckpt_path
                )

            save_every_epochs = 5
            # 每 5 个 epoch 保存验证结果和周期 checkpoint。
            if (epoch + 1) % save_every_epochs == 0:
                self.model.eval()
                self.sample_validation_patches(val_loader, step=f"epoch_{epoch + 1:04d}")

                ckpt_path = os.path.join(self.config.data.ckpt_dir, f"model_epoch_{epoch + 1:04d}")
                print("[SAVE CKPT PATH]", os.path.abspath(ckpt_path), flush=True)

                utils.logging.save_checkpoint(
                    self._checkpoint_state(epoch + 1),
                    filename=ckpt_path
                )

    def _checkpoint_state(self, epoch, best_psnr=None):
        # 实验 A 修改：保存 load_ddm_ckpt() 恢复训练所需的全部状态。
        # 这样 --resume 会真正延续训练，而不是只加载模型权重后重新开始；
        # 此项修改不会改变从头训练时的损失函数和优化目标。
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ema_helper': self.ema_helper.state_dict(),
            'step': self.step,
            'epoch': epoch,
            'params': self.args,
            'config': self.config,
            'experiment': 'A',
        }
        if best_psnr is not None:
            state['best_psnr'] = best_psnr
            state['best_epoch'] = epoch
        return state

    def ddm_losses(self, output):

        noise_output = output["noise_output"]
        e = output["e"]
        pred_fea = output["pred_fea"]
        gt_fea = output["gt_fea"]

        # pred_img = output["pred_img"]
        # high = output["high"]

        # ==================noise loss==================
        noise_loss = self.l2_loss(noise_output, e)
        # ==================scc loss==================
        scc_loss =  self.l1_loss(pred_fea, gt_fea)

        # ==================image reconstruction losses==================
        # img_l1 = self.l1_loss(pred_img, high)
        # img_mse = self.l2_loss(pred_img, high)

        # return noise_loss, scc_loss, img_l1, img_mse
        return noise_loss, scc_loss

    def validate_psnr(self, val_loader):
        total_psnr = 0.0
        total_images = 0
        self.model.eval()

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :]
                target = x[:, 3:, :, :].to(self.device)
                b, _, img_h, img_w = x_cond.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')

                model_input = torch.cat((x_cond, x_cond), dim=1).to(self.device)
                pred_x = self.model(model_input)["pred_x"][:, :, :img_h, :img_w].clamp(0.0, 1.0)

                mse = F.mse_loss(pred_x, target, reduction='none')
                mse = mse.flatten(start_dim=1).mean(dim=1)
                psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))

                total_psnr += psnr.sum().item()
                total_images += psnr.numel()

        if total_images == 0:
            return -float('inf')
        return total_psnr / total_images

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder,
                                    self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()

        with torch.no_grad():
            print('Performing validation at step: {}'.format(step))
            for i, (x, y) in enumerate(val_loader):

                x_cond = x[:, :3, :, :]
                b, _, img_h, img_w = x_cond.shape

                img_h_64 = int(64 * np.ceil(img_h / 64.0))
                img_w_64 = int(64 * np.ceil(img_w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_64 - img_w, 0, img_h_64 - img_h), 'reflect')

                x = torch.cat((x_cond, x_cond),dim=1)

                pred_x = self.model(x.to(self.device))["pred_x"][:, :, :img_h, :img_w]
                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), '{}'.format(y[0])))
