# import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import torch
# import glob
# import cv2
# import lpips
# import numpy as np
# from PIL import Image
#
#
# def ssim(prediction, target):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#     img1 = prediction.astype(np.float64)
#     img2 = target.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#     ssim_map = ((2 * mu1_mu2 + C1) *
#                 (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                        (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
#
# def calculate_ssim(target, ref):
#     '''
#     calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     img1 = np.array(target, dtype=np.float64)
#     img2 = np.array(ref, dtype=np.float64)
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')
#
#
# def calculate_psnr(target, ref):
#     img1 = np.array(target, dtype=np.float32)
#     img2 = np.array(ref, dtype=np.float32)
#     diff = img1 - img2
#     psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
#     return psnr
#
#
# def metrics(im_dir, label_dir):
#     avg_psnr = 0
#     avg_ssim = 0
#     avg_lpips = 0
#     n = 0
#     loss_fn = lpips.LPIPS(net='alex')
#     loss_fn.cuda()
#
#     for item in sorted(glob.glob(im_dir)):
#         n += 1
#         im1 = Image.open(item).convert('RGB')
#
#         name = item.split('/')[-1]
#         # name = name.split('_')[0] + '.JPG'              # for SICE
#
#         im2 = Image.open(label_dir + name).convert('RGB')
#         (h, w) = im2.size
#         im1 = im1.resize((h, w))
#         im1 = np.array(im1)
#         im2 = np.array(im2)
#         score_psnr = calculate_psnr(im1, im2)
#         score_ssim = calculate_ssim(im1, im2)
#
#         ex_p0 = lpips.im2tensor(cv2.resize(lpips.load_image(item), (h, w)))
#         ex_ref = lpips.im2tensor(lpips.load_image(label_dir + name))
#         ex_p0 = ex_p0.cuda()
#         ex_ref = ex_ref.cuda()
#         score_lpips = loss_fn.forward(ex_ref, ex_p0)
#
#         avg_psnr += score_psnr
#         avg_ssim += score_ssim
#         avg_lpips += score_lpips
#
#     avg_psnr = avg_psnr / n
#     avg_ssim = avg_ssim / n
#     avg_lpips = avg_lpips / n
#     return avg_psnr, avg_ssim, avg_lpips
#
#
# if __name__ == '__main__':
#     im_dir = './results/LOL-test/i/780.png'
#     label_dir = './dataset/PairLIE-testing-dataset/LOL-test/reference/'
#
#     # im_dir = 'results/SICE/I/*.JPG'
#     # label_dir = '../dataset/LIE/SICE-test/label/'
#
#     avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir)
#     print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
#     print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
#     print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips.item()))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import glob
import cv2
import lpips
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def ssim(prediction, target):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = [ssim(img1[:, :, i], img2[:, :, i]) for i in range(3)]
            return np.mean(ssims)
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_psnr(target, ref):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(255.0 * 255.0 / np.mean(np.square(diff)))
    return psnr


def metrics(im_dir, label_dir):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    n = 0

    loss_fn = lpips.LPIPS(net='alex')
    loss_fn.cuda()

    for item in sorted(glob.glob(im_dir)):
        name = os.path.basename(item)
        gt_path = os.path.join(label_dir, name)
        if not os.path.exists(gt_path):
            print(f"Warning: Label file {gt_path} does not exist.")
            continue

        im1 = Image.open(item).convert('RGB')
        im2 = Image.open(gt_path).convert('RGB')

        # Resize prediction to match ground truth
        im1 = im1.resize(im2.size)
        im1_np = np.array(im1)
        im2_np = np.array(im2)

        # PSNR & SSIM
        score_psnr = calculate_psnr(im1_np, im2_np)
        score_ssim = calculate_ssim(im1_np, im2_np)

        # LPIPS
        h, w = im2.size
        ex_p0 = lpips.im2tensor(cv2.resize(lpips.load_image(item), (h, w)))
        ex_ref = lpips.im2tensor(lpips.load_image(gt_path))
        ex_p0 = ex_p0.cuda()
        ex_ref = ex_ref.cuda()
        score_lpips = loss_fn.forward(ex_ref, ex_p0)

        avg_psnr += score_psnr
        avg_ssim += score_ssim
        avg_lpips += score_lpips.item()
        n += 1

    if n == 0:
        raise RuntimeError("No image pairs were processed.")

    return avg_psnr / n, avg_ssim / n, avg_lpips / n


if __name__ == '__main__':
    # im_dir = './results/zero-dce/LOL-test/*.png'
    # label_dir = './dataset/PairLIE-testing-dataset/LOL-test/reference/'

    # im_dir = r'E:\code\experience\diff_dce_2\zerodce\data\result\nll_0.019\v1\*.png'
    im_dir = r'E:\code\experience-other\code\ddm\results\0.01scc\0100\LSRW\*.png'
    # label_dir = r'E:\LLIE-dataset\LOLdataset\eval15\high'
    label_dir = r'E:\code\experience-other\code\data\reference\LSRW'
    # im_dir = r'E:\code\train_v2\result\330000\100\*.png'
    # label_dir = r'E:\LLIE-dataset\LOL-v2\stage1\Our_normal_test'

    avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir)
    # print("===> Avg.PSNR: {:.4f} dB".format(avg_psnr))
    # print("===> Avg.SSIM: {:.4f}".format(avg_ssim))
    # print("===> Avg.LPIPS: {:.4f}".format(avg_lpips))
    print("{:.4f}".format(avg_psnr))
    print("{:.4f}".format(avg_ssim))
    print("{:.4f}".format(avg_lpips))
