import os
import math
from pathlib import Path
from types import SimpleNamespace

import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from models.ddmdce_01 import Net, EMAHelper  # 你的 ddmdce.py


# ===================== 直接在代码里改这里（不用改yml） =====================
CONFIG_PATH = r"E:\code\experience-other\train_new_ddm\configs\unsupervised.yml"

# 你训练保存的二阶段权重（通常是 ckpt_dir 下的 model_latest 或你自己的best）
# yml里 ckpt_dir 是 "ckpt/stage2" :contentReference[oaicite:2]{index=2}
CKPT_PATH = r"E:\code\experience-other\train_new_ddm\ckpt\ddmdce\model_best_psnr.pth.tar"

# 输入：低光图片文件夹
INPUT_DIR = r"E:\code\experience-other\train_new_ddm\data\val\low"

# 输出：增强结果保存目录
OUTPUT_DIR = r"E:\code\experience-other\train_new_ddm\results\2"

DEVICE = "cuda"   # or "cpu"
USE_EMA = True    # yml里 model.ema=True :contentReference[oaicite:3]{index=3}
PAD_BASE = 64     # pad到64倍数，避免UNet skip连接尺寸不一致
# =====================================================================


def dict_to_ns(d):
    """dict -> SimpleNamespace，支持 config.xxx 访问"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_ns(x) for x in d]
    return d


def pad_to_multiple(x, base=64):
    """pad到base倍数，返回 pad后的x以及原始H/W"""
    _, _, h, w = x.shape
    h2 = int(base * math.ceil(h / base))
    w2 = int(base * math.ceil(w / base))
    if h2 == h and w2 == w:
        return x, h, w
    x_pad = F.pad(x, (0, w2 - w, 0, h2 - h), mode="reflect")
    return x_pad, h, w


def load_ckpt_into_model(model, ckpt_path, device, use_ema=True):
    """
    兼容两种ckpt格式：
    1) {'state_dict': ..., 'ema_helper': ...} 训练保存的
    2) 直接就是 state_dict
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    state = ckpt.get("state_dict", ckpt)

    # 兼容 DataParallel 的 module. 前缀
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            new_state[k[len("module."):]] = v if k.startswith("module.") else v
        model.load_state_dict(new_state, strict=True)

    # EMA：如果训练时保存了 ema_helper（你训练代码里是有保存的）
    if use_ema and isinstance(ckpt, dict) and "ema_helper" in ckpt:
        ema = EMAHelper(mu=0.9999)
        ema.load_state_dict(ckpt["ema_helper"])
        ema.ema(model)

    return ckpt

@torch.no_grad()
def enhance_folder(model, input_dir, output_dir, device):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    print(f"[INFO] Found {len(files)} images")
    print(f"[INFO] Save to  : {output_dir}")

    for idx, p in enumerate(files, 1):
        img = Image.open(p).convert("RGB")
        low = to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]

        # 单张低光推理：high 用占位（复制 low 最稳）
        high_dummy = low
        x = torch.cat([low, high_dummy], dim=1)  # [1,6,H,W]

        x_pad, h0, w0 = pad_to_multiple(x, base=PAD_BASE)

        # 只 forward 一次
        out = model(x_pad,)

        pred_raw = out["pred_x"][:, :, :h0, :w0]   # 可能范围很大
        # pred = torch.sigmoid(pred_raw)             # 推荐：压到 0~1
        pred = pred_raw.clamp(0, 1)   # 推荐：压到 0~1


        # 输出路径：保持相对目录结构
        rel = p.relative_to(input_dir)
        save_dir = output_dir / rel.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = rel.stem  # 不带后缀的文件名
        save_image(low,  str(save_dir / f"{stem}_low.png"))
        save_image(pred, str(save_dir / f"{stem}_pred.png"))


        # 新增：ZeroDCE 单独结果
        if "dce_x" in out:
            dce = out["dce_x"][:, :, :h0, :w0].clamp(0, 1)
        else:
            dce = None

        # 新增：打印 pred_fea 范围（判断扩散有没有崩）
        if "pred_fea" in out:
            pf = out["pred_fea"]
            print(f"[{idx}/{len(files)}] {rel} pred_fea(min,max)=({pf.min().item():.3f},{pf.max().item():.3f}) "
                  f"mean={pf.mean().item():.3f} std={pf.std().item():.3f}")

        # 保存
        save_image(low, str(save_dir / f"{stem}_low.png"))
        if dce is not None:
            save_image(dce, str(save_dir / f"{stem}_dce.png"))
        save_image(pred, str(save_dir / f"{stem}_pred.png"))

        # 打印范围（前几张 + 每20张 + 最后一张）
        if idx <= 5 or idx % 20 == 0 or idx == len(files):
            print(
                f"[{idx}/{len(files)}] {rel} "
                f"pred_raw(min,max)=({pred_raw.min().item():.3f},{pred_raw.max().item():.3f})"
            )


def main():
    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")

    # 1) 读你的 supervised.yml（注意utf-8-sig，避免你之前的GBK解码问题）
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    # 2) 转成 namespace：config.data.patch_size / config.diffusion.xxx 等
    config = dict_to_ns(cfg)
    config.device = str(device)

    # 3) 构造 args：关键是 mode != 'training'
    # 因为 ddmdce.Net 在 mode='training' 会走 load_stage1(...)，测试不需要
    args = SimpleNamespace(mode="test")

    # 4) 建模型 + 加载二阶段权重
    model = Net(args, config).to(device)
    load_ckpt_into_model(model, CKPT_PATH, device=device, use_ema=USE_EMA)
    w = model.zeroDCE.ReconNet.channel_down.conv0.weight
    print("ReconNet conv0 weight mean:", w.mean().item(), "std:", w.std().item())

    model.eval()

    # 5) 批量增强
    enhance_folder(model, INPUT_DIR, OUTPUT_DIR, device=device)


if __name__ == "__main__":
    main()
