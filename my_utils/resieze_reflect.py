import os
import numpy as np
from PIL import Image, ImageOps

def pad_reflect_to_512(img: Image.Image) -> Image.Image:
    # 先等比缩放，使得长边=512，短边<=512
    img = ImageOps.contain(img, (512, 512), method=Image.LANCZOS)

    arr = np.asarray(img)  # HWC, uint8
    h, w = arr.shape[:2]

    pad_h = 512 - h
    pad_w = 512 - w

    if pad_h < 0 or pad_w < 0:
        # 理论上 contain 后不会发生
        raise ValueError(f"Unexpected size after contain: {h}x{w}")

    # 计算上下左右 padding，居中补
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # reflect padding
    arr_pad = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    return Image.fromarray(arr_pad)

def resize_folder_to_512(input_dir: str, output_dir: str, mode: str = "pad_reflect"):
    os.makedirs(output_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in exts]

    if not files:
        print(f"No images found in: {input_dir}")
        return

    for fname in files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            with Image.open(in_path) as img:
                img = img.convert("RGB")

                if mode == "stretch":
                    out = img.resize((512, 512), Image.LANCZOS)

                elif mode == "crop":
                    out = ImageOps.fit(img, (512, 512), method=Image.LANCZOS, centering=(0.5, 0.5))

                elif mode == "pad_reflect":
                    out = pad_reflect_to_512(img)

                else:
                    raise ValueError("mode must be one of: pad_reflect, crop, stretch")

                ext = os.path.splitext(fname.lower())[1]
                if ext in {".jpg", ".jpeg"}:
                    out.save(out_path, quality=95, subsampling=0)
                else:
                    out.save(out_path)

                print(f"Saved: {out_path}")

        except Exception as e:
            print(f"[Skip] {in_path} -> {e}")

if __name__ == "__main__":
    input_dir = r"E:\LLIE-dataset\ddm_data\origin\485+\train\low"
    output_dir = r"E:\LLIE-dataset\ddm_data\reflect\486+\train\low"
    resize_folder_to_512(input_dir, output_dir, mode="pad_reflect")
