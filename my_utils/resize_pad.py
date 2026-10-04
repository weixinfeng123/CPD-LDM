import os
from PIL import Image, ImageOps

def resize_folder_to_512(
    input_dir: str,
    output_dir: str,
    mode: str = "pad"  # "pad"=等比缩放+补边到512；"crop"=等比缩放+居中裁剪到512；"stretch"=强行拉伸到512
):
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
                    # 等比缩放到至少覆盖 512×512，然后居中裁剪
                    out = ImageOps.fit(img, (512, 512), method=Image.LANCZOS, centering=(0.5, 0.5))

                elif mode == "pad":
                    # 等比缩放到最长边=512，然后补边到 512×512
                    out = ImageOps.contain(img, (512, 512), method=Image.LANCZOS)
                    out = ImageOps.pad(out, (512, 512), color=(0, 0, 0), centering=(0.5, 0.5))
                else:
                    raise ValueError("mode must be one of: pad, crop, stretch")

                # 保存：jpg 用高质量；png 保持无损
                ext = os.path.splitext(fname.lower())[1]
                if ext in {".jpg", ".jpeg"}:
                    out.save(out_path, quality=95, subsampling=0)
                else:
                    out.save(out_path)

                print(f"Saved: {out_path}")

        except Exception as e:
            print(f"[Skip] {in_path} -> {e}")

if __name__ == "__main__":
    input_dir = r"E:\LLIE-dataset\LSRW\Eval\Huawei\high"     # 改成你的输入文件夹
    output_dir = r"E:\code\experience\LSRW\val\high"      # 改成你的输出文件夹
    resize_folder_to_512(input_dir, output_dir, mode="pad")
