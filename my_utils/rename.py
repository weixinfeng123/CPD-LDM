from pathlib import Path
import re


# ============================
# 这里改成你的图片文件夹路径
# ============================
FOLDER_PATH = r"E:\code\experience-other\code\val_data\NPE"

# 重命名后的前缀
PREFIX = ""

# 编号位数，例如 4 表示 0001、0002
DIGITS = 4

# 是否递归处理子文件夹
RECURSIVE = False

# 支持的图片格式
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(path):
    """
    自然排序：
    例如 1.png, 2.png, 10.png 会按 1,2,10 排序
    """
    name = path.stem
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", name)]


def get_images(folder):
    folder = Path(folder)

    if RECURSIVE:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]

    return sorted(files, key=natural_key)


def main():
    folder = Path(FOLDER_PATH)

    if not folder.exists():
        print(f"文件夹不存在: {folder}")
        return

    images = get_images(folder)

    if len(images) == 0:
        print("该文件夹下没有找到图片")
        return

    print(f"找到图片数量: {len(images)}")

    # 第一步：先改成临时文件名，避免重名冲突
    temp_files = []
    for i, img_path in enumerate(images):
        temp_name = f"__temp_rename_{i}{img_path.suffix.lower()}"
        temp_path = img_path.with_name(temp_name)
        img_path.rename(temp_path)
        temp_files.append(temp_path)

    # 第二步：按顺序改成最终文件名
    for idx, temp_path in enumerate(temp_files, start=1):
        new_name = f"{PREFIX}{idx:0{DIGITS}d}{temp_path.suffix.lower()}"
        new_path = temp_path.with_name(new_name)
        temp_path.rename(new_path)
        print(f"{temp_path.name}  ->  {new_path.name}")

    print("\n重命名完成！")


if __name__ == "__main__":
    main()