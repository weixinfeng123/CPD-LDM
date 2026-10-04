# make_pairs_txt.py
import os
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def main(root: str, out_txt: str = "pairs.txt"):
    root = Path(root)
    low_dir = root / "data/v2-Synthetic/val/low"
    high_dir = root / "data/v2-Synthetic/val/high"

    assert low_dir.exists(), f"Not found: {low_dir}"
    assert high_dir.exists(), f"Not found: {high_dir}"

    pairs = []
    for p in low_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(low_dir)          # e.g. 0001.png or sub/0001.png
        hp = high_dir / rel                   # match by same relative path
        if hp.exists():
            pairs.append((p.relative_to(root), hp.relative_to(root)))

    assert pairs, "No image pairs found (low/high file names must match)."

    out_path = root / out_txt
    with out_path.open("w", encoding="utf-8") as f:
        for lp, hp in sorted(pairs):
            # 用 / 写入，避免 Windows 反斜杠带来转义问题
            f.write(f"{lp.as_posix()} {hp.as_posix()}\n")

    print(f"OK: wrote {len(pairs)} pairs to {out_path}")

if __name__ == "__main__":
    # 修改这里：你的数据集根目录（里面要有 low/ 和 high/）
    main(root=r"/root/autodl-tmp/ddmdce_remote_3/", out_txt="data/LOL-v2-Synthetic_val.txt")
