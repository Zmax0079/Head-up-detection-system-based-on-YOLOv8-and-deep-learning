import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import shutil
import random
from typing import Iterable, List
from config import UP_DIR, DOWN_DIR, SPLIT_DIR

random.seed(42)

# 可疑关键词：当这些词出现在 up 文件名中时，认为标签可能有误
SUSPECT_UP_KEYWORDS = {
    "down", "lower", "low", "bow", "nod", "tilt_down", "head_down", "lookdown", "lh"
}
# 允许的最大类别不平衡比（major/minor）
MAX_IMBALANCE_RATIO = 1.2


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(src_dir: Path) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for pattern in exts:
        files.extend(src_dir.glob(pattern))
    return files


def is_suspect_up(file_path: Path) -> bool:
    name = file_path.stem.lower()
    normalized = name.replace("-", "_")
    return any(k in normalized for k in SUSPECT_UP_KEYWORDS)


def filter_up_files(up_files: Iterable[Path]):
    clean, suspect = [], []
    for f in up_files:
        if is_suspect_up(f):
            suspect.append(f)
        else:
            clean.append(f)
    return clean, suspect


def maybe_balance(up_files: List[Path], down_files: List[Path]):
    """
    若类别差距过大，自动下采样多数类到不超过 MAX_IMBALANCE_RATIO。
    """
    n_up, n_down = len(up_files), len(down_files)
    major, minor = max(n_up, n_down), min(n_up, n_down)

    if minor == 0:
        raise ValueError("up/down 其中一个类别为空，无法进行划分。")

    if major / minor <= MAX_IMBALANCE_RATIO:
        return up_files, down_files, False

    target_major = int(minor * MAX_IMBALANCE_RATIO)
    if n_up > n_down:
        random.shuffle(up_files)
        up_files = up_files[:target_major]
    else:
        random.shuffle(down_files)
        down_files = down_files[:target_major]

    return up_files, down_files, True


def split_files(files: List[Path], train_ratio=0.7, val_ratio=0.15):
    files = files[:]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files


def copy_split(split_name: str, class_name: str, files: List[Path]):
    dst_dir = SPLIT_DIR / split_name / class_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)


def write_review_list(suspect_files: List[Path]):
    if not suspect_files:
        return
    review_file = SPLIT_DIR / "suspect_up_for_review.txt"
    with review_file.open("w", encoding="utf-8") as fw:
        for f in suspect_files:
            fw.write(f"{f}\n")


def main():
    print("=" * 60)
    print("分类数据集划分（含去噪+平衡策略）")
    print("=" * 60)

    clear_dir(SPLIT_DIR)

    raw_up_files = list_images(UP_DIR)
    raw_down_files = list_images(DOWN_DIR)

    up_files, suspect_up_files = filter_up_files(raw_up_files)
    down_files = raw_down_files

    print(f"原始 up 数量         : {len(raw_up_files)}")
    print(f"原始 down 数量       : {len(raw_down_files)}")
    print(f"疑似误标 up 数量      : {len(suspect_up_files)}")
    print(f"过滤后 up 有效数量     : {len(up_files)}")

    up_files, down_files, balanced = maybe_balance(up_files, down_files)

    if balanced:
        print(f"⚖️ 检测到类别差距过大，已自动下采样到最大比例 {MAX_IMBALANCE_RATIO}:1")

    print(f"用于划分 up 数量      : {len(up_files)}")
    print(f"用于划分 down 数量    : {len(down_files)}")

    up_train, up_val, up_test = split_files(up_files)
    down_train, down_val, down_test = split_files(down_files)

    copy_split("train", "up", up_train)
    copy_split("val", "up", up_val)
    copy_split("test", "up", up_test)

    copy_split("train", "down", down_train)
    copy_split("val", "down", down_val)
    copy_split("test", "down", down_test)

    write_review_list(suspect_up_files)

    print("\n✅ 划分完成")
    print(f"UP   -> train/val/test: {(len(up_train), len(up_val), len(up_test))}")
    print(f"DOWN -> train/val/test: {(len(down_train), len(down_val), len(down_test))}")
    if suspect_up_files:
        print(f"📝 已输出人工复核列表: {SPLIT_DIR / 'suspect_up_for_review.txt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
