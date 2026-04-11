import random
import shutil
import sys
from pathlib import Path
from typing import List

THIS_FILE = Path(__file__).resolve()
ROOT = None
for candidate in [THIS_FILE.parent, *THIS_FILE.parents]:
    if (candidate / "config.py").exists():
        ROOT = candidate
        break
if ROOT is None:
    ROOT = THIS_FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import UP_DIR, DOWN_DIR, SPLIT_DIR

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(src_dir: Path) -> List[Path]:
    files = []
    for p in IMAGE_PATTERNS:
        files.extend(src_dir.glob(p))
    return sorted(files)


def split_files(files: List[Path], train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
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


def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("up/down 分类数据集划分")
    print("=" * 60)
    print(f"输入 up 目录  : {UP_DIR}")
    print(f"输入 down目录: {DOWN_DIR}")
    print(f"输出 split目录: {SPLIT_DIR}")

    clear_dir(SPLIT_DIR)

    up_files = list_images(UP_DIR)
    down_files = list_images(DOWN_DIR)

    if not up_files or not down_files:
        raise ValueError(
            "up/down 有一个类别为空，请先执行 python utils/data_prep.py。"
        )

    up_train, up_val, up_test = split_files(up_files)
    down_train, down_val, down_test = split_files(down_files)

    copy_split("train", "up", up_train)
    copy_split("val", "up", up_val)
    copy_split("test", "up", up_test)

    copy_split("train", "down", down_train)
    copy_split("val", "down", down_val)
    copy_split("test", "down", down_test)

    print("\n✅ 划分完成")
    print(f"UP   -> train/val/test: {(len(up_train), len(up_val), len(up_test))}")
    print(f"DOWN -> train/val/test: {(len(down_train), len(down_val), len(down_test))}")
    print(f"比例 -> train/val/test: {TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
