# =========================
# 文件: utils/build_sequences.py
# =========================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import cv2
import numpy as np
import shutil
from tqdm import tqdm
from config import SPLIT_DIR, SEQUENCE_DIR, IMG_SIZE, SEQUENCE_LENGTH


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_sequences_for_split(split_name: str):
    for class_name in ["up", "down"]:
        src_dir = SPLIT_DIR / split_name / class_name
        dst_dir = SEQUENCE_DIR / split_name / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src_dir.glob("*.jpg"))
        if len(files) < SEQUENCE_LENGTH:
            continue

        seq_count = 0
        for i in tqdm(range(0, len(files) - SEQUENCE_LENGTH + 1), desc=f"{split_name}-{class_name}"):
            seq_imgs = []
            for j in range(SEQUENCE_LENGTH):
                img = cv2.imread(str(files[i + j]))
                if img is None:
                    break
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                seq_imgs.append(img)

            if len(seq_imgs) == SEQUENCE_LENGTH:
                arr = np.array(seq_imgs, dtype=np.uint8)
                np.save(dst_dir / f"seq_{seq_count:05d}.npy", arr)
                seq_count += 1

        print(f"✅ {split_name}/{class_name} 序列数: {seq_count}")


def main():
    print("=" * 60)
    print("构建时序序列数据")
    print("=" * 60)

    clear_dir(SEQUENCE_DIR)

    for split in ["train", "val", "test"]:
        build_sequences_for_split(split)

    print("=" * 60)
    print("✅ 时序序列构建完成")
    print("=" * 60)


if __name__ == "__main__":
    main()