# =========================
# 文件: utils/data_prep.py
# 位置: D:\01_Code\CTA\headpose_attention_detection\utils\data_prep.py
# =========================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import cv2
import shutil
from tqdm import tqdm
from config import RAW_IMAGES_DIR, RAW_LABELS_DIR, TEMP_CROPS_DIR, UP_DIR, DOWN_DIR, IMG_SIZE


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def yolo_to_xyxy(line, w, h):
    cls_id, x_c, y_c, bw, bh = map(float, line.strip().split())
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)
    return x1, y1, x2, y2


def classify_head_pose_by_bbox(x1, y1, x2, y2, img_h):
    """
    启发式分类：
    头框中心越靠上，越可能抬头
    头框越扁/越小，也倾向低头
    """
    h = y2 - y1
    w = x2 - x1
    cy = (y1 + y2) / 2

    y_ratio = cy / img_h
    aspect = h / (w + 1e-6)

    # 经验阈值，可后续调
    if y_ratio < 0.42 and aspect > 0.8:
        return "up"
    elif y_ratio > 0.50:
        return "down"
    else:
        return "up" if aspect > 0.9 else "down"


def collect_image_label_pairs():
    pairs = []
    for split in ["train", "val", "test"]:
        img_dir = RAW_IMAGES_DIR / split
        lbl_dir = RAW_LABELS_DIR / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                pairs.append((img_path, label_path))

    return pairs


def main():
    print("=" * 60)
    print("抬头/低头分类数据准备")
    print("=" * 60)
    print(f"项目根目录: {ROOT}")
    print(f"原始图像目录: {RAW_IMAGES_DIR}")
    print(f"原始标签目录: {RAW_LABELS_DIR}")
    print(f"临时裁剪目录: {TEMP_CROPS_DIR}")
    print(f"UP目录: {UP_DIR}")
    print(f"DOWN目录: {DOWN_DIR}")
    print("=" * 60)

    clear_dir(TEMP_CROPS_DIR)
    clear_dir(UP_DIR)
    clear_dir(DOWN_DIR)

    pairs = collect_image_label_pairs()
    print(f"共找到 {len(pairs)} 对图像+标签")

    up_count = 0
    down_count = 0
    crop_count = 0

    for img_path, label_path in tqdm(pairs, desc="处理样本"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(line, w, h)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, IMG_SIZE)
            label = classify_head_pose_by_bbox(x1, y1, x2, y2, h)

            save_name = f"{img_path.stem}_{i:03d}.jpg"
            temp_path = TEMP_CROPS_DIR / save_name
            cv2.imwrite(str(temp_path), crop)

            if label == "up":
                cv2.imwrite(str(UP_DIR / save_name), crop)
                up_count += 1
            else:
                cv2.imwrite(str(DOWN_DIR / save_name), crop)
                down_count += 1

            crop_count += 1

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print(f"总裁剪样本数      : {crop_count}")
    print(f"抬头样本（up）    : {up_count}")
    print(f"低头样本（down）  : {down_count}")
    print("=" * 60)
    print("输出目录：")
    print(f"  {UP_DIR}")
    print(f"  {DOWN_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()