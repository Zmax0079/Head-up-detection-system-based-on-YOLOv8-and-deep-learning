# =========================
# 文件: convert_scut.py
# 位置: D:\01_Code\CTA\headpose_attention_detection\convert_scut.py
# =========================

import shutil
from pathlib import Path
from config import DOWNLOAD_DIR, RAW_IMAGES_DIR, RAW_LABELS_DIR, RAW_YAML

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_split(part_dir: Path, split_name: str, target_split: str, prefix: str):
    """
    part_dir: PartA / PartB
    split_name: train / val / test
    target_split: train / val / test
    prefix: A or B
    """
    src_img_dir = part_dir / split_name / "images"
    src_lbl_dir = part_dir / split_name / "labels"

    dst_img_dir = RAW_IMAGES_DIR / target_split
    dst_lbl_dir = RAW_LABELS_DIR / target_split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_img_dir.exists():
        print(f"⚠️ 缺少目录: {src_img_dir}")
        return 0

    img_files = []
    for ext in IMAGE_EXTS:
        img_files.extend(src_img_dir.glob(f"*{ext}"))
        img_files.extend(src_img_dir.glob(f"*{ext.upper()}"))

    img_files = sorted(list(set(img_files)))

    count = 0
    for img_path in img_files:
        stem = img_path.stem
        label_path = src_lbl_dir / f"{stem}.txt"

        new_name = f"{prefix}_{split_name}_{stem}{img_path.suffix.lower()}"
        new_label_name = f"{prefix}_{split_name}_{stem}.txt"

        shutil.copy2(img_path, dst_img_dir / new_name)

        if label_path.exists():
            shutil.copy2(label_path, dst_lbl_dir / new_label_name)
        else:
            # 没标签也复制图片，但提醒
            print(f"⚠️ 缺少标签: {label_path}")

        count += 1

    return count


def create_yaml():
    content = f"""path: {RAW_IMAGES_DIR.parent.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: head
"""
    RAW_YAML.write_text(content, encoding="utf-8")
    print(f"📝 已生成: {RAW_YAML}")


def main():
    print("=" * 60)
    print("SCUT-HEAD 数据集转换（适配 images/labels 双层目录）")
    print("=" * 60)
    print(f"源目录: {DOWNLOAD_DIR}")
    print(f"目标目录: {RAW_IMAGES_DIR.parent}")
    print("=" * 60)

    clear_dir(RAW_IMAGES_DIR)
    clear_dir(RAW_LABELS_DIR)

    partA = DOWNLOAD_DIR / "PartA"
    partB = DOWNLOAD_DIR / "PartB"

    if not partA.exists() or not partB.exists():
        print("❌ 未找到 PartA 或 PartB，请检查下载目录")
        return

    stats = {"train": 0, "val": 0, "test": 0}

    # 推荐做法：A+B 全部混合，扩大检测样本
    for part_dir, prefix in [(partA, "A"), (partB, "B")]:
        print(f"\n📁 处理 {part_dir.name} ...")
        for split in ["train", "val", "test"]:
            num = copy_split(part_dir, split, split, prefix)
            stats[split] += num
            print(f"  ✓ {split}: {num} 张")

    create_yaml()

    print("\n" + "=" * 60)
    print("✅ 转换完成")
    print(f"train: {stats['train']} 张")
    print(f"val  : {stats['val']} 张")
    print(f"test : {stats['test']} 张")
    print(f"总计 : {sum(stats.values())} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()