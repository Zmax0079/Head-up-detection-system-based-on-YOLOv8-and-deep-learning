from pathlib import Path
from config import RAW_DIR

def count_files(folder, suffix):
    return len(list(folder.glob(f"*{suffix}"))) if folder.exists() else 0

def check_split(split):
    img_dir = RAW_DIR / split / "images"
    lbl_dir = RAW_DIR / split / "labels"

    img_count = count_files(img_dir, ".jpg") + count_files(img_dir, ".png")
    lbl_count = count_files(lbl_dir, ".txt")

    print(f"\n[{split.upper()}]")
    print(f"图片数: {img_count}")
    print(f"标签数: {lbl_count}")

    if img_count == lbl_count and img_count > 0:
        print("✅ 图片/标签匹配")
    else:
        print("⚠️ 图片/标签数量不一致")

    sample_imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:3]
    for p in sample_imgs:
        print(f"  示例: {p.name}")

def main():
    print("=" * 60)
    print("数据集检查")
    print("=" * 60)
    print(f"数据根目录: {RAW_DIR}")

    check_split("train")
    check_split("val")

    yaml_path = RAW_DIR / "data.yaml"
    print(f"\ndata.yaml 是否存在: {'✅' if yaml_path.exists() else '❌'}")

if __name__ == "__main__":
    main()