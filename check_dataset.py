import json
from collections import Counter

from config import ANNOTATION_SOURCE_DIR, BEHAVIOR_CODE_TO_NAME, BEHAVIOR_CODE_TO_BINARY
from utils.behavior_label_parser import extract_behavior_code

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("=" * 60)
    print("Student-Behavior-Dataset（Front）数据检查")
    print("=" * 60)
    print(f"检查目录: {ANNOTATION_SOURCE_DIR}")

    if not ANNOTATION_SOURCE_DIR.exists():
        print("❌ 目录不存在，请先在 config.py 修改 ANNOTATION_SOURCE_DIR")
        return

    images = {}
    labels = {}
    for p in ANNOTATION_SOURCE_DIR.rglob("*"):
        if not p.is_file():
            continue
        stem = str(p.with_suffix("")).lower()
        suffix = p.suffix.lower()
        if suffix in IMAGE_EXTS:
            images[stem] = p
        elif suffix == ".json":
            labels[stem] = p

    common_keys = sorted(set(images.keys()) & set(labels.keys()))
    only_img = set(images.keys()) - set(labels.keys())
    only_json = set(labels.keys()) - set(images.keys())

    print(f"图像总数: {len(images)}")
    print(f"标注总数: {len(labels)}")
    print(f"配对数量: {len(common_keys)}")
    print(f"仅图像无标注: {len(only_img)}")
    print(f"仅标注无图像: {len(only_json)}")

    counter_8 = Counter()
    counter_2 = Counter()
    bad_json = 0

    for k in common_keys:
        jp = labels[k]
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            bad_json += 1
            continue

        code = extract_behavior_code(data)
        if code is None:
            continue

        counter_8[code] += 1
        binary = BEHAVIOR_CODE_TO_BINARY.get(code)
        if binary:
            counter_2[binary] += 1

    print("\n8类编码统计")
    for code, cnt in sorted(counter_8.items(), key=lambda x: x[0]):
        print(f"{code:>3} ({BEHAVIOR_CODE_TO_NAME.get(code, '未知')}): {cnt}")

    print("\n2类映射统计（仅 dx+dk+tt+js+xt）")
    print(f"down: {counter_2.get('down', 0)}")
    print(f"up  : {counter_2.get('up', 0)}")

    if bad_json:
        print(f"\n⚠️ JSON 解析失败: {bad_json}")

    print("=" * 60)


if __name__ == "__main__":
    main()
