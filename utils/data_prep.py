import json
import shutil
import sys
import argparse
from pathlib import Path
from collections import Counter

import cv2
from ultralytics import YOLO
from tqdm import tqdm

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

from config import (
    ANNOTATION_SOURCE_DIR,
    UP_DIR,
    DOWN_DIR,
    TEMP_CROPS_DIR,
    BEHAVIOR_CODE_TO_BINARY,
    BEHAVIOR_CODE_TO_NAME,
    YOLO_WEIGHT_CANDIDATES,
)
from utils.behavior_label_parser import extract_behavior_code

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_image_json_pairs(root: Path):
    image_map = {}
    json_map = {}

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        stem_key = str(p.with_suffix("")).lower()
        if suffix in IMAGE_EXTS:
            image_map[stem_key] = p
        elif suffix == ".json":
            json_map[stem_key] = p

    common = sorted(set(image_map.keys()) & set(json_map.keys()))
    return [(image_map[k], json_map[k]) for k in common], image_map, json_map


def get_yolo_weight():
    for p in YOLO_WEIGHT_CANDIDATES:
        if p.exists() and p.suffix == ".pt":
            return str(p)
    return "yolov8s.pt"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def nms_boxes(cands, iou_th=0.45):
    cands = sorted(cands, key=lambda x: x[4], reverse=True)
    keep = []
    while cands:
        best = cands.pop(0)
        keep.append(best)
        cands = [c for c in cands if box_iou(best[:4], c[:4]) < iou_th]
    return keep


def person_to_head(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    head_w = int(w * 0.78)
    head_h = int(h * 0.55)
    hx1 = cx - head_w // 2
    hx2 = cx + head_w // 2
    hy1 = y1
    hy2 = y1 + head_h
    return hx1, hy1, hx2, hy2


def detect_head_boxes(detector, img_bgr, det_conf=0.15, det_imgsz=1280):
    h, w = img_bgr.shape[:2]
    result = detector.predict(
        source=img_bgr,
        conf=det_conf,
        iou=0.55,
        imgsz=det_imgsz,
        max_det=300,
        verbose=False,
    )[0]

    names = result.names if hasattr(result, "names") else {}
    person_ids, head_ids = set(), set()
    if isinstance(names, dict):
        for k, v in names.items():
            name = str(v).lower()
            if "person" in name:
                person_ids.add(int(k))
            if "head" in name:
                head_ids.add(int(k))

    cands = []
    if result.boxes is None:
        return []

    for box in result.boxes:
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        score = float(box.conf[0].item()) if box.conf is not None else 0.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if head_ids:
            if cls_id not in head_ids:
                continue
        elif person_ids:
            if cls_id not in person_ids:
                continue
            x1, y1, x2, y2 = person_to_head(x1, y1, x2, y2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        cands.append((x1, y1, x2, y2, score))

    return nms_boxes(cands, iou_th=0.45)


def save_crop(img_bgr, box, label, stem, idx):
    x1, y1, x2, y2, _ = box
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    out_dir = UP_DIR if label == "up" else DOWN_DIR
    out_name = f"{stem}_{idx:02d}.jpg"
    ok = cv2.imwrite(str(out_dir / out_name), crop)
    if ok:
        cv2.imwrite(str(TEMP_CROPS_DIR / out_name), crop)
    return ok


def copy_full_image(img_path, label):
    out_dir = UP_DIR if label == "up" else DOWN_DIR
    target_path = out_dir / img_path.name
    if target_path.exists():
        suffix_id = 1
        while True:
            alt = out_dir / f"{img_path.stem}_{suffix_id}{img_path.suffix.lower()}"
            if not alt.exists():
                target_path = alt
                break
            suffix_id += 1
    shutil.copy2(img_path, target_path)


def main():
    parser = argparse.ArgumentParser(description="根据Student-Behavior标注生成up/down数据")
    parser.add_argument("--mode", type=str, default="crop", choices=["crop", "full"],
                        help="crop: 用YOLO做人头裁切；full: 直接复制整图")
    parser.add_argument("--det-conf", type=float, default=0.15)
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--max-heads", type=int, default=8, help="每张图最多保留多少个人头")
    args = parser.parse_args()

    print("=" * 70)
    print("Student-Behavior（Front） -> up/down 数据准备")
    print("=" * 70)
    print(f"标注目录: {ANNOTATION_SOURCE_DIR}")
    print(f"输出目录: {UP_DIR.parent}")
    print(f"模式: {args.mode}")

    if not ANNOTATION_SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"未找到数据目录: {ANNOTATION_SOURCE_DIR}\n"
            "请在 config.py 修改 STUDENT_BEHAVIOR_DIR / ANNOTATION_SOURCE_DIR"
        )

    clear_dir(UP_DIR)
    clear_dir(DOWN_DIR)
    clear_dir(TEMP_CROPS_DIR)

    pairs, image_map, json_map = discover_image_json_pairs(ANNOTATION_SOURCE_DIR)
    print(f"检测到图像文件: {len(image_map)}")
    print(f"检测到JSON文件: {len(json_map)}")
    print(f"同名图像+JSON对: {len(pairs)}")

    detector = YOLO(get_yolo_weight()) if args.mode == "crop" else None

    behavior_counter = Counter()
    binary_counter = Counter()
    skipped_counter = Counter()

    for img_path, json_path in tqdm(pairs, desc="处理样本"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            skipped_counter["json_parse_error"] += 1
            continue

        code = extract_behavior_code(data)
        if code is None:
            skipped_counter["missing_behavior_code"] += 1
            continue

        behavior_counter[code] += 1
        binary_label = BEHAVIOR_CODE_TO_BINARY.get(code)
        if binary_label is None:
            skipped_counter[f"ignored_{code}"] += 1
            continue

        if args.mode == "full":
            copy_full_image(img_path, binary_label)
            binary_counter[binary_label] += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped_counter["bad_image"] += 1
            continue

        boxes = detect_head_boxes(detector, img, det_conf=args.det_conf, det_imgsz=args.det_imgsz)
        if not boxes:
            skipped_counter["no_head_detected"] += 1
            continue

        saved = 0
        for i, box in enumerate(boxes[: args.max_heads]):
            ok = save_crop(img, box, binary_label, img_path.stem, i)
            if ok:
                saved += 1

        if saved == 0:
            skipped_counter["crop_save_failed"] += 1
        else:
            binary_counter[binary_label] += saved

    print("\n" + "-" * 70)
    print("8类行为统计（按编码）")
    for code, cnt in sorted(behavior_counter.items(), key=lambda x: x[0]):
        print(f"{code:>3} ({BEHAVIOR_CODE_TO_NAME.get(code, '未知')}): {cnt}")

    print("\n二分类输出统计")
    print(f"down: {binary_counter.get('down', 0)}")
    print(f"up  : {binary_counter.get('up', 0)}")

    if skipped_counter:
        print("\n跳过样本统计")
        for k, v in skipped_counter.most_common():
            print(f"{k}: {v}")

    print("=" * 70)
    print("✅ 数据准备完成")
    print("说明：当前up映射为 tt+js+xt，down映射为 dx+dk")
    print("=" * 70)


if __name__ == "__main__":
    main()
