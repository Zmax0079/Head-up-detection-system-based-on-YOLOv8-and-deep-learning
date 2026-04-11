import sys
import cv2
import csv
import time
import torch
import argparse
from pathlib import Path
from collections import defaultdict, deque

# ===== LSTM时序缓存 =====
SEQ_LEN = 5
track_frame_buffer = defaultdict(lambda: deque(maxlen=SEQ_LEN))

from ultralytics import YOLO
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from config import DEVICE, IMG_SIZE, DEMO_VIDEO_DIR, YOLO_WEIGHT_CANDIDATES, WEIGHTS_DIR
from models.cnn_model import get_cnn_model
from models.mobilenet_model import get_mobilenet_model
from models.transformer_model import get_transformer_model
from models.rnn_model import get_rnn_model
from models.lstm_model import get_lstm_model

VIDEO_DIR = BASE_DIR / "data" / "raw" / "videos"
OUTPUT_VIDEO_DIR = DEMO_VIDEO_DIR
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_WEIGHT_MAP = {
    "cnn": WEIGHTS_DIR / "best_cnn.pth",
    "mobilenet": WEIGHTS_DIR / "best_mobilenet.pth",
    "transformer": WEIGHTS_DIR / "best_transformer.pth",
    "rnn": WEIGHTS_DIR / "best_rnn.pth",
    "lstm": WEIGHTS_DIR / "best_lstm.pth",
}

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_yolo_weight():
    for p in YOLO_WEIGHT_CANDIDATES:
        if p.exists():
            print(f"✅ 使用 YOLO 权重: {p}")
            return str(p)
    raise FileNotFoundError(
        "❌ 未找到 YOLO 权重文件，请检查：\n"
        "1) weights/yolo/best.pt\n"
        "2) weights/downloads/yolov8s.pt 或 yolov8n.pt"
    )


def load_classifier(model_name="cnn"):
    model_name = model_name.lower()
    if model_name == "cnn":
        model = get_cnn_model()
    elif model_name == "mobilenet":
        model = get_mobilenet_model()
    elif model_name == "transformer":
        model = get_transformer_model()
    elif model_name == "rnn":
        model = get_rnn_model()
    elif model_name == "lstm":
        model = get_lstm_model()
    else:
        raise ValueError(f"❌ 不支持的模型类型: {model_name}")

    weight_path = CLASSIFIER_WEIGHT_MAP[model_name]
    if not weight_path.exists():
        raise FileNotFoundError(f"❌ 未找到分类模型权重: {weight_path}")

    state_dict = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"✅ 分类模型加载成功: {model_name}")
    print(f"✅ 分类权重路径: {weight_path}")
    return model


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


def nms_boxes(candidates, iou_th=0.5):
    """candidates: [(x1,y1,x2,y2,score,box_obj)]"""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x[4], reverse=True)
    kept = []
    while candidates:
        best = candidates.pop(0)
        kept.append(best)
        remain = []
        for c in candidates:
            if box_iou(best[:4], c[:4]) < iou_th:
                remain.append(c)
        candidates = remain
    return kept


def is_valid_head_box(x1, y1, x2, y2, frame_shape):
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False

    frame_h, frame_w = frame_shape[:2]

    # 放宽远处小目标（以前最小18，后排会漏）
    if w < 10 or h < 10:
        return False

    if w > frame_w * 0.55 or h > frame_h * 0.6:
        return False

    ratio = w / (h + 1e-6)
    if ratio < 0.2 or ratio > 2.5:
        return False

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False

    return True


def classify_head_probs(crop, classifier, up_class_index=1):
    if crop is None or crop.size == 0:
        return 0.5, 0.5
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = image_transform(img_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = classifier(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    # 支持类别索引可配置，避免训练标签顺序不一致导致全判成up/down
    up_class_index = int(up_class_index)
    up_class_index = 0 if up_class_index == 0 else 1
    down_class_index = 1 - up_class_index
    return float(probs[down_class_index].item()), float(probs[up_class_index].item())


def decide_label_with_geometry(down_prob, up_prob, box_xyxy, frame_shape):
    """几何先验：低头更常出现在更低位置且框更“高瘦”，降低 low-head 被判成 up 的概率。"""
    x1, y1, x2, y2 = box_xyxy
    h, w = frame_shape[:2]

    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    ratio = bw / bh
    y_ratio = cy / max(1.0, h)

    score_down = down_prob
    score_up = up_prob

    # 低头偏置（保守一些，减少“低头被识别成抬头”）
    if y_ratio > 0.40:
        score_down += 0.08
    if ratio < 0.85:
        score_down += 0.06

    # 抬头需要更明显优势才判 up
    if score_up - score_down > 0.04:
        return "up", score_up
    return "down", score_down


def classify_with_fallback(frame, box_xyxy, classifier, up_class_index=1):
    """低头俯身时，头部可能落在更靠下区域，加入下扩fallback裁剪并融合几何先验。"""
    x1, y1, x2, y2 = box_xyxy

    crop1 = frame[y1:y2, x1:x2]
    d1, u1 = classify_head_probs(crop1, classifier, up_class_index=up_class_index)

    h, w = frame.shape[:2]
    bh = y2 - y1
    bw = x2 - x1

    fx1 = max(0, x1 - int(0.08 * bw))
    fx2 = min(w, x2 + int(0.08 * bw))
    fy1 = max(0, y1)
    fy2 = min(h, y2 + int(0.35 * bh))
    crop2 = frame[fy1:fy2, fx1:fx2]
    d2, u2 = classify_head_probs(crop2, classifier, up_class_index=up_class_index)

    # 选择更稳定的一组概率（优先 down 置信更高者）
    if max(d2, u2) > max(d1, u1) + 0.03:
        down_prob, up_prob = d2, u2
    else:
        down_prob, up_prob = d1, u1

    return decide_label_with_geometry(down_prob, up_prob, box_xyxy, frame.shape)


def extract_candidate_boxes(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []

    names = result.names if hasattr(result, "names") else {}

    # 自动识别类别语义：优先 person/head，过滤不相关物体
    person_ids = set()
    head_ids = set()
    if isinstance(names, dict):
        for k, v in names.items():
            name = str(v).lower()
            if "person" in name:
                person_ids.add(int(k))
            if "head" in name:
                head_ids.add(int(k))

    selected = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        conf = float(box.conf[0].item()) if box.conf is not None else 0.0

        # 同时存在 head/person 时，优先 head；否则退化到 person；否则不过滤
        if head_ids:
            if cls_id not in head_ids:
                continue
        elif person_ids:
            if cls_id not in person_ids:
                continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        selected.append((x1, y1, x2, y2, conf, box, cls_id, person_ids, head_ids))

    # 去重：同一个人两个框的问题
    selected = nms_boxes(selected, iou_th=0.55)
    return selected


def person_box_to_head_box(x1, y1, x2, y2):
    """将 person 框映射到上半身头肩区域，兼顾俯身（向下留更多余量）。"""
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2

    head_w = int(w * 0.78)
    # 从原来的 0.38 提高到 0.55，减轻低头俯身漏检
    head_h = int(h * 0.55)

    hx1 = cx - head_w // 2
    hx2 = cx + head_w // 2
    hy1 = y1
    hy2 = y1 + head_h
    return hx1, hy1, hx2, hy2


def get_track_id(x1, y1, x2, y2, grid_size=32):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx // grid_size, cy // grid_size)


def dedupe_by_track_grid(candidates, grid=26):
    """对已候选框再按网格去重：同一网格仅保留 det_conf 更高者。"""
    best = {}
    for c in candidates:
        x1, y1, x2, y2, det_confidence, box, cls_id, person_ids, head_ids = c
        key = get_track_id(x1, y1, x2, y2, grid_size=grid)
        if key not in best or det_confidence > best[key][4]:
            best[key] = c
    return list(best.values())


def process_video(video_path, model_name="cnn", show=True, save_video=True, det_conf=0.12, det_imgsz=1536, up_class_index=1):
    print("=" * 60)
    print("抬头率检测开始")
    print("=" * 60)
    print(f"视频路径: {video_path}")
    print(f"分类模型: {model_name}")
    print(f"设备: {DEVICE}")
    print(f"检测阈值: conf={det_conf}, imgsz={det_imgsz}")
    print(f"分类索引: up_class_index={up_class_index}, down_class_index={1-int(up_class_index)}")
    print("=" * 60)

    detector = YOLO(get_yolo_weight())
    classifier = load_classifier(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video_path = OUTPUT_VIDEO_DIR / f"output_{model_name}_{Path(video_path).stem}.mp4"
    output_csv_path = OUTPUT_VIDEO_DIR / f"output_{model_name}_{Path(video_path).stem}.csv"

    writer = None
    csv_file = open(output_csv_path, mode="w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "time_sec", "total_heads", "up_count", "head_up_rate"])

    frame_id = 0
    global_up = 0
    global_total = 0
    start_time = time.time()

    track_history = defaultdict(lambda: deque(maxlen=8))
    track_last_seen = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        frame = cv2.resize(frame, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)

        if writer is None and save_video:
            out_h, out_w = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

        # YOLO 检测
        results = detector.predict(
            source=frame,
            conf=det_conf,
            iou=0.55,
            imgsz=det_imgsz,
            max_det=400,
            verbose=False,
            device=0 if DEVICE == "cuda" else "cpu",
        )
        result = results[0]
        raw_candidates = extract_candidate_boxes(result)

        # 先把框统一映射成“最终用于分类的head框”，再做二次NMS和网格去重，避免叠框
        normalized_candidates = []
        for x1, y1, x2, y2, det_confidence, box, cls_id, person_ids, head_ids in raw_candidates:
            if (not head_ids) and person_ids and cls_id in person_ids:
                x1, y1, x2, y2 = person_box_to_head_box(x1, y1, x2, y2)
            normalized_candidates.append((x1, y1, x2, y2, det_confidence, box, cls_id, person_ids, head_ids))

        # 二次NMS（基于最终head框）+ 网格去重
        normalized_candidates = nms_boxes(normalized_candidates, iou_th=0.45)
        candidates = dedupe_by_track_grid(normalized_candidates, grid=22)

        # 清理过旧 track
        stale_keys = [k for k, v in track_last_seen.items() if frame_id - v > 40]
        for k in stale_keys:
            track_last_seen.pop(k, None)
            track_history.pop(k, None)

        total_heads = 0
        up_count = 0

        for x1, y1, x2, y2, det_confidence, box, cls_id, person_ids, head_ids in candidates:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            frame_h, frame_w = frame.shape[:2]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 轻度区域过滤，减少投影幕/墙面误检
            if cy < int(frame_h * 0.08):
                continue
            if cx < int(frame_w * 0.01) or cx > int(frame_w * 0.99):
                continue

            if not is_valid_head_box(x1, y1, x2, y2, frame.shape):
                continue

            label, cls_conf = classify_with_fallback(frame, (x1, y1, x2, y2), classifier, up_class_index=up_class_index)

            track_id = get_track_id(x1, y1, x2, y2)
            track_last_seen[track_id] = frame_id

            # 时序平滑：低置信度时优先沿用历史
            if cls_conf >= 0.80:
                track_history[track_id].append(label)
            elif cls_conf >= 0.55:
                if len(track_history[track_id]) >= 3:
                    up_votes = sum(1 for x in track_history[track_id] if x == "up")
                    down_votes = sum(1 for x in track_history[track_id] if x == "down")
                    if up_votes > down_votes:
                        label = "up"
                    elif down_votes > up_votes:
                        label = "down"
                track_history[track_id].append(label)
            else:
                if len(track_history[track_id]) >= 3:
                    up_votes = sum(1 for x in track_history[track_id] if x == "up")
                    down_votes = sum(1 for x in track_history[track_id] if x == "down")
                    if up_votes > down_votes:
                        label = "up"
                    elif down_votes > up_votes:
                        label = "down"
                    else:
                        label = "ignore"
                else:
                    label = "ignore"

            if label in ("up", "down"):
                total_heads += 1
                if label == "up":
                    up_count += 1

            color = (0, 255, 0) if label == "up" else (0, 0, 255) if label == "down" else (180, 180, 180)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} cls:{cls_conf:.2f} det:{det_confidence:.2f}",
                        (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

        head_up_rate = (up_count / total_heads * 100.0) if total_heads > 0 else 0.0
        global_up += up_count
        global_total += total_heads
        global_rate = (global_up / global_total * 100.0) if global_total > 0 else 0.0

        csv_writer.writerow([frame_id, round(frame_id / fps, 2), total_heads, up_count, round(head_up_rate, 2)])

        cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Current Heads: {total_heads}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Current Up: {up_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Current Head-Up Rate: {head_up_rate:.1f}%", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, f"Global Head-Up Rate: {global_rate:.1f}%", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)

        if show:
            cv2.imshow("Head-Up Rate Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if writer is not None:
            writer.write(frame)

        if frame_id % 30 == 0:
            print(f"处理中: {frame_id}/{total_frames} | 当前抬头率: {head_up_rate:.1f}% | 全局抬头率: {global_rate:.1f}%")

    cap.release()
    if writer is not None:
        writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    final_rate = (global_up / global_total * 100.0) if global_total > 0 else 0.0

    print("=" * 60)
    print("✅ 视频处理完成")
    print("=" * 60)
    print(f"总处理帧数: {frame_id}")
    print(f"总检测头数: {global_total}")
    print(f"总抬头数: {global_up}")
    print(f"最终抬头率: {final_rate:.2f}%")
    print(f"总耗时: {total_time:.2f} 秒")
    if save_video:
        print(f"结果视频: {output_video_path}")
    print(f"结果CSV : {output_csv_path}")
    print("=" * 60)


def auto_find_video():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"❌ 视频目录不存在: {VIDEO_DIR}")

    candidates = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        candidates.extend(VIDEO_DIR.glob(ext))

    if not candidates:
        raise FileNotFoundError(
            f"❌ 未在目录中找到视频文件:\n{VIDEO_DIR}\n"
            "请把视频放到该目录，例如:\n"
            "data/raw/videos/classroom.mp4"
        )

    candidates = sorted(candidates)
    print("检测到以下视频：")
    for i, v in enumerate(candidates, start=1):
        print(f"  [{i}] {v.name}")
    print(f"✅ 默认使用第一个视频: {candidates[0].name}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="抬头率检测主程序")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mobilenet", "transformer", "rnn", "lstm"])
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--det-conf", type=float, default=0.12, help="YOLO检测置信度阈值，后排漏检可尝试更低")
    parser.add_argument("--det-imgsz", type=int, default=1536, help="YOLO输入尺寸，后排小目标可尝试更高")
    parser.add_argument("--up-class-index", type=int, default=1, choices=[0, 1],
                        help="分类模型中up对应的类别索引。若出现‘全是抬头/全是低头’，尝试改为0")
    parser.add_argument("--noshow", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video) if args.video else auto_find_video()
    if not video_path.exists():
        raise FileNotFoundError(f"❌ 指定视频不存在: {video_path}")

    process_video(
        video_path=video_path,
        model_name=args.model,
        show=not args.noshow,
        save_video=not args.nosave,
        det_conf=args.det_conf,
        det_imgsz=args.det_imgsz,
        up_class_index=args.up_class_index,
    )


if __name__ == "__main__":
    main()
