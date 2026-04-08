# =========================
# 文件: main.py
# 路径: D:\01_Code\CTA\headpose_attention_detection\main.py
# 功能: 视频抬头率检测主程序（教室多人增强最终版）
# =========================

import os
import sys
import cv2
import csv
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
from torchvision import transforms

# =========================
# 解决 utils / models / config 导入问题
# =========================
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config import DEVICE, IMG_SIZE

# 分类模型导入
from models.cnn_model import get_cnn_model
from models.mobilenet_model import get_mobilenet_model
from models.transformer_model import get_transformer_model
from models.rnn_model import get_rnn_model
from models.lstm_model import get_lstm_model


# =========================
# 全局路径设置（按你的真实路径）
# =========================
VIDEO_DIR = BASE_DIR / "data" / "raw" / "videos"
OUTPUT_VIDEO_DIR = BASE_DIR / "results" / "demo"
OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

YOLO_WEIGHT_CANDIDATES = [
    BASE_DIR / "pretrained" / "weights" / "yolov8n.pt",
    BASE_DIR / "pretrained" / "weights" / "yolov8s.pt",
]

CLASSIFIER_WEIGHT_MAP = {
    "cnn": BASE_DIR / "results" / "weights" / "best_cnn.pth",
    "mobilenet": BASE_DIR / "results" / "weights" / "best_mobilenet.pth",
    "transformer": BASE_DIR / "results" / "weights" / "best_transformer.pth",
    "rnn": BASE_DIR / "results" / "weights" / "best_rnn.pth",
    "lstm": BASE_DIR / "results" / "weights" / "best_lstm.pth",
}


# =========================
# 图像预处理
# =========================
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# =========================
# 获取 YOLO 权重
# =========================
def get_yolo_weight():
    for p in YOLO_WEIGHT_CANDIDATES:
        if p.exists():
            print(f"✅ 使用 YOLO 权重: {p}")
            return str(p)

    raise FileNotFoundError(
        "❌ 未找到 YOLO 权重文件，请检查：\n"
        "1) pretrained/weights/yolov8n.pt\n"
        "2) pretrained/weights/yolov8s.pt"
    )


# =========================
# 加载分类模型
# =========================
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


# =========================
# 头部框过滤函数（教室小目标增强版）
# =========================
def is_valid_head_box(x1, y1, x2, y2, frame_shape):
    """
    过滤明显不合理的检测框
    教室场景优化版：放宽小目标限制
    """
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return False

    frame_h, frame_w = frame_shape[:2]

    # 太小不要
    if w < 18 or h < 18:
        return False

    # 太大不要
    if w > frame_w * 0.45 or h > frame_h * 0.45:
        return False

    # 长宽比放宽
    ratio = w / h
    if ratio < 0.35 or ratio > 1.8:
        return False

    # 越界过滤
    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False

    return True


# =========================
# 分类函数（单帧头部图像）
# =========================
def classify_head(crop, classifier):
    """
    输入:
        crop: BGR图像
        classifier: 分类模型
    输出:
        label: "up" / "down"
        conf: 置信度
    """
    if crop is None or crop.size == 0:
        return "ignore", 0.0

    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = image_transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = classifier(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

    label = "up" if pred == 1 else "down"
    return label, conf


# =========================
# 提取 YOLO 候选框
# 如果是COCO模型，只保留 person 类
# =========================
def extract_candidate_boxes(result):
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes

    names = result.names if hasattr(result, "names") else {}

    is_coco_person_model = False
    if isinstance(names, dict) and 0 in names and names[0] == "person":
        is_coco_person_model = True

    for box in result.boxes:
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1

        if is_coco_person_model:
            if cls_id != 0:
                continue

        boxes.append(box)

    return boxes


# =========================
# person框近似转head框（扩大版）
# =========================
def person_box_to_head_box(x1, y1, x2, y2):
    """
    把person框近似转换成head区域
    """
    w = x2 - x1
    h = y2 - y1

    head_h = int(h * 0.38)
    head_w = int(w * 0.75)

    cx = (x1 + x2) // 2
    hx1 = cx - head_w // 2
    hx2 = cx + head_w // 2
    hy1 = y1
    hy2 = y1 + head_h

    return hx1, hy1, hx2, hy2


# =========================
# 简易track id（用于稳定分类）
# =========================
def get_track_id(x1, y1, x2, y2, grid_size=40):
    """
    用检测框中心点近似生成一个简易ID
    不需要真正多目标跟踪，也能实现短时稳定
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx // grid_size, cy // grid_size)


# =========================
# 主视频处理函数
# =========================
def process_video(video_path, model_name="cnn", show=True, save_video=True):
    print("=" * 60)
    print("抬头率检测开始")
    print("=" * 60)
    print(f"视频路径: {video_path}")
    print(f"分类模型: {model_name}")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # 1. 加载模型
    yolo_weight = get_yolo_weight()
    detector = YOLO(yolo_weight)
    classifier = load_classifier(model_name)

    # 2. 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"❌ 无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"原视频分辨率: {orig_width}x{orig_height}")
    print(f"视频帧率: {fps:.2f}")
    print(f"总帧数: {total_frames}")

    # =========================
    # 输出视频路径
    # =========================
    output_video_path = OUTPUT_VIDEO_DIR / f"output_{model_name}_{Path(video_path).stem}.mp4"
    output_csv_path = OUTPUT_VIDEO_DIR / f"output_{model_name}_{Path(video_path).stem}.csv"

    writer = None

    # =========================
    # CSV记录
    # =========================
    csv_file = open(output_csv_path, mode="w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_id", "time_sec", "total_heads", "up_count", "head_up_rate"
    ])

    # =========================
    # 统计变量
    # =========================
    frame_id = 0
    global_up = 0
    global_total = 0
    start_time = time.time()

    # =========================
    # 分类稳定缓存（关键）
    # key: 检测框中心点近似ID
    # value: 最近几帧分类历史
    # =========================
    track_history = defaultdict(lambda: deque(maxlen=5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # =========================
        # 教室远景优化：先放大再检测
        # =========================
        frame = cv2.resize(frame, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

        if writer is None and save_video:
            out_h, out_w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (out_w, out_h))
            print(f"✅ 输出视频保存路径: {output_video_path}")
            print(f"✅ 输出视频分辨率: {out_w}x{out_h}")

        current_time_sec = frame_id / fps
        total_heads = 0
        up_count = 0

        # =========================
        # YOLO 检测（多人课堂增强）
        # =========================
        results = detector.predict(
            source=frame,
            conf=0.18,
            iou=0.45,
            imgsz=1280,
            max_det=300,
            verbose=False,
            device=0 if DEVICE == "cuda" else "cpu"
        )

        result = results[0]
        boxes = extract_candidate_boxes(result)

        # =========================
        # 遍历当前帧所有检测框
        # =========================
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det_conf = float(box.conf[0].item()) if box.conf is not None else 0.0

            names = result.names if hasattr(result, "names") else {}
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            is_coco_person_model = isinstance(names, dict) and 0 in names and names[0] == "person"

            # 如果是COCO person检测器，把人体框近似转为头部框
            if is_coco_person_model and cls_id == 0:
                x1, y1, x2, y2 = person_box_to_head_box(x1, y1, x2, y2)

            # 防止越界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # =========================
            # 教室区域过滤（减少误检）
            # =========================
            frame_h, frame_w = frame.shape[:2]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 过滤顶部区域（投影/墙面）
            if cy < int(frame_h * 0.12):
                continue

            # 过滤左右极边区域
            if cx < int(frame_w * 0.03) or cx > int(frame_w * 0.97):
                continue

            # 过滤不合理框
            if not is_valid_head_box(x1, y1, x2, y2, frame.shape):
                continue

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            # =========================
            # 分类
            # =========================
            label, cls_conf = classify_head(crop, classifier)

            # =========================
            # 生成简易track id
            # =========================
            track_id = get_track_id(x1, y1, x2, y2)

            # =========================
            # 双阈值 + 历史稳定投票
            # =========================
            raw_label = label

            if cls_conf >= 0.85:
                # 高置信度，直接写入历史
                track_history[track_id].append(raw_label)

            elif 0.60 <= cls_conf < 0.85:
                # 中等置信度：如果历史足够，则参考历史投票
                if len(track_history[track_id]) >= 3:
                    up_votes = sum(1 for x in track_history[track_id] if x == "up")
                    down_votes = sum(1 for x in track_history[track_id] if x == "down")

                    if up_votes > down_votes:
                        label = "up"
                    elif down_votes > up_votes:
                        label = "down"

                track_history[track_id].append(label)

            else:
                # 低置信度：优先沿用历史结果，否则忽略
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

            # =========================
            # 统计
            # =========================
            counted = False
            if label in ["up", "down"]:
                total_heads += 1
                counted = True
                if label == "up":
                    up_count += 1

            # =========================
            # 可视化颜色
            # =========================
            if label == "up":
                color = (0, 255, 0)
            elif label == "down":
                color = (0, 0, 255)
            elif label == "ignore":
                color = (180, 180, 180)
            else:
                color = (128, 128, 128)

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 显示文本
            text = f"{label} cls:{cls_conf:.2f} det:{det_conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2
            )

        # =========================
        # 当前帧抬头率
        # =========================
        head_up_rate = (up_count / total_heads * 100.0) if total_heads > 0 else 0.0

        global_up += up_count
        global_total += total_heads
        global_rate = (global_up / global_total * 100.0) if global_total > 0 else 0.0

        # =========================
        # 写入CSV
        # =========================
        csv_writer.writerow([
            frame_id,
            round(current_time_sec, 2),
            total_heads,
            up_count,
            round(head_up_rate, 2)
        ])

        # =========================
        # 叠加统计信息
        # =========================
        cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.putText(frame, f"Current Heads: {total_heads}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Current Up: {up_count}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Current Head-Up Rate: {head_up_rate:.1f}%", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.putText(frame, f"Global Head-Up Rate: {global_rate:.1f}%", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)

        # =========================
        # 显示 / 保存
        # =========================
        if show:
            cv2.imshow("Head-Up Rate Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if writer is not None:
            writer.write(frame)

        if frame_id % 30 == 0:
            print(f"处理中: {frame_id}/{total_frames} | 当前抬头率: {head_up_rate:.1f}% | 全局抬头率: {global_rate:.1f}%")

    # =========================
    # 结束释放
    # =========================
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


# =========================
# 自动寻找视频
# =========================
def auto_find_video():
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"❌ 视频目录不存在: {VIDEO_DIR}")

    candidates = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        candidates.extend(VIDEO_DIR.glob(ext))

    if len(candidates) == 0:
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


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser(description="抬头率检测主程序")
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "mobilenet", "transformer", "rnn", "lstm"],
                        help="选择分类模型")
    parser.add_argument("--video", type=str, default=None,
                        help="指定视频路径，不指定则自动从 data/raw/videos 中寻找")
    parser.add_argument("--noshow", action="store_true",
                        help="不显示实时窗口，仅保存输出视频")
    parser.add_argument("--nosave", action="store_true",
                        help="不保存输出视频")

    args = parser.parse_args()

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"❌ 指定视频不存在: {video_path}")
    else:
        video_path = auto_find_video()

    process_video(
        video_path=video_path,
        model_name=args.model,
        show=not args.noshow,
        save_video=not args.nosave
    )


if __name__ == "__main__":
    main()