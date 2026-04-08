# =========================
# 文件: train_yolo.py
# =========================

from ultralytics import YOLO
from config import RAW_YAML, PRETRAINED_DIR


def main():
    print("=" * 60)
    print("训练 YOLO 头部检测模型")
    print("=" * 60)
    print(f"数据配置文件: {RAW_YAML}")

    model = YOLO(str(PRETRAINED_DIR / "yolov8s.pt"))

    model.train(
        data=str(RAW_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        project="results",
        name="yolo_head_train"
    )

    print("=" * 60)
    print("✅ YOLO 训练完成")
    print("最佳权重位置:")
    print("results/yolo_head_train/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()