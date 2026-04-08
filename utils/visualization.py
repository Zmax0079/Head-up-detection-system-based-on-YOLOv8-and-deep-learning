# =========================
# 文件: utils/visualization.py
# =========================

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import FIGURES_DIR, CLASS_NAMES


def plot_train_curves(history, model_name):
    save_dir = FIGURES_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"{model_name.upper()} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.title(f"{model_name.upper()} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    save_path = save_dir / "train_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 训练曲线已保存: {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    save_dir = FIGURES_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name.upper()} Confusion Matrix")
    save_path = save_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")