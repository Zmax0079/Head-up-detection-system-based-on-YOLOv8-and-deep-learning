import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHTS_DIR, REPORTS_DIR,
    SPLIT_DIR, SEQUENCE_DIR
)
from utils.data_loader import (
    get_image_dataloaders,
    get_sequence_dataloaders
)
from utils.visualization import (
    plot_train_curves,
    plot_confusion_matrix
)

from models.cnn_model import get_cnn_model
from models.mobilenet_model import get_mobilenet_model
from models.transformer_model import get_transformer_model
from models.rnn_model import get_rnn_model
from models.lstm_model import get_lstm_model


def get_model(model_name):
    if model_name == "cnn":
        return get_cnn_model()
    if model_name == "mobilenet":
        return get_mobilenet_model()
    if model_name == "transformer":
        return get_transformer_model()
    if model_name == "rnn":
        return get_rnn_model()
    if model_name == "lstm":
        return get_lstm_model()
    raise ValueError(f"不支持的模型: {model_name}")


def _count_files(root: Path, suffix: str):
    if not root.exists():
        return 0
    return len(list(root.rglob(f"*{suffix}")))


def validate_image_split():
    train_count = _count_files(SPLIT_DIR / "train", ".jpg") + _count_files(SPLIT_DIR / "train", ".png")
    val_count = _count_files(SPLIT_DIR / "val", ".jpg") + _count_files(SPLIT_DIR / "val", ".png")
    test_count = _count_files(SPLIT_DIR / "test", ".jpg") + _count_files(SPLIT_DIR / "test", ".png")

    if min(train_count, val_count, test_count) == 0:
        raise RuntimeError(
            "分类数据为空，请先执行：\n"
            "1) python utils/data_prep.py --mode crop\n"
            "2) python utils/split_dataset.py\n"
            f"当前统计 train/val/test = {train_count}/{val_count}/{test_count}"
        )


def validate_sequence_data(auto_build=False):
    train_npy = _count_files(SEQUENCE_DIR / "train", ".npy")
    val_npy = _count_files(SEQUENCE_DIR / "val", ".npy")
    test_npy = _count_files(SEQUENCE_DIR / "test", ".npy")

    if min(train_npy, val_npy, test_npy) > 0:
        return

    if auto_build:
        print("⚠️ 检测到时序数据为空，自动执行 build_sequences ...")
        from utils.build_sequences import main as build_sequences_main
        build_sequences_main()
        train_npy = _count_files(SEQUENCE_DIR / "train", ".npy")
        val_npy = _count_files(SEQUENCE_DIR / "val", ".npy")
        test_npy = _count_files(SEQUENCE_DIR / "test", ".npy")

    if min(train_npy, val_npy, test_npy) == 0:
        raise RuntimeError(
            "时序数据为空，无法训练 RNN/LSTM。\n"
            "请先执行：python utils/build_sequences.py\n"
            f"当前统计 train/val/test = {train_npy}/{val_npy}/{test_npy}"
        )


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def main(model_name, auto_build_seq=False):
    print("=" * 60)
    print(f"开始训练模型: {model_name.upper()}")
    print(f"使用设备: {DEVICE}")
    print("=" * 60)

    is_sequence_model = model_name in ["rnn", "lstm"]

    if is_sequence_model:
        validate_sequence_data(auto_build=auto_build_seq)
        train_loader, val_loader, test_loader, train_dataset = get_sequence_dataloaders(SEQUENCE_DIR)
        print(f"训练序列数: {len(train_dataset)}")
    else:
        validate_image_split()
        train_loader, val_loader, test_loader, train_dataset = get_image_dataloaders(SPLIT_DIR)
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")

    model = get_model(model_name).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_weight_path = WEIGHTS_DIR / f"best_{model_name}.pth"

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weight_path)
            print(f"✅ 保存最佳模型: {best_weight_path}")

    print("\n" + "=" * 60)
    print("训练完成，开始测试最佳模型...")
    print("=" * 60)

    if not best_weight_path.exists():
        raise RuntimeError(f"未生成最佳模型权重: {best_weight_path}")

    model.load_state_dict(torch.load(best_weight_path, map_location=DEVICE))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    print("\n===== Final Test Results =====")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["down", "up"], zero_division=0))

    plot_train_curves(history, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)

    result_file = REPORTS_DIR / "evaluation_results.json"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    results[model_name] = {
        "accuracy": float(test_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 结果已保存到: {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["cnn", "mobilenet", "transformer", "rnn", "lstm"])
    parser.add_argument("--auto-build-seq", action="store_true",
                        help="当RNN/LSTM序列数据为空时，自动调用 utils/build_sequences.py")
    args = parser.parse_args()

    main(args.model, auto_build_seq=args.auto_build_seq)
