# =========================
# 文件: utils/compare_models.py
# =========================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import json
import matplotlib.pyplot as plt
from config import REPORTS_DIR, FIGURES_DIR


def main():
    result_file = REPORTS_DIR / "evaluation_results.json"
    if not result_file.exists():
        print("❌ 未找到 evaluation_results.json，请先训练模型")
        return

    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    save_dir = FIGURES_DIR / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        values = [results[m].get(metric, 0) for m in models]
        plt.bar(models, values)
        plt.title(f"Model Comparison - {metric.upper()}")
        plt.ylabel(metric.upper())
        plt.ylim(0, 1.0)

        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.4f}", ha="center")

        save_path = save_dir / f"{metric}_comparison.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 已保存: {save_path}")


if __name__ == "__main__":
    main()