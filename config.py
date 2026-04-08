# =========================
# 文件: config.py
# =========================

from pathlib import Path
import torch

# =========================================================
# 项目根目录
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# 数据目录
# =========================================================
DATA_DIR = BASE_DIR / "data"

# 原始 SCUT-HEAD 转换后的目录
# 结构：
# data/raw/images/train/images
# data/raw/images/train/labels
# data/raw/images/val/images
# data/raw/images/val/labels
RAW_DIR = DATA_DIR / "raw" / "images"

# 预处理目录
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_CROPS_DIR = PROCESSED_DIR / "temp_crops"
UP_DIR = PROCESSED_DIR / "up"
DOWN_DIR = PROCESSED_DIR / "down"

# 分类数据集目录
SPLIT_DIR = DATA_DIR / "split"

# 时序数据目录
SEQ_DIR = DATA_DIR / "sequences"

# =========================================================
# 模型 / 权重目录
# =========================================================
MODELS_DIR = BASE_DIR / "models"

PRETRAINED_DIR = BASE_DIR / "pretrained" / "weights"

# 注意：这里严格按你已有权重文件名来
RESNET18_WEIGHT = PRETRAINED_DIR / "resnet18-f37072fd.pth"
VIT_B16_WEIGHT = PRETRAINED_DIR / "vit_b_16-c867db91.pth"

# =========================================================
# 结果输出目录
# =========================================================
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
WEIGHTS_DIR = RESULTS_DIR / "weights"
REPORTS_DIR = RESULTS_DIR / "reports"

# 对比图目录（单独保留便于后续扩展）
COMPARISON_DIR = FIGURES_DIR

# =========================================================
# 训练参数
# =========================================================
IMG_SIZE = (224, 224)
NUM_CLASSES = 2

BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4

SEQ_LEN = 8
HIDDEN_DIM = 128
NUM_LAYERS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["down", "up"]

# =========================================================
# 自动创建必要目录
# =========================================================
AUTO_DIRS = [
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    TEMP_CROPS_DIR,
    UP_DIR,
    DOWN_DIR,
    SPLIT_DIR,
    SEQ_DIR,
    PRETRAINED_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    WEIGHTS_DIR,
    REPORTS_DIR,
]

for d in AUTO_DIRS:
    d.mkdir(parents=True, exist_ok=True)

print(f"[config] BASE_DIR = {BASE_DIR}")
print(f"[config] DEVICE   = {DEVICE}")