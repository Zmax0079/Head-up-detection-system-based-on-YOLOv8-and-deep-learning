import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, MOBILENET_WEIGHT

def get_mobilenet_model():
    model = models.mobilenet_v2(weights=None)

    if MOBILENET_WEIGHT.exists():
        state_dict = torch.load(MOBILENET_WEIGHT, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 已加载 MobileNetV2 预训练权重: {MOBILENET_WEIGHT}")
    else:
        print("⚠️ 未找到 MobileNetV2 预训练权重，将随机初始化")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model