import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, VIT_WEIGHT

def get_transformer_model(num_classes=NUM_CLASSES):
    model = models.vit_b_16(weights=None)

    if VIT_WEIGHT.exists():
        state_dict = torch.load(VIT_WEIGHT, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 已加载 ViT-B/16 预训练权重: {VIT_WEIGHT}")
    else:
        print("⚠️ 未找到 ViT 预训练权重，将随机初始化")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model