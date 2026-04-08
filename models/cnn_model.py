import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, RESNET18_WEIGHT

def get_cnn_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=None)

    if RESNET18_WEIGHT.exists():
        state_dict = torch.load(RESNET18_WEIGHT, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 已加载 ResNet18 预训练权重: {RESNET18_WEIGHT}")
    else:
        print("⚠️ 未找到 ResNet18 预训练权重，将随机初始化")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model