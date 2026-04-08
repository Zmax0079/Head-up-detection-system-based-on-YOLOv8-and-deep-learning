import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

class RNNHeadPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512

        self.rnn = nn.RNN(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.feature_extractor(x).view(B, T, self.feature_dim)
        out, _ = self.rnn(feat)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def get_rnn_model():
    return RNNHeadPoseModel()