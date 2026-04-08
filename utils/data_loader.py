# =========================
# 文件: utils/data_loader.py
# =========================

from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from config import IMG_SIZE, BATCH_SIZE, CLASS_TO_IDX, CLASS_NAMES


def get_image_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


def get_image_dataloaders(split_dir):
    train_dataset = datasets.ImageFolder(
        root=str(split_dir / "train"),
        transform=get_image_transforms(train=True)
    )
    val_dataset = datasets.ImageFolder(
        root=str(split_dir / "val"),
        transform=get_image_transforms(train=False)
    )
    test_dataset = datasets.ImageFolder(
        root=str(split_dir / "test"),
        transform=get_image_transforms(train=False)
    )

    print(f"[ImageFolder] 类别映射: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset


class SequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        root_dir = Path(root_dir)

        for class_name in CLASS_NAMES:
            class_dir = root_dir / class_name
            if not class_dir.exists():
                continue
            for npy_file in sorted(class_dir.glob("*.npy")):
                self.samples.append((npy_file, CLASS_TO_IDX[class_name]))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        arr = np.load(npy_path)  # [T,H,W,C]

        frames = []
        for i in range(arr.shape[0]):
            img = Image.fromarray(arr[i])
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames, dim=0)  # [T,C,H,W]
        return frames, label


def get_sequence_dataloaders(sequence_dir):
    train_dataset = SequenceDataset(sequence_dir / "train")
    val_dataset = SequenceDataset(sequence_dir / "val")
    test_dataset = SequenceDataset(sequence_dir / "test")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset