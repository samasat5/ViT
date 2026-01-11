import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vision_transformer import VisionTransformer

from config import (
    IMG_H, IMG_W, PATCH_SIZE, CHANNELS, 
    NUM_CLASSES, NUM_HEAD, NUM_TRANSFORMER,
    DIM_EMBED, DIM_MLP, DROPOUT_RATE,
)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    acc = total_correct / total_samples
    return avg_loss, acc


@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    acc = total_correct / total_samples
    return avg_loss, acc


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5), # how to define the weights?
            std=(0.5, 0.5, 0.5)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=128, shuffle=False)
    # model = VisionTransformer(
    #     image_size=32,
    #     patch_size=4,        # 4×4 patches → 64 patches
    #     in_channels=3,
    #     num_classes=10,
    #     dim_embed=128,
    #     num_head=4,
    #     dim_mlp=256,
    #     num_transformer=6,
    #     dropout_rate=0.1
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionTransformer(
        image_size=(IMG_H, IMG_W),
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        dim_embed=DIM_EMBED,
        dim_mlp=DIM_MLP,
        num_classes=NUM_CLASSES,
        num_head=NUM_HEAD,
        num_transformer=NUM_TRANSFORMER,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(10):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}")
        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Test  loss: {test_loss:.4f}, acc: {test_acc:.4f}")
