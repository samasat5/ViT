import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vision_transformer import VisionTransformer

from config import (
    IMAGE_SIZE, PATCH_SIZE, CHANNELS, 
    NUM_CLASSES, NUM_HEAD, NUM_TRANSFORMER,
    DIM_EMBED, DIM_MLP, DROPOUT_RATE, LOCAT,
    TASK, DPR, BIAS, PIN_MEMORY, NUM_WORKERS,
)


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, leave=False):
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)

        optimizer.zero_grad()

        logits, _ = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples # len(train_loader)
    acc = total_correct / total_samples
    return avg_loss, acc


@torch.no_grad()
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), # how to define the weights?
            std=(0.2470, 0.2435, 0.2616)
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

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader  = DataLoader(
        test_dataset, batch_size=128, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        dim_embed=DIM_EMBED,
        dim_mlp=DIM_MLP,
        num_classes=NUM_CLASSES,
        num_head=NUM_HEAD,
        num_transformer=NUM_TRANSFORMER,
        dropout_rate=DROPOUT_RATE,
        drop_path_rate=DPR,
        bias=BIAS,
        locat=LOCAT,
        task=TASK,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    num_epochs = 10
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)

        pbar.set_postfix({
            # "Train Loss": f"{train_loss:.4f}",
            "Train Acc": f"{train_acc:.4f}",
            # "Test Loss": f"{test_loss:.4f}",
            "Test Acc": f"{test_acc:.4f}"
        })

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  Test  loss: {test_loss:.4f}, acc: {test_acc:.4f}")
