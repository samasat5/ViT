import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from vision_transformer import VisionTransformer


# --------------------------
# Model related config
# --------------------------
IMAGE_SIZE = (32, 32)
PATCH_SIZE = (4, 4)
CHANNELS = 3
DIM_EMBED = 128
DIM_MLP = 4 * DIM_EMBED
NUM_HEAD = 4
NUM_TRANSFORMER = 6
NUM_CLASSES = 10
DROPOUT_RATE = 0.1
LOCAT = True
TASK = "classif"
DPR = 0.1
BIAS = False

# --------------------------
# Execution related config
# --------------------------
PIN_MEMORY = False
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
VAL_SIZE = 5000
MIN_DELTA = 1e-4
PATIENCE_INIT = 15
CHECKPOINT_PATH = "checkpoint_vit.pt"
BEST_WEIGHTS_PATH = "best_model_vit.pth"

# CIFAR10 stats classiques
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# --------------------------
# Utils
# --------------------------
@torch.no_grad()
def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# --------------------------
# Train
# --------------------------
def train(
    model,
    train_loader,
    val_dataloader,
    optimizer,          # AdamW
    scheduler,          # ReduceLROnPlateau
    checkpoint_path,
    criterion,          # loss
    patience=PATIENCE_INIT,
    min_delta=MIN_DELTA,
    load_checkpoint=False,
    grad_clip=1.0,
    device=DEVICE,
):
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {NUM_WORKERS}")
    print("")

    start_epoch = 0
    early_stopping_epoch = float("inf")
    best_loss = float("inf")
    best_weights = None
    patience_left = patience

    # Resume
    if load_checkpoint and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_loss = checkpoint.get("best_loss", best_loss)
        patience_left = checkpoint.get("patience_left", patience_left)
        start_epoch = checkpoint["epoch"] + 1

        print(f"Resumed from epoch {start_epoch}")
        print("")

    train_loss_all, val_loss_all = [], []
    train_metric_all, val_metric_all = [], []  # metric = accuracy

    for ep in tqdm(range(start_epoch, NUM_EPOCHS), leave=False):
        # ------------------ TRAIN ------------------
        model.train()
        step = 0
        current_train_loss = 0.0
        current_train_acc = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=PIN_MEMORY)
            labels = labels.to(device, non_blocking=PIN_MEMORY)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            acc = accuracy_from_logits(logits, labels)

            current_train_loss += loss.item()
            current_train_acc += acc
            step += 1

        train_loss_all.append(current_train_loss / step)
        train_metric_all.append(current_train_acc / step)

        # ------------------ VAL ------------------
        model.eval()
        step = 0
        current_val_loss = 0.0
        current_val_acc = 0.0

        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device, non_blocking=PIN_MEMORY)
                labels = labels.to(device, non_blocking=PIN_MEMORY)

                logits, _ = model(images)
                loss = criterion(logits, labels)
                acc = accuracy_from_logits(logits, labels)

                current_val_loss += loss.item()
                current_val_acc += acc
                step += 1

        val_loss_all.append(current_val_loss / step)
        val_metric_all.append(current_val_acc / step)

        print("\n" + "#" * 50)
        print(
            f"[Epoch {ep+1}] "
            f"Train Acc: {train_metric_all[-1]:.4f} | Val Acc: {val_metric_all[-1]:.4f} "
            f"| Train Loss: {train_loss_all[-1]:.4f} | Val Loss: {val_loss_all[-1]:.4f}"
        )

        # ------------------ Early stopping ------------------
        if best_loss - val_loss_all[-1] > min_delta:
            best_loss = val_loss_all[-1]
            best_weights = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                early_stopping_epoch = ep + 1
                print(f"Early stopping triggered at epoch {ep+1}")
                break

        # ------------------ Scheduler ------------------
        # ReduceLROnPlateau attend une métrique (val_loss)
        scheduler.step(val_loss_all[-1])
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        print("")

        # ------------------ Checkpoint ------------------
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "patience_left": patience_left,
            },
            checkpoint_path,
        )

    # reload best
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save(best_weights, BEST_WEIGHTS_PATH)
    else:
        print("Warning: best_weights is None (no improvement ever happened).")
        # dans ce cas on sauve quand même les poids finaux
        torch.save(model.state_dict(), BEST_WEIGHTS_PATH)

    if early_stopping_epoch == float("inf"):
        early_stopping_epoch = len(train_metric_all)

    return early_stopping_epoch, train_loss_all, val_loss_all, train_metric_all, val_metric_all


# --------------------------
# Test
# --------------------------
@torch.no_grad()
def test(model, criterion, testing_dataloader, device=DEVICE):
    model.eval()
    step = 0
    current_test_loss = 0.0
    current_test_acc = 0.0

    for images, labels in testing_dataloader:
        images = images.to(device, non_blocking=PIN_MEMORY)
        labels = labels.to(device, non_blocking=PIN_MEMORY)

        logits, _ = model(images)
        loss = criterion(logits, labels)
        acc = accuracy_from_logits(logits, labels)

        current_test_loss += loss.item()
        current_test_acc += acc
        step += 1

    test_loss = current_test_loss / step
    test_acc = current_test_acc / step

    print("\n" + "#" * 50)
    print(f"[Testing] Final Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print("\n")

    return test_acc, test_loss


# --------------------------
# Main
# --------------------------
def main(load_checkpoint=False):
    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_tf)

    # Split train/val
    train_size = len(full_train) - VAL_SIZE
    val_size = VAL_SIZE
    gen = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    # Val sans augmentation
    val_set.dataset.transform = eval_tf

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

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
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    early_ep, train_loss_all, val_loss_all, train_acc_all, val_acc_all = train(
        model=model,
        train_loader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=CHECKPOINT_PATH,
        criterion=criterion,
        patience=PATIENCE_INIT,
        min_delta=MIN_DELTA,
        load_checkpoint=load_checkpoint,
        grad_clip=1.0,
        device=DEVICE,
    )

    # Charger best et tester 1 fois
    if os.path.exists(BEST_WEIGHTS_PATH):
        model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=DEVICE))

    test_acc, test_loss = test(model, criterion, test_loader, device=DEVICE)

    print(f"Done. Early stop epoch: {early_ep}")
    print(f"Final Test Acc: {test_acc:.4f} | Final Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    main(load_checkpoint=False)
