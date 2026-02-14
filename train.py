import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from vision_transformer import VisionTransformer
import random, numpy as np
import matplotlib.pyplot as plt
import argparse

from config import CONFIG
from utils import plot_curves, seed_everything


def train(
    model,
    train_loader,
    val_dataloader,
    optimizer,          # AdamW
    scheduler,          
    checkpoint_path,
    criterion,          # loss
    metric_fn,
    patience,
    min_delta,
    device,
    pin_memory,
    num_epochs,
    best_weights_path,
    load_checkpoint=False,
    grad_clip=1.0, 
):

    start_epoch = 0
    early_stopping_epoch = float("inf")
    best_loss = float("inf")
    best_weights = None
    patience_left = patience

    train_loss_all, val_loss_all = [], []
    train_metric_all, val_metric_all = [], []  # metric = accuracy (classif), dice (seg)

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

        train_loss_all = checkpoint.get("train_loss_all", [])
        val_loss_all = checkpoint.get("val_loss_all", [])
        train_metric_all = checkpoint.get("train_metric_all", [])
        val_metric_all = checkpoint.get("val_metric_all", [])

        print(f"Resumed from epoch {start_epoch}")
        print("")

    for ep in tqdm(range(start_epoch, num_epochs), leave=False):
        # ------------------ TRAIN ------------------
        model.train()
        step = 0
        current_train_loss = 0.0
        current_train_acc = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            acc = metric_fn(logits, labels)

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
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)

                logits, _ = model(images)
                loss = criterion(logits, labels)
                acc = metric_fn(logits, labels)

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
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss_all[-1])
        else:
            scheduler.step()
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
                "train_loss": train_loss_all,
                "val_loss": val_loss_all,
                "train_metric": train_metric_all,
                "val_metric": val_metric_all,
            },
            checkpoint_path,
        )

    # reload best
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save(best_weights, best_weights_path)
    else:
        print("Warning: best_weights is None (no improvement ever happened).")
        # dans ce cas on sauve quand même les poids finaux
        torch.save(model.state_dict(), best_weights_path)

    if early_stopping_epoch == float("inf"):
        early_stopping_epoch = len(train_metric_all)

    return early_stopping_epoch, train_loss_all, val_loss_all, train_metric_all, val_metric_all


# --------------------------
# Test
# --------------------------
@torch.no_grad()
def test(model, criterion, metric_fn, testing_dataloader, device, pin_memory):
    model.eval()
    step = 0
    current_test_loss = 0.0
    current_test_acc = 0.0

    for images, labels in testing_dataloader:
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        logits, _ = model(images)
        loss = criterion(logits, labels)
        acc = metric_fn(logits, labels)

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
def main(
    seed,
    dataset,
    model,
    metric_fn,
    criterion,
    optimizer,
    scheduler,
    pin_memory,
    num_epochs, 
    batch_size, 
    val_size,
    num_workers,
    mean_dataset,
    std_dataset,
    best_weights_path,
    checkpoint_path,
    patience,
    min_delta,
    device,
    load_checkpoint=False, 
):
    
    gen = seed_everything(seed)
    # Transforms
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_dataset, std_dataset),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_dataset, std_dataset),
    ])

    DatasetClass = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    full_train = DatasetClass(root="./data", train=True, download=True, transform=train_tf)
    test_set = DatasetClass(root="./data", train=False, download=True, transform=eval_tf)

    # Split train/val
    train_size = len(full_train) - val_size
    val_size = val_size
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=gen)

    # Val sans augmentation
    val_set.dataset.transform = eval_tf

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    early_ep, train_loss_all, val_loss_all, train_acc_all, val_acc_all = train(
        model=model,
        metric_fn=metric_fn,
        train_loader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        criterion=criterion,
        patience=patience,
        min_delta=min_delta,
        load_checkpoint=load_checkpoint,
        grad_clip=1.0,
        device=device,
        pin_memory=pin_memory,
        best_weights_path=best_weights_path,
    )

    # Charger best et tester 1 fois
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))

    test_acc, test_loss = test(model, criterion, metric_fn, test_loader, device=device, pin_memory=pin_memory)

    print(f"Done. Early stop epoch: {early_ep}")
    print(f"Final Test Acc: {test_acc:.4f} | Final Test Loss: {test_loss:.4f}")

    return {
        "early_stopping_epoch": early_ep,
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEEP-L project : LocAt")
    parser.add_argument(
        "--task",
        type=str,
        default="classif",
        help="Task à réaliser: 'classif' pour classification ou 'seg' pour segmentation"
    )
    parser.add_argument("--locat", action="store_true", help="Activer LoCAT")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="Dataset à utiliser: 'cifar10' ou 'cifar100'"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour la reproductibilité"
    )
    args = parser.parse_args()

    TASK = args.task
    LOCAT = args.locat
    DATASET = args.dataset
    SEED = args.seed

    CHANNELS = CONFIG["channels"]
    DROPOUT_RATE = CONFIG["dropout"]
    DPR = CONFIG["drop_path"]
    BIAS = CONFIG["bias"]
    DEVICE = CONFIG["device"]
    PIN_MEMORY = CONFIG["pin_memory"]
    NUM_WORKERS = CONFIG["num_workers"]

    IMAGE_SIZE = CONFIG[TASK]["image_size"]
    PATCH_SIZE = CONFIG[TASK]["patch_size"]
    DIM_EMBED = CONFIG[TASK]["dim_embed"]
    DIM_MLP = CONFIG[TASK]["dim_mlp"]
    NUM_HEAD = CONFIG[TASK]["num_heads"]
    NUM_TRANSFORMER = CONFIG[TASK]["num_transformer"]
    
    NUM_EPOCHS = CONFIG[TASK]["epochs"]
    LEARNING_RATE = CONFIG[TASK]["lr"]
    WEIGHT_DECAY = CONFIG[TASK]["weight_decay"]
    VAL_SIZE = CONFIG[TASK]["val_size"]
    PATIENCE_INIT = CONFIG[TASK]["patience"]
    MIN_DELTA = CONFIG[TASK]["min_delta"]
    BATCH_SIZE = CONFIG[TASK]["batch_size"]
    BEST_WEIGHTS_PATH = f"{CONFIG[TASK]['best_weights_path'][1 if LOCAT else 0]}_{SEED}.pth"
    os.makedirs(os.path.dirname(BEST_WEIGHTS_PATH), exist_ok=True)
    CHECKPOINT_PATH = f"{CONFIG[TASK]['checkpoint_path'][1 if LOCAT else 0]}_{SEED}.pth"
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    METRIC_FN = CONFIG[TASK]["accuracy"]

    DATASET_MEAN = CONFIG[DATASET]["mean"]
    DATASET_STD = CONFIG[DATASET]["std"]
    NUM_CLASSES = CONFIG[DATASET]["num_classes"]

    print(f"Task: {TASK}")
    print(f"Locat: {LOCAT}")
    print(f"Dataset: {DATASET}")
    print(f"Seed: {SEED}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {NUM_WORKERS}")
    print("")

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
    criterion = CONFIG[TASK]["criterion"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) if TASK=="classif" else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    results = main(
        seed=SEED,
        dataset=DATASET,
        model=model, 
        metric_fn=METRIC_FN,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        load_checkpoint=False,
        pin_memory=PIN_MEMORY,
        num_epochs=NUM_EPOCHS, 
        batch_size=BATCH_SIZE, 
        val_size=VAL_SIZE,
        num_workers=NUM_WORKERS,
        mean_dataset=DATASET_MEAN,
        std_dataset=DATASET_STD,
        best_weights_path=BEST_WEIGHTS_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        patience=PATIENCE_INIT,
        min_delta=MIN_DELTA,
        device=DEVICE,
    )

    torch.save(
        results, 
        f"training_{TASK}_locat_{SEED}.pth" if LOCAT else f"training_{TASK}_{SEED}.pth"
    )
