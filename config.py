import torch
from torch import nn
from utils import accuracy_from_logits, miou_multiclass

CONFIG = {
    "bias": True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dropout": 0.1,
    "drop_path": 0.1,
    "pin_memory": True,
    "num_workers": 4,
    "channels": 3,
    "seed": 42,
    "locat": True, # local attention
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4865, 0.4409),
        "std": (0.2673, 0.2564, 0.2762),
        "num_classes": 100,
    },
    "oxford": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "num_classes": 3,
    },

    "dim_embed": 256,
    "dim_mlp": 1024, # 4 * 128
    "num_heads": 8,
    "num_transformer": 8,

    "classif": {

        "image_size": (32, 32),
        "patch_size": (4, 4),
        "val_size": 5000,

        "lr": 3e-4,
        "weight_decay": 0.05,
        "batch_size": 128, # si GPU OK, sinon 64
        "epochs": 150,
        "patience": 20,
        "min_delta": 1e-4,

        "criterion": nn.CrossEntropyLoss(label_smoothing=0.1),
        "checkpoint_path": ["./baseline/checkpoint_classif", "./locat/checkpoint_classif_locat"],
        "best_weights_path": ["./baseline/best_model_classif", "./locat/best_model_classif_locat"],
        "metric": accuracy_from_logits,
    },

    "seg": {
        
        "image_size": (256, 256),
        "patch_size": (16, 16),
        "val_size": 500,

        "lr": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 8,
        "epochs": 100,
        "patience": 20,
        "min_delta": 1e-4,

        "criterion": nn.CrossEntropyLoss(),
        "checkpoint_path": ["./baseline/checkpoint_seg", "./locat/checkpoint_seg_locat"],
        "best_weights_path": ["./baseline/best_model_seg", "./locat/best_model_seg_locat"],
        "metric": miou_multiclass,
    },
}