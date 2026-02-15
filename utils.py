import random, numpy as np, torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
from torchvision import datasets


def seed_everything(seed=42): # Et fais 3 seeds (42, 43, 44) pour moyenne/std.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator().manual_seed(seed) # for random_split
    return gen


class OxfordSegWrapper(torch.utils.data.Dataset):
    def __init__(self, root, split, image_size, mean, std, train_aug: bool, indices=None):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True
        )
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train_aug = train_aug

        # indices optionnels pour faire train/val sur le même split="trainval"
        self.indices = list(range(len(self.dataset))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, mask = self.dataset[idx]

        img = TF.resize(img, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        if self.train_aug: # on ne le fait pas en val ni en test
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long) # Oxford: 1=pet, 2=border, 3=background
        mask = mask - 1 # Oxford: 0=pet, 1=border, 2=background
        
        return img, mask
    

@torch.no_grad()
def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def miou_multiclass(logits, targets, num_classes=3, ignore_index=None, eps=1e-6):
    # logits: (B,C,H,W), targets: (B,H,W) in [0..C-1]
    preds = logits.argmax(dim=1)  # (B,H,W)

    ious = []
    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue
        pred_c = (preds == c)
        targ_c = (targets == c)

        inter = (pred_c & targ_c).sum(dim=(1,2)).float()   # (B,)
        union = (pred_c | targ_c).sum(dim=(1,2)).float()   # (B,)
        iou_c = (inter + eps) / (union + eps)              # (B,)


        # union is (B,) -> only keep samples where union>0
        valid = union > 0
        if valid.any():
            ious.append(((inter[valid] + eps) / (union[valid] + eps)))


    if len(ious) == 0:
        return 0.0
    # return torch.stack(ious).mean().item()
    return torch.cat(ious).mean().item()



def plot_curves(train_losses, val_losses, train_accs, val_accs, save_prefix="training"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_prefix}_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_prefix}_accuracy.png", dpi=150)
    plt.close()


@torch.no_grad()
def visualize_attention_once(model, loader, device, H_patches, W_patches, epoch, save_path):
    model.eval()
    images, _ = next(iter(loader))
    x = images[:1].to(device)

    logits, attn_list = model(x)

    if attn_list is None or len(attn_list) == 0 or attn_list[-1] is None:
        print("Skipping attention plot (no attention returned).")
        return

    attn = attn_list[-1]          # last layer: [B, heads, T, T]
    attn = attn[0].mean(dim=0)    # [T, T] avg heads
    cls_to_patches = attn[0, 1:]  # CLS -> patches
    heat = cls_to_patches.reshape(H_patches, W_patches).detach().cpu().numpy()

    plt.figure()
    plt.imshow(heat)
    plt.colorbar()
    plt.title(f"CLS attention (epoch {epoch})")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()