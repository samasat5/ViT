import random, numpy as np, torch
import matplotlib.pyplot as plt


def seed_everything(seed=42): # Et fais 3 seeds (42, 43, 44) pour moyenne/std.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    gen = torch.Generator().manual_seed(seed) # for random_split
    return gen

@torch.no_grad()
def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()

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