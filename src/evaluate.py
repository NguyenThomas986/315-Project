import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate(model, loader: DataLoader, device, num_classes: int) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    acc = (preds == targets).mean()
    report = classification_report(targets, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "preds": preds, "targets": targets}


def plot_loss(train_losses: list, val_losses: list, save_path: str = "loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png"):
    _, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Cluster")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_dunn_index(results: dict, best_k: int, save_path: str = "dunn_index.png"):
    ks = sorted(results.keys())
    dunns = [results[k]["dunn"] for k in ks]
    plt.figure()
    plt.plot(ks, dunns, marker="o")
    plt.axvline(best_k, color="r", linestyle="--", label=f"Best k={best_k}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Dunn Index")
    plt.title("Dunn Index by k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
