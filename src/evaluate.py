import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def evaluate(model, loader: DataLoader, device) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            targets.append(y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # targets and preds are in log1p scale; invert for interpretable metrics
    preds_orig = np.expm1(preds)
    targets_orig = np.expm1(targets)

    metrics = {
        "mae": mean_absolute_error(targets_orig, preds_orig),
        "r2": r2_score(targets_orig, preds_orig),
        "rmse": np.sqrt(np.mean((targets_orig - preds_orig) ** 2)),
    }
    return metrics, preds_orig, targets_orig


def plot_loss(train_losses: list, val_losses: list, save_path: str = "loss_curve.png"):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_predictions(targets, preds, save_path: str = "predictions.png"):
    plt.figure()
    plt.scatter(targets, preds, alpha=0.4, s=10)
    lim = max(targets.max(), preds.max())
    plt.plot([0, lim], [0, lim], "r--", linewidth=1)
    plt.xlabel("Actual Followers")
    plt.ylabel("Predicted Followers")
    plt.title("Predicted vs Actual Total Followers")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plot saved to {save_path}")
