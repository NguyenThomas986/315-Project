import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

PROFILE_NUMERIC = [
    "AVG_VIEWERS_PER_STREAM",
    "FOLLOWERS_GAINED_PER_STREAM",
    "AVERAGE_STREAM_DURATION",
    "ACTIVE_DAYS_PER_WEEK",
    "TOTAL_FOLLOWERS",
    "TOTAL_VIEWS",
]

PROFILE_CATEGORICAL = [
    "LANGUAGE",
    "MOST_STREAMED_GAME",
    "MOST_ACTIVE_DAY",
]


def profile_clusters(df: pd.DataFrame, labels: np.ndarray, save_path: str = "cluster_profiles.png"):
    df = df.copy()
    df["cluster"] = labels
    num_clusters = len(np.unique(labels))

    print("\n=== Cluster Profiles ===")
    summaries = []
    for c in range(num_clusters):
        subset = df[df["cluster"] == c]
        row = {"cluster": c, "size": len(subset)}
        for col in PROFILE_NUMERIC:
            row[col] = subset[col].mean()
        summaries.append(row)

        print(f"\nCluster {c}  ({len(subset)} streamers)")
        for col in PROFILE_NUMERIC:
            print(f"  {col:<30s} {subset[col].mean():>12,.1f}")
        for col in PROFILE_CATEGORICAL:
            top = subset[col].value_counts().index[0]
            pct = subset[col].value_counts(normalize=True).iloc[0] * 100
            print(f"  {col:<30s} {top} ({pct:.0f}%)")

    summary_df = pd.DataFrame(summaries).set_index("cluster")
    best_cluster = int(summary_df["AVG_VIEWERS_PER_STREAM"].idxmax())
    print(f"\nHighest avg viewership: Cluster {best_cluster} "
          f"({summary_df.loc[best_cluster, 'AVG_VIEWERS_PER_STREAM']:,.0f} avg viewers/stream)")

    # Bar chart comparing clusters on key metrics
    metrics = ["AVG_VIEWERS_PER_STREAM", "FOLLOWERS_GAINED_PER_STREAM", "AVERAGE_STREAM_DURATION", "ACTIVE_DAYS_PER_WEEK"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        values = [summary_df.loc[c, metric] for c in range(num_clusters)]
        bars = ax.bar([f"C{c}" for c in range(num_clusters)], values, color="steelblue")
        bars[best_cluster].set_color("darkorange")
        ax.set_title(metric.replace("_", " ").title(), fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Cluster Profiles (orange = highest avg viewers)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster profile chart saved to {save_path}")

    return summary_df, best_cluster


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
