import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from src.data import load_and_preprocess
from src.cluster import find_optimal_k, run_kmeans
from src.model import StreamerClassifier
from src.train import train_epoch, val_epoch
from src.evaluate import evaluate, plot_loss, plot_confusion_matrix, plot_dunn_index

DATA_PATH = "topstreamers.csv"


def get_args():
    parser = argparse.ArgumentParser(description="Twitch streamer clustering and classification.")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--k", type=int, default=None, help="Fix k instead of searching")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_loader(X, y, batch_size, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Load and preprocess (esports accounts filtered inside load_and_preprocess) ---
    X, df, scaler = load_and_preprocess(DATA_PATH)
    print(f"\nDataset: {len(df)} personality streamers, {X.shape[1]} features")

    # --- Step 2: K-means clustering + Dunn Index ---
    if args.k is not None:
        best_k = args.k
        labels, km_model = run_kmeans(X, best_k, seed=args.seed)
        results = None
        print(f"\nUsing fixed k={best_k}")
    else:
        print(f"\nSearching k in [{args.k_min}, {args.k_max}] using Dunn Index...")
        best_k, results = find_optimal_k(X, range(args.k_min, args.k_max + 1), seed=args.seed)
        labels = results[best_k]["labels"]
        print(f"\nBest k={best_k}")
        plot_dunn_index(results, best_k)

    for c in range(best_k):
        print(f"  Cluster {c}: {(labels == c).sum()} streamers")

    # --- Step 3: Cross-validated classification ---
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, labels), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
        test_loader = make_loader(X_test, y_test, args.batch_size)

        model = StreamerClassifier(X.shape[1], args.hidden, best_k, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_losses, val_losses = [], []
        best_val_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_loss, vl_acc = val_epoch(model, test_loader, criterion, device)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                torch.save(model.state_dict(), f"best_model_fold{fold}.pt")

        model.load_state_dict(torch.load(f"best_model_fold{fold}.pt", map_location=device))
        result = evaluate(model, test_loader, device, best_k)
        fold_accs.append(result["accuracy"])
        print(f"Fold {fold}/{args.folds} | Accuracy: {result['accuracy']:.4f}")

        if fold == 1:
            plot_loss(train_losses, val_losses, f"loss_curve_fold{fold}.png")
            plot_confusion_matrix(result["confusion_matrix"], f"confusion_matrix_fold{fold}.png")

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"\n{args.folds}-Fold CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    if mean_acc >= 0.70:
        print("Target accuracy of 70% achieved.")
    else:
        print("Below 70% target — consider tuning k, hidden dims, or epochs.")


if __name__ == "__main__":
    main()
