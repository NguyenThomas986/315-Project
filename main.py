import argparse
import torch
from torch.utils.data import DataLoader

from src.data import load_data
from src.model import StreamerMLP
from src.train import train_epoch, val_epoch
from src.evaluate import evaluate, plot_loss, plot_predictions

DATA_PATH = "topstreamers.csv"


def get_args():
    parser = argparse.ArgumentParser(description="Train a model on the Twitch top streamers dataset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, scaler, encoders = load_data(DATA_PATH, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    input_dim = train_ds[0][0].shape[0]
    model = StreamerMLP(input_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f}")

    print(f"\nBest val loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    metrics, preds, targets = evaluate(model, test_loader, device)
    print(f"\nTest Results:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:,.2f}")

    plot_loss(train_losses, val_losses)
    plot_predictions(targets, preds)


if __name__ == "__main__":
    main()
