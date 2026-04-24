import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(model, loader: DataLoader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader: DataLoader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            total_loss += criterion(model(X), y).item() * len(y)
    return total_loss / len(loader.dataset)
