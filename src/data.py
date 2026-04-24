import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

NUMERIC_COLS = [
    "AVERAGE_STREAM_DURATION",
    "FOLLOWERS_GAINED_PER_STREAM",
    "AVG_VIEWERS_PER_STREAM",
    "AVG_GAMES_PER_STREAM",
    "TOTAL_TIME_STREAMED",
    "TOTAL_VIEWS",
    "TOTAL_GAMES_STREAMED",
    "ACTIVE_DAYS_PER_WEEK",
]

CATEGORICAL_COLS = [
    "LANGUAGE",
    "TYPE",
    "MOST_STREAMED_GAME",
    "2ND_MOST_STREAMED_GAME",
    "MOST_ACTIVE_DAY",
    "DAY_WITH_MOST_FOLLOWERS_GAINED",
]

TARGET_COL = "TOTAL_FOLLOWERS"


class TwitchDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(csv_path: str, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # log-scale the target to reduce skew
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size / (1 - test_size), random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (
        TwitchDataset(X_train, y_train),
        TwitchDataset(X_val, y_val),
        TwitchDataset(X_test, y_test),
        scaler,
        encoders,
    )
