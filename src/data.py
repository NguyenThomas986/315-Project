import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

NUMERIC_COLS = [
    "AVERAGE_STREAM_DURATION",
    "FOLLOWERS_GAINED_PER_STREAM",
    "AVG_VIEWERS_PER_STREAM",
    "AVG_GAMES_PER_STREAM",
    "TOTAL_TIME_STREAMED",
    "TOTAL_FOLLOWERS",
    "TOTAL_VIEWS",
    "TOTAL_GAMES_STREAMED",
    "ACTIVE_DAYS_PER_WEEK",
]

CATEGORICAL_COLS = [
    "LANGUAGE",
    "MOST_STREAMED_GAME",
    "2ND_MOST_STREAMED_GAME",
    "MOST_ACTIVE_DAY",
    "DAY_WITH_MOST_FOLLOWERS_GAINED",
]


def load_and_preprocess(csv_path: str) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    df = pd.read_csv(csv_path)

    # Per abstract: remove esports accounts, focus on individual streamers
    df = df[df["TYPE"] == "personality"].reset_index(drop=True)

    # Fill missing second game with "None"
    df["2ND_MOST_STREAMED_GAME"] = df["2ND_MOST_STREAMED_GAME"].fillna("None")

    # Frequency encode categorical columns (count of each value / total rows)
    for col in CATEGORICAL_COLS:
        freq = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq)

    freq_cols = [c + "_freq" for c in CATEGORICAL_COLS]
    feature_cols = NUMERIC_COLS + freq_cols

    X = df[feature_cols].values.astype(np.float32)

    # Z-score normalization per abstract
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, df, scaler
