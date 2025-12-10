import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PairTradingDataset(Dataset):
    """Dataset for pair-trading experiments inspired by the MDPI article.

    The loader expects a CSV file with at least the following columns:

    - ``Date``: parseable datetime string
    - ``AssetA_Close``: closing price for asset A
    - ``AssetB_Close``: closing price for asset B

    Additional columns are ignored. The dataset builds sliding windows over the
    spread between the two assets so that the agent can observe recent context
    and predict near-term spread movements.
    """

    def __init__(self, csv_path, start_date, end_date, input_length, reward_horizon):
        self.input_length = input_length
        self.reward_horizon = reward_horizon

        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].reset_index(drop=True)

        self.asset_a = df['AssetA_Close'].values.astype(np.float32)
        self.asset_b = df['AssetB_Close'].values.astype(np.float32)

        spread = np.log(self.asset_a) - np.log(self.asset_b)
        spread_z = (spread - spread.mean()) / (spread.std() + 1e-6)
        features = np.stack([self.asset_a, self.asset_b, spread, spread_z], axis=1)

        self.windows = []
        self.spread_yesterday = []
        self.spread_today = []
        self.spread_future = []

        for idx in range(len(features) - input_length - reward_horizon):
            window = features[idx: idx + input_length]
            self.windows.append(window)
            self.spread_yesterday.append(spread[idx + input_length - 2])
            self.spread_today.append(spread[idx + input_length - 1])
            future = spread[idx + input_length: idx + input_length + reward_horizon]
            self.spread_future.append(future)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]).float(),
            torch.tensor(self.spread_yesterday[idx]).float(),
            torch.tensor(self.spread_today[idx]).float(),
            torch.from_numpy(np.array(self.spread_future[idx])).float(),
        )


def pair_batch_fn(batch):
    windows, y_days, t_days, futures = zip(*batch)
    return (
        torch.stack(windows),
        torch.stack(y_days),
        torch.stack(t_days),
        torch.stack(futures),
    )
