import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ITEM_NAMES = ["CM015Q02", "CM015Q03", "CM020Q01", "CM020Q02", "CM020Q03",
              "CM020Q04", "CM038Q03", "CM038Q05", "CM038Q06"]


class PISA12M(Dataset):

    def __init__(self):
        df = pd.read_csv('datafiles/PISA12M/cleaned.csv')

        resp_cols = [f'item_{item}'     for item in ITEM_NAMES]
        time_cols = [f'time_{item}'     for item in ITEM_NAMES]
        nact_cols = [f'nactions_{item}' for item in ITEM_NAMES]

        self.item_names = ITEM_NAMES
        self.n_users = len(df)
        self.n_items = len(ITEM_NAMES)

        resp_vals = df[resp_cols].to_numpy(dtype=float)
        # time_ / nactions_ may be absent for some items (e.g. CM038Q06); fill with 0
        time_vals = df.reindex(columns=time_cols).fillna(0).to_numpy(dtype=float)
        nact_vals = df.reindex(columns=nact_cols).fillna(0).to_numpy(dtype=float)

        self.Q = pd.read_csv('datafiles/PISA12M/Q.csv').to_numpy(copy=True)
        self.n_skills = self.Q.shape[1]
        self.n_proc = 2  # [RT, nactions]

        mask = ~np.isnan(resp_vals)
        user_idx, item_idx = np.where(mask)
        self.coords = np.stack([user_idx, item_idx], axis=1)
        self.values = resp_vals[user_idx, item_idx]
        self.values[self.values > 1] = 1
        self.RT      = time_vals   # (n_users, n_items)
        self.actions = nact_vals   # (n_users, n_items)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        u, i = self.coords[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor([self.RT[u, i], self.actions[u, i]], dtype=torch.float),
            torch.tensor(self.values[idx], dtype=torch.float),
        )

    def findIndex(self, row, column):
        matches = np.where(
            (self.coords[:, 0] == row) &
            (self.coords[:, 1] == column)
        )
        return matches[0][0] if len(matches[0]) > 0 else None
