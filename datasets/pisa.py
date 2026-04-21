import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
import torch

class PISA(Dataset):
    """
    Returns (user_idx, item_idx, resp) for each observed cell in a user by item matrix.
    """
    def __init__(self):
        self.Q = pd.read_csv('datafiles/Q matrix.csv')
        self.resp = pd.read_csv('datafiles/response.csv')
        self.RT = pd.read_csv('datafiles/RTs.csv')
        self.actions = pd.read_csv('datafiles/the number of actions.csv')

        self.resp.columns = self.resp.columns.str[:-1]
        self.RT.columns = self.RT.columns.str[:-2]
        self.actions.columns = self.actions.columns.str[:-1]

        self.Q = pd.read_csv('datafiles/Q matrix.csv').to_numpy(copy=True)
        self.RT = np.log(pd.read_csv('datafiles/RTs.csv').to_numpy(copy=True) / 1000)
        self.actions = pd.read_csv('datafiles/the number of actions.csv').to_numpy(copy=True)

        self.user_ids = np.arange(self.resp.shape[0], dtype=np.int64)
        self.item_ids = np.arange(self.resp.shape[1], dtype=np.int64)
        self.item_names = list(self.resp.columns)

        values = self.resp.to_numpy(copy=True)
        mask = ~pd.isna(values)
        user_idx, item_idx = np.where(mask)
        self.coords = np.stack([user_idx, item_idx], axis=1)
        self.values = values[user_idx, item_idx]

        self.n_users, self.n_items = self.resp.shape
        self.n_skills = self.Q.shape[1]
        self.n_proc = 2

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        u, i = self.coords[idx]
        y = self.values[idx]
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            torch.tensor([self.RT[u, i], self.actions[u, i]], dtype=torch.float),
            torch.tensor(y, dtype=torch.float),
        )
    
    def findIndex(self, row, column):
        matches = np.where(
            (self.coords[:, 0] == row) &
            (self.coords[:, 1] == column)
        )
        idx = matches[0][0] if len(matches) > 0 else None
        return idx
