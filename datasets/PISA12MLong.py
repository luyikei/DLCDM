import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets.PISA12M import ITEM_NAMES


class PISA12MLong(Dataset):

    def __init__(self):
        # --- Response data ---
        resp = pd.read_csv('datafiles/PISA12M/cleaned.csv')

        # student_id -> 0-based user index
        self.student_ids = resp['student_id'].values
        self.student_to_user = {s: idx for idx, s in enumerate(self.student_ids)}

        # Use only the 9 items covered by the Q matrix (exclude CM015Q01)
        item_cols = [f'item_{item}' for item in ITEM_NAMES]
        self.item_names = list(ITEM_NAMES)
        self.n_users = len(self.student_ids)
        self.n_items = len(item_cols)

        self.Q = pd.read_csv('datafiles/PISA12M/Q.csv').to_numpy(copy=True)
        self.n_skills = self.Q.shape[1]

        # Build (user_idx, item_idx) coordinates from non-NaN response cells
        resp_values = resp[item_cols].to_numpy(copy=True, dtype=float)
        resp_values[resp_values > 1] = 1
        mask = ~np.isnan(resp_values)
        user_idx, item_idx = np.where(mask)
        self.coords = np.stack([user_idx, item_idx], axis=1)
        self.values = resp_values[user_idx, item_idx]

        # --- Log data ---
        log = pd.read_csv('datafiles/PISA12M/log_selected.csv')

        # Combine event_name and event_type into a single event identifier.
        # When event_type is NaN, use event_name alone.
        log['event_combined'] = (
            log['event_name'] + '|' + log['event_type'].fillna('')
        ).str.rstrip('|')

        # Collapse the 1500 least-frequent combined event types into "__other__".
        #counts = log['event_combined'].value_counts()
        #keep = set(counts.iloc[:-1500].index) if len(counts) > 1500 else set(counts.index)
        #log['event_combined'] = log['event_combined'].where(
        #    log['event_combined'].isin(keep), other='__other__'
        #)

        # Build event vocabulary (sorted for reproducibility)
        event_types = sorted(log['event_combined'].unique())
        self.event_type_to_idx = {et: idx for idx, et in enumerate(event_types)}
        self.n_event_types = len(self.event_type_to_idx)

        # Learnable embedding for event types (dim=4)
        self.event_embedding = nn.Embedding(self.n_event_types, 4)

        # Map log rows to (user_idx, item_idx)
        log['user_idx'] = log['student_id'].map(self.student_to_user)
        log['item_idx'] = log['item_id'] - 1   # item_id is 1-based
        log['event_idx'] = log['event_combined'].map(self.event_type_to_idx)

        # Drop rows whose student_id is not in resp
        log = log.dropna(subset=['user_idx'])
        log['user_idx'] = log['user_idx'].astype(int)

        # Pre-group log events by (user_idx, item_idx) for O(1) lookup
        self._log_groups: dict[tuple, np.ndarray] = {}
        for (u, i), grp in log.groupby(['user_idx', 'item_idx'], sort=False):
            arr = grp[['event_idx', 'timestamp']].to_numpy(dtype=float)
            arr[:, 1] = arr[:, 1] / 1000
            self._log_groups[(int(u), int(i))] = arr

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.coords.shape[0]

    def __getitem__(self, idx):
        u, i = int(self.coords[idx, 0]), int(self.coords[idx, 1])
        y = float(self.values[idx])

        log_data = self._log_groups.get((u, i))
        if log_data is not None and len(log_data) > 0:
            event_idxs = torch.tensor(log_data[:, 0], dtype=torch.long)
            timestamps = torch.tensor(log_data[:, 1], dtype=torch.float)
            embeddings = self.event_embedding(event_idxs)          # (T, 4)
            log_matrix = torch.cat([embeddings,
                                    timestamps.unsqueeze(1)], dim=1)  # (T, 5)
        else:
            event_idxs = torch.zeros(0, dtype=torch.long)
            log_matrix = torch.zeros(0, 5)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i, dtype=torch.long),
            log_matrix,    # (T, 5)  embedded features + timestamp
            event_idxs,    # (T,)    raw event type indices for next-event loss
            torch.tensor(y, dtype=torch.float),
        )

    def findIndex(self, row, column):
        matches = np.where(
            (self.coords[:, 0] == row) &
            (self.coords[:, 1] == column)
        )
        return matches[0][0] if len(matches[0]) > 0 else None
