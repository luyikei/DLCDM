import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from datasets.PISA12R import PISA12R, ITEM_NAMES


def _make_cleaned_csv(n_users=5, nan_fraction=0.2, seed=0):
    """Return a DataFrame matching the format PISA12R expects."""
    rng = np.random.default_rng(seed)
    data = {"student_id": [f"user_{i}" for i in range(n_users)]}

    resp_vals = rng.integers(0, 2, size=(n_users, len(ITEM_NAMES))).astype(float)
    time_vals = rng.uniform(1000, 60000, size=(n_users, len(ITEM_NAMES)))
    nact_vals = rng.integers(1, 20, size=(n_users, len(ITEM_NAMES))).astype(float)

    # Introduce NaNs in responses
    nan_mask = rng.random(size=(n_users, len(ITEM_NAMES))) < nan_fraction
    resp_vals[nan_mask] = np.nan

    for j, item in enumerate(ITEM_NAMES):
        data[f"resp_{item}"] = resp_vals[:, j]
        data[f"time_{item}"] = time_vals[:, j]
        data[f"nactions_{item}"] = nact_vals[:, j]

    return pd.DataFrame(data)


def _make_Q_csv(n_skills=3):
    """Return a Q-matrix DataFrame with shape (len(ITEM_NAMES), n_skills)."""
    data = {f"Factor{k+1}": [0] * len(ITEM_NAMES) for k in range(n_skills)}
    # Give each item at least one skill
    for j in range(len(ITEM_NAMES)):
        data[f"Factor{(j % n_skills) + 1}"][j] = 1
    return pd.DataFrame(data)


N_USERS = 5
N_SKILLS = 3
_CLEANED = _make_cleaned_csv(n_users=N_USERS)
_Q = _make_Q_csv(n_skills=N_SKILLS)


def _read_csv_side_effect(path, *args, **kwargs):
    if "cleaned" in path:
        return _CLEANED.copy()
    if "/Q" in path or "Q.csv" in path:
        return _Q.copy()
    raise FileNotFoundError(f"Unexpected path: {path}")


@patch("datasets.PISA12R.pd.read_csv", side_effect=_read_csv_side_effect)
class TestPISA12R(unittest.TestCase):

    def _ds(self, mock_read):
        return PISA12R()

    # ------------------------------------------------------------------
    # Initialization / metadata
    # ------------------------------------------------------------------

    def test_n_users(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.n_users, N_USERS)

    def test_n_items(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.n_items, len(ITEM_NAMES))

    def test_n_skills(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.n_skills, N_SKILLS)

    def test_n_proc(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.n_proc, 2)

    def test_item_names(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.item_names, ITEM_NAMES)

    # ------------------------------------------------------------------
    # Length / coordinates
    # ------------------------------------------------------------------

    def test_len_matches_non_nan_responses(self, mock_read):
        ds = self._ds(mock_read)
        resp_cols = [f"resp_{item}" for item in ITEM_NAMES]
        expected = int(np.sum(~_CLEANED[resp_cols].isna().to_numpy()))
        self.assertEqual(len(ds), expected)

    def test_coords_shape(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.coords.shape, (len(ds), 2))

    def test_coords_within_bounds(self, mock_read):
        ds = self._ds(mock_read)
        self.assertTrue((ds.coords[:, 0] < N_USERS).all())
        self.assertTrue((ds.coords[:, 1] < len(ITEM_NAMES)).all())

    # ------------------------------------------------------------------
    # __getitem__ return format
    # ------------------------------------------------------------------

    def test_getitem_returns_four_tensors(self, mock_read):
        ds = self._ds(mock_read)
        out = ds[0]
        self.assertEqual(len(out), 4)

    def test_getitem_user_idx_dtype(self, mock_read):
        ds = self._ds(mock_read)
        u, i, proc, resp = ds[0]
        self.assertEqual(u.dtype, torch.long)

    def test_getitem_item_idx_dtype(self, mock_read):
        ds = self._ds(mock_read)
        u, i, proc, resp = ds[0]
        self.assertEqual(i.dtype, torch.long)

    def test_getitem_proc_shape_and_dtype(self, mock_read):
        ds = self._ds(mock_read)
        u, i, proc, resp = ds[0]
        self.assertEqual(proc.shape, (2,))
        self.assertEqual(proc.dtype, torch.float)

    def test_getitem_resp_scalar_dtype(self, mock_read):
        ds = self._ds(mock_read)
        u, i, proc, resp = ds[0]
        self.assertEqual(resp.shape, ())
        self.assertEqual(resp.dtype, torch.float)

    def test_getitem_resp_is_zero_or_one(self, mock_read):
        ds = self._ds(mock_read)
        for idx in range(len(ds)):
            _, _, _, resp = ds[idx]
            self.assertIn(resp.item(), {0.0, 1.0})

    def test_getitem_proc_rt_is_log_scale(self, mock_read):
        """RT stored as log(ms/1000) so all values should be finite."""
        ds = self._ds(mock_read)
        for idx in range(len(ds)):
            _, _, proc, _ = ds[idx]
            self.assertTrue(torch.isfinite(proc[0]))

    def test_getitem_indices_in_bounds(self, mock_read):
        ds = self._ds(mock_read)
        for idx in range(len(ds)):
            u, i, _, _ = ds[idx]
            self.assertGreaterEqual(u.item(), 0)
            self.assertLess(u.item(), N_USERS)
            self.assertGreaterEqual(i.item(), 0)
            self.assertLess(i.item(), len(ITEM_NAMES))

    # ------------------------------------------------------------------
    # findIndex
    # ------------------------------------------------------------------

    def test_find_index_known_pair(self, mock_read):
        ds = self._ds(mock_read)
        u, i = int(ds.coords[0, 0]), int(ds.coords[0, 1])
        result = ds.findIndex(u, i)
        self.assertEqual(result, 0)

    def test_find_index_missing_pair_returns_none(self, mock_read):
        """A (user, item) pair with a NaN response should not be in coords."""
        ds = self._ds(mock_read)
        resp_cols = [f"resp_{item}" for item in ITEM_NAMES]
        nan_mask = _CLEANED[resp_cols].isna().to_numpy()
        nan_positions = list(zip(*np.where(nan_mask)))
        if nan_positions:
            u, i = nan_positions[0]
            self.assertIsNone(ds.findIndex(u, i))
        else:
            self.skipTest("No NaN entries in synthetic data to test missing pair")

    def test_find_index_all_observed_coords_found(self, mock_read):
        ds = self._ds(mock_read)
        for k in range(len(ds)):
            u, i = int(ds.coords[k, 0]), int(ds.coords[k, 1])
            result = ds.findIndex(u, i)
            self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # Q matrix
    # ------------------------------------------------------------------

    def test_Q_shape(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.Q.shape, (len(ITEM_NAMES), N_SKILLS))

    # ------------------------------------------------------------------
    # RT and actions arrays
    # ------------------------------------------------------------------

    def test_RT_shape(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.RT.shape, (N_USERS, len(ITEM_NAMES)))

    def test_actions_shape(self, mock_read):
        ds = self._ds(mock_read)
        self.assertEqual(ds.actions.shape, (N_USERS, len(ITEM_NAMES)))


class TestPISA12RFoldAlignment(unittest.TestCase):
    """Diagnostic tests using real data files to check folds vs dataset alignment."""

    CLEANED_PATH = "datafiles/PISA12R/cleaned.csv"
    FOLDS_PATH   = "datafiles/PISA12R/loo_cv_fold_PISA12R.csv"

    @classmethod
    def setUpClass(cls):
        import os
        if not os.path.exists(cls.CLEANED_PATH) or not os.path.exists(cls.FOLDS_PATH):
            raise unittest.SkipTest("Real data files not found; skipping alignment tests")
        cls.ds    = PISA12R()
        cls.folds = pd.read_csv(cls.FOLDS_PATH)

    def test_cleaned_csv_has_resp_columns(self):
        """cleaned.csv must use resp_<item> column names."""
        raw = pd.read_csv(self.CLEANED_PATH, nrows=0)
        missing = [f"resp_{item}" for item in ITEM_NAMES if f"resp_{item}" not in raw.columns]
        self.assertEqual(missing, [],
            f"cleaned.csv is missing resp_ columns: {missing}\n"
            f"Actual columns: {list(raw.columns)}")

    def test_folds_columns_match_item_names(self):
        """Fold CSV columns must match ITEM_NAMES exactly."""
        fold_items = list(self.folds.columns)
        self.assertEqual(fold_items, ITEM_NAMES,
            f"Folds columns: {fold_items}\nITEM_NAMES:    {ITEM_NAMES}")

    def test_no_none_from_findIndex_in_folds(self):
        """Every (user, item) entry in the folds CSV must resolve to a dataset index."""
        none_entries = []
        for fold_col_idx, item_name in enumerate(self.folds.columns):
            item_idx = ITEM_NAMES.index(item_name) if item_name in ITEM_NAMES else None
            for user_idx in range(len(self.folds)):
                fold_val = self.folds.iloc[user_idx, fold_col_idx]
                if pd.isna(fold_val):
                    continue
                if item_idx is None:
                    none_entries.append((user_idx, fold_col_idx, item_name, "item not in ITEM_NAMES"))
                    continue
                result = self.ds.findIndex(user_idx, item_idx)
                if result is None:
                    none_entries.append((user_idx, item_idx, item_name, "NaN in cleaned.csv"))

        if none_entries:
            # Summarise by item
            from collections import Counter
            by_item = Counter(e[2] for e in none_entries)
            self.fail(
                f"{len(none_entries)} fold entries resolve to None.\n"
                f"Breakdown by item: {dict(by_item)}\n"
                f"First 10 entries (user_idx, item_idx, item_name, reason):\n"
                + "\n".join(str(e) for e in none_entries[:10])
            )


if __name__ == "__main__":
    unittest.main()
