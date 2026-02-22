from __future__ import annotations

import numpy as np
import pandas as pd

from dl_alpha_bench.cv import PurgedKFoldSplitter, WalkForwardSplitter, validate_no_time_overlap


def test_purged_kfold_blocks_forbidden_zone() -> None:
    ts = pd.date_range("2023-01-01", periods=20, freq="D")
    splitters = PurgedKFoldSplitter(n_splits=4, label_horizon=3, embargo=2)
    splits = splitters.split(pd.Series(ts))

    assert len(splits) == 4
    for split in splits:
        report = validate_no_time_overlap(pd.Series(ts), split, label_horizon=3, embargo=2)
        assert report.passed, report.details


def test_walk_forward_is_chronological() -> None:
    ts = pd.Series(pd.date_range("2023-01-01", periods=30, freq="D"))
    splitter = WalkForwardSplitter(train_window=10, valid_window=5, step=5)
    splits = splitter.split(ts)
    assert splits
    for split in splits:
        train_max = ts.iloc[split.train_idx].max()
        valid_min = ts.iloc[split.valid_idx].min()
        assert train_max < valid_min


def test_walk_forward_purges_boundary_for_label_horizon() -> None:
    ts = pd.Series(pd.date_range("2024-01-01", periods=40, freq="D"))
    splitter = WalkForwardSplitter(train_window=20, valid_window=5, step=5, label_horizon=10)
    splits = splitter.split(ts)
    assert splits
    for split in splits:
        report = validate_no_time_overlap(ts, split, label_horizon=10, embargo=0)
        assert report.passed, report.details


def test_purged_kfold_has_non_empty_train_and_valid() -> None:
    ts = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    splitter = PurgedKFoldSplitter(n_splits=5, label_horizon=2, embargo=1)
    for split in splitter.split(ts):
        assert split.train_idx.size > 0
        assert split.valid_idx.size > 0
        assert np.intersect1d(split.train_idx, split.valid_idx).size == 0
