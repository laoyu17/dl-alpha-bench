"""Leakage-safe splitters for financial ML experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Split:
    train_idx: np.ndarray
    valid_idx: np.ndarray


@dataclass
class LeakageReport:
    passed: bool
    details: list[str]


class PurgedKFoldSplitter:
    def __init__(self, n_splits: int = 5, label_horizon: int = 1, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.label_horizon = max(1, int(label_horizon))
        self.embargo = max(0, int(embargo))

    def split(self, timestamps: pd.Series | np.ndarray) -> list[Split]:
        ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
        unique_ts = np.array(sorted(ts.unique()))
        folds = np.array_split(unique_ts, self.n_splits)
        ts_to_pos = {v: i for i, v in enumerate(unique_ts)}
        pos_arr = ts.map(ts_to_pos).to_numpy(dtype=int)

        result: list[Split] = []
        for fold in folds:
            if len(fold) == 0:
                continue
            valid_pos = np.array([ts_to_pos[t] for t in fold], dtype=int)
            vmin = int(valid_pos.min())
            vmax = int(valid_pos.max())
            purge_left = max(0, vmin - self.label_horizon + 1)
            embargo_right = min(len(unique_ts) - 1, vmax + self.embargo)

            valid_mask = np.isin(pos_arr, valid_pos)
            forbidden_mask = (pos_arr >= purge_left) & (pos_arr <= embargo_right)
            train_mask = ~valid_mask & ~forbidden_mask

            result.append(
                Split(
                    train_idx=np.where(train_mask)[0],
                    valid_idx=np.where(valid_mask)[0],
                )
            )
        return result


class WalkForwardSplitter:
    def __init__(
        self,
        train_window: int,
        valid_window: int,
        step: int | None = None,
        label_horizon: int = 1,
    ):
        self.train_window = int(train_window)
        self.valid_window = int(valid_window)
        self.step = int(step or valid_window)
        self.label_horizon = max(1, int(label_horizon))
        if self.train_window <= 0 or self.valid_window <= 0 or self.step <= 0:
            raise ValueError("window sizes must be positive")

    def split(self, timestamps: pd.Series | np.ndarray) -> list[Split]:
        ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
        unique_ts = np.array(sorted(ts.unique()))
        ts_to_pos = {v: i for i, v in enumerate(unique_ts)}
        pos_arr = ts.map(ts_to_pos).to_numpy(dtype=int)

        splits: list[Split] = []
        start = 0
        while True:
            train_start = start
            raw_train_end = train_start + self.train_window - 1
            valid_start = raw_train_end + 1
            valid_end = valid_start + self.valid_window - 1
            if valid_end >= len(unique_ts):
                break

            # Purge boundary samples whose forward labels would leak into valid window.
            train_end = raw_train_end - (self.label_horizon - 1)
            if train_end < train_start:
                start += self.step
                continue

            train_pos = np.arange(train_start, train_end + 1)
            valid_pos = np.arange(valid_start, valid_end + 1)
            train_mask = np.isin(pos_arr, train_pos)
            valid_mask = np.isin(pos_arr, valid_pos)

            splits.append(
                Split(
                    train_idx=np.where(train_mask)[0],
                    valid_idx=np.where(valid_mask)[0],
                )
            )
            start += self.step

        return splits


def validate_no_time_overlap(
    timestamps: pd.Series | np.ndarray,
    split: Split,
    label_horizon: int = 1,
    embargo: int = 0,
) -> LeakageReport:
    ts = pd.to_datetime(pd.Series(timestamps)).reset_index(drop=True)
    train_ts = set(ts.iloc[split.train_idx])
    valid_ts = sorted(set(ts.iloc[split.valid_idx]))

    details: list[str] = []
    if train_ts.intersection(valid_ts):
        details.append("train/valid timestamps overlap")

    if valid_ts:
        unique_ts = np.array(sorted(ts.unique()))
        ts_to_pos = {v: i for i, v in enumerate(unique_ts)}
        valid_pos = np.array([ts_to_pos[t] for t in valid_ts], dtype=int)
        vmin, vmax = int(valid_pos.min()), int(valid_pos.max())
        purge_left = max(0, vmin - int(label_horizon) + 1)
        embargo_right = min(len(unique_ts) - 1, vmax + int(embargo))
        forbidden = set(unique_ts[purge_left : embargo_right + 1])
        leakage = train_ts.intersection(forbidden)
        if leakage:
            details.append(f"found {len(leakage)} train timestamps in forbidden zone")

    return LeakageReport(passed=not details, details=details)
