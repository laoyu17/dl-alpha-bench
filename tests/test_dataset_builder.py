from __future__ import annotations

import pandas as pd

from dl_alpha_bench.dataset import DatasetBuilder


def _make_frame() -> pd.DataFrame:
    rows = []
    for symbol in ["AAA", "BBB"]:
        close = 100.0
        for d in pd.date_range("2023-01-01", periods=30, freq="D"):
            close += 1
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": d,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1000,
                    "value": close * 1000,
                    "tradable": True,
                }
            )
    return pd.DataFrame(rows)


def test_dataset_builder_generates_features_and_labels() -> None:
    frame = _make_frame()
    cfg = {
        "feature_generators": ["ret_1", "ret_5", "vol_z", "spread"],
        "feature_columns": ["feat_ret_1", "feat_ret_5", "feat_vol_z", "feat_spread"],
        "label_horizons": [1, 3],
    }
    ds = DatasetBuilder().build(frame, cfg)

    assert "label_fwd_ret_1" in ds.frame.columns
    assert "label_fwd_ret_3" in ds.frame.columns
    assert ds.features().shape[1] == 4
    assert ds.labels().shape[1] == 2
    assert ds.mask().dtype == bool


def test_dataset_builder_applies_mask_by_default() -> None:
    frame = _make_frame()
    frame.loc[frame.index[10:20], "tradable"] = False
    cfg = {
        "feature_generators": ["ret_1"],
        "feature_columns": ["feat_ret_1"],
        "label_horizons": [1],
    }

    ds = DatasetBuilder().build(frame, cfg)
    assert ds.frame["tradable"].all()


def test_dataset_builder_can_keep_untradable_rows_when_apply_mask_disabled() -> None:
    frame = _make_frame()
    frame.loc[frame.index[10:20], "tradable"] = False
    cfg = {
        "feature_generators": ["ret_1"],
        "feature_columns": ["feat_ret_1"],
        "label_horizons": [1],
        "apply_mask": False,
    }

    ds = DatasetBuilder().build(frame, cfg)
    assert not ds.frame["tradable"].all()
