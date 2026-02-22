from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_alpha_bench.dataset import DatasetBuilder


def _make_micro_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | str | pd.Timestamp | bool]] = []
    for symbol in ["AAA", "BBB"]:
        close = 100.0
        for ts in pd.date_range("2024-01-01", periods=120, freq="min"):
            close += float(rng.normal(0, 0.05))
            spread = abs(float(rng.normal(0.02, 0.005)))
            bid_size1 = float(rng.integers(100, 3000))
            ask_size1 = float(rng.integers(100, 3000))
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": ts,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": float(rng.integers(1000, 6000)),
                    "value": close * float(rng.integers(1000, 6000)),
                    "bid_price1": close - spread / 2,
                    "ask_price1": close + spread / 2,
                    "bid_size1": bid_size1,
                    "ask_size1": ask_size1,
                    "trade_count": float(rng.integers(10, 400)),
                    "tradable": True,
                }
            )
    return pd.DataFrame(rows)


def test_microstructure_feature_generation() -> None:
    frame = _make_micro_frame()
    cfg = {
        "strict_feature_requirements": True,
        "feature_generators": [
            "ret_1",
            "spread",
            "obi_l1",
            "trade_intensity",
            "short_vol",
            "microprice_bias",
            "ofi_l1",
        ],
        "feature_params": {
            "trade_intensity_window": 10,
            "short_vol_window": 15,
            "ofi_window": 12,
        },
        "feature_columns": [
            "feat_ret_1",
            "feat_spread",
            "feat_obi_l1",
            "feat_trade_intensity",
            "feat_short_vol",
            "feat_microprice_bias",
            "feat_ofi_l1",
        ],
        "label_horizons": [3],
    }

    ds = DatasetBuilder().build(frame, cfg)
    assert set(cfg["feature_columns"]).issubset(ds.frame.columns)
    assert ds.features().shape[1] == len(cfg["feature_columns"])
    assert np.isfinite(ds.features()).all()


def test_microstructure_strict_missing_columns_raises() -> None:
    frame = _make_micro_frame().drop(columns=["bid_size1"])
    cfg = {
        "strict_feature_requirements": True,
        "feature_generators": ["obi_l1"],
        "feature_columns": ["feat_obi_l1"],
        "label_horizons": [1],
    }

    with pytest.raises(ValueError, match="obi_l1"):
        DatasetBuilder().build(frame, cfg)
