from __future__ import annotations

import pandas as pd
import pytest

from dl_alpha_bench.data import CorporateActionAdjuster


def _price_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "timestamp": "2024-01-01",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
            },
            {
                "symbol": "AAA",
                "timestamp": "2024-01-02",
                "open": 10.1,
                "high": 10.6,
                "low": 9.9,
                "close": 10.3,
            },
        ]
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def test_adjuster_deduplicates_repeated_actions_without_row_expansion() -> None:
    frame = _price_frame()
    actions = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-01-01", "adjust_factor": 2.0},
            {"symbol": "AAA", "timestamp": "2024-01-01", "adjust_factor": 2.0},
        ]
    )

    out = CorporateActionAdjuster().apply(frame, actions)
    assert len(out) == len(frame)
    assert out.loc[0, "close"] == pytest.approx(20.4)
    assert out.loc[1, "close"] == pytest.approx(10.3)


def test_adjuster_rejects_conflicting_action_factors() -> None:
    frame = _price_frame()
    actions = pd.DataFrame(
        [
            {"symbol": "AAA", "timestamp": "2024-01-01", "adjust_factor": 2.0},
            {"symbol": "AAA", "timestamp": "2024-01-01", "adjust_factor": 1.5},
        ]
    )

    with pytest.raises(ValueError, match="conflicting adjust_factor"):
        CorporateActionAdjuster().apply(frame, actions)
