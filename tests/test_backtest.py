from __future__ import annotations

import pandas as pd
import pytest

from dl_alpha_bench.backtest import EventBacktester


def test_backtest_runs_and_returns_summary() -> None:
    ts = pd.date_range("2023-01-01", periods=10, freq="D")
    rows = []
    for t in ts:
        for idx, symbol in enumerate(["A", "B", "C", "D", "E"]):
            rows.append(
                {
                    "timestamp": t,
                    "symbol": symbol,
                    "score": float(5 - idx),
                    "ret": float(0.01 - idx * 0.002),
                }
            )
    frame = pd.DataFrame(rows)
    summary = EventBacktester(top_frac=0.2, trading_cost=0.0).run(frame, "score", "ret")
    assert summary.n_periods == 10
    assert summary.total_return > -1.0


def test_backtest_keeps_long_short_disjoint_with_large_top_frac() -> None:
    ts = pd.date_range("2024-01-01", periods=2, freq="D")
    scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    rets = [0.05, 0.04, 0.03, -0.02, -0.03]
    symbols = ["A", "B", "C", "D", "E"]

    rows = []
    for stamp in ts:
        for symbol, score, ret in zip(symbols, scores, rets):
            rows.append(
                {
                    "timestamp": stamp,
                    "symbol": symbol,
                    "score": score,
                    "ret": ret,
                }
            )
    frame = pd.DataFrame(rows)
    summary = EventBacktester(top_frac=0.8, trading_cost=0.0).run(frame, "score", "ret")

    expected_period_ret = 0.07
    expected_total = (1.0 + expected_period_ret) ** 2 - 1.0
    assert summary.n_periods == 2
    assert summary.total_return == pytest.approx(expected_total)
