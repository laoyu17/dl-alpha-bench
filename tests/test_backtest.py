from __future__ import annotations

import pandas as pd

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
