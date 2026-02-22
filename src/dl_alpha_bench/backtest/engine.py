"""Lightweight event-driven style backtest for cross-sectional signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    avg_turnover: float
    n_periods: int


class EventBacktester:
    def __init__(self, top_frac: float = 0.2, trading_cost: float = 0.0005):
        self.top_frac = float(top_frac)
        self.trading_cost = float(trading_cost)

    def run(self, frame: pd.DataFrame, score_col: str, ret_col: str) -> BacktestSummary:
        data = frame[["timestamp", "symbol", score_col, ret_col]].copy()
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        period_rets: list[float] = []
        turnovers: list[float] = []
        prev_long: set[str] = set()
        prev_short: set[str] = set()

        for ts, grp in data.groupby("timestamp"):
            _ = ts
            grp = grp.dropna(subset=[score_col, ret_col])
            if grp.empty:
                continue

            n = max(1, int(len(grp) * self.top_frac))
            ranked = grp.sort_values(score_col, ascending=False)
            long_part = ranked.head(n)
            short_part = ranked.tail(n)

            long_symbols = set(long_part["symbol"].astype(str).tolist())
            short_symbols = set(short_part["symbol"].astype(str).tolist())

            long_ret = float(long_part[ret_col].mean()) if not long_part.empty else 0.0
            short_ret = float(short_part[ret_col].mean()) if not short_part.empty else 0.0
            gross = long_ret - short_ret

            if prev_long or prev_short:
                changed = len(prev_long.symmetric_difference(long_symbols)) + len(
                    prev_short.symmetric_difference(short_symbols)
                )
                denom = max(1, len(prev_long) + len(prev_short))
                turnover = changed / denom
            else:
                turnover = 1.0

            cost = turnover * self.trading_cost
            net = gross - cost
            period_rets.append(net)
            turnovers.append(turnover)
            prev_long, prev_short = long_symbols, short_symbols

        if not period_rets:
            return BacktestSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)

        rets = np.array(period_rets, dtype=float)
        equity = np.cumprod(1 + rets)
        total_return = float(equity[-1] - 1.0)
        annual_return = float((1 + total_return) ** (252 / len(rets)) - 1)
        std = float(np.std(rets))
        sharpe = float(np.mean(rets) / std * np.sqrt(252)) if std > 0 else 0.0

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_drawdown = float(np.min(dd))

        return BacktestSummary(
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            avg_turnover=float(np.mean(turnovers)),
            n_periods=len(rets),
        )
