"""Feature-level explainability metrics for factor interviews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .metrics import information_coefficient, rank_information_coefficient


@dataclass
class FeatureExplainabilityRow:
    feature: str
    ic_mean: float
    rank_ic_mean: float
    ic_ir: float
    rank_ic_ir: float
    quantile_top_mean: float
    quantile_bottom_mean: float
    quantile_spread_mean: float
    quantile_positive_ratio: float
    n_obs: int
    n_periods: int
    n_periods_used: int


def summarize_feature_explainability(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    timestamp_column: str = "timestamp",
    n_quantiles: int = 5,
) -> list[dict[str, Any]]:
    if label_column not in frame.columns:
        raise ValueError(f"label column `{label_column}` not found")

    quantiles = max(2, int(n_quantiles))
    rows: list[FeatureExplainabilityRow] = []
    for feature in feature_columns:
        if feature not in frame.columns:
            continue
        row = _summarize_single_feature(
            frame=frame,
            feature=feature,
            label_column=label_column,
            timestamp_column=timestamp_column,
            n_quantiles=quantiles,
        )
        rows.append(row)

    out = [asdict(row) for row in rows]
    out.sort(key=lambda item: _abs_or_neg_inf(item["rank_ic_mean"]), reverse=True)
    return out


def _summarize_single_feature(
    frame: pd.DataFrame,
    feature: str,
    label_column: str,
    timestamp_column: str,
    n_quantiles: int,
) -> FeatureExplainabilityRow:
    sub = frame[[timestamp_column, feature, label_column]].dropna().copy()
    if sub.empty:
        return FeatureExplainabilityRow(
            feature=feature,
            ic_mean=float("nan"),
            rank_ic_mean=float("nan"),
            ic_ir=float("nan"),
            rank_ic_ir=float("nan"),
            quantile_top_mean=float("nan"),
            quantile_bottom_mean=float("nan"),
            quantile_spread_mean=float("nan"),
            quantile_positive_ratio=float("nan"),
            n_obs=0,
            n_periods=0,
            n_periods_used=0,
        )

    ic_list: list[float] = []
    rank_ic_list: list[float] = []
    top_ret_list: list[float] = []
    bottom_ret_list: list[float] = []
    spread_list: list[float] = []

    grouped = sub.groupby(timestamp_column)
    for _, grp in grouped:
        x = grp[feature].to_numpy(dtype=float)
        y = grp[label_column].to_numpy(dtype=float)

        if len(grp) >= 2:
            ic_list.append(information_coefficient(x, y))
            rank_ic_list.append(rank_information_coefficient(x, y))

        bucket = _make_quantile_bucket(grp[feature], n_quantiles)
        if bucket is None:
            continue

        quant_ret = grp[label_column].groupby(bucket).mean().sort_index()
        if len(quant_ret) < 2:
            continue
        top_ret = float(quant_ret.iloc[-1])
        bottom_ret = float(quant_ret.iloc[0])

        top_ret_list.append(top_ret)
        bottom_ret_list.append(bottom_ret)
        spread_list.append(top_ret - bottom_ret)

    return FeatureExplainabilityRow(
        feature=feature,
        ic_mean=_nanmean(ic_list),
        rank_ic_mean=_nanmean(rank_ic_list),
        ic_ir=_information_ratio(ic_list),
        rank_ic_ir=_information_ratio(rank_ic_list),
        quantile_top_mean=_nanmean(top_ret_list),
        quantile_bottom_mean=_nanmean(bottom_ret_list),
        quantile_spread_mean=_nanmean(spread_list),
        quantile_positive_ratio=_positive_ratio(spread_list),
        n_obs=int(len(sub)),
        n_periods=int(sub[timestamp_column].nunique()),
        n_periods_used=int(len(spread_list)),
    )


def _make_quantile_bucket(values: pd.Series, n_quantiles: int) -> pd.Series | None:
    if values.nunique(dropna=True) < 2:
        return None
    effective_q = min(n_quantiles, int(values.notna().sum()))
    if effective_q < 2:
        return None

    ranked = values.rank(method="first")
    try:
        bucket = pd.qcut(ranked, q=effective_q, labels=False, duplicates="drop")
    except ValueError:
        return None
    if bucket.nunique(dropna=True) < 2:
        return None
    return bucket


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(values))


def _information_ratio(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    std = float(arr.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(arr.mean() / std)


def _positive_ratio(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > 0))


def _abs_or_neg_inf(value: float) -> float:
    if value is None:
        return float("-inf")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if not np.isfinite(val):
        return float("-inf")
    return abs(val)
