from __future__ import annotations

import numpy as np
import pandas as pd

from dl_alpha_bench.eval import summarize_feature_explainability


def _build_panel_frame() -> pd.DataFrame:
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    timestamps = pd.date_range("2024-01-01", periods=30, freq="D")
    symbols = ["A", "B", "C", "D", "E", "F"]

    for ts in timestamps:
        for i, symbol in enumerate(symbols):
            x = float(i + 1)
            y = 0.01 * x
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "feat_pos": x,
                    "feat_neg": -x,
                    "label_fwd_ret_1": y,
                }
            )
    return pd.DataFrame(rows)


def test_feature_explainability_sign_and_sort() -> None:
    frame = _build_panel_frame()
    rows = summarize_feature_explainability(
        frame=frame,
        feature_columns=["feat_pos", "feat_neg"],
        label_column="label_fwd_ret_1",
        timestamp_column="timestamp",
        n_quantiles=3,
    )

    by_feature = {row["feature"]: row for row in rows}
    assert set(by_feature) == {"feat_pos", "feat_neg"}
    assert by_feature["feat_pos"]["ic_mean"] > 0
    assert by_feature["feat_neg"]["ic_mean"] < 0
    assert by_feature["feat_pos"]["quantile_spread_mean"] > 0
    assert by_feature["feat_neg"]["quantile_spread_mean"] < 0


def test_feature_explainability_handles_missing_columns() -> None:
    frame = _build_panel_frame()
    rows = summarize_feature_explainability(
        frame=frame,
        feature_columns=["feat_pos", "feat_missing"],
        label_column="label_fwd_ret_1",
    )

    assert len(rows) == 1
    assert rows[0]["feature"] == "feat_pos"
    assert np.isfinite(rows[0]["rank_ic_mean"])
