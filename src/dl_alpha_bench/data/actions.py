"""Corporate action interface."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CorporateActionAdjuster:
    """Interface to apply corporate action adjustments.

    The default behavior is identity so the pipeline stays deterministic
    when no adjustment table is provided.
    """

    def apply(self, frame: pd.DataFrame, actions: pd.DataFrame | None = None) -> pd.DataFrame:
        if actions is None or actions.empty:
            return frame
        normalized_actions = self._normalize_actions(actions)
        merged = frame.merge(normalized_actions, on=["symbol", "timestamp"], how="left")
        factor = merged.get("adjust_factor", 1.0).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            if col in merged.columns:
                merged[col] = merged[col] * factor
        return merged.drop(columns=["adjust_factor"], errors="ignore")

    def _normalize_actions(self, actions: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "timestamp", "adjust_factor"}
        if not required.issubset(actions.columns):
            missing = sorted(required - set(actions.columns))
            raise ValueError(f"corporate actions missing required columns: {missing}")

        out = actions[["symbol", "timestamp", "adjust_factor"]].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        if out["timestamp"].isna().any():
            raise ValueError("corporate actions contains invalid timestamp values")

        out["adjust_factor"] = pd.to_numeric(out["adjust_factor"], errors="coerce")
        if out["adjust_factor"].isna().any():
            raise ValueError("corporate actions contains non-numeric adjust_factor values")

        conflicts = out.groupby(["symbol", "timestamp"])["adjust_factor"].nunique(dropna=False)
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            symbol, timestamp = conflicts.index[0]
            raise ValueError(
                "corporate actions contains conflicting adjust_factor values "
                f"for symbol={symbol}, timestamp={timestamp}"
            )

        return out.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
