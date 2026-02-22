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
        merged = frame.merge(actions, on=["symbol", "timestamp"], how="left")
        factor = merged.get("adjust_factor", 1.0).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            if col in merged.columns:
                merged[col] = merged[col] * factor
        return merged.drop(columns=["adjust_factor"], errors="ignore")
