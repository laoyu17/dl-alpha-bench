"""Runtime data contract validator."""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ("symbol", "timestamp", "close")
SORT_KEYS = ("symbol", "timestamp")


def validate_data_contract(frame: pd.DataFrame) -> None:
    """Validate minimal runtime data contract before dataset construction."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("input data must be a pandas DataFrame")

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"input data missing required columns: {missing}")

    if frame["symbol"].isna().any():
        raise ValueError("input data contains null symbol values")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("input data contains unparseable timestamp values")

    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any():
        raise ValueError("input data contains non-numeric close values")

    ordered = frame.assign(timestamp=timestamps).sort_values(list(SORT_KEYS), kind="stable")
    if not ordered.index.equals(frame.index):
        raise ValueError("input data must be sorted by symbol, timestamp")

    if "tradable" in frame.columns:
        invalid_values = [
            value
            for value in frame["tradable"].dropna().tolist()
            if not _is_bool_like(value)
        ]
        if invalid_values:
            sample = invalid_values[0]
            raise ValueError(
                "tradable column must be boolean-like (true/false/0/1), "
                f"but got value: {sample!r}"
            )


def _is_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return value in (0, 1)
    if isinstance(value, float):
        return value in (0.0, 1.0)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "false", "0", "1"}
    return False
