"""Canonical data contracts for financial datasets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CanonicalColumns:
    symbol: str = "symbol"
    timestamp: str = "timestamp"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"
    value: str = "value"
    tradable: str = "tradable"


@dataclass(frozen=True)
class DatasetContract:
    feature_columns: list[str]
    label_columns: list[str]
    index_columns: list[str] = field(default_factory=lambda: ["symbol", "timestamp"])
    mask_columns: list[str] = field(default_factory=lambda: ["tradable"])


DEFAULT_COLUMNS = CanonicalColumns()
