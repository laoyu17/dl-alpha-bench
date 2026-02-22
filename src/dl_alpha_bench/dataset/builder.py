"""Dataset builder for aligned feature/label/mask outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dl_alpha_bench.data import CorporateActionAdjuster, DatasetContract


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_columns: list[str]
    mask_column: str

    def features(self) -> np.ndarray:
        return self.frame[self.feature_columns].to_numpy(dtype=float)

    def labels(self) -> np.ndarray:
        return self.frame[self.label_columns].to_numpy(dtype=float)

    def mask(self) -> np.ndarray:
        return self.frame[self.mask_column].to_numpy(dtype=bool)


class DatasetBuilder:
    def __init__(self, adjuster: CorporateActionAdjuster | None = None):
        self.adjuster = adjuster or CorporateActionAdjuster()

    def build(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        contract: DatasetContract | None = None,
        actions: pd.DataFrame | None = None,
    ) -> DatasetBundle:
        contract = contract or DatasetContract(feature_columns=[], label_columns=[])
        frame = raw.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        frame = self.adjuster.apply(frame, actions)
        frame = self._build_features(frame, config)
        frame, label_cols = self._build_labels(frame, config)

        feature_cols = config.get("feature_columns") or contract.feature_columns
        if not feature_cols:
            feature_cols = [c for c in frame.columns if c.startswith("feat_")]

        mask_col = config.get("mask_column", "tradable")
        apply_mask = bool(config.get("apply_mask", True))
        if mask_col not in frame.columns:
            frame[mask_col] = True
        frame[mask_col] = frame[mask_col].fillna(False).astype(bool)

        required = ["symbol", "timestamp", mask_col] + feature_cols + label_cols
        frame = frame[required].dropna(subset=feature_cols + label_cols).reset_index(drop=True)
        if apply_mask:
            frame = frame[frame[mask_col]].reset_index(drop=True)

        return DatasetBundle(
            frame=frame,
            feature_columns=feature_cols,
            label_columns=label_cols,
            mask_column=mask_col,
        )

    def _build_features(self, frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
        methods = config.get("feature_generators", ["ret_1", "ret_5", "vol_z"])
        params = config.get("feature_params", {})
        strict = bool(config.get("strict_feature_requirements", True))

        vol_z_window = int(params.get("vol_z_window", 20))
        trade_intensity_window = int(params.get("trade_intensity_window", 20))
        short_vol_window = int(params.get("short_vol_window", 20))
        ofi_window = int(params.get("ofi_window", 10))
        grouped = frame.groupby("symbol", group_keys=False)

        if "ret_1" in methods:
            frame["feat_ret_1"] = grouped["close"].pct_change(1)
        if "ret_5" in methods:
            frame["feat_ret_5"] = grouped["close"].pct_change(5)
        if "vol_z" in methods and "volume" in frame.columns:
            rolling_mean = (
                grouped["volume"].rolling(vol_z_window).mean().reset_index(level=0, drop=True)
            )
            rolling_std = (
                grouped["volume"].rolling(vol_z_window).std().reset_index(level=0, drop=True)
            )
            frame["feat_vol_z"] = (frame["volume"] - rolling_mean) / rolling_std.replace(0, np.nan)
        if "spread" in methods and self._has_columns(
            frame,
            ["high", "low", "close"],
            "spread",
            strict,
        ):
            frame["feat_spread"] = (
                frame["high"] - frame["low"]
            ) / frame["close"].replace(0, np.nan)
        if "obi_l1" in methods and self._has_columns(
            frame,
            ["bid_size1", "ask_size1"],
            "obi_l1",
            strict,
        ):
            depth = (frame["bid_size1"] + frame["ask_size1"]).replace(0, np.nan)
            frame["feat_obi_l1"] = (frame["bid_size1"] - frame["ask_size1"]) / depth
        if "trade_intensity" in methods and self._has_columns(
            frame,
            ["trade_count"],
            "trade_intensity",
            strict,
        ):
            rolling_count = grouped["trade_count"].rolling(trade_intensity_window).mean()
            rolling_count = rolling_count.reset_index(level=0, drop=True).replace(0, np.nan)
            frame["feat_trade_intensity"] = frame["trade_count"] / rolling_count
        if "short_vol" in methods and self._has_columns(frame, ["close"], "short_vol", strict):
            log_close = np.log(frame["close"].replace(0, np.nan))
            log_ret = log_close.groupby(frame["symbol"]).diff()
            rolling_std = log_ret.groupby(frame["symbol"]).rolling(short_vol_window).std()
            frame["feat_short_vol"] = rolling_std.reset_index(level=0, drop=True)
        if "microprice_bias" in methods and self._has_columns(
            frame,
            ["bid_price1", "ask_price1", "bid_size1", "ask_size1", "close"],
            "microprice_bias",
            strict,
        ):
            depth = (frame["bid_size1"] + frame["ask_size1"]).replace(0, np.nan)
            microprice = (
                frame["ask_price1"] * frame["bid_size1"] + frame["bid_price1"] * frame["ask_size1"]
            ) / depth
            frame["feat_microprice_bias"] = microprice / frame["close"].replace(0, np.nan) - 1.0
        if "ofi_l1" in methods and self._has_columns(
            frame,
            ["bid_size1", "ask_size1"],
            "ofi_l1",
            strict,
        ):
            bid_delta = grouped["bid_size1"].diff()
            ask_delta = grouped["ask_size1"].diff()
            ofi_raw = bid_delta - ask_delta
            scale = (
                ofi_raw.abs()
                .groupby(frame["symbol"])
                .rolling(ofi_window)
                .mean()
                .reset_index(level=0, drop=True)
                .replace(0, np.nan)
            )
            frame["feat_ofi_l1"] = ofi_raw / scale

        return frame

    def _has_columns(
        self,
        frame: pd.DataFrame,
        required_cols: list[str],
        feature_name: str,
        strict: bool,
    ) -> bool:
        missing = [col for col in required_cols if col not in frame.columns]
        if not missing:
            return True
        if strict:
            raise ValueError(
                f"feature `{feature_name}` requires columns {required_cols}, missing: {missing}"
            )
        return False

    def _build_labels(
        self,
        frame: pd.DataFrame,
        config: dict[str, Any],
    ) -> tuple[pd.DataFrame, list[str]]:
        horizons = config.get("label_horizons", [1])
        price_col = config.get("label_price_col", "close")
        grouped = frame.groupby("symbol", group_keys=False)
        label_cols: list[str] = []
        for horizon in horizons:
            col = f"label_fwd_ret_{int(horizon)}"
            shifted = grouped[price_col].shift(-int(horizon))
            frame[col] = shifted / frame[price_col] - 1.0
            label_cols.append(col)
        return frame, label_cols
