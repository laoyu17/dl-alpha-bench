"""Data connectors for market data providers."""

from __future__ import annotations

import importlib
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import pandas as pd

from .errors import DataAuthError, DataConnectorError, DataRateLimitError, DataTemporaryError
from .resilience import RateLimiter, RetryConfig, call_with_retry, default_retryable_error

CANONICAL_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value",
    "tradable",
]
OPTIONAL_MICRO_COLUMNS = [
    "bid_price1",
    "ask_price1",
    "bid_size1",
    "ask_size1",
    "trade_count",
]
AUTH_KEYWORDS = (
    "auth",
    "password",
    "token",
    "permission",
    "forbidden",
    "unauthorized",
    "login",
    "credential",
)
RATE_LIMIT_KEYWORDS = (
    "429",
    "too many requests",
    "rate limit",
    "throttle",
    "quota",
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(key in lowered for key in keywords)


def _load_sdk(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise DataConnectorError(
            f"Missing provider SDK `{module_name}`. Install hint: {install_hint}"
        ) from exc


def _normalize_price_frame(raw: Any, symbol: str) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    if not isinstance(raw, pd.DataFrame):
        raise DataConnectorError(f"Provider returned unsupported type: {type(raw)!r}")

    data = raw.copy()
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    elif isinstance(data.index, pd.DatetimeIndex):
        idx_name = data.index.name or "timestamp"
        data = data.reset_index().rename(columns={idx_name: "timestamp"})

    data = data.rename(
        columns={
            "datetime": "timestamp",
            "time": "timestamp",
            "date": "timestamp",
            "code": "symbol",
            "order_book_id": "symbol",
            "money": "value",
            "total_turnover": "value",
        }
    )
    if "timestamp" not in data.columns and "index" in data.columns:
        data["timestamp"] = data["index"]

    if "symbol" not in data.columns:
        data["symbol"] = symbol

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise DataConnectorError(f"Provider output missing required columns: {missing}")

    if "value" not in data.columns:
        data["value"] = data["close"] * data["volume"]
    if "tradable" not in data.columns:
        if "paused" in data.columns:
            data["tradable"] = ~pd.Series(data["paused"]).fillna(0).astype(bool)
        else:
            data["tradable"] = True

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    numeric_cols = ["open", "high", "low", "close", "volume", "value"] + OPTIONAL_MICRO_COLUMNS
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    keep_cols = CANONICAL_COLUMNS + [col for col in OPTIONAL_MICRO_COLUMNS if col in data.columns]
    return data[keep_cols].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


class DataConnector(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_universe(
        self,
        start: str,
        end: str,
        freq: str,
        symbols: list[str],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV-like data using canonical columns."""

    def fetch_corporate_actions(self, start: str, end: str, symbols: list[str]) -> pd.DataFrame:
        _ = (start, end, symbols)
        return pd.DataFrame(columns=["symbol", "timestamp", "adjust_factor"])


class _OnlineConnector(DataConnector):
    def __init__(self, name: str, retry_config: RetryConfig, min_interval_sec: float):
        self.name = name
        self._retry_config = retry_config
        self._rate_limiter = RateLimiter(min_interval_sec=min_interval_sec, sleep_fn=time.sleep)

    def _call_provider(self, op_name: str, fn: Callable[[], Any]) -> Any:
        def guarded() -> Any:
            self._rate_limiter.acquire()
            return fn()

        try:
            return call_with_retry(
                guarded,
                retry_config=self._retry_config,
                is_retryable=default_retryable_error,
            )
        except Exception as exc:  # noqa: BLE001
            raise self._classify_error(exc, op_name) from exc

    def _classify_error(self, exc: Exception, op_name: str) -> DataConnectorError:
        text = str(exc)
        prefix = f"{self.name}.{op_name} failed"
        if _contains_any(text, AUTH_KEYWORDS):
            return DataAuthError(f"{prefix}: {exc}")
        if _contains_any(text, RATE_LIMIT_KEYWORDS):
            return DataRateLimitError(f"{prefix}: {exc}")
        if default_retryable_error(exc):
            return DataTemporaryError(f"{prefix}: {exc}")
        return DataConnectorError(f"{prefix}: {exc}")


class JoinQuantConnector(_OnlineConnector):
    name = "joinquant"

    def __init__(
        self,
        user_env: str = "JOINQUANT_USER",
        pwd_env: str = "JOINQUANT_PASSWORD",
        min_interval_sec: float | None = None,
        max_retries: int | None = None,
        base_delay_sec: float | None = None,
        max_delay_sec: float | None = None,
        sdk: Any | None = None,
    ):
        resolved_max_retries = (
            max_retries
            if max_retries is not None
            else _env_int("JOINQUANT_MAX_RETRIES", 3)
        )
        retry_cfg = RetryConfig(
            max_retries=resolved_max_retries,
            base_delay_sec=(
                base_delay_sec
                if base_delay_sec is not None
                else _env_float("JOINQUANT_BASE_DELAY_SEC", 0.5)
            ),
            max_delay_sec=(
                max_delay_sec
                if max_delay_sec is not None
                else _env_float("JOINQUANT_MAX_DELAY_SEC", 4.0)
            ),
        )
        interval = (
            min_interval_sec
            if min_interval_sec is not None
            else _env_float("JOINQUANT_MIN_INTERVAL_SEC", 0.25)
        )
        super().__init__(name="joinquant", retry_config=retry_cfg, min_interval_sec=interval)

        self.user = os.getenv(user_env, "")
        self.password = os.getenv(pwd_env, "")
        self._sdk = sdk or _load_sdk("jqdatasdk", "pip install jqdatasdk")
        self._authed = False

    def _ensure_auth(self) -> None:
        if self._authed:
            return
        if not self.user or not self.password:
            raise DataAuthError(
                "joinquant.auth failed: missing JOINQUANT_USER / JOINQUANT_PASSWORD"
            )
        if not hasattr(self._sdk, "auth"):
            raise DataConnectorError("joinquant.auth failed: SDK has no auth()")
        self._call_provider("auth", lambda: self._sdk.auth(self.user, self.password))
        self._authed = True

    def fetch_universe(
        self,
        start: str,
        end: str,
        freq: str,
        symbols: list[str],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        self._ensure_auth()
        if not symbols:
            raise DataConnectorError("joinquant.fetch_universe requires non-empty symbols")

        requested_fields = fields or [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "money",
            "paused",
            "bid_price1",
            "ask_price1",
            "bid_size1",
            "ask_size1",
            "trade_count",
        ]

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            raw = self._fetch_single_symbol(symbol, start, end, freq, requested_fields)
            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                continue
            frames.append(_normalize_price_frame(raw, symbol))

        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _fetch_single_symbol(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str,
        fields: list[str],
    ) -> pd.DataFrame:
        base_kwargs = {
            "start_date": start,
            "end_date": end,
            "frequency": freq,
            "fields": fields,
            "fq": "pre",
        }
        variations = [
            {**base_kwargs, "panel": False, "skip_paused": False},
            {**base_kwargs, "panel": False},
            base_kwargs,
        ]

        last_error: DataConnectorError | None = None
        for kwargs in variations:
            try:
                return self._call_provider(
                    "get_price",
                    lambda kwargs=kwargs: self._sdk.get_price(symbol, **kwargs),
                )
            except DataConnectorError as exc:
                if "unexpected keyword argument" in str(exc).lower():
                    last_error = exc
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise DataConnectorError("joinquant.get_price failed with no concrete exception")

    def fetch_corporate_actions(self, start: str, end: str, symbols: list[str]) -> pd.DataFrame:
        self._ensure_auth()
        if not symbols or not hasattr(self._sdk, "get_extras"):
            return super().fetch_corporate_actions(start, end, symbols)

        raw = self._call_provider(
            "get_extras",
            lambda: self._sdk.get_extras(
                "factor",
                symbols,
                start_date=start,
                end_date=end,
                df=True,
            ),
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return super().fetch_corporate_actions(start, end, symbols)

        factors = raw.stack(dropna=False).rename("adjust_factor").reset_index()
        factors = factors.rename(
            columns={
                factors.columns[0]: "timestamp",
                factors.columns[1]: "symbol",
            }
        )
        factors["timestamp"] = pd.to_datetime(factors["timestamp"])
        return factors[["symbol", "timestamp", "adjust_factor"]].dropna(subset=["adjust_factor"])


class RiceQuantConnector(_OnlineConnector):
    name = "ricequant"

    def __init__(
        self,
        token_env: str = "RICEQUANT_TOKEN",
        user_env: str = "RICEQUANT_USER",
        pwd_env: str = "RICEQUANT_PASSWORD",
        min_interval_sec: float | None = None,
        max_retries: int | None = None,
        base_delay_sec: float | None = None,
        max_delay_sec: float | None = None,
        sdk: Any | None = None,
    ):
        resolved_max_retries = (
            max_retries
            if max_retries is not None
            else _env_int("RICEQUANT_MAX_RETRIES", 3)
        )
        retry_cfg = RetryConfig(
            max_retries=resolved_max_retries,
            base_delay_sec=(
                base_delay_sec
                if base_delay_sec is not None
                else _env_float("RICEQUANT_BASE_DELAY_SEC", 0.5)
            ),
            max_delay_sec=(
                max_delay_sec
                if max_delay_sec is not None
                else _env_float("RICEQUANT_MAX_DELAY_SEC", 4.0)
            ),
        )
        interval = (
            min_interval_sec
            if min_interval_sec is not None
            else _env_float("RICEQUANT_MIN_INTERVAL_SEC", 0.2)
        )
        super().__init__(name="ricequant", retry_config=retry_cfg, min_interval_sec=interval)

        self.token = os.getenv(token_env, "")
        self.user = os.getenv(user_env, "")
        self.password = os.getenv(pwd_env, "")
        self._sdk = sdk or _load_sdk("rqdatac", "pip install rqdatac")
        self._authed = False

    def _ensure_auth(self) -> None:
        if self._authed:
            return
        if not hasattr(self._sdk, "init"):
            raise DataConnectorError("ricequant.init failed: SDK has no init()")

        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if self.token:
            candidates.append(((), {"token": self.token}))
            candidates.append(((self.token,), {}))
        if self.user and self.password:
            candidates.append(((), {"username": self.user, "password": self.password}))
            candidates.append(((self.user, self.password), {}))
        candidates.append(((), {}))

        last_error: Exception | None = None
        for args, kwargs in candidates:
            try:
                self._call_provider(
                    "init",
                    lambda args=args, kwargs=kwargs: self._sdk.init(*args, **kwargs),
                )
                self._authed = True
                return
            except DataConnectorError as exc:
                last_error = exc
                if "unexpected keyword argument" in str(exc).lower():
                    continue
                if "missing" in str(exc).lower() and kwargs:
                    continue
                break

        if last_error is not None:
            raise last_error
        raise DataAuthError("ricequant.init failed: no valid credential method")

    def fetch_universe(
        self,
        start: str,
        end: str,
        freq: str,
        symbols: list[str],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        self._ensure_auth()
        if not symbols:
            raise DataConnectorError("ricequant.fetch_universe requires non-empty symbols")

        requested_fields = fields or [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "total_turnover",
            "bid_price1",
            "ask_price1",
            "bid_size1",
            "ask_size1",
            "trade_count",
        ]

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            raw = self._fetch_single_symbol(symbol, start, end, freq, requested_fields)
            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                continue
            normalized = _normalize_price_frame(raw, symbol)
            frames.append(normalized)

        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _fetch_single_symbol(
        self,
        symbol: str,
        start: str,
        end: str,
        freq: str,
        fields: list[str],
    ) -> pd.DataFrame:
        base_kwargs = {
            "start_date": start,
            "end_date": end,
            "frequency": freq,
            "fields": fields,
        }
        variations = [
            {**base_kwargs, "adjust_type": "pre"},
            base_kwargs,
        ]

        last_error: DataConnectorError | None = None
        for kwargs in variations:
            try:
                return self._call_provider(
                    "get_price",
                    lambda kwargs=kwargs: self._sdk.get_price(symbol, **kwargs),
                )
            except DataConnectorError as exc:
                if "unexpected keyword argument" in str(exc).lower():
                    last_error = exc
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise DataConnectorError("ricequant.get_price failed with no concrete exception")

    def fetch_corporate_actions(self, start: str, end: str, symbols: list[str]) -> pd.DataFrame:
        self._ensure_auth()
        if not symbols or not hasattr(self._sdk, "get_ex_factor"):
            return super().fetch_corporate_actions(start, end, symbols)

        raw = self._call_provider(
            "get_ex_factor",
            lambda: self._sdk.get_ex_factor(order_book_ids=symbols, start_date=start, end_date=end),
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return super().fetch_corporate_actions(start, end, symbols)

        frame = raw.copy()
        if isinstance(frame.index, pd.MultiIndex) or isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index()

        frame = frame.rename(
            columns={
                "order_book_id": "symbol",
                "date": "timestamp",
                "datetime": "timestamp",
                "ex_cum_factor": "adjust_factor",
                "factor": "adjust_factor",
            }
        )
        if "timestamp" not in frame.columns and "index" in frame.columns:
            frame["timestamp"] = frame["index"]

        if "adjust_factor" not in frame.columns and "timestamp" in frame.columns:
            value_cols = [col for col in frame.columns if col in symbols]
            if value_cols:
                frame = frame[["timestamp"] + value_cols].melt(
                    id_vars=["timestamp"],
                    value_vars=value_cols,
                    var_name="symbol",
                    value_name="adjust_factor",
                )

        required_cols = {"symbol", "timestamp", "adjust_factor"}
        if not required_cols.issubset(frame.columns):
            return super().fetch_corporate_actions(start, end, symbols)

        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame[["symbol", "timestamp", "adjust_factor"]].dropna(subset=["adjust_factor"])


class LocalMockConnector(DataConnector):
    """Connector used by tests and local demos."""

    name = "mock"

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.copy()

    def fetch_universe(
        self,
        start: str,
        end: str,
        freq: str,
        symbols: list[str],
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        _ = freq
        frame = self.frame
        mask = (frame["timestamp"] >= pd.Timestamp(start)) & (
            frame["timestamp"] <= pd.Timestamp(end)
        )
        if symbols:
            mask &= frame["symbol"].isin(symbols)
        data = frame.loc[mask].reset_index(drop=True)
        if fields:
            skip_cols = {"symbol", "timestamp", "tradable"}
            keep = ["symbol", "timestamp", "tradable"] + [
                c for c in fields if c in data.columns and c not in skip_cols
            ]
            if "value" in data.columns and "value" not in keep:
                keep.append("value")
            data = data[keep]
        return data
