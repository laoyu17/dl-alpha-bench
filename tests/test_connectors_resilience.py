from __future__ import annotations

import pandas as pd
import pytest

from dl_alpha_bench.data import (
    DataAuthError,
    DataRateLimitError,
    JoinQuantConnector,
    RiceQuantConnector,
)


class _JoinQuantSdkRetryThenOk:
    def __init__(self):
        self.auth_calls = 0
        self.price_calls = 0

    def auth(self, user: str, password: str) -> int:
        self.auth_calls += 1
        if not user or not password:
            raise RuntimeError("auth failed")
        return 0

    def get_price(self, symbol: str, **kwargs) -> pd.DataFrame:
        _ = (symbol, kwargs)
        self.price_calls += 1
        if self.price_calls == 1:
            raise RuntimeError("429 too many requests")

        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame(
            {
                "open": [10.0, 10.2, 10.1],
                "high": [10.3, 10.4, 10.2],
                "low": [9.9, 10.0, 9.95],
                "close": [10.1, 10.25, 10.0],
                "volume": [1000, 1200, 1100],
                "money": [10100, 12300, 11000],
                "paused": [0, 0, 1],
            },
            index=idx,
        )


class _JoinQuantSdkAuthFail:
    def auth(self, user: str, password: str) -> int:
        _ = (user, password)
        raise RuntimeError("unauthorized invalid password")

    def get_price(self, symbol: str, **kwargs) -> pd.DataFrame:
        _ = (symbol, kwargs)
        return pd.DataFrame()


class _RiceQuantSdkRetryThenOk:
    def __init__(self):
        self.inited = False
        self.price_calls = 0

    def init(self, *args, **kwargs) -> None:
        _ = (args, kwargs)
        self.inited = True

    def get_price(self, symbol: str, **kwargs) -> pd.DataFrame:
        _ = (symbol, kwargs)
        self.price_calls += 1
        if self.price_calls == 1:
            raise ConnectionError("temporary connection reset")

        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        return pd.DataFrame(
            {
                "open": [20.0, 20.2, 20.4],
                "high": [20.3, 20.4, 20.5],
                "low": [19.8, 20.0, 20.2],
                "close": [20.1, 20.3, 20.45],
                "volume": [2000, 2100, 1900],
                "total_turnover": [40200, 42630, 38855],
            },
            index=idx,
        )


class _JoinQuantSdkAlways429:
    def auth(self, user: str, password: str) -> int:
        _ = (user, password)
        return 0

    def get_price(self, symbol: str, **kwargs) -> pd.DataFrame:
        _ = (symbol, kwargs)
        raise RuntimeError("429 too many requests")


def test_joinquant_retry_and_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_JQ_USER", "u")
    monkeypatch.setenv("TEST_JQ_PWD", "p")
    sdk = _JoinQuantSdkRetryThenOk()
    connector = JoinQuantConnector(
        user_env="TEST_JQ_USER",
        pwd_env="TEST_JQ_PWD",
        min_interval_sec=0.0,
        max_retries=2,
        base_delay_sec=0.0,
        max_delay_sec=0.0,
        sdk=sdk,
    )

    frame = connector.fetch_universe(
        start="2024-01-01",
        end="2024-01-03",
        freq="1d",
        symbols=["000001.XSHE"],
    )

    assert sdk.auth_calls == 1
    assert sdk.price_calls == 2
    assert set(["symbol", "timestamp", "value", "tradable"]).issubset(frame.columns)
    assert frame["symbol"].nunique() == 1
    assert frame["tradable"].tolist() == [True, True, False]


def test_joinquant_auth_error_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_JQ_USER", "u")
    monkeypatch.setenv("TEST_JQ_PWD", "bad")
    connector = JoinQuantConnector(
        user_env="TEST_JQ_USER",
        pwd_env="TEST_JQ_PWD",
        min_interval_sec=0.0,
        max_retries=0,
        base_delay_sec=0.0,
        max_delay_sec=0.0,
        sdk=_JoinQuantSdkAuthFail(),
    )

    with pytest.raises(DataAuthError):
        connector.fetch_universe(
            start="2024-01-01",
            end="2024-01-03",
            freq="1d",
            symbols=["000001.XSHE"],
        )


def test_ricequant_retry_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_RQ_TOKEN", "demo-token")
    connector = RiceQuantConnector(
        token_env="TEST_RQ_TOKEN",
        min_interval_sec=0.0,
        max_retries=2,
        base_delay_sec=0.0,
        max_delay_sec=0.0,
        sdk=_RiceQuantSdkRetryThenOk(),
    )

    frame = connector.fetch_universe(
        start="2024-01-01",
        end="2024-01-03",
        freq="1d",
        symbols=["000001.XSHE"],
    )

    assert not frame.empty
    assert "value" in frame.columns
    assert frame["value"].iloc[0] == pytest.approx(40200)


def test_joinquant_rate_limit_classified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_JQ_USER", "u")
    monkeypatch.setenv("TEST_JQ_PWD", "p")
    connector = JoinQuantConnector(
        user_env="TEST_JQ_USER",
        pwd_env="TEST_JQ_PWD",
        min_interval_sec=0.0,
        max_retries=0,
        base_delay_sec=0.0,
        max_delay_sec=0.0,
        sdk=_JoinQuantSdkAlways429(),
    )

    with pytest.raises(DataRateLimitError):
        connector.fetch_universe(
            start="2024-01-01",
            end="2024-01-03",
            freq="1d",
            symbols=["000001.XSHE"],
        )
