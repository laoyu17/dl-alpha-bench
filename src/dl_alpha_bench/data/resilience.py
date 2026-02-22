"""Retry and rate-limit helpers for provider integrations."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 3
    base_delay_sec: float = 0.5
    max_delay_sec: float = 4.0
    backoff_factor: float = 2.0
    jitter_sec: float = 0.05


class RateLimiter:
    def __init__(
        self,
        min_interval_sec: float,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._last_call_ts: float | None = None

    def acquire(self) -> None:
        if self.min_interval_sec <= 0:
            return
        now = self._time_fn()
        if self._last_call_ts is not None:
            wait = self.min_interval_sec - (now - self._last_call_ts)
            if wait > 0:
                self._sleep_fn(wait)
        self._last_call_ts = self._time_fn()


def default_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    text = str(exc).lower()
    retry_keywords = (
        "timeout",
        "temporar",
        "connection",
        "429",
        "too many requests",
        "rate limit",
        "busy",
        "try again",
        "timed out",
        "service unavailable",
    )
    return any(key in text for key in retry_keywords)


def call_with_retry(
    fn: Callable[[], T],
    retry_config: RetryConfig,
    is_retryable: Callable[[Exception], bool] = default_retryable_error,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    max_retries = max(0, int(retry_config.max_retries))
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not is_retryable(exc):
                raise
            delay = min(
                retry_config.max_delay_sec,
                retry_config.base_delay_sec * (retry_config.backoff_factor ** attempt),
            )
            if retry_config.jitter_sec > 0:
                delay += random.uniform(0.0, retry_config.jitter_sec)
            sleep_fn(delay)

    raise RuntimeError("call_with_retry reached an unexpected branch")
