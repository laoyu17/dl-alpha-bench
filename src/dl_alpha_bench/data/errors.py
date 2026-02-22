"""Typed exceptions for data connectors."""

from __future__ import annotations


class DataConnectorError(RuntimeError):
    """Base connector failure."""


class DataAuthError(DataConnectorError):
    """Authentication failed."""


class DataRateLimitError(DataConnectorError):
    """Provider rate limit exceeded."""


class DataTemporaryError(DataConnectorError):
    """Temporary provider/network error."""
