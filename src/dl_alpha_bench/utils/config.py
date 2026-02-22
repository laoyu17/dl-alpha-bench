"""Config helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=False)


def stable_hash(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
