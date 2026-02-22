"""GUI entrypoint helpers."""

from __future__ import annotations


def main() -> None:
    from .app import main as _main

    _main()


__all__ = ["main"]
