from __future__ import annotations

import importlib


def test_gui_module_exposes_main_callable() -> None:
    module = importlib.import_module("dl_alpha_bench.gui")
    assert hasattr(module, "main")
    assert callable(module.main)
