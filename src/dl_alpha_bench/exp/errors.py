"""Experiment-specific exceptions."""

from __future__ import annotations


class ExperimentError(RuntimeError):
    """Base runtime error for experiment orchestration."""


class ConfigValidationError(ExperimentError):
    """Configuration or input data contract validation failed."""


class LeakageGuardError(ExperimentError):
    """Experiment execution is blocked by leakage guard."""
