"""Model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Regressor(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> list[float]:
        """Train and return loss history."""

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values."""
