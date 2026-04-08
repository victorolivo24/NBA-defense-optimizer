"""Abstract model interface for defensive scheme recommendation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseSchemeModel(ABC):
    """Common interface so the modeling backend can be swapped later."""

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, features: pd.DataFrame):
        """Generate predictions for new lineup observations."""

    @abstractmethod
    def feature_importance(self) -> pd.DataFrame:
        """Return a feature importance view for downstream explainability."""
