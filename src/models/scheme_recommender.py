"""Baseline linear lineup outcome model implementation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseSchemeModel


class LinearSchemeRecommender(BaseSchemeModel):
    """
    Baseline regressor for lineup defensive outcome prediction.

    This can later be replaced with:
    - one model per scheme with expected points saved outputs
    - a ranking model over candidate defensive coverages
    - a scheme-aware model once true scheme supervision exists
    """

    def __init__(self) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-4, 4, 100)),
        )
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series, sample_weight: pd.Series | None = None) -> None:
        self.feature_names = list(features.columns)

        if sample_weight is not None:
            self.model.fit(features, target, ridgecv__sample_weight=sample_weight)
        else:
            self.model.fit(features, target)

    def predict(self, features: pd.DataFrame):
        return self.model.predict(features)

    def feature_importance(self) -> pd.DataFrame:
        coefficients = self.model.named_steps["ridgecv"].coef_
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": coefficients,
            }
        ).assign(abs_importance=lambda df: df["importance"].abs()).sort_values("abs_importance", ascending=False).drop(
            columns="abs_importance"
        )

    def _scaled_feature_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        scaler = self.model.named_steps["standardscaler"]
        scaled_features = scaler.transform(features)
        return pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return SHAP values for the supplied feature rows."""
        ridge_model = self.model.named_steps["ridgecv"]
        scaled_features = self._scaled_feature_frame(features)
        explainer = shap.LinearExplainer(ridge_model, scaled_features)
        shap_values = explainer.shap_values(scaled_features)
        return pd.DataFrame(shap_values, columns=features.columns, index=features.index)

    def plot_shap_summary(self, features: pd.DataFrame, output_path: str = "data/processed/plots/shap_summary.png") -> None:
        """Generate and save a SHAP summary plot."""
        ridge_model = self.model.named_steps["ridgecv"]
        scaled_features = self._scaled_feature_frame(features)
        explainer = shap.LinearExplainer(ridge_model, scaled_features)
        shap_values = explainer.shap_values(scaled_features)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, scaled_features, show=False)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(2.0)
