"""Baseline XGBoost lineup outcome model implementation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

from .base import BaseSchemeModel


class XGBoostSchemeRecommender(BaseSchemeModel):
    """
    Baseline regressor for lineup defensive outcome prediction.

    This can later be replaced with:
    - one model per scheme with expected points saved outputs
    - a ranking model over candidate defensive coverages
    - a scheme-aware model once true scheme supervision exists
    """

    def __init__(self) -> None:
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self.feature_names = list(features.columns)
        self.model.fit(features, target)

    def predict(self, features: pd.DataFrame):
        return self.model.predict(features)

    def feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return SHAP values for the supplied feature rows."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)
        return pd.DataFrame(shap_values, columns=features.columns, index=features.index)

    def plot_shap_summary(self, features: pd.DataFrame, output_path: str = "data/processed/plots/shap_summary.png") -> None:
        """Generate and save a SHAP summary plot."""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(features)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, features, show=False)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
