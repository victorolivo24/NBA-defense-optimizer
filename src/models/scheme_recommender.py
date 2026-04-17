"""Baseline XGBoost lineup outcome model implementation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import RandomizedSearchCV
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
            objective="reg:squarederror",
            random_state=42,
        )
        self.feature_names: list[str] = []

    def fit(self, features: pd.DataFrame, target: pd.Series, sample_weight: pd.Series | None = None) -> None:
        self.feature_names = list(features.columns)
        
        param_grid = {
            "max_depth": [2, 3, 4],
            "min_child_weight": [1, 3, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "reg_alpha": [1, 5, 10],
            "reg_lambda": [1, 5, 10],
            "n_estimators": [50, 100, 150],
            "subsample": [0.5, 0.7],
            "colsample_bytree": [0.5, 0.8],
        }
        
        base_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
        )
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=30,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42,
        )
        
        if sample_weight is not None:
            search.fit(features, target, sample_weight=sample_weight)
        else:
            search.fit(features, target)
        self.model = search.best_estimator_

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
        plt.show(block=False)
        plt.pause(2.0)
