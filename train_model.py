"""CLI entry point for Phase 3 baseline model training."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import build_training_dataset
from src.models import (
    DEFAULT_TARGET_COLUMN,
    TARGET_OPTIONS,
    export_feature_importance,
    train_baseline_regressor,
)
from src.models.training import prepare_training_matrices


def plot_actual_vs_predicted(y_true, y_pred, output_path: str = "data/processed/plots/actual_vs_predicted.png"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Defensive Rating')
    plt.ylabel('Predicted Defensive Rating')
    plt.title('Actual vs Predicted Defensive Rating')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    season = "2024-25"
    target_column = DEFAULT_TARGET_COLUMN
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    dataset = build_training_dataset(session_factory=session_factory, season=season)
    artifacts = train_baseline_regressor(dataset=dataset, target_column=target_column)

    print(f"Target: {target_column}")
    print(f"Description: {TARGET_OPTIONS[target_column]}")
    print(f"Train rows: {artifacts.train_rows}")
    print(f"Test rows: {artifacts.test_rows}")
    print(f"MAE: {artifacts.metrics['mae']:.4f}")
    print(f"RMSE: {artifacts.metrics['rmse']:.4f}")

    importance_path = export_feature_importance(
        artifacts,
        output_path="data/processed/feature_importance.csv",
    )
    print(f"Wrote feature importance to {importance_path}")

    # Generate Visualizations
    import os
    import numpy as np
    
    os.makedirs("data/processed/plots", exist_ok=True)
    
    # 1. SHAP Summary Plot
    features, _ = prepare_training_matrices(dataset, target_column=target_column)
    artifacts.model.plot_shap_summary(features, "data/processed/plots/shap_summary.png")
    print("Wrote SHAP summary plot to data/processed/plots/shap_summary.png")

    # 2. Actual vs Predicted Plot
    # Re-split data with the same random state to get the test set targets and predictions
    _, x_test, _, y_test = train_test_split(
        features,
        dataset[target_column].dropna().astype(float),
        test_size=0.25,
        random_state=42,
    )
    predictions = artifacts.model.predict(x_test)
    plot_actual_vs_predicted(y_test, predictions, "data/processed/plots/actual_vs_predicted.png")
    print("Wrote Actual vs Predicted plot to data/processed/plots/actual_vs_predicted.png")
