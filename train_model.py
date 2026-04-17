"""CLI entry point for Phase 3 baseline model training."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import DEFAULT_MIN_LINEUP_MINUTES, build_training_dataset
from src.models import (
    DEFAULT_TARGET_COLUMN,
    TARGET_OPTIONS,
    export_feature_importance,
    train_baseline_regressor,
)
from src.models.training import prepare_training_matrices


def plot_actual_vs_predicted(y_train_true, y_train_pred, y_test_true, y_test_pred, output_path: str = "data/processed/plots/actual_vs_predicted.png"):
    """Plot actual vs predicted values for both train and test sets."""
    plt.figure(figsize=(10, 8))
    
    # Plot Training Set
    sns.scatterplot(x=y_train_true, y=y_train_pred, alpha=0.4, label='Training Data', color='blue')
    # Plot Test Set
    sns.scatterplot(x=y_test_true, y=y_test_pred, alpha=0.8, label='Test Data', color='orange')
    
    # Plot perfect prediction line
    min_val = min(y_train_true.min(), y_test_true.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train_true.max(), y_test_true.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (y=x)', linewidth=2)
    
    plt.xlabel('Actual Defensive Rating')
    plt.ylabel('Predicted Defensive Rating')
    plt.title('Actual vs Predicted Defensive Rating (Train & Test)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)


if __name__ == "__main__":
    seasons = ["2022-23", "2023-24", "2024-25"]
    target_column = DEFAULT_TARGET_COLUMN
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    
    datasets = []
    for season in seasons:
        ds = build_training_dataset(
            session_factory=session_factory,
            season=season,
            min_minutes=DEFAULT_MIN_LINEUP_MINUTES,
        )
        datasets.append(ds)
    dataset = pd.concat(datasets, ignore_index=True)
    
    artifacts = train_baseline_regressor(dataset=dataset, target_column=target_column)

    print(f"Target: {target_column}")
    print(f"Description: {TARGET_OPTIONS[target_column]}")
    print(f"Train rows: {artifacts.train_rows}")
    print(f"Test rows: {artifacts.test_rows}")
    print("--- Train Metrics ---")
    print(f"MAE:  {artifacts.metrics['train_mae']:.4f}")
    print(f"RMSE: {artifacts.metrics['train_rmse']:.4f}")
    print(f"R2:   {artifacts.metrics['train_r2']:.4f}")
    print("--- Test Metrics ---")
    print(f"MAE:  {artifacts.metrics['test_mae']:.4f}")
    print(f"RMSE: {artifacts.metrics['test_rmse']:.4f}")
    print(f"R2:   {artifacts.metrics['test_r2']:.4f}")

    importance_path = export_feature_importance(
        artifacts,
        output_path="data/processed/feature_importance.csv",
    )
    print(f"Wrote feature importance to {importance_path}")

    importance_df = pd.read_csv(importance_path)
    print("\n--- Top 10 Feature Importances ---")
    print(importance_df.head(10).to_string(index=False))

    # Generate Visualizations
    import os
    import numpy as np
    
    os.makedirs("data/processed/plots", exist_ok=True)
    
    # 1. SHAP Summary Plot
    features, _, _ = prepare_training_matrices(dataset, target_column=target_column)
    artifacts.model.plot_shap_summary(features, "data/processed/plots/shap_summary.png")
    print("Wrote SHAP summary plot to data/processed/plots/shap_summary.png")

    # 2. Actual vs Predicted Plot
    # Re-split data with the same random state to get the train/test set targets and predictions
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        dataset[target_column].dropna().astype(float),
        test_size=0.25,
        random_state=42,
    )
    pred_train = artifacts.model.predict(x_train)
    pred_test = artifacts.model.predict(x_test)
    plot_actual_vs_predicted(y_train, pred_train, y_test, pred_test, "data/processed/plots/actual_vs_predicted.png")
    print("Wrote Actual vs Predicted plot to data/processed/plots/actual_vs_predicted.png")
