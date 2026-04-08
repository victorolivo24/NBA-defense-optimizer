"""CLI entry point for Phase 3 baseline model training."""

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import build_training_dataset
from src.models import (
    DEFAULT_TARGET_COLUMN,
    TARGET_OPTIONS,
    export_feature_importance,
    train_baseline_regressor,
)


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
