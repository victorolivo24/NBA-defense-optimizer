"""Training and evaluation helpers for the baseline lineup outcome model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.features import build_training_dataset

from .scheme_recommender import XGBoostSchemeRecommender

DEFAULT_TARGET_COLUMN = "defensive_rating_target"
TARGET_OPTIONS = {
    "defensive_rating_target": "Baseline surrogate target: lineup defensive rating.",
    "opponent_ppp_target": "Alternative surrogate target: opponent points per possession.",
}
NON_FEATURE_COLUMNS = {
    "lineup_id",
    "lineup_key",
    "lineup_name",
    "season",
    "team_abbreviation",
    "defensive_rating_target",
    "opponent_ppp_target",
}


@dataclass
class TrainingArtifacts:
    """Container for the trained model, selected features, and evaluation metrics."""

    model: XGBoostSchemeRecommender
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, float]
    train_rows: int
    test_rows: int


def build_model_dataset(
    session_factory,
    season: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    min_minutes: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create the numeric feature matrix and target vector for baseline training."""
    if target_column not in TARGET_OPTIONS:
        raise ValueError(
            f"Unsupported target column '{target_column}'. Expected one of: {sorted(TARGET_OPTIONS)}"
        )

    dataset = build_training_dataset(
        session_factory=session_factory,
        season=season,
        min_minutes=min_minutes,
    )
    return prepare_training_matrices(dataset, target_column=target_column)


def prepare_training_matrices(
    dataset: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert the engineered lineup dataset into numeric features and a target series."""
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    filtered = dataset.dropna(subset=[target_column]).copy()
    if filtered.empty:
        raise ValueError(f"No rows remain after dropping null values for '{target_column}'.")

    coerced_numeric_columns: dict[str, pd.Series] = {}
    for column in filtered.columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        numeric_series = pd.to_numeric(filtered[column], errors="coerce")
        if numeric_series.notna().any():
            coerced_numeric_columns[column] = numeric_series

    feature_columns = list(coerced_numeric_columns.keys())
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for model training.")

    features = pd.DataFrame(coerced_numeric_columns, index=filtered.index)
    features = features.fillna(features.mean(numeric_only=True))
    features = features.fillna(0.0)
    target = filtered[target_column].astype(float)
    return features, target


def train_baseline_regressor(
    dataset: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainingArtifacts:
    """Train and evaluate the baseline XGBoost regressor on the engineered lineup dataset."""
    features, target = prepare_training_matrices(dataset, target_column=target_column)
    if len(features) < 4:
        raise ValueError("At least 4 rows are required to train and evaluate the baseline model.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBoostSchemeRecommender()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
    }

    return TrainingArtifacts(
        model=model,
        target_column=target_column,
        feature_columns=list(features.columns),
        metrics=metrics,
        train_rows=len(x_train),
        test_rows=len(x_test),
    )


def export_feature_importance(artifacts: TrainingArtifacts, output_path: str) -> Path:
    """Write the baseline model's feature importance table to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.model.feature_importance().to_csv(path, index=False)
    return path
