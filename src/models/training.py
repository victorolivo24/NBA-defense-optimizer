"""Training and evaluation helpers for the baseline lineup outcome model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import DEFAULT_MIN_LINEUP_MINUTES, build_training_dataset

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
    "defensive_rating_target_source",
    "defensive_rating_target",
    "opponent_ppp_target",
    "possessions",
    "minutes_played",
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
    feature_mins: dict[str, float] | None = None
    feature_maxs: dict[str, float] | None = None


def build_model_dataset(
    session_factory,
    season: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    min_minutes: float = DEFAULT_MIN_LINEUP_MINUTES,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create the numeric feature matrix, target vector, and weights for baseline training."""
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
    weight_column: str = "possessions",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convert the engineered lineup dataset into numeric features, a target series, and weights."""
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

    filtered = dataset.dropna(subset=[target_column]).copy()
    if filtered.empty:
        raise ValueError(f"No rows remain after dropping null values for '{target_column}'.")

    if weight_column in filtered.columns and filtered[weight_column].notna().any():
        weights = filtered[weight_column].fillna(1.0).astype(float)
    elif "minutes_played" in filtered.columns and filtered["minutes_played"].notna().any():
        weights = filtered["minutes_played"].fillna(1.0).astype(float)
    else:
        weights = pd.Series(1.0, index=filtered.index)

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
    features = features.fillna(features.median(numeric_only=True))
    features = features.fillna(0.0)
    target = filtered[target_column].astype(float)
    return features, target, weights


def train_baseline_regressor(
    dataset: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    test_size: float = 0.25,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    search_iterations: int = 30,
) -> TrainingArtifacts:
    """Train and evaluate the baseline XGBoost regressor on the engineered lineup dataset."""
    features, target, weights = prepare_training_matrices(dataset, target_column=target_column)
    if len(features) < 4:
        raise ValueError("At least 4 rows are required to train and evaluate the baseline model.")

    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        features,
        target,
        weights,
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBoostSchemeRecommender(
        tune_hyperparameters=tune_hyperparameters,
        search_iterations=search_iterations,
        search_n_jobs=1,
    )
    model.fit(x_train, y_train, sample_weight=w_train)
    
    train_predictions = model.predict(x_train)
    predictions = model.predict(x_test)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_predictions)),
        "train_rmse": float(root_mean_squared_error(y_train, train_predictions)),
        "train_r2": float(r2_score(y_train, train_predictions)),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(root_mean_squared_error(y_test, predictions)),
        "test_r2": float(r2_score(y_test, predictions)),
    }

    # Use 5th and 95th percentiles for robust scaling (ignoring outliers)
    feature_mins = features.quantile(0.05).to_dict()
    feature_maxs = features.quantile(0.95).to_dict()

    return TrainingArtifacts(
        model=model,
        target_column=target_column,
        feature_columns=list(features.columns),
        metrics=metrics,
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
    )


def export_feature_importance(artifacts: TrainingArtifacts, output_path: str) -> Path:
    """Write the baseline model's feature importance table to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.model.feature_importance().to_csv(path, index=False)
    return path
