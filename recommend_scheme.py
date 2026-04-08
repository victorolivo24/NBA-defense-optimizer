"""CLI entry point for Phase 4 scheme recommendation."""

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import build_training_dataset
from src.models import DEFAULT_TARGET_COLUMN, recommend_scheme, train_baseline_regressor


if __name__ == "__main__":
    season = "2024-25"
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    dataset = build_training_dataset(session_factory=session_factory, season=season)
    artifacts = train_baseline_regressor(dataset=dataset, target_column=DEFAULT_TARGET_COLUMN)

    lineup_row = dataset.iloc[0]
    recommendation = recommend_scheme(lineup_row, artifacts)

    print(f"Recommended scheme: {recommendation.recommended_scheme}")
    print(f"Predicted target value: {recommendation.predicted_value:.4f}")
    print("Scheme ranking:")
    print(recommendation.ranked_schemes.to_string(index=False))
    if not recommendation.explanation.empty:
        print("Top explanation rows:")
        print(recommendation.explanation.head(10).to_string(index=False))
