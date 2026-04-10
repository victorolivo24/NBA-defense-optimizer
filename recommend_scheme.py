"""CLI entry point for Phase 4 scheme simulation and recommendation."""

import json
from pathlib import Path

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import build_training_dataset
from src.models import DEFAULT_TARGET_COLUMN, recommend_scheme, train_baseline_regressor


if __name__ == "__main__":
    season = "2024-25"
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    dataset = build_training_dataset(session_factory=session_factory, season=season)
    artifacts = train_baseline_regressor(dataset=dataset, target_column=DEFAULT_TARGET_COLUMN)

    dynamic_profiles = None
    try:
        path = Path("data/processed/dynamic_scheme_profiles.json")
        if path.exists():
            with open(path, "r") as f:
                dynamic_profiles = json.load(f)
            print(f"Loaded dynamic scheme profiles from {path}.\n")
    except Exception as e:
        print(f"Warning: Failed to load dynamic profiles ({e}). Falling back to hardcoded defaults.\n")

    lineup_row = dataset.iloc[0]
    recommendation = recommend_scheme(lineup_row, artifacts, scheme_profiles=dynamic_profiles)

    print(f"Recommended scheme: {recommendation.recommended_scheme}")
    print(f"Predicted target value: {recommendation.predicted_value:.4f}")
    print("Scheme ranking:")
    print(recommendation.ranked_schemes.to_string(index=False))
    if not recommendation.explanation.empty:
        print("Top explanation rows:")
        print(recommendation.explanation.head(10).to_string(index=False))
