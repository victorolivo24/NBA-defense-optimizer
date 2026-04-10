"""CLI script to compare hardcoded vs dynamic scheme profiles."""

import json
from pathlib import Path

import pandas as pd

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import build_training_dataset
from src.models import (
    DEFAULT_TARGET_COLUMN,
    DEFAULT_SCHEME_PROFILES,
    recommend_scheme,
    train_baseline_regressor,
)


def load_dynamic_profiles(path: str = "data/processed/dynamic_scheme_profiles.json") -> dict[str, dict[str, float]]:
    """Load the dynamically calculated scheme profiles from JSON."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {path}. Have you run calculate_scheme_deltas.py?")
        return {}


if __name__ == "__main__":
    season = "2024-25"
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    
    print("Loading data and training baseline model...")
    dataset = build_training_dataset(session_factory=session_factory, season=season, min_minutes=10.0)
    artifacts = train_baseline_regressor(dataset=dataset, target_column=DEFAULT_TARGET_COLUMN)
    
    dynamic_profiles = load_dynamic_profiles()
    
    if not dynamic_profiles:
        print("Exiting because dynamic profiles are missing.")
        exit(1)

    print("\n--- Comparing Profiles on Top 3 Most Used Lineups ---")
    
    # Let's pick a few top lineups to compare
    test_lineups = dataset.sort_values("minutes_played", ascending=False).head(3)
    
    for idx, (_, lineup_row) in enumerate(test_lineups.iterrows(), start=1):
        lineup_name = lineup_row.get("lineup_name", lineup_row["lineup_key"])
        print(f"\n[{idx}] Lineup: {lineup_name}")
        print(f"Minutes Played: {lineup_row['minutes_played']:.1f}")
        print(f"Actual Defensive Rating: {lineup_row['defensive_rating_target']:.2f}")
        
        # Test Hardcoded
        rec_hardcoded = recommend_scheme(lineup_row, artifacts, scheme_profiles=DEFAULT_SCHEME_PROFILES)
        
        # Test Dynamic
        rec_dynamic = recommend_scheme(lineup_row, artifacts, scheme_profiles=dynamic_profiles)
        
        print("\n  >> Hardcoded Profiles Results:")
        print(f"  Recommended: {rec_hardcoded.recommended_scheme} (Predicted DRtg: {rec_hardcoded.predicted_value:.2f})")
        print("  Rankings:")
        for _, row in rec_hardcoded.ranked_schemes.iterrows():
            print(f"    - {row['scheme']}: {row['predicted_value']:.2f}")
            
        print("\n  >> Dynamic Profiles Results:")
        print(f"  Recommended: {rec_dynamic.recommended_scheme} (Predicted DRtg: {rec_dynamic.predicted_value:.2f})")
        print("  Rankings:")
        for _, row in rec_dynamic.ranked_schemes.iterrows():
            print(f"    - {row['scheme']}: {row['predicted_value']:.2f}")

        print("-" * 60)
