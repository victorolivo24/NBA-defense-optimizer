"""CLI entry point for Phase 2 feature dataset generation."""

from src.features import DEFAULT_MIN_LINEUP_MINUTES, DEFAULT_MIN_PLAY_TYPE_POSSESSIONS, export_training_dataset


if __name__ == "__main__":
    output_path = export_training_dataset(
        season=["2022-23", "2023-24", "2024-25"],
        min_minutes=DEFAULT_MIN_LINEUP_MINUTES,
        min_play_type_possessions=DEFAULT_MIN_PLAY_TYPE_POSSESSIONS,
    )
    print(f"Wrote training dataset to {output_path}")
