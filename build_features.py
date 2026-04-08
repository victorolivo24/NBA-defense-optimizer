"""CLI entry point for Phase 2 feature dataset generation."""

from src.features import export_training_dataset


if __name__ == "__main__":
    output_path = export_training_dataset(season="2024-25")
    print(f"Wrote training dataset to {output_path}")
