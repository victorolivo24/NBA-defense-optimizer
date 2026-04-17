"""CLI entry point for database initialization and starter ingestion."""

from src.database.ingest import IngestConfig, run_ingestion


if __name__ == "__main__":
    seasons = ["2022-23", "2023-24", "2024-25"]
    for season in seasons:
        run_ingestion(IngestConfig(season=season))
