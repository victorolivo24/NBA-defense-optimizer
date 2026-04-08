"""CLI entry point for database initialization and starter ingestion."""

from src.database.ingest import IngestConfig, run_ingestion


if __name__ == "__main__":
    run_ingestion(IngestConfig())
