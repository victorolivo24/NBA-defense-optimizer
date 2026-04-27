"""Fetch and persist season-level external defensive feature caches used by experiments."""

from __future__ import annotations

import argparse

from src.database.ingest import (
    IngestConfig,
    fetch_lineup_basic_defense_stats,
    fetch_player_defense_stats,
    _write_lineup_basic_profiles,
    _write_player_defense_profiles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh player and lineup defensive feature caches for one or more seasons.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25"],
        help="Season strings to fetch.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=1.5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    for season in args.seasons:
        config = IngestConfig(season=season, sleep_seconds=args.sleep_seconds)
        players_df = fetch_player_defense_stats(config)
        _write_player_defense_profiles(players_df, season)

        lineup_basic_df = fetch_lineup_basic_defense_stats(config)
        _write_lineup_basic_profiles(lineup_basic_df, season)

        print(f"Refreshed feature caches for {season}")
