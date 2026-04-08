"""Feature engineering utilities for lineup-level training datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.database.schema import LineupMetric, LineupPlayer, Player

DEFAULT_PLAY_TYPES = [
    "Isolation",
    "Pick and Roll Ball Handler",
    "Pick and Roll Roll Man",
    "Spot Up",
]


def build_training_dataset(
    session_factory,
    season: str,
    play_types: list[str] | None = None,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    """Build one model-ready row per lineup from the normalized database tables."""
    play_types = play_types or DEFAULT_PLAY_TYPES

    session = session_factory()
    try:
        lineups = (
            session.execute(
                select(LineupMetric)
                .where(LineupMetric.season == season)
                .where((LineupMetric.minutes_played.is_(None)) | (LineupMetric.minutes_played >= min_minutes))
                .options(
                    selectinload(LineupMetric.players)
                    .selectinload(LineupPlayer.player)
                    .selectinload(Player.defensive_play_types)
                )
            )
            .scalars()
            .all()
        )

        rows = [_build_lineup_feature_row(lineup, play_types) for lineup in lineups]
        return pd.DataFrame(rows)
    finally:
        session.close()


def export_training_dataset(
    season: str,
    output_path: str = "data/processed/lineup_training_dataset.csv",
    database_url: str = DEFAULT_DATABASE_URL,
    play_types: list[str] | None = None,
    min_minutes: float = 0.0,
) -> Path:
    """Build the training dataset and write it to disk."""
    session_factory = create_session_factory(database_url)
    dataset = build_training_dataset(
        session_factory=session_factory,
        season=season,
        play_types=play_types,
        min_minutes=min_minutes,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    return path


def _build_lineup_feature_row(lineup: LineupMetric, play_types: list[str]) -> dict[str, float | int | str | None]:
    links = sorted(lineup.players, key=lambda link: link.slot)
    players = [link.player for link in links if link.player is not None]

    row: dict[str, float | int | str | None] = {
        "lineup_id": lineup.id,
        "lineup_key": lineup.lineup_key,
        "lineup_name": lineup.lineup_name,
        "season": lineup.season,
        "team_abbreviation": lineup.team_abbreviation,
        "minutes_played": lineup.minutes_played,
        "possessions": lineup.possessions,
        "lineup_size": len(players),
        "guard_count": _count_positions(players, "G"),
        "forward_count": _count_positions(players, "F"),
        "center_count": _count_positions(players, "C"),
        "avg_height_inches": _mean([_height_to_inches(player.height) for player in players]),
        "avg_weight": _mean([player.weight for player in players]),
        "defensive_rating_target": lineup.defensive_rating,
        "opponent_ppp_target": lineup.opponent_ppp,
    }

    for play_type in play_types:
        normalized_name = _normalize_play_type_name(play_type)
        ppp_values = []
        possession_values = []
        percentile_values = []

        for player in players:
            match = next(
                (metric for metric in player.defensive_play_types if metric.play_type == play_type),
                None,
            )
            if match is None:
                continue
            ppp_values.append(match.ppp_allowed)
            possession_values.append(match.possessions)
            percentile_values.append(match.percentile)

        row[f"{normalized_name}_player_count"] = len(ppp_values)
        row[f"{normalized_name}_ppp_mean"] = _mean(ppp_values)
        row[f"{normalized_name}_ppp_min"] = _min(ppp_values)
        row[f"{normalized_name}_ppp_max"] = _max(ppp_values)
        row[f"{normalized_name}_possessions_mean"] = _mean(possession_values)
        row[f"{normalized_name}_percentile_mean"] = _mean(percentile_values)
        row[f"{normalized_name}_percentile_min"] = _min(percentile_values)

    return row


def _count_positions(players: list[Player], token: str) -> int:
    return sum(1 for player in players if player.position and token in player.position)


def _height_to_inches(height: str | None) -> int | None:
    if not height or "-" not in height:
        return None

    feet_str, inches_str = height.split("-", maxsplit=1)
    try:
        return int(feet_str) * 12 + int(inches_str)
    except ValueError:
        return None


def _normalize_play_type_name(play_type: str) -> str:
    return (
        play_type.lower()
        .replace("&", "and")
        .replace("/", " ")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "_")
    )


def _mean(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _min(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _max(values: list[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return max(filtered)
