"""Starter ingestion pipeline for NBA defensive data."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, synergyplaytypes
from sqlalchemy import select

from .connection import DEFAULT_DATABASE_URL, create_session_factory, initialize_database
from .schema import DefensivePlayType, Player

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    season: str = "2024-25"
    season_type: str = "Regular Season"
    sleep_seconds: float = 1.5
    database_url: str = DEFAULT_DATABASE_URL


def fetch_player_defense_stats(config: IngestConfig) -> pd.DataFrame:
    """
    Fetch basic player defense stats.

    The NBA stats API is rate-limited aggressively. Keep the sleep between
    requests unless you replace this with explicit retry and backoff logic.
    """
    endpoint = leaguedashplayerstats.LeagueDashPlayerStats(
        season=config.season,
        season_type_all_star=config.season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Defense",
    )
    df = endpoint.get_data_frames()[0]
    time.sleep(config.sleep_seconds)
    return df


def fetch_defensive_play_type_stats(config: IngestConfig) -> pd.DataFrame:
    """
    Fetch defensive play-type data.

    This endpoint can be finicky, and field availability may vary by season.
    Keep the sleep delay to reduce the chance of being blocked mid-download.
    """
    endpoint = synergyplaytypes.SynergyPlayTypes(
        league_id_nullable="00",
        per_mode_simple="Totals",
        player_or_team_abbreviation="P",
        season=config.season,
        season_type_all_star=config.season_type,
        type_grouping_nullable="defensive",
    )
    df = endpoint.get_data_frames()[0]
    time.sleep(config.sleep_seconds)
    return df


def upsert_players(session_factory, players_df: pd.DataFrame) -> None:
    """Insert or update player records from the player defense dataset."""
    session = session_factory()
    try:
        for row in players_df.to_dict(orient="records"):
            nba_player_id = int(row["PLAYER_ID"])
            existing = session.execute(
                select(Player).where(Player.nba_player_id == nba_player_id)
            ).scalar_one_or_none()

            payload = {
                "nba_player_id": nba_player_id,
                "full_name": row.get("PLAYER_NAME", "Unknown"),
                "team_abbreviation": row.get("TEAM_ABBREVIATION"),
                "position": row.get("POSITION"),
                "height": row.get("HEIGHT"),
                "weight": _safe_int(row.get("WEIGHT")),
                "is_active": True,
            }

            if existing is None:
                session.add(Player(**payload))
            else:
                for field, value in payload.items():
                    setattr(existing, field, value)

        session.commit()
        LOGGER.info("Upserted %s player records.", len(players_df))
    finally:
        session.close()


def upsert_defensive_play_types(
    session_factory, play_types_df: pd.DataFrame, season: str
) -> None:
    """Insert or update defensive play-type metrics for each player."""
    session = session_factory()
    try:
        player_lookup = {
            player.nba_player_id: player.id
            for player in session.execute(select(Player)).scalars().all()
        }

        inserted = 0
        for row in play_types_df.to_dict(orient="records"):
            nba_player_id = _safe_int(row.get("PLAYER_ID"))
            if nba_player_id is None or nba_player_id not in player_lookup:
                continue

            play_type = row.get("PLAY_TYPE") or row.get("PlayType") or "Unknown"
            existing = session.execute(
                select(DefensivePlayType).where(
                    DefensivePlayType.player_id == player_lookup[nba_player_id],
                    DefensivePlayType.season == season,
                    DefensivePlayType.play_type == play_type,
                )
            ).scalar_one_or_none()

            payload = {
                "player_id": player_lookup[nba_player_id],
                "season": season,
                "play_type": play_type,
                "possessions": _safe_float(row.get("POSS") or row.get("POSS_PCT")),
                "points_allowed": _safe_float(row.get("PTS")),
                "ppp_allowed": _safe_float(row.get("PPP")),
                "frequency_pct": _safe_float(row.get("FREQ") or row.get("POSS_PCT")),
                "percentile": _safe_float(row.get("PERCENTILE")),
                "source": "nba_api_synergy",
            }

            if existing is None:
                session.add(DefensivePlayType(**payload))
            else:
                for field, value in payload.items():
                    setattr(existing, field, value)
            inserted += 1

        session.commit()
        LOGGER.info("Upserted %s defensive play-type records.", inserted)
    finally:
        session.close()


def run_ingestion(config: IngestConfig | None = None) -> None:
    """Run the starter ingestion workflow."""
    config = config or IngestConfig()
    initialize_database(config.database_url)
    session_factory = create_session_factory(config.database_url)

    LOGGER.info("Fetching player defense stats for %s.", config.season)
    players_df = fetch_player_defense_stats(config)
    upsert_players(session_factory, players_df)

    LOGGER.info("Fetching defensive play-type stats for %s.", config.season)
    try:
        play_types_df = fetch_defensive_play_type_stats(config)
        upsert_defensive_play_types(session_factory, play_types_df, config.season)
    except Exception as exc:
        LOGGER.warning(
            "Play-type ingestion failed. This is common with unstable NBA endpoints: %s",
            exc,
        )

    LOGGER.info("Ingestion complete.")


def _safe_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
