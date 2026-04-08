"""Starter ingestion pipeline for NBA defensive data."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from nba_api.stats.endpoints import leaguedashlineups, leaguedashplayerstats, synergyplaytypes
from sqlalchemy import select

from .connection import DEFAULT_DATABASE_URL, create_session_factory, initialize_database
from .schema import DefensivePlayType, LineupMetric, LineupPlayer, Player

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    season: str = "2024-25"
    season_type: str = "Regular Season"
    sleep_seconds: float = 1.5
    max_retries: int = 3
    backoff_seconds: float = 2.0
    database_url: str = DEFAULT_DATABASE_URL
    raw_data_dir: str = "data/raw"

def fetch_player_defense_stats(config: IngestConfig) -> pd.DataFrame:
    """Fetch basic player defense stats with retry and backoff."""
    return _fetch_dataframe(
        endpoint_name="player_defense",
        endpoint_factory=leaguedashplayerstats.LeagueDashPlayerStats,
        endpoint_kwargs={
            "season": config.season,
            "season_type_all_star": config.season_type,
            "per_mode_detailed": "PerGame",
            "measure_type_detailed_defense": "Defense",
        },
        config=config,
    )


def fetch_defensive_play_type_stats(config: IngestConfig) -> pd.DataFrame:
    """Fetch defensive play-type data with retry and backoff."""
    return _fetch_dataframe(
        endpoint_name="defensive_play_types",
        endpoint_factory=synergyplaytypes.SynergyPlayTypes,
        endpoint_kwargs={
            "league_id_nullable": "00",
            "per_mode_simple": "Totals",
            "player_or_team_abbreviation": "P",
            "season": config.season,
            "season_type_all_star": config.season_type,
            "type_grouping_nullable": "defensive",
        },
        config=config,
    )


def fetch_lineup_defense_stats(config: IngestConfig) -> pd.DataFrame:
    """Fetch five-man lineup defense data with retry and backoff."""
    return _fetch_dataframe(
        endpoint_name="lineup_defense",
        endpoint_factory=leaguedashlineups.LeagueDashLineups,
        endpoint_kwargs={
            "season": config.season,
            "season_type_all_star": config.season_type,
            "per_mode_detailed": "PerGame",
            "measure_type_detailed_defense": "Defense",
            "group_quantity": 5,
        },
        config=config,
    )


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


def upsert_lineups(session_factory, lineups_df: pd.DataFrame, season: str) -> None:
    """Insert or update lineup-level defensive snapshots and lineup membership."""
    session = session_factory()
    try:
        player_lookup = {
            player.nba_player_id: player
            for player in session.execute(select(Player)).scalars().all()
        }

        upserted = 0
        for row in normalize_lineup_records(lineups_df, season):
            lineup = session.execute(
                select(LineupMetric).where(
                    LineupMetric.lineup_key == row["lineup_key"],
                    LineupMetric.season == season,
                )
            ).scalar_one_or_none()

            lineup_payload = {
                "lineup_key": row["lineup_key"],
                "lineup_name": row["lineup_name"],
                "season": season,
                "team_abbreviation": row["team_abbreviation"],
                "minutes_played": row["minutes_played"],
                "possessions": row["possessions"],
                "defensive_rating": row["defensive_rating"],
                "opponent_ppp": row["opponent_ppp"],
                "observed_scheme": None,
                "recommended_scheme": None,
            }

            if lineup is None:
                lineup = LineupMetric(**lineup_payload)
                session.add(lineup)
                session.flush()
            else:
                for field, value in lineup_payload.items():
                    setattr(lineup, field, value)

            existing_slots = {link.slot: link for link in lineup.players}
            for slot, nba_player_id in enumerate(row["player_ids"], start=1):
                player = player_lookup.get(nba_player_id)
                if player is None:
                    player = Player(
                        nba_player_id=nba_player_id,
                        full_name=f"Unknown Player {nba_player_id}",
                        is_active=True,
                    )
                    session.add(player)
                    session.flush()
                    player_lookup[nba_player_id] = player

                link = existing_slots.get(slot)
                if link is None:
                    session.add(
                        LineupPlayer(lineup_id=lineup.id, player_id=player.id, slot=slot)
                    )
                else:
                    link.player_id = player.id

            upserted += 1

        session.commit()
        LOGGER.info("Upserted %s lineup records.", upserted)
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

    LOGGER.info("Fetching lineup defense stats for %s.", config.season)
    try:
        lineup_df = fetch_lineup_defense_stats(config)
        upsert_lineups(session_factory, lineup_df, config.season)
    except Exception as exc:
        LOGGER.warning(
            "Lineup ingestion failed. This is common with unstable NBA endpoints: %s",
            exc,
        )

    LOGGER.info("Ingestion complete.")


def normalize_lineup_records(lineups_df: pd.DataFrame, season: str) -> list[dict[str, Any]]:
    """Normalize lineup endpoint rows into database-ready dictionaries."""
    normalized: list[dict[str, Any]] = []

    for row in lineups_df.to_dict(orient="records"):
        lineup_key = str(
            row.get("GROUP_ID")
            or row.get("LINEUP_ID")
            or row.get("LINEUP_KEY")
            or row.get("GROUP_NAME")
            or ""
        ).strip()
        player_ids = parse_lineup_player_ids(lineup_key)

        if len(player_ids) != 5:
            LOGGER.debug("Skipping lineup without five player ids: %s", lineup_key)
            continue

        possessions = _safe_float(
            row.get("POSS")
            or row.get("POSSessions")
            or row.get("DEF_POSS")
            or row.get("POSS_EST")
        )
        defensive_rating = _safe_float(
            row.get("DEF_RATING") or row.get("DEFRTG") or row.get("DEF_RTG")
        )

        normalized.append(
            {
                "lineup_key": lineup_key,
                "lineup_name": row.get("GROUP_NAME") or row.get("LINEUP_NAME"),
                "season": season,
                "team_abbreviation": row.get("TEAM_ABBREVIATION") or row.get("TEAM_ABBREV"),
                "minutes_played": _safe_float(row.get("MIN") or row.get("MINUTES")),
                "possessions": possessions,
                "defensive_rating": defensive_rating,
                "opponent_ppp": _derive_opponent_ppp(defensive_rating),
                "player_ids": player_ids,
            }
        )

    return normalized


def parse_lineup_player_ids(lineup_key: str) -> list[int]:
    """Parse player ids from a lineup key such as '201939-202691-203110-203952-1626172'."""
    if not lineup_key:
        return []

    candidate_parts = [part.strip() for part in str(lineup_key).split("-")]
    if not candidate_parts:
        return []

    player_ids: list[int] = []
    for part in candidate_parts:
        parsed = _safe_int(part)
        if parsed is None:
            return []
        player_ids.append(parsed)
    return player_ids


def save_raw_frame(
    df: pd.DataFrame, endpoint_name: str, config: IngestConfig, metadata: dict[str, Any] | None = None
) -> Path:
    """Persist a raw endpoint snapshot to disk for reproducibility and debugging."""
    raw_dir = Path(config.raw_data_dir) / config.season / endpoint_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = raw_dir / f"{timestamp}.json"

    payload = {
        "endpoint": endpoint_name,
        "season": config.season,
        "season_type": config.season_type,
        "captured_at_utc": timestamp,
        "record_count": len(df),
        "metadata": metadata or {},
        "records": df.to_dict(orient="records"),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _fetch_dataframe(
    endpoint_name: str,
    endpoint_factory: Callable[..., Any],
    endpoint_kwargs: dict[str, Any],
    config: IngestConfig,
) -> pd.DataFrame:
    """
    Fetch a dataframe from an nba_api endpoint, save the raw snapshot, and sleep.

    The NBA stats API is rate-limited aggressively. Keep the sleep between
    requests unless you replace this with explicit retry and backoff logic.
    """
    last_error: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            endpoint = endpoint_factory(**endpoint_kwargs)
            df = endpoint.get_data_frames()[0]
            raw_path = save_raw_frame(df, endpoint_name, config, metadata=endpoint_kwargs)
            LOGGER.info("Saved raw %s snapshot to %s", endpoint_name, raw_path)
            time.sleep(config.sleep_seconds)
            return df
        except Exception as exc:
            last_error = exc
            if attempt == config.max_retries:
                break

            wait_seconds = config.backoff_seconds * attempt
            LOGGER.warning(
                "Attempt %s/%s failed for %s: %s. Retrying in %.1f seconds.",
                attempt,
                config.max_retries,
                endpoint_name,
                exc,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to fetch {endpoint_name} after {config.max_retries} attempts"
    ) from last_error


def _derive_opponent_ppp(defensive_rating: float | None) -> float | None:
    """Convert defensive rating to an opponent PPP proxy when possible."""
    if defensive_rating is None:
        return None
    return defensive_rating / 100.0


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
