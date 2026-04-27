"""Feature engineering utilities for lineup-level training datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.database.schema import DefensivePlayType, LineupMetric, LineupPlayer, Player

DEFAULT_PLAY_TYPES = [
    "Isolation",
    "Pick and Roll Ball Handler",
    "Pick and Roll Roll Man",
    "Spot Up",
]
DEFAULT_MIN_LINEUP_MINUTES = 25.0
DEFAULT_MIN_LINEUP_POSSESSIONS = 50.0
DEFAULT_MIN_PLAY_TYPE_POSSESSIONS = 10.0

_PLAYER_DEFENSE_PROFILES_CACHE: dict[str, dict[str, dict[str, float | None]]] | None = None
_LINEUP_BASIC_PROFILES_CACHE: dict[str, dict[str, dict[str, float | str | None]]] | None = None

PLAYER_BASIC_STATS = (
    "dreb",
    "stl",
    "blk",
    "def_ws",
    "opp_pts_off_tov",
    "opp_pts_2nd_chance",
    "opp_pts_fb",
    "opp_pts_paint",
)


def _load_json_cache(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _get_player_defense_profiles(season: str) -> dict[str, dict[str, float | None]]:
    global _PLAYER_DEFENSE_PROFILES_CACHE
    if _PLAYER_DEFENSE_PROFILES_CACHE is None:
        _PLAYER_DEFENSE_PROFILES_CACHE = _load_json_cache("data/processed/player_defense_profiles.json")
    return _PLAYER_DEFENSE_PROFILES_CACHE.get(season, {})


def _get_lineup_basic_profiles(season: str) -> dict[str, dict[str, float | str | None]]:
    global _LINEUP_BASIC_PROFILES_CACHE
    if _LINEUP_BASIC_PROFILES_CACHE is None:
        _LINEUP_BASIC_PROFILES_CACHE = _load_json_cache("data/processed/lineup_basic_defense_profiles.json")
    return _LINEUP_BASIC_PROFILES_CACHE.get(season, {})


def build_training_dataset(
    session_factory,
    season: str,
    play_types: list[str] | None = None,
    min_minutes: float = DEFAULT_MIN_LINEUP_MINUTES,
    min_possessions: float = DEFAULT_MIN_LINEUP_POSSESSIONS,
    min_play_type_possessions: float = DEFAULT_MIN_PLAY_TYPE_POSSESSIONS,
) -> pd.DataFrame:
    """Build one model-ready row per lineup from the normalized database tables."""
    play_types = play_types or DEFAULT_PLAY_TYPES

    session = session_factory()
    try:
        statement = (
            select(LineupMetric)
            .where(LineupMetric.season == season)
            .options(
                selectinload(LineupMetric.players)
                .selectinload(LineupPlayer.player)
                .selectinload(Player.defensive_play_types)
            )
        )
        if min_minutes > 0:
            statement = statement.where(LineupMetric.minutes_played >= min_minutes)
        if min_possessions > 0:
            statement = statement.where(LineupMetric.possessions >= min_possessions)

        lineups = session.execute(statement).scalars().all()

        rows = [
            _build_lineup_feature_row(
                lineup,
                play_types,
                min_play_type_possessions=min_play_type_possessions,
            )
            for lineup in lineups
        ]
        return pd.DataFrame(rows)
    finally:
        session.close()


def export_training_dataset(
    season: str | list[str],
    output_path: str = "data/processed/lineup_training_dataset.csv",
    database_url: str = DEFAULT_DATABASE_URL,
    play_types: list[str] | None = None,
    min_minutes: float = DEFAULT_MIN_LINEUP_MINUTES,
    min_possessions: float = DEFAULT_MIN_LINEUP_POSSESSIONS,
    min_play_type_possessions: float = DEFAULT_MIN_PLAY_TYPE_POSSESSIONS,
) -> Path:
    """Build the training dataset and write it to disk."""
    session_factory = create_session_factory(database_url)

    seasons_list = [season] if isinstance(season, str) else season
    datasets = []
    for s in seasons_list:
        dataset = build_training_dataset(
            session_factory=session_factory,
            season=s,
            play_types=play_types,
            min_minutes=min_minutes,
            min_possessions=min_possessions,
            min_play_type_possessions=min_play_type_possessions,
        )
        datasets.append(dataset)

    final_dataset = pd.concat(datasets, ignore_index=True)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(path, index=False)
    return path


def build_synthetic_lineup_row(
    players: list[Player],
    *,
    season: str,
    play_types: list[str] | None = None,
    min_play_type_possessions: float = DEFAULT_MIN_PLAY_TYPE_POSSESSIONS,
    lineup_key: str | None = None,
    lineup_name: str | None = None,
    team_abbreviation: str | None = None,
) -> dict[str, float | int | str | None]:
    """Build one model-ready row for a user-specified five-player lineup."""
    play_types = play_types or DEFAULT_PLAY_TYPES
    if len(players) != 5:
        raise ValueError("A synthetic lineup row requires exactly 5 players.")

    resolved_team = team_abbreviation or _resolve_team_abbreviation(players)
    resolved_name = lineup_name or " - ".join(player.full_name for player in players)
    resolved_key = lineup_key or "custom:" + "|".join(str(player.nba_player_id) for player in players)

    return _build_player_feature_row(
        players,
        play_types=play_types,
        season=season,
        min_play_type_possessions=min_play_type_possessions,
        lineup_id=None,
        lineup_key=resolved_key,
        lineup_name=resolved_name,
        team_abbreviation=resolved_team,
        minutes_played=None,
        possessions=None,
        defensive_rating=None,
        opponent_ppp=None,
    )


def _build_lineup_feature_row(
    lineup: LineupMetric,
    play_types: list[str],
    *,
    min_play_type_possessions: float,
) -> dict[str, float | int | str | None]:
    links = sorted(lineup.players, key=lambda link: link.slot)
    players = [link.player for link in links if link.player is not None]
    return _build_player_feature_row(
        players,
        play_types=play_types,
        season=lineup.season,
        min_play_type_possessions=min_play_type_possessions,
        lineup_id=lineup.id,
        lineup_key=lineup.lineup_key,
        lineup_name=lineup.lineup_name,
        team_abbreviation=lineup.team_abbreviation,
        minutes_played=lineup.minutes_played,
        possessions=lineup.possessions,
        defensive_rating=lineup.defensive_rating,
        opponent_ppp=lineup.opponent_ppp,
    )


def _build_player_feature_row(
    players: list[Player],
    *,
    play_types: list[str],
    season: str,
    min_play_type_possessions: float,
    lineup_id: int | None,
    lineup_key: str,
    lineup_name: str | None,
    team_abbreviation: str | None,
    minutes_played: float | None,
    possessions: float | None,
    defensive_rating: float | None,
    opponent_ppp: float | None,
) -> dict[str, float | int | str | None]:
    player_profiles = _get_player_defense_profiles(season)
    def_ratings = []
    player_basic_aggregates: dict[str, list[float]] = {stat: [] for stat in PLAYER_BASIC_STATS}

    for player in players:
        pid_str = str(player.nba_player_id)
        profile = player_profiles.get(pid_str, {})
        rating = profile.get("def_rating")
        def_ratings.append(115.0 if rating is None else float(rating))
        for stat in PLAYER_BASIC_STATS:
            value = profile.get(stat)
            if value is not None:
                player_basic_aggregates[stat].append(float(value))

    avg_rating = _mean(def_ratings) if def_ratings else 115.0
    best_rating = min(def_ratings) if def_ratings else 115.0
    worst_rating = max(def_ratings) if def_ratings else 115.0

    row: dict[str, float | int | str | None] = {
        "lineup_id": lineup_id,
        "lineup_key": lineup_key,
        "lineup_name": lineup_name,
        "season": season,
        "team_abbreviation": team_abbreviation,
        "minutes_played": minutes_played,
        "possessions": possessions,
        "lineup_size": len(players),
        "guard_count": _count_positions(players, "G"),
        "forward_count": _count_positions(players, "F"),
        "center_count": _count_positions(players, "C"),
        "avg_height_inches": _mean([_height_to_inches(player.height) for player in players]),
        "avg_weight": _mean([player.weight for player in players]),
        "defensive_rating_target": defensive_rating,
        "defensive_rating_target_source": _resolve_target_source(
            defensive_rating=defensive_rating,
            possessions=possessions,
            minutes_played=minutes_played,
        ),
        "opponent_ppp_target": opponent_ppp,
        "avg_player_def_rtg": avg_rating,
        "best_player_def_rtg": best_rating,
        "worst_player_def_rtg": worst_rating,
    }

    for stat, values in player_basic_aggregates.items():
        row[f"avg_player_{stat}"] = _mean(values)
        row[f"max_player_{stat}"] = _max(values)

    _add_lineup_basic_features(
        row=row,
        season=season,
        lineup_key=lineup_key,
        possessions=possessions,
    )

    for play_type in play_types:
        normalized_name = _normalize_play_type_name(play_type)
        ppp_values = []
        possession_values = []
        percentile_values = []

        for player in players:
            match = _find_play_type_metric(
                player.defensive_play_types,
                play_type=play_type,
                season=season,
                min_possessions=min_play_type_possessions,
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


def _add_lineup_basic_features(
    *,
    row: dict[str, float | int | str | None],
    season: str,
    lineup_key: str,
    possessions: float | None,
) -> None:
    profile = _get_lineup_basic_profiles(season).get(lineup_key)
    if not profile:
        row["lineup_opp_efg_pct"] = None
        row["lineup_opp_tov_rate"] = None
        row["lineup_opp_orb_rate"] = None
        row["lineup_opp_fta_rate"] = None
        return

    fgm = _safe_float(profile.get("fgm"))
    fga = _safe_float(profile.get("fga"))
    fg3m = _safe_float(profile.get("fg3m"))
    fta = _safe_float(profile.get("fta"))
    oreb = _safe_float(profile.get("oreb"))
    dreb = _safe_float(profile.get("dreb"))
    tov = _safe_float(profile.get("tov"))

    row["lineup_opp_efg_pct"] = _safe_ratio((fgm or 0.0) + 0.5 * (fg3m or 0.0), fga)
    row["lineup_opp_tov_rate"] = _safe_ratio(tov, possessions)
    row["lineup_opp_orb_rate"] = _safe_ratio(oreb, (oreb or 0.0) + (dreb or 0.0))
    row["lineup_opp_fta_rate"] = _safe_ratio(fta, fga)


def _find_play_type_metric(
    metrics: list[DefensivePlayType],
    *,
    play_type: str,
    season: str,
    min_possessions: float,
) -> DefensivePlayType | None:
    return next(
        (
            metric
            for metric in metrics
            if metric.play_type == play_type
            and metric.season == season
            and metric.possessions is not None
            and metric.possessions >= min_possessions
        ),
        None,
    )


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


def _resolve_team_abbreviation(players: list[Player]) -> str | None:
    teams = {player.team_abbreviation for player in players if player.team_abbreviation}
    if len(teams) == 1:
        return next(iter(teams))
    if not teams:
        return None
    return "MIX"


def _resolve_target_source(
    *,
    defensive_rating: float | None,
    possessions: float | None,
    minutes_played: float | None,
) -> str | None:
    if defensive_rating is None:
        return None
    if possessions is not None:
        return "api_defensive_rating"
    if minutes_played is not None:
        return "fallback_points_allowed_per_48"
    return "synthetic_no_actual_target"


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _safe_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
