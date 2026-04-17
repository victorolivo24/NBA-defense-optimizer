"""Helpers for running a presentation-friendly scheme recommendation demo."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.database.ingest import parse_lineup_player_ids
from src.database.schema import Player
from src.features import build_synthetic_lineup_row, build_training_dataset
from src.features.lineup_dataset import _resolve_team_abbreviation
from src.models import (
    DEFAULT_TARGET_COLUMN,
    SchemeRecommendation,
    prepare_training_matrices,
    recommend_scheme,
    train_baseline_regressor,
)
from src.models.training import TrainingArtifacts


CASE_STUDY_LABELS = (
    ("Most Used Lineup", "minutes_played", True),
    ("Best Actual Defense", "defensive_rating_target", False),
    ("Worst Actual Defense", "defensive_rating_target", True),
    ("Best Isolation Defense", "isolation_ppp_mean", False),
)
OVERLAP_WEIGHTS = {
    5: 1.0,
    4: 0.7,
    3: 0.4,
    2: 0.15,
}
NON_BLEND_COLUMNS = {
    "lineup_id",
    "lineup_key",
    "lineup_name",
    "season",
    "team_abbreviation",
    "case_label",
    "defensive_rating_target_source",
}


@dataclass
class DemoResult:
    """Presentation-friendly recommendation output for one lineup."""

    case_label: str
    lineup_key: str
    lineup_name: str
    team_abbreviation: str | None
    minutes_played: float | None
    actual_target: float | None
    actual_target_source: str | None
    lineup_source: str | None
    baseline_prediction: float
    recommendation: SchemeRecommendation


def train_demo_artifacts(
    season: str = "2024-25",
    database_url: str = DEFAULT_DATABASE_URL,
    target_column: str = DEFAULT_TARGET_COLUMN,
    min_minutes: float = 0.0,
) -> tuple[pd.DataFrame, TrainingArtifacts]:
    """Load the engineered lineup dataset and fit the baseline model for demo use."""
    session_factory = create_session_factory(database_url)
    dataset = build_training_dataset(
        session_factory=session_factory,
        season=season,
        min_minutes=min_minutes,
    )
    artifacts = train_baseline_regressor(dataset=dataset, target_column=target_column)
    return dataset, artifacts


def load_players_for_demo(
    player_names: list[str],
    *,
    season: str,
    database_url: str = DEFAULT_DATABASE_URL,
) -> list[Player]:
    """Resolve exactly five user-supplied player names from the database."""
    if len(player_names) != 5:
        raise ValueError("Please supply exactly 5 player names.")

    session_factory = create_session_factory(database_url)
    session = session_factory()
    try:
        players = (
            session.execute(select(Player).options(selectinload(Player.defensive_play_types)))
            .scalars()
            .all()
        )
        resolved = [_resolve_player_name(name, players) for name in player_names]

        duplicate_ids = {player.nba_player_id for player in resolved}
        if len(duplicate_ids) != len(resolved):
            raise ValueError("Player input contains duplicates. Please supply 5 distinct players.")

        missing_play_types = [
            player.full_name
            for player in resolved
            if not any(metric.season == season for metric in player.defensive_play_types)
        ]
        if missing_play_types:
            joined = ", ".join(missing_play_types)
            raise ValueError(f"Missing defensive play-type data for season {season}: {joined}")

        return resolved
    finally:
        session.close()


def make_custom_lineup_demo_frame(
    player_names: list[str],
    *,
    season: str,
    dataset: pd.DataFrame | None = None,
    database_url: str = DEFAULT_DATABASE_URL,
) -> pd.DataFrame:
    """Build a one-row demo dataset from five user-specified players."""
    players = load_players_for_demo(player_names, season=season, database_url=database_url)
    if dataset is not None and not dataset.empty:
        matched = _build_historical_or_blended_lineup(players, dataset, season=season)
        if matched is not None:
            matched["case_label"] = "Custom Lineup"
            return pd.DataFrame([matched])

    row = build_synthetic_lineup_row(players, season=season)
    row["lineup_source"] = "synthetic_player_profile"
    frame = pd.DataFrame([row])
    frame["case_label"] = "Custom Lineup"
    return frame


def select_lineups(
    dataset: pd.DataFrame,
    *,
    lineup_key: str | None = None,
    team: str | None = None,
    search: str | None = None,
    min_minutes: float = 0.0,
    limit: int = 5,
) -> pd.DataFrame:
    """Filter the engineered dataset down to the lineups a user wants to demo."""
    filtered = dataset.copy()

    if min_minutes > 0:
        filtered = filtered[filtered["minutes_played"].fillna(0) >= min_minutes]

    if lineup_key:
        filtered = filtered[filtered["lineup_key"] == lineup_key]

    if team:
        filtered = filtered[
            filtered["team_abbreviation"].fillna("").str.upper() == team.strip().upper()
        ]

    if search:
        needle = search.strip().casefold()
        filtered = filtered[
            filtered["lineup_name"].fillna("").str.casefold().str.contains(needle, regex=False)
        ]

    return filtered.sort_values("minutes_played", ascending=False).head(limit).reset_index(drop=True)


def build_default_case_studies(dataset: pd.DataFrame, min_minutes: float = 10.0) -> pd.DataFrame:
    """Pick a handful of interesting lineups so the demo works without manual input."""
    eligible = dataset[dataset["minutes_played"].fillna(0) >= min_minutes].copy()
    if eligible.empty:
        eligible = dataset.copy()

    cases: list[pd.Series] = []
    used_keys: set[str] = set()

    for label, metric, descending in CASE_STUDY_LABELS:
        candidates = eligible.dropna(subset=[metric]).sort_values(metric, ascending=not descending)
        candidates = candidates[~candidates["lineup_key"].isin(used_keys)]
        if candidates.empty:
            continue

        selected = candidates.iloc[0].copy()
        selected["case_label"] = label
        cases.append(selected)
        used_keys.add(str(selected["lineup_key"]))

    if not cases and not dataset.empty:
        fallback = dataset.iloc[0].copy()
        fallback["case_label"] = "Sample Lineup"
        cases.append(fallback)

    return pd.DataFrame(cases).reset_index(drop=True)


def plot_recommendation_results(result: DemoResult, output_dir: str = "data/processed/plots") -> str:
    """Generate and save a bar chart comparing the schemes."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = "".join([c if c.isalnum() else "_" for c in result.lineup_name])
    output_path = os.path.join(output_dir, f"recommendation_{safe_name}.png")

    df = result.recommendation.ranked_schemes

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="scheme", y="predicted_value", data=df, hue="scheme", palette="viridis", legend=False)

    # Add actual baseline if it exists as a baseline line
    plt.axhline(y=result.baseline_prediction, color='r', linestyle='--', label=f'Baseline ({result.baseline_prediction:.1f})')

    # Add labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(
            f"{df['predicted_value'].iloc[i]:.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha='center',
            va='bottom',
            fontsize=10,
            color='black',
            xytext=(0, 5),
            textcoords='offset points',
        )

    plt.ylim(min(df["predicted_value"]) * 0.95, max(df["predicted_value"]) * 1.05)
    plt.title(f"Predicted Defensive Rating by Scheme\n{result.lineup_name}")
    plt.ylabel("Predicted Defensive Rating (Lower is Better)")
    plt.xlabel("Defensive Scheme")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2.0)

    return output_path


def run_demo_for_lineups(
    lineups: pd.DataFrame,
    artifacts: TrainingArtifacts,
    *,
    case_label_column: str | None = None,
    scheme_profiles: dict[str, dict[str, float]] | None = None,
) -> list[DemoResult]:
    """Run recommendations for one or more selected lineup rows."""
    results: list[DemoResult] = []
    for index, (_, lineup_row) in enumerate(lineups.iterrows(), start=1):
        recommendation = recommend_scheme(lineup_row, artifacts, scheme_profiles=scheme_profiles)
        baseline_prediction = _predict_baseline(lineup_row, artifacts)
        case_label = (
            str(lineup_row[case_label_column])
            if case_label_column and case_label_column in lineup_row
            else f"Lineup {index}"
        )
        results.append(
            DemoResult(
                case_label=case_label,
                lineup_key=str(lineup_row["lineup_key"]),
                lineup_name=str(lineup_row.get("lineup_name") or lineup_row["lineup_key"]),
                team_abbreviation=_optional_str(lineup_row.get("team_abbreviation")),
                minutes_played=_optional_float(lineup_row.get("minutes_played")),
                actual_target=_optional_float(lineup_row.get(artifacts.target_column)),
                actual_target_source=_optional_str(lineup_row.get("defensive_rating_target_source")),
                lineup_source=_optional_str(lineup_row.get("lineup_source")),
                baseline_prediction=baseline_prediction,
                recommendation=recommendation,
            )
        )

    return results


def format_demo_result(result: DemoResult, top_explanations: int = 5) -> str:
    """Format one demo result as presentation-friendly plain text."""
    lines = [
        f"{result.case_label}",
        f"Lineup: {result.lineup_name}",
    ]

    meta = []
    if result.team_abbreviation:
        meta.append(f"Team: {result.team_abbreviation}")
    if result.minutes_played is not None:
        meta.append(f"Minutes: {result.minutes_played:.1f}")
    if result.actual_target is not None:
        target_label = "Actual target"
        if result.actual_target_source == "fallback_points_allowed_per_48":
            target_label = "Actual target (fallback per 48)"
        elif result.actual_target_source == "api_defensive_rating":
            target_label = "Actual target (API defensive rating)"
        elif result.actual_target_source == "weighted_historical_overlap":
            target_label = "Estimated historical target (weighted overlap)"
        meta.append(f"{target_label}: {result.actual_target:.2f}")
    if result.lineup_source:
        meta.append(f"Source: {result.lineup_source}")
    meta.append(f"Baseline prediction: {result.baseline_prediction:.2f}")
    lines.append(" | ".join(meta))
    lines.append(f"Recommended scheme: {result.recommendation.recommended_scheme}")
    lines.append("Scheme ranking:")
    lines.append(result.recommendation.ranked_schemes.to_string(index=False))

    explanation = result.recommendation.explanation.head(top_explanations)
    if not explanation.empty:
        lines.append("Top explanation rows:")
        lines.append(explanation.to_string(index=False))

    plot_path = plot_recommendation_results(result)
    lines.append(f"\n=> Bar chart generated at: {plot_path}")

    return "\n".join(lines)


def _predict_baseline(lineup_row: pd.Series, artifacts: TrainingArtifacts) -> float:
    lineup_frame = lineup_row.to_frame().T
    if artifacts.target_column not in lineup_frame.columns:
        lineup_frame[artifacts.target_column] = 0.0
    else:
        lineup_frame[artifacts.target_column] = pd.to_numeric(
            lineup_frame[artifacts.target_column],
            errors="coerce",
        ).fillna(0.0)
    other_target = "opponent_ppp_target" if artifacts.target_column == "defensive_rating_target" else "defensive_rating_target"
    if other_target not in lineup_frame.columns:
        lineup_frame[other_target] = 0.0
    else:
        lineup_frame[other_target] = pd.to_numeric(
            lineup_frame[other_target],
            errors="coerce",
        ).fillna(0.0)
    features, _, _ = prepare_training_matrices(lineup_frame, target_column=artifacts.target_column)
    features = features.reindex(columns=artifacts.feature_columns, fill_value=0.0)
    return float(artifacts.model.predict(features)[0])


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _resolve_player_name(name: str, players: list[Player]) -> Player:
    needle = _normalize_name(name)

    exact_matches = [player for player in players if _normalize_name(player.full_name) == needle]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(_format_ambiguity_message(name, exact_matches))

    partial_matches = [player for player in players if needle in _normalize_name(player.full_name)]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        raise ValueError(_format_ambiguity_message(name, partial_matches))

    raise ValueError(f"Could not find a player matching '{name}'.")


def _format_ambiguity_message(name: str, matches: list[Player]) -> str:
    options = ", ".join(sorted(player.full_name for player in matches[:5]))
    return f"Player name '{name}' is ambiguous. Try one of: {options}"


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _build_historical_or_blended_lineup(
    players: list[Player],
    dataset: pd.DataFrame,
    *,
    season: str,
) -> dict[str, object] | None:
    desired_ids = [player.nba_player_id for player in players]
    desired_set = set(desired_ids)
    same_team = _resolve_team_abbreviation(players)

    working = dataset.copy()
    working["_parsed_ids"] = working["lineup_key"].map(parse_lineup_player_ids)
    working["_overlap_count"] = working["_parsed_ids"].map(lambda ids: len(set(ids) & desired_set))
    working = working[working["_overlap_count"] >= 2].copy()
    if working.empty:
        return None

    if same_team and same_team != "MIX":
        same_team_matches = working[working["team_abbreviation"] == same_team].copy()
        if not same_team_matches.empty:
            working = same_team_matches

    exact_matches = working[working["_overlap_count"] == 5].copy()
    if not exact_matches.empty:
        exact_row = exact_matches.sort_values("minutes_played", ascending=False).iloc[0].copy()
        exact_row["lineup_source"] = "exact_historical_lineup"
        return exact_row.to_dict()

    working["_weight"] = working.apply(_overlap_weight, axis=1)
    working = working[working["_weight"] > 0].copy()
    if working.empty:
        return None

    synthetic = build_synthetic_lineup_row(players, season=season)
    blended = dict(synthetic)
    weighted = _weighted_feature_average(working)

    for column, value in weighted.items():
        blended[column] = value

    blended["lineup_name"] = " - ".join(player.full_name for player in players)
    blended["team_abbreviation"] = same_team
    blended["lineup_source"] = _resolve_blend_label(working)
    if pd.notna(weighted.get("defensive_rating_target")):
        blended["defensive_rating_target_source"] = "weighted_historical_overlap"
    return blended


def _overlap_weight(row: pd.Series) -> float:
    overlap = int(row["_overlap_count"])
    overlap_weight = OVERLAP_WEIGHTS.get(overlap, 0.0)
    minutes = float(row.get("minutes_played") or 0.0)
    minute_weight = 1.0 + min(minutes, 500.0) / 500.0
    return overlap_weight * minute_weight


def _weighted_feature_average(candidates: pd.DataFrame) -> dict[str, float]:
    weights = candidates["_weight"].astype(float)
    numeric_columns: dict[str, float] = {}
    for column in candidates.columns:
        if column in NON_BLEND_COLUMNS or column.startswith("_"):
            continue

        series = pd.to_numeric(candidates[column], errors="coerce")
        valid = series.notna() & weights.notna()
        if not valid.any():
            continue

        numeric_columns[column] = float((series[valid] * weights[valid]).sum() / weights[valid].sum())

    return numeric_columns


def _resolve_blend_label(candidates: pd.DataFrame) -> str:
    max_overlap = int(candidates["_overlap_count"].max())
    if max_overlap >= 4:
        return "weighted_historical_overlap_4_plus"
    if max_overlap == 3:
        return "weighted_historical_overlap_3"
    return "weighted_historical_overlap_2"
