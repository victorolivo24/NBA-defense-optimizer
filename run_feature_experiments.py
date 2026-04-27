"""Run comparative feature-set experiments for lineup defensive rating prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import DEFAULT_MIN_LINEUP_MINUTES, build_training_dataset
from src.models import train_baseline_regressor

SEASONS = ["2022-23", "2023-24", "2024-25"]
TARGET_COLUMN = "defensive_rating_target"
OUTPUT_PATH = Path("data/processed/feature_experiment_results.csv")
IMPORTANCE_OUTPUT_PATH = Path("data/processed/feature_experiment_importances.csv")

COMMON_CONTEXT_COLUMNS = [
    "guard_count",
    "forward_count",
    "center_count",
    "avg_height_inches",
    "avg_weight",
]
PLAYER_DEF_RTG_COLUMNS = [
    "avg_player_def_rtg",
    "best_player_def_rtg",
    "worst_player_def_rtg",
]
PLAY_TYPE_PREFIXES = [
    "isolation_",
    "pick_and_roll_ball_handler_",
    "pick_and_roll_roll_man_",
    "spot_up_",
]
PLAYER_BASIC_PREFIXES = [
    "avg_player_dreb",
    "max_player_dreb",
    "avg_player_stl",
    "max_player_stl",
    "avg_player_blk",
    "max_player_blk",
    "avg_player_def_ws",
    "max_player_def_ws",
    "avg_player_opp_pts_off_tov",
    "max_player_opp_pts_off_tov",
    "avg_player_opp_pts_2nd_chance",
    "max_player_opp_pts_2nd_chance",
    "avg_player_opp_pts_fb",
    "max_player_opp_pts_fb",
    "avg_player_opp_pts_paint",
    "max_player_opp_pts_paint",
]
FOUR_FACTOR_COLUMNS = [
    "lineup_opp_efg_pct",
    "lineup_opp_tov_rate",
    "lineup_opp_orb_rate",
    "lineup_opp_fta_rate",
]

EXPERIMENTS = [
    {
        "name": "current_baseline_like",
        "description": "Context + play-type features + aggregated player defensive rating.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS + PLAYER_DEF_RTG_COLUMNS,
        "prefixes": PLAY_TYPE_PREFIXES,
    },
    {
        "name": "play_types_only",
        "description": "Context + Synergy play-type aggregates only.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS,
        "prefixes": PLAY_TYPE_PREFIXES,
    },
    {
        "name": "player_def_rating_only",
        "description": "Context + aggregated player defensive ratings only.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS + PLAYER_DEF_RTG_COLUMNS,
        "prefixes": [],
    },
    {
        "name": "player_basic_only",
        "description": "Context + aggregated player basic defensive box-score features only.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS,
        "prefixes": PLAYER_BASIC_PREFIXES,
    },
    {
        "name": "player_basic_plus_advanced",
        "description": "Context + aggregated player basic defensive box-score features + aggregated player defensive rating.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS + PLAYER_DEF_RTG_COLUMNS,
        "prefixes": PLAYER_BASIC_PREFIXES,
    },
    {
        "name": "play_types_plus_player_basic",
        "description": "Context + play-type aggregates + player basic defensive box-score features.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS,
        "prefixes": PLAY_TYPE_PREFIXES + PLAYER_BASIC_PREFIXES,
    },
    {
        "name": "all_valid_features",
        "description": "Context + play-type aggregates + player defensive rating + player basic defensive box-score features.",
        "valid_for_production": True,
        "columns": COMMON_CONTEXT_COLUMNS + PLAYER_DEF_RTG_COLUMNS,
        "prefixes": PLAY_TYPE_PREFIXES + PLAYER_BASIC_PREFIXES,
    },
    {
        "name": "four_factors_leakage_control",
        "description": "Context + same-stint lineup four factors. Upper-bound leakage control, not valid for production recommendation.",
        "valid_for_production": False,
        "columns": COMMON_CONTEXT_COLUMNS + FOUR_FACTOR_COLUMNS,
        "prefixes": [],
    },
    {
        "name": "all_features_plus_four_factors_leakage_control",
        "description": "All engineered features plus same-stint lineup four factors. Upper-bound leakage control, not valid for production recommendation.",
        "valid_for_production": False,
        "columns": COMMON_CONTEXT_COLUMNS + PLAYER_DEF_RTG_COLUMNS + FOUR_FACTOR_COLUMNS,
        "prefixes": PLAY_TYPE_PREFIXES + PLAYER_BASIC_PREFIXES,
    },
]
TUNED_EXPERIMENTS = {
    experiment["name"]
    for experiment in EXPERIMENTS
    if experiment["valid_for_production"]
}


def build_combined_dataset() -> pd.DataFrame:
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    datasets = [
        build_training_dataset(
            session_factory=session_factory,
            season=season,
            min_minutes=DEFAULT_MIN_LINEUP_MINUTES,
        )
        for season in SEASONS
    ]
    return pd.concat(datasets, ignore_index=True)


def matching_columns(dataset: pd.DataFrame, *, columns: list[str], prefixes: list[str]) -> list[str]:
    matched = [column for column in columns if column in dataset.columns]
    for prefix in prefixes:
        matched.extend(column for column in dataset.columns if column.startswith(prefix))
    # preserve order, remove duplicates
    seen: set[str] = set()
    ordered: list[str] = []
    for column in matched:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def build_experiment_dataset(dataset: pd.DataFrame, experiment: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    selected = matching_columns(
        dataset,
        columns=list(experiment["columns"]),
        prefixes=list(experiment["prefixes"]),
    )
    keep = [
        "lineup_id",
        "lineup_key",
        "lineup_name",
        "season",
        "team_abbreviation",
        "defensive_rating_target_source",
        "defensive_rating_target",
        "opponent_ppp_target",
        "possessions",
        "minutes_played",
        *selected,
    ]
    available_keep = [column for column in keep if column in dataset.columns]
    return dataset[available_keep].copy(), selected


if __name__ == "__main__":
    dataset = build_combined_dataset()
    results: list[dict[str, object]] = []
    importances: list[pd.DataFrame] = []

    for experiment in EXPERIMENTS:
        experiment_dataset, selected_features = build_experiment_dataset(dataset, experiment)
        model_variants = [("fixed", False)]
        if experiment["name"] in TUNED_EXPERIMENTS:
            model_variants.append(("tuned", True))

        for model_mode, tune_hyperparameters in model_variants:
            artifacts = train_baseline_regressor(
                dataset=experiment_dataset,
                target_column=TARGET_COLUMN,
                tune_hyperparameters=tune_hyperparameters,
            )
            results.append(
                {
                    "experiment": experiment["name"],
                    "model_mode": model_mode,
                    "valid_for_production": experiment["valid_for_production"],
                    "feature_count": len(selected_features),
                    "description": experiment["description"],
                    "train_rows": artifacts.train_rows,
                    "test_rows": artifacts.test_rows,
                    "train_mae": artifacts.metrics["train_mae"],
                    "train_rmse": artifacts.metrics["train_rmse"],
                    "train_r2": artifacts.metrics["train_r2"],
                    "test_mae": artifacts.metrics["test_mae"],
                    "test_rmse": artifacts.metrics["test_rmse"],
                    "test_r2": artifacts.metrics["test_r2"],
                    "features": ", ".join(selected_features),
                }
            )
            importance_df = artifacts.model.feature_importance().copy()
            importance_df["experiment"] = experiment["name"]
            importance_df["model_mode"] = model_mode
            importances.append(importance_df)
            print(
                f"{experiment['name']} ({model_mode}): test_r2={artifacts.metrics['test_r2']:.4f}, "
                f"test_mae={artifacts.metrics['test_mae']:.4f}, test_rmse={artifacts.metrics['test_rmse']:.4f}"
            )

    results_df = pd.DataFrame(results).sort_values(
        ["valid_for_production", "test_r2"], ascending=[False, False]
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote experiment results to {OUTPUT_PATH}")

    if importances:
        importance_results = pd.concat(importances, ignore_index=True)
        importance_results.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)
        print(f"Wrote experiment importances to {IMPORTANCE_OUTPUT_PATH}")
