from pathlib import Path

import pandas as pd

from src.database.connection import create_session_factory, initialize_database
from src.database.ingest import upsert_defensive_play_types, upsert_lineups, upsert_players
from src.features import build_training_dataset, export_training_dataset


def test_build_training_dataset_aggregates_lineup_features(tmp_path: Path):
    session_factory = _seed_feature_test_database(tmp_path)

    dataset = build_training_dataset(session_factory, season="2024-25")

    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["lineup_size"] == 5
    assert row["guard_count"] == 2
    assert row["forward_count"] == 2
    assert row["center_count"] == 1
    assert row["avg_height_inches"] == 78.0
    assert row["avg_weight"] == 221.0
    assert row["isolation_player_count"] == 5
    assert round(row["isolation_ppp_mean"], 3) == 0.98
    assert round(row["isolation_ppp_max"], 3) == 1.1
    assert round(row["spot_up_percentile_mean"], 1) == 67.0
    assert round(row["defensive_rating_target"], 1) == 108.4


def test_build_training_dataset_respects_min_minutes_filter(tmp_path: Path):
    session_factory = _seed_feature_test_database(tmp_path)

    dataset = build_training_dataset(session_factory, season="2024-25", min_minutes=40.0)

    assert dataset.empty


def test_export_training_dataset_writes_csv(tmp_path: Path):
    session_factory = _seed_feature_test_database(tmp_path)
    database_url = f"sqlite:///{tmp_path / 'phase2.sqlite'}"
    output_path = tmp_path / "features.csv"

    export_training_dataset(
        season="2024-25",
        output_path=str(output_path),
        database_url=database_url,
    )

    assert output_path.exists()
    exported = pd.read_csv(output_path)
    assert len(exported) == 1
    assert exported.loc[0, "lineup_key"] == "201939-202691-203110-203952-1626172"


def _seed_feature_test_database(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'phase2.sqlite'}"
    initialize_database(database_url)
    session_factory = create_session_factory(database_url)

    players_df = pd.DataFrame(
        [
            {"PLAYER_ID": 201939, "PLAYER_NAME": "Stephen Curry", "TEAM_ABBREVIATION": "GSW", "POSITION": "G", "HEIGHT": "6-2", "WEIGHT": 185},
            {"PLAYER_ID": 202691, "PLAYER_NAME": "Klay Thompson", "TEAM_ABBREVIATION": "GSW", "POSITION": "G", "HEIGHT": "6-6", "WEIGHT": 215},
            {"PLAYER_ID": 203110, "PLAYER_NAME": "Draymond Green", "TEAM_ABBREVIATION": "GSW", "POSITION": "F", "HEIGHT": "6-6", "WEIGHT": 230},
            {"PLAYER_ID": 203952, "PLAYER_NAME": "Andrew Wiggins", "TEAM_ABBREVIATION": "GSW", "POSITION": "F", "HEIGHT": "6-7", "WEIGHT": 197},
            {"PLAYER_ID": 1626172, "PLAYER_NAME": "Kevon Looney", "TEAM_ABBREVIATION": "GSW", "POSITION": "C", "HEIGHT": "6-9", "WEIGHT": 278},
        ]
    )
    upsert_players(session_factory, players_df)

    play_types_df = pd.DataFrame(
        [
            {"PLAYER_ID": 201939, "PLAY_TYPE": "Isolation", "POSS": 20, "PTS": 18, "PPP": 0.90, "FREQ": 8, "PERCENTILE": 70},
            {"PLAYER_ID": 202691, "PLAY_TYPE": "Isolation", "POSS": 22, "PTS": 21, "PPP": 0.95, "FREQ": 10, "PERCENTILE": 65},
            {"PLAYER_ID": 203110, "PLAY_TYPE": "Isolation", "POSS": 18, "PTS": 18, "PPP": 1.00, "FREQ": 7, "PERCENTILE": 60},
            {"PLAYER_ID": 203952, "PLAY_TYPE": "Isolation", "POSS": 16, "PTS": 17.6, "PPP": 1.10, "FREQ": 6, "PERCENTILE": 55},
            {"PLAYER_ID": 1626172, "PLAY_TYPE": "Isolation", "POSS": 19, "PTS": 18.05, "PPP": 0.95, "FREQ": 7, "PERCENTILE": 58},
            {"PLAYER_ID": 201939, "PLAY_TYPE": "Spot Up", "POSS": 30, "PTS": 27.9, "PPP": 0.93, "FREQ": 12, "PERCENTILE": 72},
            {"PLAYER_ID": 202691, "PLAY_TYPE": "Spot Up", "POSS": 28, "PTS": 28.0, "PPP": 1.00, "FREQ": 11, "PERCENTILE": 68},
            {"PLAYER_ID": 203110, "PLAY_TYPE": "Spot Up", "POSS": 26, "PTS": 25.0, "PPP": 0.96, "FREQ": 10, "PERCENTILE": 66},
            {"PLAYER_ID": 203952, "PLAY_TYPE": "Spot Up", "POSS": 25, "PTS": 23.0, "PPP": 0.92, "FREQ": 9, "PERCENTILE": 64},
            {"PLAYER_ID": 1626172, "PLAY_TYPE": "Spot Up", "POSS": 12, "PTS": 12.5, "PPP": 1.04, "FREQ": 5, "PERCENTILE": 65},
        ]
    )
    upsert_defensive_play_types(session_factory, play_types_df, "2024-25")

    lineups_df = pd.DataFrame(
        [
            {
                "GROUP_ID": "201939-202691-203110-203952-1626172",
                "GROUP_NAME": "Sample Lineup",
                "TEAM_ABBREVIATION": "GSW",
                "MIN": 32.5,
                "POSS": 72.0,
                "DEF_RATING": 108.4,
            }
        ]
    )
    upsert_lineups(session_factory, lineups_df, "2024-25")
    return session_factory
