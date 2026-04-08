from pathlib import Path

import pandas as pd

from src.database.connection import create_session_factory, initialize_database
from src.database.ingest import (
    IngestConfig,
    normalize_lineup_records,
    parse_lineup_player_ids,
    save_raw_frame,
    upsert_defensive_play_types,
    upsert_lineups,
    upsert_players,
)
from src.database.schema import DefensivePlayType, LineupMetric, LineupPlayer, Player


def test_parse_lineup_player_ids_returns_five_ids():
    player_ids = parse_lineup_player_ids("201939-202691-203110-203952-1626172")

    assert player_ids == [201939, 202691, 203110, 203952, 1626172]


def test_normalize_lineup_records_skips_invalid_rows():
    df = pd.DataFrame(
        [
            {
                "GROUP_ID": "201939-202691-203110-203952-1626172",
                "GROUP_NAME": "Curry-Thompson-Green-Wiggins-Looney",
                "TEAM_ABBREVIATION": "GSW",
                "MIN": 48.0,
                "POSS": 100.0,
                "DEF_RATING": 110.2,
            },
            {
                "GROUP_ID": "not-a-lineup",
                "GROUP_NAME": "Invalid",
            },
        ]
    )

    records = normalize_lineup_records(df, "2024-25")

    assert len(records) == 1
    assert records[0]["team_abbreviation"] == "GSW"
    assert records[0]["opponent_ppp"] == 1.102


def test_save_raw_frame_writes_snapshot(tmp_path: Path):
    df = pd.DataFrame([{"PLAYER_ID": 1, "PLAYER_NAME": "Test Player"}])
    config = IngestConfig(raw_data_dir=str(tmp_path), sleep_seconds=0)

    output_path = save_raw_frame(df, "player_defense", config, metadata={"source": "test"})

    assert output_path.exists()
    assert output_path.suffix == ".json"
    assert "player_defense" in str(output_path)
    assert "2024-25" in str(output_path)


def test_database_upserts_persist_players_play_types_and_lineups(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'phase1.sqlite'}"
    initialize_database(database_url)
    session_factory = create_session_factory(database_url)

    players_df = pd.DataFrame(
        [
            {
                "PLAYER_ID": 201939,
                "PLAYER_NAME": "Stephen Curry",
                "TEAM_ABBREVIATION": "GSW",
                "POSITION": "G",
                "HEIGHT": "6-2",
                "WEIGHT": 185,
            },
            {
                "PLAYER_ID": 202691,
                "PLAYER_NAME": "Klay Thompson",
                "TEAM_ABBREVIATION": "GSW",
                "POSITION": "G",
                "HEIGHT": "6-6",
                "WEIGHT": 215,
            },
        ]
    )
    upsert_players(session_factory, players_df)

    play_types_df = pd.DataFrame(
        [
            {
                "PLAYER_ID": 201939,
                "PLAY_TYPE": "Isolation",
                "POSS": 25,
                "PTS": 24,
                "PPP": 0.96,
                "FREQ": 11.0,
                "PERCENTILE": 67.0,
            }
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

    session = session_factory()
    try:
        assert session.query(Player).count() == 5
        assert session.query(DefensivePlayType).count() == 1
        assert session.query(LineupMetric).count() == 1
        assert session.query(LineupPlayer).count() == 5

        lineup = session.query(LineupMetric).one()
        assert lineup.lineup_name == "Sample Lineup"
        assert lineup.team_abbreviation == "GSW"
        assert lineup.opponent_ppp == 1.084
    finally:
        session.close()
