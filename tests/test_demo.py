import pandas as pd

from src.database.schema import DefensivePlayType, Player
from src.demo.workflow import build_default_case_studies, format_demo_result, select_lineups
from src.features import build_synthetic_lineup_row
from src.models.recommendation import SchemeRecommendation


def test_select_lineups_filters_team_and_search():
    dataset = pd.DataFrame(
        [
            {"lineup_key": "a", "lineup_name": "J. Brunson - J. Hart", "team_abbreviation": "NYK", "minutes_played": 18.0},
            {"lineup_key": "b", "lineup_name": "J. Tatum - J. Brown", "team_abbreviation": "BOS", "minutes_played": 20.0},
            {"lineup_key": "c", "lineup_name": "J. Brunson - K. Towns", "team_abbreviation": "NYK", "minutes_played": 12.0},
        ]
    )

    selected = select_lineups(dataset, team="nyk", search="brunson", min_minutes=15.0, limit=5)

    assert list(selected["lineup_key"]) == ["a"]


def test_build_default_case_studies_creates_distinct_labels():
    dataset = pd.DataFrame(
        [
            {
                "lineup_key": "a",
                "lineup_name": "Lineup A",
                "minutes_played": 30.0,
                "defensive_rating_target": 105.0,
                "isolation_ppp_mean": 0.90,
            },
            {
                "lineup_key": "b",
                "lineup_name": "Lineup B",
                "minutes_played": 25.0,
                "defensive_rating_target": 99.0,
                "isolation_ppp_mean": 1.10,
            },
            {
                "lineup_key": "c",
                "lineup_name": "Lineup C",
                "minutes_played": 20.0,
                "defensive_rating_target": 118.0,
                "isolation_ppp_mean": 0.88,
            },
        ]
    )

    case_studies = build_default_case_studies(dataset, min_minutes=10.0)

    assert set(case_studies["case_label"]).issubset(
        {
            "Most Used Lineup",
            "Best Actual Defense",
            "Worst Actual Defense",
            "Best Isolation Defense",
        }
    )
    assert len(case_studies) == 3
    assert case_studies["lineup_key"].is_unique


def test_format_demo_result_contains_recommendation_summary():
    recommendation = SchemeRecommendation(
        recommended_scheme="Switch",
        predicted_value=108.2,
        ranked_schemes=pd.DataFrame(
            [
                {"scheme": "Switch", "predicted_value": 108.2},
                {"scheme": "Drop", "predicted_value": 109.7},
            ]
        ),
        explanation=pd.DataFrame(
            [{"scheme": "Switch", "feature": "isolation_ppp_mean", "adjustment": -0.05}]
        ),
    )
    from src.demo.workflow import DemoResult

    result = DemoResult(
        case_label="Case Study",
        lineup_key="abc",
        lineup_name="Sample Lineup",
        team_abbreviation="BOS",
        minutes_played=18.5,
        actual_target=106.1,
        actual_target_source="fallback_points_allowed_per_48",
        baseline_prediction=107.4,
        recommendation=recommendation,
    )

    formatted = format_demo_result(result)

    assert "Recommended scheme: Switch" in formatted
    assert "Sample Lineup" in formatted
    assert "Scheme ranking:" in formatted
    assert "Actual target (fallback per 48)" in formatted


def test_build_synthetic_lineup_row_aggregates_five_players():
    players = []
    for index in range(5):
        player = Player(
            nba_player_id=100 + index,
            full_name=f"Player {index}",
            team_abbreviation="NYK",
            position="G" if index < 2 else ("F" if index < 4 else "C"),
            height="6-6",
            weight=220,
        )
        player.defensive_play_types = [
            DefensivePlayType(
                season="2024-25",
                play_type="Isolation",
                ppp_allowed=0.9 + index * 0.01,
                possessions=20 + index,
                percentile=60 + index,
            ),
            DefensivePlayType(
                season="2024-25",
                play_type="Pick and Roll Ball Handler",
                ppp_allowed=1.0 + index * 0.01,
                possessions=15 + index,
                percentile=55 + index,
            ),
            DefensivePlayType(
                season="2024-25",
                play_type="Pick and Roll Roll Man",
                ppp_allowed=1.1 + index * 0.01,
                possessions=10 + index,
                percentile=50 + index,
            ),
            DefensivePlayType(
                season="2024-25",
                play_type="Spot Up",
                ppp_allowed=0.95 + index * 0.01,
                possessions=18 + index,
                percentile=58 + index,
            ),
        ]
        players.append(player)

    row = build_synthetic_lineup_row(players, season="2024-25")

    assert row["lineup_size"] == 5
    assert row["guard_count"] == 2
    assert row["forward_count"] == 2
    assert row["center_count"] == 1
    assert row["team_abbreviation"] == "NYK"
    assert row["defensive_rating_target"] is None
    assert row["defensive_rating_target_source"] is None
    assert row["isolation_player_count"] == 5
