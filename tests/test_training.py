import pandas as pd

from src.models import prepare_training_matrices, train_baseline_regressor


def test_prepare_training_matrices_selects_numeric_feature_columns():
    dataset = pd.DataFrame(
        [
            {
                "lineup_key": "A",
                "lineup_name": "Alpha",
                "season": "2024-25",
                "team_abbreviation": "AAA",
                "minutes_played": 20.0,
                "lineup_size": 5,
                "guard_count": 2,
                "avg_height_inches": 78.0,
                "isolation_ppp_mean": 0.95,
                "defensive_rating_target": 108.0,
                "opponent_ppp_target": 1.08,
            },
            {
                "lineup_key": "B",
                "lineup_name": "Beta",
                "season": "2024-25",
                "team_abbreviation": "BBB",
                "minutes_played": 22.0,
                "lineup_size": 5,
                "guard_count": 1,
                "avg_height_inches": 79.0,
                "isolation_ppp_mean": 1.02,
                "defensive_rating_target": 111.0,
                "opponent_ppp_target": 1.11,
            },
        ]
    )

    features, target = prepare_training_matrices(dataset)

    assert list(features.columns) == [
        "minutes_played",
        "lineup_size",
        "guard_count",
        "avg_height_inches",
        "isolation_ppp_mean",
    ]
    assert list(target) == [108.0, 111.0]


def test_train_baseline_regressor_returns_metrics():
    rows = []
    for idx in range(8):
        rows.append(
            {
                "lineup_id": idx + 1,
                "lineup_key": f"L{idx}",
                "lineup_name": f"Lineup {idx}",
                "season": "2024-25",
                "team_abbreviation": "AAA",
                "minutes_played": 15.0 + idx,
                "possessions": 40.0 + idx,
                "lineup_size": 5,
                "guard_count": 2 if idx % 2 == 0 else 1,
                "forward_count": 2,
                "center_count": 1,
                "avg_height_inches": 77.0 + (idx % 3),
                "avg_weight": 215.0 + idx,
                "isolation_player_count": 5,
                "isolation_ppp_mean": 0.90 + idx * 0.02,
                "isolation_ppp_max": 1.00 + idx * 0.02,
                "spot_up_percentile_mean": 60.0 + idx,
                "defensive_rating_target": 105.0 + idx * 1.5,
                "opponent_ppp_target": 1.05 + idx * 0.015,
            }
        )

    dataset = pd.DataFrame(rows)
    artifacts = train_baseline_regressor(dataset, target_column="defensive_rating_target")

    assert artifacts.target_column == "defensive_rating_target"
    assert artifacts.train_rows > 0
    assert artifacts.test_rows > 0
    assert "mae" in artifacts.metrics
    assert "rmse" in artifacts.metrics
    assert len(artifacts.feature_columns) > 0


def test_prepare_training_matrices_uses_median_imputation():
    dataset = pd.DataFrame(
        [
            {
                "lineup_key": "A",
                "lineup_name": "Alpha",
                "season": "2024-25",
                "team_abbreviation": "AAA",
                "minutes_played": 20.0,
                "lineup_size": 5,
                "guard_count": 2,
                "avg_height_inches": 78.0,
                "isolation_ppp_mean": 1.0,
                "defensive_rating_target": 108.0,
                "opponent_ppp_target": 1.08,
            },
            {
                "lineup_key": "B",
                "lineup_name": "Beta",
                "season": "2024-25",
                "team_abbreviation": "BBB",
                "minutes_played": 22.0,
                "lineup_size": 5,
                "guard_count": 2,
                "avg_height_inches": 79.0,
                "isolation_ppp_mean": 2.0,
                "defensive_rating_target": 109.0,
                "opponent_ppp_target": 1.09,
            },
            {
                "lineup_key": "C",
                "lineup_name": "Gamma",
                "season": "2024-25",
                "team_abbreviation": "CCC",
                "minutes_played": 24.0,
                "lineup_size": 5,
                "guard_count": 1,
                "avg_height_inches": 80.0,
                "isolation_ppp_mean": 100.0,
                "defensive_rating_target": 110.0,
                "opponent_ppp_target": 1.10,
            },
            {
                "lineup_key": "D",
                "lineup_name": "Delta",
                "season": "2024-25",
                "team_abbreviation": "DDD",
                "minutes_played": 26.0,
                "lineup_size": 5,
                "guard_count": 1,
                "avg_height_inches": 81.0,
                "isolation_ppp_mean": None,
                "defensive_rating_target": 111.0,
                "opponent_ppp_target": 1.11,
            },
        ]
    )

    features, _ = prepare_training_matrices(dataset)

    assert features.loc[3, "isolation_ppp_mean"] == 2.0
