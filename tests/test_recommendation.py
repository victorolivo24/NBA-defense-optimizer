import pandas as pd

from src.models.recommendation import apply_scheme_profile, recommend_scheme
from src.models.training import TrainingArtifacts


class DummyModel:
    def predict(self, features: pd.DataFrame):
        return (
            features["isolation_ppp_mean"]
            + features["pick_and_roll_ball_handler_ppp_mean"]
            + features["pick_and_roll_roll_man_ppp_mean"]
            + features["spot_up_ppp_mean"]
        ).to_numpy()

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.copy()


def test_apply_scheme_profile_adjusts_only_matching_columns():
    features = pd.DataFrame([{"isolation_ppp_mean": 1.0, "spot_up_ppp_mean": 1.1}])

    adjusted = apply_scheme_profile(features, {"isolation_ppp_mean": -0.1, "missing": 1.0})

    assert adjusted.loc[0, "isolation_ppp_mean"] == 0.9
    assert adjusted.loc[0, "spot_up_ppp_mean"] == 1.1


def test_apply_scheme_profile_weights_ppp_adjustments_by_volume_and_matchup():
    features = pd.DataFrame(
        [
            {
                "pick_and_roll_roll_man_ppp_mean": 1.20,
                "pick_and_roll_roll_man_possessions_mean": 18.0,
            }
        ]
    )
    artifacts = TrainingArtifacts(
        model=DummyModel(),
        target_column="defensive_rating_target",
        feature_columns=list(features.columns),
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_means={
            "pick_and_roll_roll_man_ppp_mean": 1.00,
            "pick_and_roll_roll_man_possessions_mean": 9.0,
        },
    )

    adjusted = apply_scheme_profile(features, {"pick_and_roll_roll_man_ppp_mean": 0.10}, artifacts)

    assert round(adjusted.loc[0, "pick_and_roll_roll_man_ppp_mean"], 6) == 1.44


def test_apply_scheme_profile_penalties_are_not_masked_for_strong_lineups():
    features = pd.DataFrame(
        [
            {
                "pick_and_roll_roll_man_ppp_mean": 0.80,
                "pick_and_roll_roll_man_possessions_mean": 9.0,
            }
        ]
    )
    artifacts = TrainingArtifacts(
        model=DummyModel(),
        target_column="defensive_rating_target",
        feature_columns=list(features.columns),
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_means={
            "pick_and_roll_roll_man_ppp_mean": 1.00,
            "pick_and_roll_roll_man_possessions_mean": 9.0,
        },
    )

    adjusted = apply_scheme_profile(features, {"pick_and_roll_roll_man_ppp_mean": 0.10}, artifacts)

    assert round(adjusted.loc[0, "pick_and_roll_roll_man_ppp_mean"], 6) == 0.88


def test_apply_scheme_profile_amplifies_drop_for_anchor_big_lineups():
    features = pd.DataFrame(
        [
            {
                "pick_and_roll_roll_man_ppp_mean": 0.90,
                "pick_and_roll_roll_man_ppp_min": 0.60,
                "pick_and_roll_roll_man_possessions_mean": 12.0,
                "pick_and_roll_ball_handler_percentile_mean": 0.30,
                "pick_and_roll_ball_handler_percentile_min": 0.10,
                "pick_and_roll_ball_handler_ppp_mean": 1.00,
                "pick_and_roll_ball_handler_possessions_mean": 10.0,
                "spot_up_ppp_mean": 1.05,
                "spot_up_possessions_mean": 10.0,
            }
        ]
    )
    artifacts = TrainingArtifacts(
        model=DummyModel(),
        target_column="defensive_rating_target",
        feature_columns=list(features.columns),
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_means={
            "pick_and_roll_roll_man_ppp_mean": 1.00,
            "pick_and_roll_roll_man_ppp_min": 0.80,
            "pick_and_roll_roll_man_possessions_mean": 10.0,
            "pick_and_roll_ball_handler_percentile_mean": 0.50,
            "pick_and_roll_ball_handler_percentile_min": 0.25,
            "pick_and_roll_ball_handler_ppp_mean": 1.00,
            "pick_and_roll_ball_handler_possessions_mean": 10.0,
            "spot_up_ppp_mean": 1.00,
            "spot_up_possessions_mean": 10.0,
        },
    )

    drop_adjusted = apply_scheme_profile(
        features,
        {
            "pick_and_roll_roll_man_ppp_mean": -0.10,
            "pick_and_roll_ball_handler_ppp_mean": 0.10,
        },
        artifacts,
        scheme_name="Drop",
    )
    neutral_adjusted = apply_scheme_profile(
        features,
        {
            "pick_and_roll_roll_man_ppp_mean": -0.10,
            "pick_and_roll_ball_handler_ppp_mean": 0.10,
        },
        artifacts,
    )

    assert drop_adjusted.loc[0, "pick_and_roll_roll_man_ppp_mean"] < neutral_adjusted.loc[0, "pick_and_roll_roll_man_ppp_mean"]
    assert drop_adjusted.loc[0, "pick_and_roll_ball_handler_ppp_mean"] < neutral_adjusted.loc[0, "pick_and_roll_ball_handler_ppp_mean"]


def test_apply_scheme_profile_penalizes_switch_for_drop_style_lineups():
    features = pd.DataFrame(
        [
            {
                "isolation_ppp_mean": 0.90,
                "isolation_ppp_min": 0.70,
                "isolation_possessions_mean": 10.0,
                "pick_and_roll_ball_handler_ppp_mean": 1.00,
                "pick_and_roll_ball_handler_possessions_mean": 10.0,
                "pick_and_roll_ball_handler_percentile_mean": 0.35,
                "pick_and_roll_ball_handler_percentile_min": 0.10,
                "pick_and_roll_roll_man_ppp_mean": 0.90,
                "pick_and_roll_roll_man_ppp_min": 0.60,
                "pick_and_roll_roll_man_possessions_mean": 12.0,
            }
        ]
    )
    artifacts = TrainingArtifacts(
        model=DummyModel(),
        target_column="defensive_rating_target",
        feature_columns=list(features.columns),
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_means={
            "isolation_ppp_mean": 1.00,
            "isolation_ppp_min": 0.80,
            "isolation_possessions_mean": 10.0,
            "pick_and_roll_ball_handler_ppp_mean": 1.00,
            "pick_and_roll_ball_handler_possessions_mean": 10.0,
            "pick_and_roll_ball_handler_percentile_mean": 0.50,
            "pick_and_roll_ball_handler_percentile_min": 0.25,
            "pick_and_roll_roll_man_ppp_mean": 1.00,
            "pick_and_roll_roll_man_ppp_min": 0.80,
            "pick_and_roll_roll_man_possessions_mean": 10.0,
        },
    )

    switch_adjusted = apply_scheme_profile(
        features,
        {
            "isolation_ppp_mean": -0.10,
            "pick_and_roll_ball_handler_ppp_mean": -0.10,
            "pick_and_roll_roll_man_ppp_mean": 0.10,
        },
        artifacts,
        scheme_name="Switch",
    )
    neutral_adjusted = apply_scheme_profile(
        features,
        {
            "isolation_ppp_mean": -0.10,
            "pick_and_roll_ball_handler_ppp_mean": -0.10,
            "pick_and_roll_roll_man_ppp_mean": 0.10,
        },
        artifacts,
    )

    assert switch_adjusted.loc[0, "isolation_ppp_mean"] > neutral_adjusted.loc[0, "isolation_ppp_mean"]
    assert switch_adjusted.loc[0, "pick_and_roll_ball_handler_ppp_mean"] > neutral_adjusted.loc[0, "pick_and_roll_ball_handler_ppp_mean"]
    assert switch_adjusted.loc[0, "pick_and_roll_roll_man_ppp_mean"] > neutral_adjusted.loc[0, "pick_and_roll_roll_man_ppp_mean"]


def test_recommend_scheme_centers_global_scheme_bias():
    lineup_row = pd.Series(
        {
            "lineup_size": 5,
            "guard_count": 2,
            "isolation_ppp_mean": 1.00,
            "isolation_possessions_mean": 10.0,
            "pick_and_roll_ball_handler_ppp_mean": 1.00,
            "pick_and_roll_ball_handler_possessions_mean": 10.0,
            "pick_and_roll_roll_man_ppp_mean": 1.00,
            "pick_and_roll_roll_man_possessions_mean": 10.0,
            "spot_up_ppp_mean": 1.00,
            "spot_up_possessions_mean": 10.0,
            "spot_up_percentile_mean": 50.0,
            "defensive_rating_target": 110.0,
            "opponent_ppp_target": 1.10,
        }
    )
    artifacts = TrainingArtifacts(
        model=DummyModel(),
        target_column="defensive_rating_target",
        feature_columns=[
            "lineup_size",
            "guard_count",
            "isolation_ppp_mean",
            "isolation_possessions_mean",
            "pick_and_roll_ball_handler_ppp_mean",
            "pick_and_roll_ball_handler_possessions_mean",
            "pick_and_roll_roll_man_ppp_mean",
            "pick_and_roll_roll_man_possessions_mean",
            "spot_up_ppp_mean",
            "spot_up_possessions_mean",
            "spot_up_percentile_mean",
        ],
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_means={
            "lineup_size": 5.0,
            "guard_count": 2.0,
            "isolation_ppp_mean": 1.00,
            "isolation_possessions_mean": 10.0,
            "pick_and_roll_ball_handler_ppp_mean": 1.00,
            "pick_and_roll_ball_handler_possessions_mean": 10.0,
            "pick_and_roll_roll_man_ppp_mean": 1.00,
            "pick_and_roll_roll_man_possessions_mean": 10.0,
            "spot_up_ppp_mean": 1.00,
            "spot_up_possessions_mean": 10.0,
            "spot_up_percentile_mean": 50.0,
        },
    )

    recommendation = recommend_scheme(lineup_row, artifacts)

    assert set(recommendation.ranked_schemes["scheme"]) == {"Switch", "Drop", "Zone"}
    assert recommendation.ranked_schemes["predicted_value"].round(6).nunique() == 1


def test_recommend_scheme_returns_lowest_scoring_profile():
    lineup_row = pd.Series(
        {
            "minutes_played": 20.0,
            "lineup_size": 5,
            "guard_count": 2,
            "isolation_ppp_mean": 1.02,
            "isolation_possessions_mean": 6.0,
            "pick_and_roll_ball_handler_ppp_mean": 1.01,
            "pick_and_roll_ball_handler_possessions_mean": 7.0,
            "pick_and_roll_roll_man_ppp_mean": 1.18,
            "pick_and_roll_roll_man_possessions_mean": 20.0,
            "spot_up_ppp_mean": 1.05,
            "spot_up_possessions_mean": 11.0,
            "spot_up_percentile_mean": 45.0,
            "defensive_rating_target": 110.0,
            "opponent_ppp_target": 1.10,
        }
    )
    artifacts = TrainingArtifacts(
        model=DummyModel(),
        target_column="defensive_rating_target",
        feature_columns=[
            "minutes_played",
            "lineup_size",
            "guard_count",
            "isolation_ppp_mean",
            "isolation_possessions_mean",
            "pick_and_roll_ball_handler_ppp_mean",
            "pick_and_roll_ball_handler_possessions_mean",
            "pick_and_roll_roll_man_ppp_mean",
            "pick_and_roll_roll_man_possessions_mean",
            "spot_up_ppp_mean",
            "spot_up_possessions_mean",
            "spot_up_percentile_mean",
        ],
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_means={
            "lineup_size": 5.0,
            "guard_count": 2.0,
            "isolation_ppp_mean": 1.0,
            "isolation_possessions_mean": 10.0,
            "pick_and_roll_ball_handler_ppp_mean": 1.0,
            "pick_and_roll_ball_handler_possessions_mean": 10.0,
            "pick_and_roll_roll_man_ppp_mean": 1.0,
            "pick_and_roll_roll_man_possessions_mean": 10.0,
            "spot_up_ppp_mean": 1.0,
            "spot_up_possessions_mean": 10.0,
            "spot_up_percentile_mean": 50.0,
        },
    )

    recommendation = recommend_scheme(lineup_row, artifacts)

    assert recommendation.recommended_scheme == "Drop"
    assert not recommendation.explanation.empty
