import pandas as pd

from src.models.recommendation import (
    apply_scheme_profile,
    calculate_scheme_fit_adjustment,
    recommend_scheme,
)
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


class ConstantModel:
    def predict(self, features: pd.DataFrame):
        return pd.Series([100.0] * len(features), index=features.index).to_numpy()

    def explain(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.copy()


def test_apply_scheme_profile_adjusts_only_matching_columns():
    features = pd.DataFrame([{"isolation_ppp_mean": 1.0, "spot_up_ppp_mean": 1.1}])

    adjusted = apply_scheme_profile(features, {"isolation_ppp_mean": -0.1, "missing": 1.0})

    assert adjusted.loc[0, "isolation_ppp_mean"] == 0.9
    assert adjusted.loc[0, "spot_up_ppp_mean"] == 1.1


def test_recommend_scheme_returns_lowest_scoring_profile():
    lineup_row = pd.Series(
        {
            "minutes_played": 20.0,
            "lineup_size": 5,
            "guard_count": 2,
            "isolation_ppp_mean": 1.00,
            "pick_and_roll_ball_handler_ppp_mean": 1.05,
            "pick_and_roll_roll_man_ppp_mean": 1.10,
            "spot_up_ppp_mean": 1.08,
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
            "pick_and_roll_ball_handler_ppp_mean",
            "pick_and_roll_roll_man_ppp_mean",
            "spot_up_ppp_mean",
        ],
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
    )

    recommendation = recommend_scheme(lineup_row, artifacts)

    assert recommendation.recommended_scheme == "Switch"
    assert list(recommendation.ranked_schemes["scheme"]) == ["Switch", "Drop", "Zone"]
    assert not recommendation.explanation.empty


def test_calculate_scheme_fit_adjustment_rewards_good_scheme_fit():
    base_features = pd.DataFrame(
        [
            {
                "pick_and_roll_roll_man_ppp_mean": 0.92,
                "pick_and_roll_ball_handler_ppp_mean": 0.98,
                "spot_up_ppp_mean": 1.02,
            }
        ]
    )
    artifacts = TrainingArtifacts(
        model=ConstantModel(),
        target_column="defensive_rating_target",
        feature_columns=list(base_features.columns),
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_mins={
            "pick_and_roll_roll_man_ppp_mean": 0.8,
            "pick_and_roll_ball_handler_ppp_mean": 0.9,
            "spot_up_ppp_mean": 0.95,
        },
        feature_maxs={
            "pick_and_roll_roll_man_ppp_mean": 1.2,
            "pick_and_roll_ball_handler_ppp_mean": 1.2,
            "spot_up_ppp_mean": 1.15,
        },
    )

    drop_adjustment = calculate_scheme_fit_adjustment(
        base_features,
        {
            "pick_and_roll_roll_man_ppp_mean": -0.258,
            "pick_and_roll_ball_handler_ppp_mean": 0.069,
            "spot_up_ppp_mean": 0.122,
        },
        artifacts,
    )
    zone_adjustment = calculate_scheme_fit_adjustment(
        base_features,
        {
            "isolation_ppp_mean": -0.009,
            "spot_up_ppp_mean": 0.001,
            "spot_up_percentile_mean": -0.001,
        },
        artifacts,
    )

    assert drop_adjustment > zone_adjustment
    assert drop_adjustment > 1.0


def test_recommend_scheme_uses_scheme_fit_to_create_clearer_separation():
    lineup_row = pd.Series(
        {
            "minutes_played": 20.0,
            "lineup_size": 5,
            "guard_count": 2,
            "isolation_ppp_mean": 1.00,
            "pick_and_roll_ball_handler_ppp_mean": 0.97,
            "pick_and_roll_roll_man_ppp_mean": 0.89,
            "spot_up_ppp_mean": 1.04,
            "defensive_rating_target": 110.0,
            "opponent_ppp_target": 1.10,
        }
    )
    artifacts = TrainingArtifacts(
        model=ConstantModel(),
        target_column="defensive_rating_target",
        feature_columns=[
            "minutes_played",
            "lineup_size",
            "guard_count",
            "isolation_ppp_mean",
            "pick_and_roll_ball_handler_ppp_mean",
            "pick_and_roll_roll_man_ppp_mean",
            "spot_up_ppp_mean",
        ],
        metrics={"mae": 0.0, "rmse": 0.0},
        train_rows=1,
        test_rows=1,
        feature_mins={
            "isolation_ppp_mean": 0.85,
            "pick_and_roll_ball_handler_ppp_mean": 0.9,
            "pick_and_roll_roll_man_ppp_mean": 0.8,
            "spot_up_ppp_mean": 0.95,
        },
        feature_maxs={
            "isolation_ppp_mean": 1.15,
            "pick_and_roll_ball_handler_ppp_mean": 1.15,
            "pick_and_roll_roll_man_ppp_mean": 1.2,
            "spot_up_ppp_mean": 1.15,
        },
    )
    scheme_profiles = {
        "Drop": {
            "pick_and_roll_roll_man_ppp_mean": -0.258,
            "pick_and_roll_ball_handler_ppp_mean": 0.069,
            "spot_up_ppp_mean": 0.122,
        },
        "Zone": {
            "isolation_ppp_mean": -0.009,
            "spot_up_ppp_mean": 0.001,
        },
    }

    recommendation = recommend_scheme(lineup_row, artifacts, scheme_profiles=scheme_profiles)

    assert recommendation.recommended_scheme == "Drop"
    ranked = recommendation.ranked_schemes.set_index("scheme")
    assert ranked.loc["Drop", "scheme_fit_adjustment"] > ranked.loc["Zone", "scheme_fit_adjustment"]
    assert ranked.loc["Zone", "predicted_value"] - ranked.loc["Drop", "predicted_value"] > 1.0
