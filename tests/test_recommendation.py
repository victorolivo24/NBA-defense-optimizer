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
