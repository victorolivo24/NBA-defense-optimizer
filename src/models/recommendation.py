"""Recommendation utilities for scoring candidate defensive schemes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .training import TrainingArtifacts, prepare_training_matrices

DEFAULT_SCHEME_PROFILES: dict[str, dict[str, float]] = {
    "Drop": {
        "pick_and_roll_roll_man_ppp_mean": -0.08,
        "pick_and_roll_ball_handler_ppp_mean": 0.03,
        "spot_up_ppp_mean": 0.02,
    },
    "Switch": {
        "isolation_ppp_mean": -0.05,
        "pick_and_roll_ball_handler_ppp_mean": -0.04,
        "pick_and_roll_roll_man_ppp_mean": 0.05,
    },
    "Zone": {
        "isolation_ppp_mean": -0.03,
        "spot_up_ppp_mean": 0.06,
        "spot_up_percentile_mean": -4.0,
    },
}


@dataclass
class SchemeRecommendation:
    """Recommendation output for a single lineup row."""

    recommended_scheme: str
    predicted_value: float
    ranked_schemes: pd.DataFrame
    explanation: pd.DataFrame


def recommend_scheme(
    lineup_row: pd.DataFrame | pd.Series,
    artifacts: TrainingArtifacts,
    scheme_profiles: dict[str, dict[str, float]] | None = None,
) -> SchemeRecommendation:
    """Score candidate defensive schemes for a lineup and return the best option."""
    scheme_profiles = scheme_profiles or DEFAULT_SCHEME_PROFILES
    lineup_frame = _coerce_lineup_frame(lineup_row)
    base_features, _ = prepare_training_matrices(
        _ensure_targets_exist(lineup_frame),
        target_column=artifacts.target_column,
    )
    base_features = base_features[artifacts.feature_columns]

    scored_rows = []
    explanation_rows = []

    for scheme_name, adjustments in scheme_profiles.items():
        scenario_features = apply_scheme_profile(base_features, adjustments)
        prediction = float(artifacts.model.predict(scenario_features)[0])
        scored_rows.append(
            {
                "scheme": scheme_name,
                "predicted_value": prediction,
            }
        )

        for feature, delta in adjustments.items():
            if feature in scenario_features.columns and delta != 0:
                explanation_rows.append(
                    {
                        "scheme": scheme_name,
                        "feature": feature,
                        "adjustment": delta,
                    }
                )

    ranked = pd.DataFrame(scored_rows).sort_values("predicted_value", ascending=True).reset_index(drop=True)
    explanation = pd.DataFrame(explanation_rows)

    if hasattr(artifacts.model, "explain"):
        shap_frame = artifacts.model.explain(base_features)
        shap_long = (
            shap_frame.iloc[[0]]
            .T.reset_index()
            .rename(columns={"index": "feature", 0: "baseline_shap_value"})
            .sort_values("baseline_shap_value", key=lambda col: col.abs(), ascending=False)
        )
        explanation = explanation.merge(shap_long, on="feature", how="left")
        explanation = explanation.sort_values(
            ["scheme", "baseline_shap_value"],
            key=lambda col: col.abs() if col.name == "baseline_shap_value" else col,
            ascending=[True, False],
        )

    best = ranked.iloc[0]
    return SchemeRecommendation(
        recommended_scheme=str(best["scheme"]),
        predicted_value=float(best["predicted_value"]),
        ranked_schemes=ranked,
        explanation=explanation,
    )


def apply_scheme_profile(features: pd.DataFrame, adjustments: dict[str, float]) -> pd.DataFrame:
    """Apply additive scheme-specific adjustments to a lineup feature row."""
    adjusted = features.copy()
    for feature_name, delta in adjustments.items():
        if feature_name in adjusted.columns:
            adjusted.loc[:, feature_name] = adjusted[feature_name] + delta
    return adjusted


def _coerce_lineup_frame(lineup_row: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(lineup_row, pd.Series):
        return lineup_row.to_frame().T
    return lineup_row.copy()


def _ensure_targets_exist(lineup_frame: pd.DataFrame) -> pd.DataFrame:
    frame = lineup_frame.copy()
    if "defensive_rating_target" not in frame.columns:
        frame["defensive_rating_target"] = 0.0
    if "opponent_ppp_target" not in frame.columns:
        frame["opponent_ppp_target"] = 0.0
    return frame
