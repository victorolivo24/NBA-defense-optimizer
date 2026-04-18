"""Simulation and recommendation utilities for candidate defensive schemes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .scheme_profiles import DEFAULT_SCHEME_PROFILES
from .training import TrainingArtifacts, prepare_training_matrices

LOWER_IS_BETTER_TOKENS = ("ppp", "rating", "turnover")
HIGHER_IS_BETTER_TOKENS = ("percentile", "count", "size", "height", "weight")
MATCHUP_SCALE_MIN = 0.5
MATCHUP_SCALE_MAX = 1.5
VOLUME_SCALE_MIN = 0.25
VOLUME_SCALE_MAX = 2.5
SCHEME_FIT_SCALE_MIN = 0.35
SCHEME_FIT_SCALE_MAX = 2.0


@dataclass
class SchemeRecommendation:
    """Simulation and recommendation output for a single lineup row."""

    recommended_scheme: str
    predicted_value: float
    ranked_schemes: pd.DataFrame
    explanation: pd.DataFrame


def recommend_scheme(
    lineup_row: pd.DataFrame | pd.Series,
    artifacts: TrainingArtifacts,
    scheme_profiles: dict[str, dict[str, float]] | None = None,
) -> SchemeRecommendation:
    """Simulate candidate defensive schemes for a lineup and return the best option."""
    scheme_profiles = scheme_profiles or DEFAULT_SCHEME_PROFILES
    lineup_frame = _coerce_lineup_frame(lineup_row)
    base_features, _, _ = prepare_training_matrices(
        _ensure_targets_exist(lineup_frame),
        target_column=artifacts.target_column,
    )
    base_features = base_features.reindex(columns=artifacts.feature_columns, fill_value=0.0)
    reference_features = _build_reference_feature_frame(base_features, artifacts)
    reference_prediction = float(artifacts.model.predict(reference_features)[0])

    scored_rows = []
    explanation_rows = []

    for scheme_name, adjustments in scheme_profiles.items():
        effective_adjustments = _compute_effective_adjustments(
            base_features,
            adjustments,
            artifacts,
            scheme_name=scheme_name,
        )
        scenario_features = _apply_adjustments(base_features, effective_adjustments)

        reference_adjustments = _compute_effective_adjustments(
            reference_features,
            adjustments,
            artifacts,
            scheme_name=scheme_name,
        )
        reference_scenario = _apply_adjustments(reference_features, reference_adjustments)
        scheme_bias = float(artifacts.model.predict(reference_scenario)[0]) - reference_prediction

        raw_prediction = float(artifacts.model.predict(scenario_features)[0])
        debiased_prediction = raw_prediction - scheme_bias
        scored_rows.append(
            {
                "scheme": scheme_name,
                "predicted_value": debiased_prediction,
                "raw_predicted_value": raw_prediction,
                "scheme_bias_correction": scheme_bias,
            }
        )

        for feature, scaled_delta in effective_adjustments.items():
            if feature in scenario_features.columns and float(scaled_delta.iloc[0]) != 0.0:
                explanation_rows.append(
                    {
                        "scheme": scheme_name,
                        "feature": feature,
                        "adjustment": float(scaled_delta.iloc[0]),
                        "base_adjustment": adjustments[feature],
                    }
                )

    ranked = pd.DataFrame(scored_rows).sort_values("predicted_value", ascending=True).reset_index(drop=True)
    explanation = pd.DataFrame(explanation_rows)

    if hasattr(artifacts.model, "explain"):
        shap_frame = artifacts.model.explain(base_features)
        shap_long = shap_frame.iloc[[0]].T.reset_index()
        if len(shap_long.columns) >= 2:
            shap_long.columns = ["feature", "baseline_shap_value"]
            shap_long = shap_long.sort_values(
                "baseline_shap_value",
                key=lambda col: col.abs(),
                ascending=False,
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


def apply_scheme_profile(
    features: pd.DataFrame,
    adjustments: dict[str, float],
    artifacts: TrainingArtifacts | None = None,
    scheme_name: str | None = None,
) -> pd.DataFrame:
    """
    Apply additive simulator adjustments to lineup features.

    The simulation is lineup-aware in three ways:
    1. PPP and percentile adjustments are weighted by the lineup's actual play-type volume.
    2. Benefits and penalties scale with the lineup's baseline matchup quality on that feature.
    3. Penalties are never hidden by a generic talent mask.
    """
    effective_adjustments = _compute_effective_adjustments(
        features,
        adjustments,
        artifacts,
        scheme_name=scheme_name,
    )
    return _apply_adjustments(features, effective_adjustments)


def _compute_effective_adjustments(
    features: pd.DataFrame,
    adjustments: dict[str, float],
    artifacts: TrainingArtifacts | None,
    scheme_name: str | None = None,
) -> dict[str, pd.Series]:
    effective_adjustments: dict[str, pd.Series] = {}

    for feature_name, delta in adjustments.items():
        if feature_name not in features.columns:
            continue

        base_delta = pd.Series(delta, index=features.index, dtype=float)
        volume_scale = _volume_scale_for_feature(features, feature_name, artifacts)
        matchup_scale = _matchup_scale_for_feature(features, feature_name, delta, artifacts)
        scheme_fit_scale = _scheme_fit_scale(
            features,
            scheme_name=scheme_name,
            feature_name=feature_name,
            delta=delta,
            artifacts=artifacts,
        )
        effective_adjustments[feature_name] = base_delta * volume_scale * matchup_scale * scheme_fit_scale

    return effective_adjustments


def _scheme_fit_scale(
    features: pd.DataFrame,
    *,
    scheme_name: str | None,
    feature_name: str,
    delta: float,
    artifacts: TrainingArtifacts | None,
) -> pd.Series:
    if scheme_name == "Drop":
        fit_scale = _drop_fit_scale(features, artifacts)
    elif scheme_name == "Switch":
        fit_scale = _switch_fit_scale(features, artifacts)
    else:
        return pd.Series(1.0, index=features.index, dtype=float)

    fit_scale = fit_scale.pow(2).clip(lower=SCHEME_FIT_SCALE_MIN, upper=SCHEME_FIT_SCALE_MAX)

    if _is_beneficial_adjustment(feature_name, delta):
        return fit_scale

    return (1.0 / fit_scale.clip(lower=1e-6)).clip(
        lower=SCHEME_FIT_SCALE_MIN,
        upper=SCHEME_FIT_SCALE_MAX,
    )


def _drop_fit_scale(features: pd.DataFrame, artifacts: TrainingArtifacts | None) -> pd.Series:
    # Drop schemes are most natural for lineups with a strong interior anchor and weaker
    # point-of-attack containment. That lets the simulator prefer drop for Brook/Giannis-like units.
    roll_man_anchor = _feature_advantage(
        features,
        feature_name="pick_and_roll_roll_man_ppp_min",
        artifacts=artifacts,
    )
    roll_man_mean = _feature_advantage(
        features,
        feature_name="pick_and_roll_roll_man_ppp_mean",
        artifacts=artifacts,
    )
    point_of_attack = _feature_advantage(
        features,
        feature_name="pick_and_roll_ball_handler_percentile_mean",
        artifacts=artifacts,
    )
    point_of_attack_floor = _feature_advantage(
        features,
        feature_name="pick_and_roll_ball_handler_percentile_min",
        artifacts=artifacts,
    )

    return (
        (roll_man_anchor * 2.0)
        + roll_man_mean
        + (1.0 / point_of_attack.clip(lower=1e-6))
        + (1.0 / point_of_attack_floor.clip(lower=1e-6))
    ) / 5.0


def _switch_fit_scale(features: pd.DataFrame, artifacts: TrainingArtifacts | None) -> pd.Series:
    # Switch schemes should get extra credit for lineups with strong POA/isolation defenders,
    # and less credit for lineups that look more like classic drop units.
    point_of_attack = _feature_advantage(
        features,
        feature_name="pick_and_roll_ball_handler_percentile_mean",
        artifacts=artifacts,
    )
    point_of_attack_floor = _feature_advantage(
        features,
        feature_name="pick_and_roll_ball_handler_percentile_min",
        artifacts=artifacts,
    )
    isolation_stop = _feature_advantage(
        features,
        feature_name="isolation_ppp_min",
        artifacts=artifacts,
    )
    roll_man_anchor = _feature_advantage(
        features,
        feature_name="pick_and_roll_roll_man_ppp_min",
        artifacts=artifacts,
    )

    return (
        point_of_attack
        + point_of_attack_floor
        + isolation_stop
        + (1.0 / roll_man_anchor.clip(lower=1e-6))
    ) / 4.0


def _feature_advantage(
    features: pd.DataFrame,
    *,
    feature_name: str,
    artifacts: TrainingArtifacts | None,
) -> pd.Series:
    if feature_name not in features.columns:
        return pd.Series(1.0, index=features.index, dtype=float)

    current_values = pd.to_numeric(features[feature_name], errors="coerce").fillna(0.0)
    reference_value = _reference_feature_value(feature_name, artifacts)
    if reference_value is None or reference_value == 0:
        return pd.Series(1.0, index=features.index, dtype=float)

    if _is_higher_better_feature(feature_name):
        ratio = current_values / reference_value
    else:
        ratio = reference_value / current_values.clip(lower=1e-6)

    return ratio.clip(lower=MATCHUP_SCALE_MIN, upper=MATCHUP_SCALE_MAX)


def _is_beneficial_adjustment(feature_name: str, delta: float) -> bool:
    if _is_higher_better_feature(feature_name):
        return delta > 0
    return delta < 0


def _apply_adjustments(features: pd.DataFrame, adjustments: dict[str, pd.Series]) -> pd.DataFrame:
    adjusted = features.copy()
    for feature_name, scaled_delta in adjustments.items():
        if feature_name not in adjusted.columns:
            continue
        adjusted[feature_name] = adjusted[feature_name].astype(float) + scaled_delta.astype(float)
    return adjusted


def _volume_scale_for_feature(
    features: pd.DataFrame,
    feature_name: str,
    artifacts: TrainingArtifacts | None,
) -> pd.Series:
    paired_volume_feature = _paired_possessions_feature_name(feature_name)
    if paired_volume_feature is None or paired_volume_feature not in features.columns:
        return pd.Series(1.0, index=features.index, dtype=float)

    current_volume = pd.to_numeric(features[paired_volume_feature], errors="coerce").fillna(0.0)
    reference_volume = _reference_feature_value(paired_volume_feature, artifacts)
    if reference_volume is None or reference_volume <= 0:
        return pd.Series(1.0, index=features.index, dtype=float)

    return (current_volume / reference_volume).clip(lower=VOLUME_SCALE_MIN, upper=VOLUME_SCALE_MAX)


def _matchup_scale_for_feature(
    features: pd.DataFrame,
    feature_name: str,
    delta: float,
    artifacts: TrainingArtifacts | None,
) -> pd.Series:
    current_values = pd.to_numeric(features[feature_name], errors="coerce").fillna(0.0)
    reference_value = _reference_feature_value(feature_name, artifacts)
    if reference_value is None or reference_value == 0:
        return pd.Series(1.0, index=features.index, dtype=float)

    higher_is_better = _is_higher_better_feature(feature_name)
    is_beneficial = (delta > 0) if higher_is_better else (delta < 0)

    if higher_is_better:
        ratio = current_values / reference_value
    else:
        ratio = reference_value / current_values.clip(lower=1e-6)

    if not is_beneficial:
        ratio = 1.0 / ratio.clip(lower=1e-6)

    return ratio.clip(lower=MATCHUP_SCALE_MIN, upper=MATCHUP_SCALE_MAX)


def _paired_possessions_feature_name(feature_name: str) -> str | None:
    if feature_name.endswith("_ppp_mean"):
        return feature_name.replace("_ppp_mean", "_possessions_mean")
    if feature_name.endswith("_percentile_mean"):
        return feature_name.replace("_percentile_mean", "_possessions_mean")
    return None


def _reference_feature_value(feature_name: str, artifacts: TrainingArtifacts | None) -> float | None:
    if artifacts is None:
        return None

    if artifacts.feature_means and feature_name in artifacts.feature_means:
        return float(artifacts.feature_means[feature_name])

    if artifacts.feature_mins and artifacts.feature_maxs:
        feature_min = artifacts.feature_mins.get(feature_name)
        feature_max = artifacts.feature_maxs.get(feature_name)
        if feature_min is not None and feature_max is not None:
            return float((feature_min + feature_max) / 2.0)

    return None


def _build_reference_feature_frame(
    base_features: pd.DataFrame,
    artifacts: TrainingArtifacts,
) -> pd.DataFrame:
    reference_row: dict[str, float] = {}
    for column in base_features.columns:
        reference_value = _reference_feature_value(column, artifacts)
        if reference_value is None:
            reference_value = float(pd.to_numeric(base_features[column], errors="coerce").fillna(0.0).iloc[0])
        reference_row[column] = reference_value
    return pd.DataFrame([reference_row], columns=base_features.columns)


def _is_higher_better_feature(feature_name: str) -> bool:
    lowered = feature_name.lower()
    if any(token in lowered for token in HIGHER_IS_BETTER_TOKENS):
        return True
    if any(token in lowered for token in LOWER_IS_BETTER_TOKENS):
        return False
    return False


def _coerce_lineup_frame(lineup_row: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(lineup_row, pd.Series):
        return lineup_row.to_frame().T
    return lineup_row.copy()


def _ensure_targets_exist(lineup_frame: pd.DataFrame) -> pd.DataFrame:
    frame = lineup_frame.copy()
    if "defensive_rating_target" not in frame.columns:
        frame["defensive_rating_target"] = 0.0
    else:
        frame["defensive_rating_target"] = pd.to_numeric(
            frame["defensive_rating_target"],
            errors="coerce",
        ).fillna(0.0)
    if "opponent_ppp_target" not in frame.columns:
        frame["opponent_ppp_target"] = 0.0
    else:
        frame["opponent_ppp_target"] = pd.to_numeric(
            frame["opponent_ppp_target"],
            errors="coerce",
        ).fillna(0.0)
    return frame
