"""Simulation and recommendation utilities for candidate defensive schemes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .scheme_profiles import DEFAULT_SCHEME_PROFILES
from .training import TrainingArtifacts, prepare_training_matrices


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

    scored_rows = []
    explanation_rows = []

    for scheme_name, adjustments in scheme_profiles.items():
        scenario_features = apply_scheme_profile(base_features, adjustments, artifacts)
        prediction = float(artifacts.model.predict(scenario_features)[0])
        scored_rows.append(
            {
                "scheme": scheme_name,
                "predicted_value": prediction,
            }
        )

        for feature, delta in adjustments.items():
            if feature in scenario_features.columns and delta != 0:
                # We record the original intended delta for the explanation table
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
    artifacts: TrainingArtifacts | None = None
) -> pd.DataFrame:
    """
    Apply additive simulator adjustments to a lineup feature row, scaled by baseline talent.
    
    Why this approach?
    Applying flat scheme adjustments (e.g. -0.317 PPP) artificially inflates bad lineups, making them 
    look like elite defenses just by changing the scheme.
    
    Talent-Scaling Math:
    1. We use the 5th and 95th percentiles from the training dataset (stored in artifacts as min/max)
       to compute a `talent_score` between 0.0 (worst) and 1.0 (elite). Clipping at these percentiles 
       prevents extreme outliers from stretching the scale and making average defenders look elite.
    2. If the scheme delta is beneficial (improves the defense), the lineup receives a percentage 
       of that benefit equal to their talent_score (floored at 10% so bad teams get *some* effect).
    3. If the scheme delta is detrimental (exposes a weakness), the lineup receives the penalty 
       scaled by (1.0 - talent_score). Elite defenders mask the scheme's weaknesses, while bad 
       defenders are fully exposed by it.
    """
    adjusted = features.copy()
    for feature_name, delta in adjustments.items():
        if feature_name not in adjusted.columns:
            continue
            
        if artifacts and hasattr(artifacts, 'feature_mins') and hasattr(artifacts, 'feature_maxs') and artifacts.feature_mins and artifacts.feature_maxs:
            f_min = artifacts.feature_mins.get(feature_name)
            f_max = artifacts.feature_maxs.get(feature_name)
            
            if f_min is not None and f_max is not None and f_max > f_min:
                current_vals = adjusted[feature_name].clip(lower=f_min, upper=f_max)
                
                # Determine if higher values are better for this specific feature
                # Typically, percentile, count, and size metrics are "higher is better"
                # PPP metrics are "lower is better"
                is_higher_better = any(token in feature_name for token in ["percentile", "count", "size"])
                
                if is_higher_better:
                    talent_score = (current_vals - f_min) / (f_max - f_min)
                else:
                    # For PPP allowed, lower is better (so if current == f_min, score = 1.0)
                    talent_score = (f_max - current_vals) / (f_max - f_min)
                    
                # delta > 0 is beneficial if higher is better, else delta < 0 is beneficial
                is_beneficial = (delta > 0) if is_higher_better else (delta < 0)
                
                if is_beneficial:
                    # Elite gets full benefit, bad gets minimum 10% benefit
                    effective_scaler = talent_score.clip(lower=0.1)
                else:
                    # Elite masks the penalty (gets 10%), bad gets full penalty (100%)
                    effective_scaler = (1.0 - talent_score).clip(lower=0.1)
                    
                scaled_delta = delta * effective_scaler
                adjusted[feature_name] = adjusted[feature_name] + scaled_delta
            else:
                adjusted[feature_name] = adjusted[feature_name] + delta
        else:
            adjusted[feature_name] = adjusted[feature_name] + delta
            
    return adjusted


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
