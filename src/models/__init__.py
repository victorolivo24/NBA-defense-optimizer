"""Model package exports."""

from .base import BaseSchemeModel
from .recommendation import (
    SchemeRecommendation,
    apply_scheme_profile,
    recommend_scheme,
)
from .scheme_recommender import XGBoostSchemeRecommender
from .scheme_profiles import DEFAULT_SCHEME_PROFILES
from .training import (
    DEFAULT_TARGET_COLUMN,
    TARGET_OPTIONS,
    TrainingArtifacts,
    build_model_dataset,
    export_feature_importance,
    prepare_training_matrices,
    train_baseline_regressor,
)

__all__ = [
    "BaseSchemeModel",
    "XGBoostSchemeRecommender",
    "DEFAULT_SCHEME_PROFILES",
    "DEFAULT_TARGET_COLUMN",
    "SchemeRecommendation",
    "TARGET_OPTIONS",
    "TrainingArtifacts",
    "apply_scheme_profile",
    "build_model_dataset",
    "export_feature_importance",
    "prepare_training_matrices",
    "recommend_scheme",
    "train_baseline_regressor",
]
