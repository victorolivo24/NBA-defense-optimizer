"""Model package exports."""

from .base import BaseSchemeModel
from .scheme_recommender import XGBoostSchemeRecommender
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
    "DEFAULT_TARGET_COLUMN",
    "TARGET_OPTIONS",
    "TrainingArtifacts",
    "build_model_dataset",
    "export_feature_importance",
    "prepare_training_matrices",
    "train_baseline_regressor",
]
