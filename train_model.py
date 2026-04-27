"""CLI entry point for baseline model training and diagnostics generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.features import DEFAULT_MIN_LINEUP_MINUTES, build_training_dataset
from src.models import (
    DEFAULT_TARGET_COLUMN,
    TARGET_OPTIONS,
    export_feature_importance,
    train_baseline_regressor,
)
from src.models.training import prepare_training_matrices

PLOTS_DIR = Path("data/processed/plots")


def plot_actual_vs_predicted(
    y_train_true: pd.Series,
    y_train_pred: pd.Series,
    y_test_true: pd.Series,
    y_test_pred: pd.Series,
    metrics: dict[str, float] | None = None,
    output_path: Path = PLOTS_DIR / "actual_vs_predicted.png",
) -> None:
    """Plot actual vs predicted values for both train and test sets."""
    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=y_train_true, y=y_train_pred, alpha=0.35, label="Training", color="#457b9d")
    sns.scatterplot(x=y_test_true, y=y_test_pred, alpha=0.8, label="Test", color="#e76f51")

    min_val = min(y_train_true.min(), y_test_true.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train_true.max(), y_test_true.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Perfect Prediction", linewidth=1.8)

    if metrics:
        stats_text = (
            "--- Train ---\n"
            f"MAE:  {metrics['train_mae']:.4f}\n"
            f"RMSE: {metrics['train_rmse']:.4f}\n"
            f"R²:   {metrics['train_r2']:.4f}\n"
            "--- Test ---\n"
            f"MAE:  {metrics['test_mae']:.4f}\n"
            f"RMSE: {metrics['test_rmse']:.4f}\n"
            f"R²:   {metrics['test_r2']:.4f}"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        plt.gca().text(
            0.03,
            0.97,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
            family="monospace",
        )

    plt.xlabel("Actual Defensive Rating")
    plt.ylabel("Predicted Defensive Rating")
    plt.title("Actual vs Predicted Defensive Rating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_feature_importance_bar(
    importance_df: pd.DataFrame,
    output_path: Path = PLOTS_DIR / "feature_importance_bar.png",
    top_n: int = 15,
) -> None:
    """Plot the top feature importances as a horizontal bar chart."""
    top_features = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x="importance", y="feature", hue="feature", palette="crest", legend=False)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_residuals_vs_predicted(
    y_test_true: pd.Series,
    y_test_pred: pd.Series,
    output_path: Path = PLOTS_DIR / "residuals_vs_predicted.png",
) -> None:
    """Plot residuals against predicted values for the test set."""
    residuals = y_test_true - y_test_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_pred, y=residuals, color="#264653", alpha=0.75)
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Defensive Rating")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_residual_distribution(
    y_test_true: pd.Series,
    y_test_pred: pd.Series,
    output_path: Path = PLOTS_DIR / "residual_distribution.png",
) -> None:
    """Plot the distribution of test residuals."""
    residuals = y_test_true - y_test_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=25, color="#2a9d8f")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_prediction_error_boxplot(
    y_test_true: pd.Series,
    y_test_pred: pd.Series,
    output_path: Path = PLOTS_DIR / "prediction_error_boxplot.png",
) -> None:
    """Plot absolute and signed prediction errors for the test set."""
    residuals = y_test_true - y_test_pred
    plot_df = pd.DataFrame(
        {
            "metric": ["Residual"] * len(residuals) + ["Absolute Error"] * len(residuals),
            "value": list(residuals) + list(residuals.abs()),
        }
    )
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_df, x="metric", y="value", hue="metric", palette="Set2", legend=False)
    plt.title("Prediction Error Summary")
    plt.xlabel("")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_target_distribution(
    y_train_true: pd.Series,
    y_test_true: pd.Series,
    output_path: Path = PLOTS_DIR / "target_distribution.png",
) -> None:
    """Plot the train/test target distributions."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(y_train_true, fill=True, alpha=0.35, label="Training", color="#457b9d")
    sns.kdeplot(y_test_true, fill=True, alpha=0.35, label="Test", color="#e76f51")
    plt.xlabel("Defensive Rating")
    plt.ylabel("Density")
    plt.title("Target Distribution by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_top_feature_correlation_heatmap(
    features: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_path: Path = PLOTS_DIR / "top_feature_correlation_heatmap.png",
    top_n: int = 12,
) -> None:
    """Plot a correlation heatmap for the top model features."""
    top_features = [feature for feature in importance_df["feature"].head(top_n) if feature in features.columns]
    if len(top_features) < 2:
        return
    corr = features[top_features].corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title(f"Correlation Heatmap of Top {len(top_features)} Features")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_shap_bar(
    shap_frame: pd.DataFrame,
    output_path: Path = PLOTS_DIR / "shap_bar.png",
    top_n: int = 15,
) -> None:
    """Plot mean absolute SHAP values as a bar chart."""
    mean_abs_shap = shap_frame.abs().mean().sort_values(ascending=False).head(top_n).iloc[::-1]
    plot_df = pd.DataFrame({"feature": mean_abs_shap.index, "mean_abs_shap": mean_abs_shap.values})
    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_df, x="mean_abs_shap", y="feature", hue="feature", palette="mako", legend=False)
    plt.title(f"Top {top_n} Mean Absolute SHAP Values")
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()


def export_prediction_table(
    y_train_true: pd.Series,
    y_train_pred: pd.Series,
    y_test_true: pd.Series,
    y_test_pred: pd.Series,
    output_path: Path = Path("data/processed/model_predictions.csv"),
) -> None:
    """Export actual/predicted values and residuals for train and test rows."""
    train_df = pd.DataFrame(
        {
            "split": "train",
            "actual": y_train_true.to_numpy(),
            "predicted": y_train_pred,
        }
    )
    test_df = pd.DataFrame(
        {
            "split": "test",
            "actual": y_test_true.to_numpy(),
            "predicted": y_test_pred,
        }
    )
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined["residual"] = combined["actual"] - combined["predicted"]
    combined["absolute_error"] = combined["residual"].abs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)


if __name__ == "__main__":
    seasons = ["2022-23", "2023-24", "2024-25"]
    target_column = DEFAULT_TARGET_COLUMN
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)

    datasets = []
    for season in seasons:
        ds = build_training_dataset(
            session_factory=session_factory,
            season=season,
            min_minutes=DEFAULT_MIN_LINEUP_MINUTES,
        )
        datasets.append(ds)
    dataset = pd.concat(datasets, ignore_index=True)

    artifacts = train_baseline_regressor(dataset=dataset, target_column=target_column)

    print(f"Target: {target_column}")
    print(f"Description: {TARGET_OPTIONS[target_column]}")
    print(f"Train rows: {artifacts.train_rows}")
    print(f"Test rows: {artifacts.test_rows}")
    print("--- Train Metrics ---")
    print(f"MAE:  {artifacts.metrics['train_mae']:.4f}")
    print(f"RMSE: {artifacts.metrics['train_rmse']:.4f}")
    print(f"R2:   {artifacts.metrics['train_r2']:.4f}")
    print("--- Test Metrics ---")
    print(f"MAE:  {artifacts.metrics['test_mae']:.4f}")
    print(f"RMSE: {artifacts.metrics['test_rmse']:.4f}")
    print(f"R2:   {artifacts.metrics['test_r2']:.4f}")

    importance_path = export_feature_importance(
        artifacts,
        output_path="data/processed/feature_importance.csv",
    )
    print(f"Wrote feature importance to {importance_path}")

    importance_df = pd.read_csv(importance_path)
    print("\n--- Top 10 Feature Importances ---")
    print(importance_df.head(10).to_string(index=False))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    features, target, _ = prepare_training_matrices(dataset, target_column=target_column)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )
    pred_train = artifacts.model.predict(x_train)
    pred_test = artifacts.model.predict(x_test)

    # XGBoost-style diagnostic suite
    plot_actual_vs_predicted(y_train, pred_train, y_test, pred_test, metrics=artifacts.metrics)
    plot_feature_importance_bar(importance_df)
    plot_residuals_vs_predicted(y_test, pred_test)
    plot_residual_distribution(y_test, pred_test)
    plot_prediction_error_boxplot(y_test, pred_test)
    plot_target_distribution(y_train, y_test)
    plot_top_feature_correlation_heatmap(features, importance_df)
    export_prediction_table(y_train, pred_train, y_test, pred_test)

    artifacts.model.plot_shap_summary(features, str(PLOTS_DIR / "shap_summary.png"))
    shap_frame = artifacts.model.explain(features)
    plot_shap_bar(shap_frame)

    print(f"Wrote actual vs predicted plot to {PLOTS_DIR / 'actual_vs_predicted.png'}")
    print(f"Wrote feature importance bar plot to {PLOTS_DIR / 'feature_importance_bar.png'}")
    print(f"Wrote residuals vs predicted plot to {PLOTS_DIR / 'residuals_vs_predicted.png'}")
    print(f"Wrote residual distribution plot to {PLOTS_DIR / 'residual_distribution.png'}")
    print(f"Wrote prediction error boxplot to {PLOTS_DIR / 'prediction_error_boxplot.png'}")
    print(f"Wrote target distribution plot to {PLOTS_DIR / 'target_distribution.png'}")
    print(f"Wrote top feature correlation heatmap to {PLOTS_DIR / 'top_feature_correlation_heatmap.png'}")
    print(f"Wrote SHAP summary plot to {PLOTS_DIR / 'shap_summary.png'}")
    print(f"Wrote SHAP bar plot to {PLOTS_DIR / 'shap_bar.png'}")
    print("Wrote predictions table to data/processed/model_predictions.csv")
