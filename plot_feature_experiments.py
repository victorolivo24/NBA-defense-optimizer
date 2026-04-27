"""Create a readable summary plot from feature experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_PATH = Path("data/processed/feature_experiment_results.csv")
OUTPUT_PATH = Path("data/processed/plots/feature_experiment_results_summary.png")


def select_best_runs(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values("test_r2", ascending=False).copy()
    return ordered.groupby("experiment", as_index=False).first()


def label_row(row: pd.Series) -> str:
    suffix = "valid" if str(row["valid_for_production"]).lower() == "true" else "control"
    return f"{row['experiment']} ({row['model_mode']}, {suffix})"


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing experiment results file: {INPUT_PATH}")

    sns.set_theme(style="whitegrid")
    results = pd.read_csv(INPUT_PATH)
    best_runs = select_best_runs(results)
    best_runs["plot_label"] = best_runs.apply(label_row, axis=1)
    best_runs = best_runs.sort_values("test_r2", ascending=True)

    palette = {
        True: "#2a9d8f",
        False: "#b56576",
        "True": "#2a9d8f",
        "False": "#b56576",
    }
    colors = [palette.get(value, "#577590") for value in best_runs["valid_for_production"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=True)

    metrics = [
        ("test_r2", "Test R²", False),
        ("test_mae", "Test MAE", True),
        ("test_rmse", "Test RMSE", True),
    ]

    for ax, (metric, title, lower_is_better) in zip(axes, metrics):
        ax.barh(best_runs["plot_label"], best_runs[metric], color=colors)
        ax.set_title(title)
        ax.set_xlabel(title)
        if lower_is_better:
            ax.invert_xaxis()
        for index, value in enumerate(best_runs[metric]):
            text_x = value + (0.003 if not lower_is_better else -0.003)
            ha = "left" if not lower_is_better else "right"
            ax.text(text_x, index, f"{value:.3f}", va="center", ha=ha, fontsize=9)

    axes[0].set_ylabel("Feature Family (best mode)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.suptitle("Feature Experiment Summary", fontsize=16)
    fig.text(
        0.5,
        0.02,
        "Green = valid production feature set. Red = leakage control / non-production comparison.",
        ha="center",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Wrote feature experiment summary plot to {OUTPUT_PATH}")
