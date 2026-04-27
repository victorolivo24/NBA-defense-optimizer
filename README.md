# Dynamic Lineup-Dependent Defensive Scheme Optimizer

This repository implements a lineup-aware defensive scheme simulator and recommendation engine for NBA five-man units. It ingests live `nba_api` data, stores it in SQLite, engineers lineup features, trains an XGBoost regressor on lineup defensive outcome, and simulates `Drop`, `Switch`, and `Zone` to recommend the lowest-risk coverage.

Important framing:

- This is not a supervised defensive scheme classifier.
- The public NBA API does not expose possession-level labels for `Drop`, `Switch`, or `Zone`.
- The project therefore predicts lineup defensive outcome first, then simulates how candidate coverages may change that outcome.

## Grader Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the core pipeline:

```bash
python ingest.py
python build_features.py
python train_model.py
python src/features/calculate_scheme_deltas.py
python demo.py --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns"
```

If you want the feature-ablation branch results as well:

```bash
python refresh_feature_caches.py
python run_feature_experiments.py
python plot_feature_experiments.py
```

Recommended review order:

1. `README.md`
2. `docs/demo_instructions.txt`
3. `docs/implementation_log.md`
4. `docs/feature_experiment_results.md`

## What the Project Does

Primary input:

- five player names

Current workflow:

1. Resolve the five players from the database.
2. Try to find an exact historical 5-man lineup match.
3. If no exact match exists, build a weighted blend from historical lineups with `2` to `4` overlapping players.
4. If there is not enough historical evidence, fall back to a synthetic lineup row built from player defensive profiles.
5. Predict the lineup's baseline defensive outcome with XGBoost.
6. Simulate `Drop`, `Switch`, and `Zone`.
7. Return the ranked recommendation and explanation rows.

Primary output:

- recommended scheme
- ranked scheme table
- baseline predicted defensive rating
- historical or estimated target when applicable
- explanation rows showing the most relevant scheme adjustments

## Current Architecture

```text
NBA-defense-optimizer/
|-- data/
|   |-- processed/
|   |   |-- feature_experiment_importances.csv
|   |   |-- feature_experiment_results.csv
|   |   |-- feature_importance.csv
|   |   |-- lineup_training_dataset.csv
|   |   |-- model_predictions.csv
|   |   `-- plots/
|   `-- raw/
|-- docs/
|   |-- demo_instructions.txt
|   |-- feature_experiment_results.md
|   `-- implementation_log.md
|-- notebooks/
|-- src/
|   |-- database/
|   |   |-- connection.py
|   |   |-- ingest.py
|   |   `-- schema.py
|   |-- demo/
|   |   `-- workflow.py
|   |-- features/
|   |   |-- calculate_scheme_deltas.py
|   |   `-- lineup_dataset.py
|   `-- models/
|       |-- base.py
|       |-- recommendation.py
|       |-- scheme_profiles.py
|       |-- scheme_recommender.py
|       `-- training.py
|-- build_features.py
|-- demo.py
|-- ingest.py
|-- plot_feature_experiments.py
|-- project-proposal.txt
|-- recommend_scheme.py
|-- refresh_feature_caches.py
|-- requirements.txt
|-- run_feature_experiments.py
`-- train_model.py
```

## Core Pipeline

1. `ingest.py`
   - pulls player defense, Synergy defensive play types, and lineup data from `nba_api`
   - writes raw endpoint snapshots to `data/raw/<season>/<endpoint>/`
   - normalizes records into SQLite with SQLAlchemy
2. `build_features.py`
   - joins players, defensive play types, and lineups into one model-ready dataset
   - writes `data/processed/lineup_training_dataset.csv`
3. `train_model.py`
   - trains the baseline XGBoost regressor
   - reports `R^2`, `MAE`, and `RMSE`
   - writes feature importance, prediction exports, and diagnostic plots
4. `src/features/calculate_scheme_deltas.py`
   - derives dynamic scheme profiles from the current database
   - writes `data/processed/dynamic_scheme_profiles.json`
5. `demo.py`
   - runs the player-input or historical-lineup demo

## Data Model

Core tables:

- `players`
- `defensive_play_types`
- `lineup_metrics`
- `lineup_players`

Why this schema:

- players remain normalized and reusable
- lineups map to five players through `lineup_players`
- the feature layer stays separate from ingestion and model training
- the model can be swapped later without changing the database design

## Target and Recommendation Logic

Default training target:

- `defensive_rating_target`

Target source:

- lineup ingestion uses the advanced `LeagueDashLineups` view
- that provides real lineup `DEF_RATING`, `PACE`, and `POSS`
- fallback per-48 targets are only used if those advanced fields are missing

Recommendation logic:

1. train an outcome model for lineup defensive rating
2. apply scheme-specific feature adjustments for `Drop`, `Switch`, and `Zone`
3. rescore the lineup under each simulated scheme
4. choose the scheme with the lowest predicted defensive cost

Scheme profile source priority:

1. `data/processed/dynamic_scheme_profiles.json`
2. `src/models/scheme_profiles.py`

## Final Production Model Outputs

`python train_model.py` writes:

- `data/processed/feature_importance.csv`
- `data/processed/model_predictions.csv`
- `data/processed/plots/actual_vs_predicted.png`
- `data/processed/plots/feature_importance_bar.png`
- `data/processed/plots/residuals_vs_predicted.png`
- `data/processed/plots/residual_distribution.png`
- `data/processed/plots/prediction_error_boxplot.png`
- `data/processed/plots/target_distribution.png`
- `data/processed/plots/top_feature_correlation_heatmap.png`
- `data/processed/plots/shap_summary.png`
- `data/processed/plots/shap_bar.png`

Current verified branch metrics from `train_model.py`:

- train rows: `565`
- test rows: `189`
- train `R^2 = 0.2242`
- train `MAE = 9.3111`
- train `RMSE = 11.9291`
- test `R^2 = 0.1319`
- test `MAE = 10.3263`
- test `RMSE = 12.8919`

## Feature Experiments and Literature

This branch also includes a structured feature experiment pass to test whether the model could beat the original defensive-rating-driven feature set.

Main experiment scripts:

- `refresh_feature_caches.py`
- `run_feature_experiments.py`
- `plot_feature_experiments.py`

Main outputs:

- `data/processed/feature_experiment_results.csv`
- `data/processed/feature_experiment_importances.csv`
- `data/processed/plots/feature_experiment_results_summary.png`
- `docs/feature_experiment_results.md`

What was tested:

- player defensive rating aggregates
- Synergy play-type features
- player basic defensive box-score features
- mixed advanced plus basic feature sets
- lineup four-factor style leakage controls

Best valid feature-family result:

- `player_def_rating_only`, tuned
- test `R^2 = 0.1132`
- test `MAE = 10.3952`
- test `RMSE = 13.0303`

What that means:

- aggregated player defensive rating features remained the strongest valid predictive signal
- player basic defensive box-score features helped, but did not beat the rating-only core
- Synergy play-type features were weaker than expected on their own
- adding more features did not materially lift `R^2`

Literature used to justify these experiments:

- Four Factors and efficiency: https://arxiv.org/abs/2305.13032
- unseen lineup prediction from player-level summaries: https://arxiv.org/abs/2303.04963
- sparse and noisy lineup data: https://arxiv.org/abs/2601.15000

## Attempts, Failures, and Pivots

Key pivots in the project:

- The project was reframed from a defensive scheme classifier to a simulator and recommendation engine because public scheme labels do not exist.
- The original lineup target used a fallback per-48 calculation because the wrong lineup endpoint view was ingested.
- Ingestion was corrected to use the advanced lineup endpoint, which exposed real `DEF_RATING` and `POSS`.
- The demo originally relied too heavily on synthetic custom lineups and was changed to prefer:
  - exact historical lineup match
  - weighted historical overlap
  - synthetic fallback only as a last resort
- Feature engineering was hardened with minute and possession thresholds plus median imputation to reduce noise.

These decisions are documented in detail in:

- `docs/implementation_log.md`
- `docs/feature_experiment_results.md`

## Running the Demo

Player-input demo:

```bash
python demo.py --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns"
```

Other example lineups:

```bash
python demo.py --players "Stephen Curry" "Brandin Podziemski" "Andrew Wiggins" "Draymond Green" "Kevon Looney"
python demo.py --players "Jrue Holiday" "Derrick White" "Jaylen Brown" "Jayson Tatum" "Kristaps Porzingis"
```

Historical-lineup mode:

```bash
python demo.py
python demo.py --team BOS --limit 3
python demo.py --search Brunson
```

## Limitations

- no public possession-level scheme labels
- no opponent-aware recommendation layer yet
- scheme profiles are still proxy-based, even when generated dynamically
- custom unseen lineups remain less certain than exact historical matches
- lineup-level NBA defensive outcomes are sparse and noisy, which keeps the achievable `R^2` relatively low

The project should therefore be described as:

- a lineup-aware defensive scheme simulator and recommendation engine

Not as:

- a supervised defensive coverage classifier

## Documentation Map

Use these files based on what you need:

- `docs/demo_instructions.txt`
  - step-by-step runbook for graders and operators
- `docs/implementation_log.md`
  - project history, failures, pivots, and technical rationale
- `docs/feature_experiment_results.md`
  - literature-backed feature-ablation study and why some feature ideas underperformed
- `project-proposal.txt`
  - original proposal updated to match the final framing
