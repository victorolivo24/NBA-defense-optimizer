# Dynamic Lineup-Dependent Defensive Scheme Optimizer

Starter boilerplate for a Rutgers Data Science project that recommends the best NBA defensive scheme for a live five-man lineup using lineup context, defensive play-type data, and a swappable modeling layer.

## Project Goal

This project aims to move from descriptive NBA analytics to prescriptive decision support. Given a five-man lineup and a defensive objective, the system will aggregate player-level defensive tendencies, lineup-level performance, and matchup context to recommend a scheme such as `Drop`, `Switch`, or `Zone`.

The initial scaffold in this repository is designed to support that workflow end to end:

1. Pull raw defensive data from `nba_api`.
2. Normalize it into a relational SQLite database using SQLAlchemy.
3. Build lineup-level training data from player and play-type tables.
4. Plug that data into a model layer that can later be replaced or expanded.
5. Explain predictions with SHAP after the first model is trained.

## Repository Layout

```text
NBA-defense-optimizer/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── ingest.py
│   │   └── schema.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       └── scheme_recommender.py
├── ingest.py
├── project-proposal.txt
└── requirements.txt
```

## What Was Added

### 1. Database layer

The database scaffold lives in [src/database/schema.py](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/src/database/schema.py) and [src/database/connection.py](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/src/database/connection.py).

It includes:

- A `Player` table for core player identity fields.
- A `DefensivePlayType` table for player-level defensive PPP allowed by play type.
- A `LineupMetric` table for lineup-level defensive performance and scheme labels.
- A `LineupPlayer` association table that maps a lineup to its five players.

Why this shape:

- It keeps player data normalized and reusable.
- It supports SQL joins for feature engineering, which aligns with your course plan.
- It leaves room to add matchup, possession, or scheme-outcome tables later without rewriting the base schema.

Important note:

- SQLite cannot cleanly enforce "exactly five players per lineup" with a simple declarative constraint, so the schema enforces uniqueness per lineup-player pair and exposes a `slot` field from `1` to `5`. The exact five-player validation should happen in the ingestion or lineup-building logic.

### 2. Ingestion scaffold

The ingestion logic lives in [src/database/ingest.py](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/src/database/ingest.py), with a thin CLI entry point in [ingest.py](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/ingest.py).

This starter script:

- Creates the SQLite database.
- Fetches basic player defensive stats from `nba_api`.
- Attempts to fetch defensive play-type data.
- Normalizes the raw responses into Pandas DataFrames.
- Upserts data into the SQLAlchemy models.

Rate-limit warning:

- The NBA stats endpoints are sensitive to rapid repeated requests.
- The scaffold intentionally includes `time.sleep()` between API calls.
- Do not remove those delays unless you replace them with a more robust retry and backoff strategy.

### 3. Model layer placeholder

The model scaffold lives in [src/models/base.py](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/src/models/base.py) and [src/models/scheme_recommender.py](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/src/models/scheme_recommender.py).

This is intentionally modular:

- `BaseSchemeModel` defines a small interface for training, prediction, and feature importance.
- `XGBoostSchemeRecommender` is a placeholder implementation target for the first real model.
- You can later swap XGBoost for Random Forest, LightGBM, or a custom ensemble without rewriting the database or ingestion code.

### 4. Dependency list

[requirements.txt](/c:/Users/victo/Downloads/cs210/NBA-defense-optimizer/requirements.txt) includes the core libraries you requested:

- `nba_api`
- `pandas`
- `sqlalchemy`
- `xgboost`
- `shap`
- `scikit-learn`

## Current Plan

This scaffold is the starting point. The next implementation steps should be:

1. Expand ingestion to collect stable historical windows instead of a single snapshot.
2. Add lineup source tables or a lineup ingestion step from NBA lineup endpoints.
3. Engineer lineup features by aggregating the five linked players' defensive play-type profiles.
4. Define a target variable for scheme recommendation.
5. Train a baseline `XGBoost` regressor or classifier.
6. Add SHAP-based explanation outputs for coach-readable recommendations.

## How To Run

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Initialize the database and pull starter data:

```bash
python ingest.py
```

By default, the script writes to:

```text
data/processed/nba_defense.sqlite
```

## Notes On Data Quality

- Some `nba_api` endpoints change behavior without warning.
- Some defensive play-type endpoints are sparse or season-dependent.
- You should expect to harden the ingestion layer with retries, logging, and checkpointing once you begin full data collection.

## Proposal Alignment

The scaffold reflects the project proposal in `project-proposal.txt`:

- Relational storage using SQL concepts and joins.
- API-based data acquisition and Pandas transformation.
- A tree-based modeling path using XGBoost.
- Explainability support through SHAP.
- A modular pipeline that can evolve from a class project scaffold into a full recommendation engine.
