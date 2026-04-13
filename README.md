# Dynamic Lineup-Dependent Defensive Scheme Optimizer

Starter boilerplate for a Rutgers Data Science project that builds a lineup-aware defensive scheme simulator and recommendation engine using lineup context, defensive play-type data, and a swappable modeling layer.

## Project Goal

This project aims to move from descriptive NBA analytics to prescriptive decision support. Given a five-man lineup and a defensive objective, the system will aggregate player-level defensive tendencies, lineup-level performance, and matchup context to simulate how a scheme such as `Drop`, `Switch`, or `Zone` may perform, then rank the available options.

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
│   ├── features/
│   │   ├── __init__.py
│   │   └── lineup_dataset.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── recommendation.py
│       ├── scheme_recommender.py
│       └── training.py
├── build_features.py
├── ingest.py
├── project-proposal.txt
├── recommend_scheme.py
├── train_model.py
└── requirements.txt
```

## What Was Added

### 1. Database layer

The database scaffold lives in `src/database/schema.py` and `src/database/connection.py`.

It includes:

- A `Player` table for core player identity fields.
- A `DefensivePlayType` table for player-level defensive PPP allowed by play type.
- A `LineupMetric` table for lineup-level defensive performance and optional scheme metadata.
- A `LineupPlayer` association table that maps a lineup to its five players.

Why this shape:

- It keeps player data normalized and reusable.
- It supports SQL joins for feature engineering, which aligns with your course plan.
- It leaves room to add matchup, possession, or scheme-outcome tables later without rewriting the base schema.

Important note:

- SQLite cannot cleanly enforce "exactly five players per lineup" with a simple declarative constraint, so the schema enforces uniqueness per lineup-player pair and exposes a `slot` field from `1` to `5`. The exact five-player validation should happen in the ingestion or lineup-building logic.

### 2. Ingestion scaffold

The ingestion logic lives in `src/database/ingest.py`, with a thin CLI entry point in `ingest.py`.

Phase 1 now includes:

- Creates the SQLite database.
- Fetches basic player defensive stats from `nba_api`.
- Fetches defensive play-type data.
- Fetches five-man lineup defensive data.
- Saves raw endpoint snapshots to `data/raw/<season>/<endpoint>/`.
- Normalizes the raw responses into Pandas DataFrames.
- Upserts data into the SQLAlchemy models.
- Retries unstable API requests with backoff.

Rate-limit warning:

- The NBA stats endpoints are sensitive to rapid repeated requests.
- The scaffold intentionally includes `time.sleep()` between API calls.
- Do not remove those delays unless you replace them with a more robust retry and backoff strategy.

Raw snapshot rationale:

- API payloads are saved before further transformation.
- This gives you reproducibility when an endpoint changes later.
- It also makes debugging easier when a downstream normalization step breaks.

### 3. Model layer placeholder

The model scaffold lives in `src/models/base.py` and `src/models/scheme_recommender.py`.

This is intentionally modular:

- `BaseSchemeModel` defines a small interface for training, prediction, and feature importance.
- `XGBoostSchemeRecommender` is the current baseline lineup outcome model.
- You can later swap XGBoost for Random Forest, LightGBM, or a custom ensemble without rewriting the database or ingestion code.

### 4. Feature engineering layer

Phase 2 adds a dedicated feature pipeline in `src/features/lineup_dataset.py`.

This layer:

- Joins lineup records to the five linked players.
- Pulls each player's defensive play-type profile.
- Aggregates lineup-level features such as position counts, average size, and play-type PPP summaries.
- Produces model-ready rows with targets like `defensive_rating_target` and `opponent_ppp_target`.
- Exports the dataset to CSV through `build_features.py`.

### 5. Dependency list

`requirements.txt` includes the core libraries you requested:

- `nba_api`
- `pandas`
- `sqlalchemy`
- `xgboost`
- `shap`
- `scikit-learn`

## Phase 1 Status

Phase 1 is the data-ingestion hardening phase. The implementation goal is:

1. Pull player defense, defensive play-type, and lineup defense data.
2. Save raw snapshots locally for reproducibility.
3. Load normalized records into SQLite.
4. Test ingestion helpers and database writes before moving to feature engineering.

Phase 1 deliverables now implemented:

1. Retry and backoff aware endpoint fetch logic.
2. Raw JSON persistence under `data/raw/`.
3. Five-man lineup ingestion support.
4. Automated tests for parsing, raw persistence, and SQL upserts.

## Phase 2 Status

Phase 2 is the feature engineering phase. The implementation goal is:

1. Join player, play-type, and lineup tables into a single training dataset.
2. Aggregate five-man lineup features that a model can consume directly.
3. Keep the feature layer separate from the model so the modeling logic stays swappable.
4. Test the generated dataset before starting model training.

Phase 2 deliverables now implemented:

1. A dedicated feature package under `src/features/`.
2. A lineup dataset builder that produces model-ready Pandas rows.
3. A CSV export entry point for offline training workflows.
4. Automated tests for lineup feature aggregation and dataset export.

## Phase 3 Status

Phase 3 is the baseline target-definition and model-training phase. The implementation goal is:

1. Decide on a target the model can learn from with the currently available data.
2. Keep that target definition explicit so it can be replaced later with true scheme-aware supervision.
3. Train and evaluate a baseline model on the engineered lineup dataset.
4. Test the training pipeline before moving into recommendation logic.

Phase 3 deliverables now implemented:

1. A training helper module in `src/models/training.py`.
2. Explicit target options for `defensive_rating_target` and `opponent_ppp_target`.
3. A baseline XGBoost training and evaluation path.
4. A CLI entry point in `train_model.py`.
5. Automated tests for matrix preparation and baseline training.

Current target note:

- The project does not yet have true historical defensive-scheme labels.
- Phase 3 therefore uses lineup defensive outcome targets as a surrogate learning problem.
- The default baseline target is `defensive_rating_target`.
- Lineup ingestion now uses the advanced `LeagueDashLineups` view so the API can return lineup `DEF_RATING`, `PACE`, and `POSS`.
- The fallback target from opponent points allowed per 48 minutes is now only used when the advanced lineup view still fails to provide a direct defensive rating.
- This is not a direct scheme classifier; it is the first defensible supervised training step for the simulator.

Live NBA API validation note:

- The ingestion code is wired to the real `nba_api` endpoints and has now been validated end to end in this repo environment.
- The original blocker was a broken shell proxy configuration plus mismatched Synergy endpoint arguments.
- `python ingest.py` now successfully writes raw snapshots and populates the SQLite database with live player, play-type, and lineup data.

## Phase 4 Status

Phase 4 is the simulation, recommendation, and explanation phase. The implementation goal is:

1. Score candidate defensive schemes for a lineup using the baseline model.
2. Keep scheme logic explicit and editable while true scheme labels are still unavailable.
3. Return a ranked recommendation rather than only a raw predicted outcome.
4. Expose explanation data to support coach-readable output.

Phase 4 deliverables now implemented:

1. A recommendation layer in `src/models/recommendation.py`.
2. Explicit simulator profiles for `Drop`, `Switch`, and `Zone`.
3. SHAP-backed explanation support through the trained XGBoost model.
4. A CLI entry point in `recommend_scheme.py`.
5. Automated tests for scheme adjustment and recommendation ranking.

Phase 4 limitation:

- The recommendation engine currently applies explicit feature adjustments for each scheme profile and scores the adjusted lineup with the baseline outcome model.
- This is a practical simulation-and-recommendation workflow built on real data, but it is still a proxy for true historical scheme outcome modeling.
- Once you have scheme-aware labels or possession-level scheme tagging, these simulator profiles should be replaced by learned scheme-specific training targets.

## Current Plan

After Phase 4, the next implementation steps should be:

1. Expand ingestion to collect stable historical windows instead of a single snapshot.
2. Add any missing context tables needed for matchup or game-window features.
3. Replace heuristic simulator profiles with scheme-aware training data.
4. Add richer context such as coaching directives, opponent archetypes, and game state.
5. Package a recommendation interface for course demos.
6. Add persistence for recommendation runs and explanations.

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

Raw API snapshots are written under:

```text
data/raw/<season>/<endpoint>/
```

Run the Phase 1 test suite with:

```bash
pytest
```

Build the Phase 2 training dataset with:

```bash
python build_features.py
```

Train the Phase 3 baseline lineup outcome model with:

```bash
python train_model.py
```

Generate a Phase 4 simulated scheme recommendation with:

```bash
python recommend_scheme.py
```

Run the presentation demo with five player names:

```bash
python demo.py --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns"
```

When a historical lineup row uses the fallback target instead of a possession-based defensive rating, the demo labels it explicitly as `Actual target (fallback per 48)`.

If you do not supply player names, the demo falls back to curated case studies:

```bash
python demo.py
```

You can also filter the demo to a team or lineup search string:

```bash
python demo.py --team BOS
python demo.py --search Brunson
```

The matching notebook walkthrough lives in:

```text
notebooks/demo_workflow.ipynb
```

## Notes On Data Quality

- Some `nba_api` endpoints change behavior without warning.
- Some defensive play-type endpoints are sparse or season-dependent.
- You should expect to keep refining retries, logging, and checkpointing as you begin full data collection.
- Some lineup endpoint fields vary slightly by season, so normalization helpers should stay defensive.

## Proposal Alignment

The scaffold reflects the project proposal in `project-proposal.txt`:

- Relational storage using SQL concepts and joins.
- API-based data acquisition and Pandas transformation.
- A tree-based modeling path using XGBoost.
- Explainability support through SHAP.
- A modular pipeline that can evolve from a class project scaffold into a full simulation and recommendation engine.

## Study Log

The running implementation history, modeling choices, failures, fixes, and current metrics are documented in:

```text
docs/implementation_log.md
```

Use that file as the project study guide before presentations or interviews.

## Positioning Note

This repository should be described as a lineup-aware defensive scheme simulator and recommendation engine, not as a supervised defensive scheme classifier.

Why:

- The public NBA stats API exposes lineup outcomes and aggregated defensive play-type data.
- It does not expose possession-level labels that say a defense was in `Drop`, `Switch`, or `Zone` on a given play.
- The current model therefore learns lineup defensive outcomes from real data, then the simulator layer applies explicit scheme profiles to estimate how different coverages may perform.

That framing is both accurate and defendable for a course project.
