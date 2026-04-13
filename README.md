# Dynamic Lineup-Dependent Defensive Scheme Optimizer

This repository implements a lineup-aware defensive scheme simulator and recommendation engine for NBA five-man units. It uses live `nba_api` data, a relational SQLite store, lineup-level feature engineering, a baseline XGBoost regressor, and a scheme simulation layer that ranks `Drop`, `Switch`, and `Zone`.

Important framing:

- This is not a supervised defensive scheme classifier.
- The public NBA API does not expose possession-level labels for `Drop`, `Switch`, or `Zone`.
- The project therefore predicts lineup defensive outcome first, then simulates how candidate coverages may change that outcome.

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
|   `-- raw/
|-- docs/
|   |-- demo_instructions.txt
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
|-- project-proposal.txt
|-- recommend_scheme.py
|-- requirements.txt
`-- train_model.py
```

## Data Pipeline

The project uses live NBA data and keeps the pipeline modular:

1. `ingest.py`
   - pulls player defense, Synergy defensive play types, and lineup data from `nba_api`
   - writes raw endpoint snapshots to `data/raw/<season>/<endpoint>/`
   - normalizes records into SQLite with SQLAlchemy
2. `build_features.py`
   - joins players, defensive play types, and lineups into one model-ready dataset
   - writes `data/processed/lineup_training_dataset.csv`
3. `train_model.py`
   - trains the baseline XGBoost regressor
   - reports `MAE` and `RMSE`
   - writes feature importance output
4. `src/features/calculate_scheme_deltas.py`
   - derives dynamic scheme profiles from the current season database
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
- the feature layer stays separate from the ingestion and model layers
- the model can be swapped later without changing the database design

## Targets and Labels

Default training target:

- `defensive_rating_target`

Target source:

- lineup ingestion now uses the advanced `LeagueDashLineups` view
- that gives real lineup `DEF_RATING`, `PACE`, and `POSS`
- fallback per-48 targets are only used if those advanced fields are missing

Important distinction in demo output:

- `Actual target (API defensive rating)`
  - true historical target for an exact lineup match
- `Estimated historical target (weighted overlap)`
  - weighted estimate from similar historical lineups, not a true exact-lineup actual
- `Actual target (fallback per 48)`
  - backup target if the advanced lineup view did not provide a usable defensive rating

## Scheme Recommendation Logic

The final recommendation is not learned end to end from scheme labels.

Instead:

1. the baseline model predicts lineup defensive outcome
2. the recommendation layer adjusts the lineup features for each candidate scheme
3. the model rescored each scheme-adjusted lineup
4. the lowest predicted defensive cost wins

Scheme profile source priority:

1. `data/processed/dynamic_scheme_profiles.json`
2. `src/models/scheme_profiles.py`

That means:

- dynamic JSON profiles override the hardcoded defaults
- hardcoded profiles remain the fallback if the JSON has not been generated

## Player-Input Demo Behavior

The demo supports real player input:

```bash
python demo.py --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns"
```

Possible `Source` values in the output:

- `exact_historical_lineup`
- `weighted_historical_overlap_4_plus`
- `weighted_historical_overlap_3`
- `weighted_historical_overlap_2`
- `synthetic_player_profile`

Why this matters:

- exact historical matches are the most trustworthy
- overlap-based estimates are more realistic than a pure synthetic lineup
- synthetic rows are still supported for unseen combinations, but are the weakest evidence tier

## Dynamic Scheme Profiles

The dynamic scheme profile generator lives in:

```text
src/features/calculate_scheme_deltas.py
```

It writes:

```text
data/processed/dynamic_scheme_profiles.json
```

These deltas are data-driven proxies, not supervised labels.

Current proxy logic:

- `Drop`
  - derived from high-volume roll-man defenders
- `Switch`
  - derived from high-volume ball-handler defenders
- `Zone`
  - derived from lineups with high spot-up possession exposure

This gives the simulator a transparent adjustment layer while true public scheme labels remain unavailable.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python ingest.py
python build_features.py
python train_model.py
python src/features/calculate_scheme_deltas.py
python demo.py --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns"
```

Historical-lineup demo mode:

```bash
python demo.py
python demo.py --team BOS --limit 3
python demo.py --search Brunson
```

Single-lineup recommender example:

```bash
python recommend_scheme.py
```

## Verified Current State

Current repo behavior that has been validated:

- live `nba_api` ingestion works in this repo
- lineup ingestion uses the advanced endpoint for real `DEF_RATING`
- feature engineering exports a model-ready training dataset
- baseline training runs with XGBoost
- player-input demo supports exact historical lookup, weighted overlap estimation, and synthetic fallback

Recent verified training metrics:

- train rows: `1545`
- test rows: `515`
- `MAE = 14.5110`
- `RMSE = 18.8088`

## Key Limitations

- no public possession-level scheme labels
- no opponent-aware recommendation layer yet
- scheme profiles are still proxy-based, even when they are generated dynamically
- custom unseen lineups remain less certain than exact historical matches

The project should therefore be described as:

- a lineup-aware defensive scheme simulator and recommendation engine

Not as:

- a supervised defensive coverage classifier

## Documentation

Full implementation history, failures, fixes, and interview notes:

```text
docs/implementation_log.md
```

Short runbook for presentation day:

```text
docs/demo_instructions.txt
```

Original proposal, updated to match the current framing:

```text
project-proposal.txt
```
