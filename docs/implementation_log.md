# Implementation Log

This document records what the system is trying to do, what was implemented, what failed, what changed, and what the current metrics mean.

## Project Framing

Current project description:

- `Lineup-Aware Defensive Scheme Simulator and Recommendation Engine`

Why this framing changed:

- The original idea was closer to a defensive scheme classifier.
- The public NBA stats API does not expose possession-level labels that say a defense was in `Drop`, `Switch`, or `Zone`.
- Because those labels do not exist in the public data, the project cannot honestly claim to be a supervised scheme classifier.
- The current system instead:
  - predicts lineup-level defensive outcome with ML
  - simulates how candidate coverages may change that outcome
  - recommends the lowest predicted defensive-cost option

## End-to-End Pipeline

Current pipeline:

1. Ingest defensive player stats, defensive play-type stats, and five-man lineup stats from `nba_api`.
2. Persist raw JSON snapshots under `data/raw/` for reproducibility.
3. Normalize data into SQLite with SQLAlchemy.
4. Build lineup-level features from the five players in each lineup.
5. Train an XGBoost regressor on lineup defensive outcome.
6. Simulate candidate schemes by adjusting lineup features using explicit scheme profiles.
7. Recommend the scheme with the lowest predicted defensive outcome.

Current demo flow:

1. User enters five player names.
2. The app resolves those names from the `players` table.
3. It tries to find an exact historical 5-man lineup match in the engineered lineup dataset.
4. If no exact match exists, it builds a weighted blend from historical lineups that share `2` to `4` of the same players.
5. If there is not enough historical overlap evidence, it falls back to a synthetic lineup feature row built from the five players.
6. It predicts the baseline defensive outcome.
7. It simulates `Drop`, `Switch`, and `Zone`.
8. It prints the ranked recommendation, explanation rows, and the lineup source used.

Important display note:

- historical exact matches should show `Actual target (API defensive rating)` when available
- overlap-based custom lineup estimates should show `Estimated historical target (weighted overlap)`
- fallback targets should show `Actual target (fallback per 48)`

## Inputs and Outputs

Primary input:

- five player names

Supporting inputs:

- season, for example `2024-25`
- optional existing lineup search or team filters for the historical demo mode

Feature inputs used by the model:

- lineup size
- guard, forward, and center counts
- average height and weight
- player defensive rating aggregates
- player defensive box-score aggregates
- Synergy play-type feature aggregates such as:
  - `isolation_ppp_mean`
  - `pick_and_roll_ball_handler_ppp_mean`
  - `pick_and_roll_roll_man_ppp_mean`
  - `spot_up_ppp_mean`
  - possession and percentile summaries for each play type

Primary model output:

- `defensive_rating_target`

Recommendation output:

- recommended scheme
- ranked schemes with predicted values
- SHAP-backed explanation rows for the baseline feature profile

## Database and Data Model

Core tables:

- `players`
- `defensive_play_types`
- `lineup_metrics`
- `lineup_players`

Important schema decision:

- lineups are mapped to players through a many-to-many association table with a `slot` field from `1` to `5`

Why:

- it keeps players reusable across lineups
- it supports SQL joins for feature engineering
- it keeps the model layer separate from ingestion

## Ingestion Notes

What ingestion does:

- fetches player defense from `LeagueDashPlayerStats`
- fetches defensive play types from `SynergyPlayTypes`
- fetches five-man lineups from `LeagueDashLineups`
- writes raw snapshots before transformation
- upserts normalized records into SQLite

Additional cache outputs on the feature-experiment branch:

- `data/processed/player_defense_profiles.json`
- `data/processed/lineup_basic_defense_profiles.json`

Important operational note:

- the NBA API is rate-limit sensitive
- `time.sleep()` was intentionally kept in the ingestion path to avoid being blocked

## Major Failures and Fixes

### 1. API connectivity failed in script even though notebook code worked

Observed failure:

- notebook requests worked
- `ingest.py` failed with request refusal before data returned

Root cause:

- broken proxy environment variables in the shell
- mismatched `SynergyPlayTypes` arguments in the script
- live lineup `GROUP_ID` parsing assumptions were too strict

Fix:

- temporarily clear proxy env vars during NBA requests
- use supported `SynergyPlayTypes` parameters
- normalize live play-type labels
- update lineup parsing to handle live hyphen-wrapped `GROUP_ID` values

Result:

- live ingestion succeeded
- raw snapshots were written
- database was populated from real API data

### 2. Project originally overclaimed scheme supervision

Observed issue:

- the code and docs implied the model might learn actual defensive scheme choice

Root cause:

- no public possession-level scheme labels exist in the NBA API

Fix:

- reframe the project as a simulator and recommendation engine
- keep ML focused on lineup defensive outcome
- make scheme logic explicit and heuristic

Result:

- framing is more honest and technically defensible

### 3. Demo originally selected historical lineups by minutes instead of taking user players

Observed issue:

- the original demo was useful for showing existing lineups
- it did not match the intended product workflow

Fix:

- add player-name lookup
- support player-input custom lineups

Result:

- the demo is now player-input driven

### 4. Pure synthetic player-input lineups were not realistic enough

Observed issue:

- the first version of player-input mode built a synthetic lineup row only from the five players' defensive profiles
- that could produce unrealistic predictions for real lineups, especially when the exact 5-man unit already existed historically

Why this happened:

- the synthetic row used player-level aggregates only
- it did not reuse lineup-level evidence from `lineup_metrics`

Fix:

- player-input mode now tries three strategies in order:
  1. exact 5-man historical lineup match
  2. weighted historical overlap blend from similar lineups
  3. pure synthetic player-profile fallback only if no meaningful lineup history exists

Current overlap weights:

- 5 shared players: `1.0`
- 4 shared players: `0.7`
- 3 shared players: `0.4`
- 2 shared players: `0.15`

Result:

- player-input recommendations now stay closer to real historical lineup behavior
- exact known lineups can surface a real target
- unseen lineups still work through the synthetic fallback, but only after historical evidence is exhausted

### 5. Training pipeline was too vulnerable to noise

Observed issues:

- low-minute lineups could create unstable lineup targets
- low-possession play-type stats could create misleading player defensive profiles
- mean imputation was vulnerable to skew from outliers

Fix:

- set a minimum lineup minutes threshold for training data
- set a minimum play-type possessions threshold before a play-type stat is included
- switch missing-value imputation from mean to median

Current defaults:

- `DEFAULT_MIN_LINEUP_MINUTES = 10.0`
- `DEFAULT_MIN_PLAY_TYPE_POSSESSIONS = 10.0`

Result:

- less noise from tiny samples
- more stable training rows
- less distortion from extreme outliers

### 6. We initially pulled the wrong lineup endpoint view for the target

Observed issue:

- every lineup training row used the fallback target
- lineup targets looked inflated and unstable

Root cause:

- ingestion originally called `LeagueDashLineups` with `measure_type_detailed_defense='Defense'`
- that returned a box-score-style lineup table with fields like `MIN`, `PTS`, `FGM`, and `FGA`
- it did not return the advanced lineup fields we needed, such as `DEF_RATING`, `PACE`, and `POSS`

How we discovered it:

- we directly queried the endpoint with different parameter settings
- `measure_type_detailed_defense='Defense'` returned no possession-based advanced columns
- `measure_type_detailed_defense='Advanced'` returned:
  - `DEF_RATING`
  - `OFF_RATING`
  - `NET_RATING`
  - `PACE`
  - `POSS`

Fix:

- switch lineup ingestion to the advanced lineup view
- keep the fallback target only as a backup when advanced fields are still missing

Result:

- the project now trains on real API-provided lineup defensive rating instead of the fallback-only target

## Modeling Decisions

Baseline model:

- `XGBoostSchemeRecommender`

Why XGBoost:

- handles nonlinear interactions well
- fits tabular data
- supports feature importance and SHAP explanation

What the model predicts:

- lineup defensive outcome, not scheme label

Target definition:

- default target: `defensive_rating_target`

How we make the target source explicit:

- the engineered dataset includes `defensive_rating_target_source`
- values are labeled as either:
  - `api_defensive_rating`
  - `fallback_points_allowed_per_48`

## Recommendation Logic

Where the default heuristic lives:

- `src/models/scheme_profiles.py`

Where the dynamic override comes from:

- `data/processed/dynamic_scheme_profiles.json`

How recommendation works:

1. build or load a lineup feature row
2. run the baseline model
3. apply scheme profile adjustments for each candidate scheme
4. rescore each scheme-adjusted lineup
5. choose the scheme with the lowest predicted value

Important caveat:

- the scheme profile adjustments are not learned from true labeled historical scheme data
- they are heuristic or proxy-driven, depending on whether the hardcoded or generated profiles are used

## Dynamic Scheme Profiles

Where the JSON comes from:

- `src/features/calculate_scheme_deltas.py`

What it writes:

- `data/processed/dynamic_scheme_profiles.json`

Why this file exists:

- the hardcoded profiles in `src/models/scheme_profiles.py` are manual basketball heuristics
- `calculate_scheme_deltas.py` is an attempt to replace those hand-set values with data-driven deltas from the current database

How the generator works at a high level:

1. Pull defensive play-type data for the selected season from SQLite.
2. Filter out tiny samples by requiring at least `10` possessions.
3. Build a proxy group of defenders or lineups for each scheme.
4. Compare those proxy groups against league-average values.
5. Store the difference as the scheme delta.

Core proxy logic:

- `Drop`
  - uses high-volume roll-man defenders as the main proxy
- `Switch`
  - uses high-volume ball-handler defenders as the main proxy
- `Zone`
  - uses lineups with high spot-up possession exposure as the proxy

Important caveat:

- this is still heuristic
- it is data-driven, but not truly supervised on scheme labels

## Feature Experiment Branch

Branch goal:

- test whether the model could be improved by replacing or supplementing the defensive-rating-driven features

Motivation:

- baseline feature importance showed very heavy reliance on:
  - `avg_player_def_rtg`
  - `best_player_def_rtg`
  - `worst_player_def_rtg`

What was added:

1. Player basic defensive box-score features
- defensive rebounds
- steals
- blocks
- defensive win shares
- opponent points off turnovers
- opponent second-chance points
- opponent fast break points
- opponent paint points

2. Lineup four-factor style features
- `lineup_opp_efg_pct`
- `lineup_opp_tov_rate`
- `lineup_opp_orb_rate`
- `lineup_opp_fta_rate`

3. Experiment infrastructure
- `refresh_feature_caches.py`
- `run_feature_experiments.py`
- `plot_feature_experiments.py`

Literature used to justify the experiments:

- Four Factors and efficiency: https://arxiv.org/abs/2305.13032
- unseen lineup prediction from player-level summaries: https://arxiv.org/abs/2303.04963
- sparse and noisy lineup data: https://arxiv.org/abs/2601.15000

Detailed experiment results live in:

- `docs/feature_experiment_results.md`

High-level conclusion:

- aggregated player defensive rating features remained the strongest valid predictive signal
- player defensive box-score features were useful supplements
- Synergy play-type features were weak on their own
- more features did not materially lift `R^2`

## Evaluation and Metrics

Training metrics used:

- `R^2`
- `MAE`
- `RMSE`

Current verified production-model run from `train_model.py`:

- train rows: `565`
- test rows: `189`
- train `R^2 = 0.2242`
- train `MAE = 9.3111`
- train `RMSE = 11.9291`
- test `R^2 = 0.1319`
- test `MAE = 10.3263`
- test `RMSE = 12.8919`

Best valid feature-family ablation result:

- `player_def_rating_only`, tuned
- test `R^2 = 0.1132`
- test `MAE = 10.3952`
- test `RMSE = 13.0303`

What these metrics mean:

- the model is judged on how close its predicted lineup defensive outcome is to the stored historical target
- these are baseline outcome-model metrics, not direct scheme recommendation accuracy metrics
- the low `R^2` is likely partly structural because lineup-level NBA defensive outcomes are sparse and noisy

## Final Plots and Artifacts

Main training artifacts:

- `data/processed/feature_importance.csv`
- `data/processed/model_predictions.csv`

Main production-model plots:

- `data/processed/plots/actual_vs_predicted.png`
- `data/processed/plots/feature_importance_bar.png`
- `data/processed/plots/residuals_vs_predicted.png`
- `data/processed/plots/residual_distribution.png`
- `data/processed/plots/prediction_error_boxplot.png`
- `data/processed/plots/target_distribution.png`
- `data/processed/plots/top_feature_correlation_heatmap.png`
- `data/processed/plots/shap_summary.png`
- `data/processed/plots/shap_bar.png`

Feature experiment outputs:

- `data/processed/feature_experiment_results.csv`
- `data/processed/feature_experiment_importances.csv`
- `data/processed/plots/feature_experiment_results_summary.png`

## What Is Still Weak

Known limitations:

- no true public scheme labels
- no opponent-aware recommendation layer
- scheme profiles are still proxy-based, even when generated dynamically
- historical lineup targets remain noisy
- unseen custom lineups are weaker than exact historical matches

Largest unresolved weakness:

- label quality for the final recommendation problem is weaker than the mechanics of ingestion, feature engineering, or model training
