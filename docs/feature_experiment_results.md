# Feature Experiment Results

This document records the branch-level feature experiments run to test whether the model can be improved by replacing or supplementing the aggregated player defensive rating features.

## Quick Summary

If you only read one conclusion from this document, it is this:

- aggregated player defensive rating features remained the strongest valid predictive signal
- player basic defensive box-score features helped, but did not beat the rating-driven core
- Synergy play-type features were weaker than expected on their own
- broader feature sets did not materially lift `R^2`
- the low `R^2` appears to be partly structural because lineup-level defensive outcomes are sparse and noisy

Main result files:

- `data/processed/feature_experiment_results.csv`
- `data/processed/feature_experiment_importances.csv`
- `data/processed/plots/feature_experiment_results_summary.png`

## Goal

The main concern on this branch was that feature importance was being driven too heavily by:

- `avg_player_def_rtg`
- `best_player_def_rtg`
- `worst_player_def_rtg`

The objective was to test whether alternative feature families could outperform those variables or at least materially improve `R^2`, `MAE`, and `RMSE`.

## Experimental Design

Target:

- `defensive_rating_target`

Seasons:

- `2022-23`
- `2023-24`
- `2024-25`

Comparison approach:

- build a combined multiseason lineup dataset
- engineer additional player basic defensive box-score features from the player defense endpoint
- engineer optional lineup four-factor style features from the basic lineup endpoint
- compare multiple feature families under the same train/test split

Two experiment modes were used:

- `fixed`
  - same untuned XGBoost configuration for feature-family comparison
- `tuned`
  - hyperparameter search enabled for valid production feature sets

Important validity distinction:

- player-level defensive rating, player box-score defensive features, and Synergy play-type aggregates are valid pre-lineup predictors
- same-stint lineup four-factor features are not valid for production recommendation, because they are derived from the lineup's observed box-score outcome and therefore act as a leakage control rather than a deployable predictor

## Literature and Rationale

The tested feature families were motivated by three strands of prior work:

1. Four Factors and efficiency
- Dean Oliver's Four Factors remain the standard box-score decomposition of basketball efficiency.
- The recent paper *Dean Oliver's Four Factors Revisited* explicitly studies how four factors relate to offensive, defensive, and net rating, and notes that the relationship is non-linear.
- Source: https://arxiv.org/abs/2305.13032

2. Unseen lineup prediction from player-level summaries
- *Predicting Elite NBA Lineups Using Individual Player Order Statistics* shows that lineup performance can be predicted from individual player information even when the lineup has never played together.
- Source: https://arxiv.org/abs/2303.04963

3. Sparse and noisy lineup data
- *Lineup Regularized Adjusted Plus-Minus (L-RAPM)* highlights that NBA lineups are highly sparse, with the average lineup seeing only about `25-30` possessions, which naturally limits predictive power.
- Source: https://arxiv.org/abs/2601.15000

These sources motivated three different feature directions on this branch:

- defensive rating aggregates
- player box-score defensive features
- four-factor style lineup features

## Engineered Feature Families

Added on this branch:

1. Player basic defensive box-score features
- aggregated from `player_defense`
- includes:
  - `dreb`
  - `stl`
  - `blk`
  - `def_ws`
  - `opp_pts_off_tov`
  - `opp_pts_2nd_chance`
  - `opp_pts_fb`
  - `opp_pts_paint`

2. Lineup four-factor style features
- derived from `lineup_defense_basic`
- includes:
  - `lineup_opp_efg_pct`
  - `lineup_opp_tov_rate`
  - `lineup_opp_orb_rate`
  - `lineup_opp_fta_rate`

3. Refreshable season caches
- `data/processed/player_defense_profiles.json`
- `data/processed/lineup_basic_defense_profiles.json`

These were generated for:

- `2022-23`
- `2023-24`
- `2024-25`

## Results

The CSV stores one row per experiment run.
The importance file stores per-feature importances for each experiment and model mode.

### Best valid runs by test R2

| Experiment | Mode | Test R2 | Test MAE | Test RMSE | Notes |
|---|---:|---:|---:|---:|---|
| `player_def_rating_only` | tuned | `0.1132` | `10.3952` | `13.0303` | best valid feature family |
| `current_baseline_like` | tuned | `0.1119` | `10.4613` | `13.0401` | player defensive ratings plus Synergy play types |
| `player_basic_plus_advanced` | fixed | `0.1095` | `10.6548` | `13.0577` | best untuned mixed feature set |
| `player_basic_plus_advanced` | tuned | `0.1064` | `10.4586` | `13.0802` | box-score defensive features help, but do not beat rating-only |
| `all_valid_features` | tuned | `0.1026` | `10.5556` | `13.1083` | more features did not outperform the leaner tuned sets |

### Weakest valid runs

| Experiment | Mode | Test R2 | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| `play_types_only` | fixed | `-0.0442` | `11.6303` | `14.1397` |
| `play_types_only` | tuned | `0.0229` | `11.0697` | `13.6776` |

### Leakage controls

| Experiment | Mode | Test R2 | Interpretation |
|---|---:|---:|---|
| `four_factors_leakage_control` | fixed | `-0.0035` | four-factor ratios alone were not enough to reconstruct defensive rating well |
| `all_features_plus_four_factors_leakage_control` | fixed | `0.0958` | adding same-stint four-factor inputs did not beat the best tuned valid feature sets |

### Standard tuned branch run

The standard `train_model.py` run on this branch, with the expanded engineered feature set and tuning enabled, produced:

- test `R^2 = 0.1319`
- test `MAE = 10.3263`
- test `RMSE = 12.8919`

That is the strongest tuned end-to-end branch result so far.

## Why Each Attempt Was Run

`player_def_rating_only`
- Why we tried it:
  - feature importance in the baseline pipeline was already heavily concentrated in player defensive rating aggregates
- What happened:
  - it produced the best valid feature-family result
- Conclusion:
  - those aggregates remain the strongest compact signal for this target

`current_baseline_like`
- Why we tried it:
  - this was the original richer production-style set using player defensive rating plus Synergy play-type features
- What happened:
  - it nearly tied the best result, but did not clearly beat the leaner rating-only set
- Conclusion:
  - Synergy features add context, but not enough to overturn the rating-driven core

`play_types_only`
- Why we tried it:
  - basketball logic suggests that Isolation, Ball Handler, Roll Man, and Spot Up defense should matter for coverage choice
- What happened:
  - it was the weakest valid feature family
- Conclusion:
  - public Synergy defensive summaries are not strong enough by themselves to predict lineup defensive rating well

`player_basic_only`
- Why we tried it:
  - steals, blocks, rebounds, defensive win shares, and opponent scoring splits are intuitive defensive signals
- What happened:
  - it was more useful than play-type-only features, but weaker than the defensive-rating-driven sets
- Conclusion:
  - box-score defense helps, but does not replace the advanced defensive rating aggregates

`player_basic_plus_advanced`
- Why we tried it:
  - combine intuitive box-score defensive features with player defensive rating to see whether the mix can outperform the leaner set
- What happened:
  - this was competitive, but still slightly below the best rating-only result
- Conclusion:
  - box-score features are reasonable supplements, not clear replacements

`all_valid_features`
- Why we tried it:
  - test whether the full combined set benefits from having more information
- What happened:
  - it did not outperform the leaner tuned sets
- Conclusion:
  - more features added redundancy and noise rather than a clear gain

`four_factors_leakage_control`
- Why we tried it:
  - literature strongly links four factors to efficiency, so it was a useful stress test
- What happened:
  - it did not perform well in this setup
- Conclusion:
  - in this project, same-stint four-factor features were not useful as a standalone predictor and are not valid deployment features anyway

## What the Results Suggest

### 1. Aggregated player defensive rating still carries the most signal

The strongest valid tuned experiment was:

- `player_def_rating_only`

Its top importances were:

- `best_player_def_rtg`
- `avg_player_def_rtg`
- `worst_player_def_rtg`

This confirms the original concern: those features remain highly influential.

### 2. Player basic defensive box-score features help, but do not replace defensive rating

Adding player-level defensive box-score features such as:

- steals
- blocks
- defensive rebounds
- defensive win shares
- opponent points off turnovers / second chance / fast break / paint

did improve some models, especially:

- `player_basic_plus_advanced`

However, the box-score-only version still underperformed the tuned rating-only model.

### 3. Synergy play-type features do not carry enough signal by themselves

The weakest valid runs were:

- `play_types_only`

This suggests that public Synergy defensive play-type summaries are useful as supporting context, but not strong enough alone to predict lineup defensive rating well.

### 4. More features is not automatically better

The fully combined valid set:

- `all_valid_features`

did not outperform the leaner tuned feature families. This suggests feature redundancy and noise are real issues in this problem.

### 5. Low R2 is at least partly structural, not just a modeling mistake

The lineup literature and the branch results point to the same conclusion:

- lineup data is sparse
- lineup defensive outcomes are noisy
- unseen or rare combinations are hard to predict well from public data alone

The fact that even broader valid feature sets do not materially lift `R^2` above the low-`0.10s` range supports the argument that the public-data ceiling may be low for this target.

## Feature Importance Follow-Up

The tuned branch run from `train_model.py` produced a more mixed top-10 than the earlier branch state:

- `best_player_def_rtg`
- `worst_player_def_rtg`
- `avg_player_def_ws`
- `avg_player_opp_pts_fb`
- `isolation_percentile_mean`
- `lineup_opp_orb_rate`
- `pick_and_roll_ball_handler_percentile_min`
- `avg_player_def_rtg`
- `pick_and_roll_roll_man_ppp_mean`
- `avg_player_blk`

That means the model is still influenced strongly by player defensive rating aggregates, but the new branch features are now contributing real signal.

## Files Added or Updated for This Branch

New or updated branch artifacts:

- `refresh_feature_caches.py`
- `run_feature_experiments.py`
- `data/processed/player_defense_profiles.json`
- `data/processed/lineup_basic_defense_profiles.json`
- `data/processed/feature_experiment_results.csv`
- `data/processed/feature_experiment_importances.csv`

Code paths updated:

- `src/database/ingest.py`
- `src/features/lineup_dataset.py`
- `src/models/scheme_recommender.py`
- `src/models/training.py`

## Current Conclusion

Based on the branch experiments so far:

- the strongest valid predictive signal still comes from aggregated player defensive rating features
- player basic defensive box-score features are useful supplements
- Synergy play-type features help less than expected and do not perform well on their own
- four-factor style lineup outcome features are not suitable as production predictors in this setup and did not become dominant even as leakage controls
- the public-data ceiling for this target appears low, which is consistent with the sparse and noisy nature of lineup-level NBA defensive outcomes
