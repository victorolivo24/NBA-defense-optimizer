# Implementation Log

This document is the interview and presentation log for the project. It records what the system is trying to do, what was actually implemented, what failed, what was changed, and what the current metrics mean.

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

Interview version:

- The model predicts lineup defensive outcome.
- The recommendation layer is heuristic and simulation-based.
- The project is ML-assisted recommendation, not end-to-end supervised scheme classification.

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

- when a historical lineup target came from the fallback formula instead of a possession-based defensive rating field, the demo should label it as `Actual target (fallback per 48)`

## Inputs and Outputs

Primary input:

- five player names

Supporting inputs:

- season, for example `2024-25`
- optional existing lineup search or team filters for fallback demo mode

Feature inputs used by the model:

- lineup size
- guard, forward, and center counts
- average height and weight
- play-type feature aggregates such as:
  - `isolation_ppp_mean`
  - `pick_and_roll_ball_handler_ppp_mean`
  - `pick_and_roll_roll_man_ppp_mean`
  - `spot_up_ppp_mean`
  - possession and percentile summaries for each play type

Primary model output:

- `defensive_rating_target`

Alternative target:

- `opponent_ppp_target`

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

- framing is more honest and easier to defend academically

### 3. Demo originally selected historical lineups by minutes instead of taking user players

Observed issue:

- default demo was useful for presentation
- but it did not reflect the intended product workflow

Fix:

- add player-name lookup
- build synthetic lineup rows from five chosen players
- run the recommendation directly on that synthetic lineup

Result:

- the demo is now player-input driven

### 3b. Pure synthetic player-input lineups were not realistic enough

Observed issue:

- the first version of player-input mode built a synthetic lineup row only from the five players' defensive profiles
- that could produce unrealistic predictions for real lineups, especially when the exact 5-man unit already existed in the historical lineup data
- custom lineups also had no historical `actual target`, which made the output harder to trust and harder to explain

Why this happened:

- the synthetic row used player-level aggregates
- it did not try to reuse real lineup-level evidence from `lineup_metrics`
- so the model was forced to extrapolate from player traits alone

Fix:

- player-input mode now tries three strategies in order:
  1. exact 5-man historical lineup match
  2. weighted historical overlap blend from similar lineups
  3. pure synthetic player-profile fallback only if no meaningful lineup history exists

How the overlap blend works:

- parse the five input player IDs
- scan historical lineups for shared players
- ignore lineups with fewer than 2 shared players
- weight matches by overlap strength and lineup minutes

Current overlap weights:

- 5 shared players: `1.0`
- 4 shared players: `0.7`
- 3 shared players: `0.4`
- 2 shared players: `0.15`

Why we chose this direction:

- exact 5-man matches should dominate if they exist
- 4-man overlap is strong evidence
- 3-man overlap is usable but weaker
- 2-man overlap provides only light contextual evidence and should not dominate the estimate

What this improves:

- player-input recommendations now stay closer to real historical lineup behavior
- exact known lineups can surface a real `actual target`
- unseen lineups still work through the synthetic fallback, but only after historical evidence is exhausted

How this shows up in the demo:

- the output now includes a `Source` field
- common values include:
  - `exact_historical_lineup`
  - `weighted_historical_overlap_4_plus`
  - `weighted_historical_overlap_3`
  - `weighted_historical_overlap_2`
  - `synthetic_player_profile`

### 4. Training pipeline was too vulnerable to noise

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

Why this matters:

- it reduces noise from tiny samples
- it makes the training rows more stable
- it reduces the impact of extreme outliers during imputation

### 5. We initially pulled the wrong lineup endpoint view for the target

Observed issue:

- every lineup training row used the fallback target
- `288 / 288` training rows were labeled `fallback_points_allowed_per_48`
- `0 / 2000` stored lineup rows had possessions
- the demo showed inflated historical targets like `161.12` for some OKC lineups

Root cause:

- the ingestion code was calling `LeagueDashLineups` with `measure_type_detailed_defense='Defense'`
- that returned a box-score-style lineup table with fields like `MIN`, `PTS`, `FGM`, and `FGA`
- it did not return the advanced lineup fields we actually needed, such as `DEF_RATING`, `PACE`, and `POSS`

How we discovered it:

- we directly queried the endpoint in the repo environment with two parameter settings
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

Why this matters:

- it replaces a proxy target with a real API-provided lineup defensive rating whenever possible
- it should reduce extreme target values caused by the per-48 fallback formula

Post-fix verification:

- `python ingest.py` succeeded with the advanced lineup view
- rebuilt feature dataset size: `2060` rows
- rebuilt training dataset source breakdown:
  - `api_defensive_rating = 2060`
  - `fallback_points_allowed_per_48 = 0`
- database verification after the fix:
  - `3149` total lineup rows stored
  - `2000` lineup rows with possessions present from the advanced endpoint

Interpretation:

- the API does provide real lineup `DEF_RATING` and `POSS` when queried through the advanced lineup view
- the earlier fallback-only dataset was caused by using the wrong lineup endpoint configuration, not by an unavoidable API limitation

## Modeling Decisions

Baseline model:

- `XGBoostSchemeRecommender`

Why XGBoost:

- handles nonlinear interactions well
- reasonable for tabular features
- supports feature importance and SHAP explanation

What the model predicts:

- lineup defensive outcome, not scheme label

Target definition:

- default target: `defensive_rating_target`
- fallback target source: points allowed per 48 minutes when direct defensive rating is unavailable from the live lineup endpoint

Why this fallback exists:

- live lineup data did not reliably expose a direct defensive rating field in all cases

How we now surface this clearly:

- the engineered dataset includes `defensive_rating_target_source`
- values are labeled as either:
  - `api_defensive_rating`
  - `fallback_points_allowed_per_48`
- the demo output now prints that distinction directly instead of showing a generic `actual target`

## Recommendation Logic

Where the heuristic lives:

- `src/models/scheme_profiles.py`

Where the CLI entry point lives:

- `recommend_scheme.py`

How `recommend_scheme.py` works step by step:

1. It creates a database session using `DEFAULT_DATABASE_URL`.
2. It builds the full lineup training dataset for the selected season.
3. It trains the baseline XGBoost model by calling `train_baseline_regressor(...)`.
4. It tries to load `data/processed/dynamic_scheme_profiles.json`.
5. If that JSON exists, it passes those dynamic profiles into the recommendation layer.
6. If that JSON does not exist, it falls back to `DEFAULT_SCHEME_PROFILES` from `src/models/scheme_profiles.py`.
7. It selects one lineup row from the dataset, currently `dataset.iloc[0]`, as the example lineup.
8. It calls `recommend_scheme(lineup_row, artifacts, scheme_profiles=dynamic_profiles)`.
9. It prints:
   - the recommended scheme
   - the predicted target value
   - the ranked scheme table
   - the top explanation rows

Important current behavior:

- `recommend_scheme.py` is an example CLI, not the player-input demo.
- It uses an existing historical lineup from the dataset.
- The player-input path lives in `demo.py`.

How the actual recommendation function works:

1. `src/models/recommendation.py` converts the lineup row into a one-row DataFrame.
2. It makes sure the expected target columns exist.
3. It converts that row into the same numeric feature layout used by the trained model.
4. For each candidate scheme:
   - copy the baseline feature row
   - apply the scheme adjustments
   - run the trained model on the adjusted row
5. It stores one predicted value per scheme.
6. It sorts the schemes from lowest predicted defensive cost to highest.
7. It selects the top row as the final recommendation.
8. If SHAP is available on the model, it adds baseline SHAP values to the explanation output.

How recommendation works:

1. build or load a lineup feature row
2. run baseline prediction
3. apply additive scheme profile adjustments
4. rescore each scheme-adjusted lineup
5. choose the scheme with the lowest predicted value

Important caveat:

- the scheme profile adjustments are hand-set basketball heuristics
- they are not learned from labeled historical scheme data

Current profile source priority:

1. `data/processed/dynamic_scheme_profiles.json` if it exists
2. `src/models/scheme_profiles.py` otherwise

This means:

- the JSON file is an override
- the hardcoded Python profiles are the default fallback

How `apply_scheme_profile(...)` works:

- It does not blindly add flat deltas anymore.
- It uses `feature_mins` and `feature_maxs` from the training artifacts to compute a rough talent score for the lineup on each affected feature.
- Beneficial changes are scaled up for stronger lineups.
- Harmful changes are dampened for stronger lineups and hit weaker lineups more directly.
- If the needed percentile bounds are unavailable, it falls back to a simple additive adjustment.

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

Important caveat:

- this is still heuristic
- it is data-driven, but not truly supervised on scheme labels
- the script never observes actual possession-level `Drop`, `Switch`, or `Zone` calls
- instead, it uses basketball-informed proxy groups to estimate what those schemes should affect

### How Drop deltas are chosen

Function:

- `calculate_drop_deltas(...)`

Core assumption:

- Drop coverage is anchored by bigs who defend the roll man well

Proxy used:

- players with top-20-percent possession volume in `Pick and Roll Roll Man`

Why this proxy:

- those are the players most frequently involved in defending roll-man actions
- in practice, that tends to approximate the bigs who would matter most in a Drop scheme

What the script measures:

- `pick_and_roll_roll_man_ppp_mean`
  - compares the best quarter of those high-volume roll-man defenders to the average of that group
  - this is treated as the main Drop benefit
- `pick_and_roll_ball_handler_ppp_mean`
  - compares the worst quarter of those same defenders on ball-handler defense to the group average
  - this represents the idea that Drop can concede more pull-up or ball-handler comfort
- `spot_up_ppp_mean`
  - compares the worst quarter of those same defenders on Spot Up defense to the group average
  - this reflects the idea that Drop can create more kick-out or late-contest perimeter looks

Interpretation:

- Drop is modeled as helping the roll man matchup but potentially hurting ball-handler and spot-up outcomes

### How Switch deltas are chosen

Function:

- `calculate_switch_deltas(...)`

Core assumption:

- Switch-heavy defenses depend on perimeter defenders who can survive on ball handlers and in isolation

Proxy used:

- players with top-40-percent possession volume in `Pick and Roll Ball Handler`

Why this proxy:

- these players are treated as guards and wings who are repeatedly involved in point-of-attack defense

What the script measures:

- `isolation_ppp_mean`
  - compares the best quarter of those perimeter defenders in Isolation to the group average
  - this is the main Switch benefit
- `pick_and_roll_ball_handler_ppp_mean`
  - compares the best quarter on ball-handler defense to the group average
  - this represents the second main Switch benefit
- `pick_and_roll_roll_man_ppp_mean`
  - compares the worst quarter of those same defenders against Roll Man outcomes to the group average
  - this represents Switch creating interior or mismatch problems

Interpretation:

- Switch is modeled as helping isolation and ball-handler defense while risking roll-man mismatch damage

### How Zone deltas are chosen

Function:

- `calculate_zone_deltas(...)`

Core assumption:

- Zone is a team-level shell, so it should be inferred from lineup behavior rather than individual defender type

Proxy used:

- lineups with the highest spot-up possession volume
- specifically, the top 15 percent of lineups by `spot_up_possessions_mean`

Why this proxy:

- Zone tends to wall off penetration and concede more perimeter spot-up looks
- lineups that live in that kind of defensive profile are used as a stand-in for zone-like behavior

What the script measures:

- `isolation_ppp_mean`
  - compares those zone-proxy lineups to the average lineup
- `spot_up_ppp_mean`
  - compares those zone-proxy lineups to the average lineup
- `spot_up_percentile_mean`
  - compares those zone-proxy lineups to the average lineup

Interpretation:

- Zone is modeled as reducing direct isolation pressure while usually increasing perimeter spot-up exposure

### How the delta values are created

The script does not pick values from thin air.

For each relevant feature:

1. compute the average value for the proxy group
2. compute a stronger or weaker cohort inside that proxy group, usually the top or bottom quartile
3. subtract the group average from the cohort value
4. round the difference to 3 decimals

That difference becomes the scheme delta written into the JSON.

Example interpretation:

- if elite roll-man defenders allow meaningfully less PPP than the average high-volume roll-man defender, that gap becomes a negative Drop adjustment on `pick_and_roll_roll_man_ppp_mean`

### What "important to switch, zone, drop" means in this system

The script treats these features as the important ones:

- `Isolation`
- `Pick and Roll Ball Handler`
- `Pick and Roll Roll Man`
- `Spot Up`
- `Spot Up percentile`

Why these matter:

- they are the defensive play types we actually ingest from Synergy
- they map reasonably well to common coverage tradeoffs
- they are the clearest bridge between lineup personnel and scheme behavior

So the system is not saying:

- "these are the only things that matter in basketball"

It is saying:

- "given the public features we actually have, these are the most defensible levers for scheme simulation"

### Interview version

If asked how the JSON is generated:

- "We use a data-driven proxy script that derives scheme adjustments from the current season database. For Drop, we proxy bigs by high roll-man defensive volume. For Switch, we proxy perimeter defenders by high ball-handler defensive volume. For Zone, we proxy team shell behavior using lineups that face the most spot-up volume. We then compare stronger or weaker cohorts inside those proxy groups to league-average values and store those differences as scheme deltas in `dynamic_scheme_profiles.json`."

Why those heuristics were used:

- public data does not include scheme labels
- the project still needed an interpretable simulation layer for `Drop`, `Switch`, and `Zone`

## Evaluation and Metrics

Training metrics used:

- `MAE`
- `RMSE`

Recent verified training run after stricter filtering changes:

- after switching lineup ingestion to the advanced lineup view:
  - train rows: `1545`
  - test rows: `515`
  - `MAE = 14.5110`
  - `RMSE = 18.8088`

What these metrics mean:

- the model is judged on how close its predicted lineup defensive outcome is to the stored historical target
- these are baseline outcome-model metrics, not direct scheme recommendation accuracy metrics

## Verified Demo Example

Verified player-input demo run:

```bash
python demo.py --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns"
```

Current interpretation of the player-input path:

- if the exact lineup exists historically, the demo should label it as `Source: exact_historical_lineup`
- if the lineup is estimated from overlapping historical evidence, the demo should label it as one of:
  - `weighted_historical_overlap_4_plus`
  - `weighted_historical_overlap_3`
  - `weighted_historical_overlap_2`
- only if no useful lineup history exists should it label the result as `synthetic_player_profile`

What this demonstrates:

- the demo can resolve five named players from the database
- reuse exact historical lineup evidence when available
- build a weighted overlap estimate when the exact lineup has not played together
- fall back to a synthetic lineup profile only as a last resort
- score all candidate schemes
- return a recommendation with explanation rows

## What Is Still Weak

Known limitations:

- no true public scheme labels
- no possession-level coaching coverage ground truth
- heuristic scheme profiles are manually specified
- current recommendation evaluation is weaker than the baseline model evaluation
- historical lineup targets can still be noisy even after thresholding

If asked what the biggest weakness is:

- the biggest weakness is label quality for the final recommendation problem, not the mechanics of ingestion or model training

## Best Interview Answers

If asked "What does the ML do?":

- It predicts lineup-level defensive outcome, primarily defensive rating.

If asked "What does the heuristic do?":

- It simulates how candidate coverages might shift the lineup feature profile before rescoring.

If asked "Why not call it a classifier?":

- Because the public NBA API does not provide true scheme labels, so a supervised scheme classifier would overstate what the data supports.

If asked "Why did you add filtering thresholds?":

- Because tiny lineup samples and one-off play-type possessions create unstable targets and toxic outliers.

If asked "Why median imputation instead of mean?":

- Because the mean is more sensitive to extreme outliers, which was exactly the failure mode we were worried about.

## Recent Commit Milestones

Important milestones in repo history:

- `5af749e` `Fix live NBA API ingestion and proxy handling`
- `88da58d` `Add Phase 3 baseline training pipeline`
- `a3a630d` `score drop, switch, and zone, feature adjustments, rank schemes`
- `4dc97b5` `Extract scheme simulator profiles`
- `e2733a1` `Reframe project as simulation and recommendation engine`
- `7914ede` `add demo modules`
