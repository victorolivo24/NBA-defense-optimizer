# Codebase Guide: Dynamic Lineup-Dependent Defensive Scheme Optimizer

This guide provides a comprehensive overview of the `NBA-defense-optimizer` project, its architecture, data flows, and internal workings. It serves as the primary technical reference for engineers working on the system.

---

## 1. High-Level Overview

**Purpose:** 
The project is a prescriptive recommendation engine that shifts from traditional descriptive NBA analytics to simulated strategic decision support. Given a 5-man lineup, the system evaluates how well the lineup would perform defensively under different coverages (e.g., "Drop", "Switch", "Zone") and ranks them.

**How it works:**
Since true possession-level defensive scheme labels are not publicly available, the system acts as a simulator. It fetches raw player, play-type, and lineup defensive metrics from the `nba_api`, engineers cohesive lineup-level features, and trains a baseline XGBoost regression model to predict overall defensive outcomes (e.g., defensive rating). To recommend a scheme, it applies heuristic "adjustments" to the lineup's features based on the chosen scheme (simulating how "Drop" affects Pick-and-Roll defense, for example), scores the adjusted features through the baseline model, and ranks the results to minimize opponent points allowed.

---

## 2. Architecture

The project follows a standard Data Science pipeline layered architecture:

- **Ingestion / Database Layer:** Relational DB (SQLite) managed via SQLAlchemy, populated by fetching from the public `nba_api`.
- **Feature Layer:** Translates normalized relational data into a flat, numeric Pandas DataFrame ready for machine learning.
- **Modeling Layer:** A baseline regression model (XGBoost) trained on the feature dataset, wrapped in a generic interface.
- **Simulation / Recommendation Layer:** Applies explicit scheme profiles to feature inputs, runs them through the trained model, and uses SHAP to provide explainability.
- **Presentation Layer:** CLI scripts (`demo.py`, `recommend_scheme.py`) that serve as entry points.

---

## 3. Directory & File Breakdown

```text
NBA-defense-optimizer/
├── data/                       # Local data storage (ignored by git)
│   ├── raw/                    # Raw JSON responses dumped from nba_api
│   └── processed/              # SQLite database and compiled feature CSVs
├── src/
│   ├── database/               # Relational data layer
│   │   ├── connection.py       # Engine and session factory setup
│   │   ├── ingest.py           # NBA API fetching, backoff logic, and upsert routines
│   │   └── schema.py           # SQLAlchemy declarative ORM models
│   ├── features/               # Feature engineering logic
│   │   └── lineup_dataset.py   # Joins DB tables to calculate lineup-level metrics 
│   ├── models/                 # ML and simulation layer
│   │   ├── base.py             # BaseSchemeModel abstract interface
│   │   ├── recommendation.py   # Simulates features and scores recommendations 
│   │   ├── scheme_profiles.py  # Explicit numerical adjustments for Drop/Switch/Zone
│   │   ├── scheme_recommender.py # XGBoost implementation with SHAP explainability
│   │   └── training.py         # Data prep, train/test split, and evaluation logic
│   └── demo/                   # CLI presentation helpers
│       └── workflow.py         # Filtering, formatting, and executing demo case studies
├── ingest.py                   # CLI: Runs database ingestion phase
├── build_features.py           # CLI: Runs feature dataset export phase
├── train_model.py              # CLI: Trains baseline model and outputs metrics
├── recommend_scheme.py         # CLI: Tests a single recommendation
└── demo.py                     # CLI: Presentation-ready demo with multiple filtering options
```

---

## 4. Core Workflows

### Phase 1: Data Ingestion (`ingest.py`)
1. Database is initialized in SQLite.
2. `src.database.ingest` requests data from `nba_api` endpoints (`LeagueDashPlayerStats`, `SynergyPlayTypes`, `LeagueDashLineups`).
3. Raw JSON responses are timestamped and saved to `data/raw/` for reproducibility.
4. Pandas DataFrames are normalized and upserted into SQLAlchemy ORM models.

### Phase 2: Feature Engineering (`build_features.py`)
1. Queries the `LineupMetric` table and eagerly loads its 5 players and their defensive play-type metrics.
2. Calculates lineup aggregates (e.g., average height, count of guards, mean Pick-and-Roll PPP allowed).
3. Produces a flat, numeric `pd.DataFrame` containing the features and a surrogate target (`defensive_rating_target`).
4. Can export to `data/processed/lineup_training_dataset.csv`.

### Phase 3: Model Training (`train_model.py`)
1. Reads the numeric feature dataset.
2. Handles missing values (fills with mean/0) and extracts `features` and `target`.
3. Splits data into train/test sets and fits `XGBoostSchemeRecommender`.
4. Outputs standard regression metrics (MAE, RMSE) and saves feature importance.

### Phase 4: Recommendation (`demo.py` / `recommend_scheme.py`)
1. Takes a 5-man lineup row.
2. For each scheme in `scheme_profiles.py` (Drop, Switch, Zone), creates an adjusted copy of the lineup's features (e.g., Zone slightly worsens spot-up defense but improves isolation defense).
3. Evaluates all adjusted feature rows through the baseline model.
4. Ranks schemes based on the lowest predicted defensive rating (lower is better).
5. Explains the result using baseline SHAP values combined with the adjustment magnitudes.

---

## 5. Key Components

- **`XGBoostSchemeRecommender`**: Found in `src/models/scheme_recommender.py`. Implements `BaseSchemeModel`. It uses `XGBRegressor` and integrates `shap.TreeExplainer` for insights.
- **`recommend_scheme`**: The core function in `src/models/recommendation.py`. Orchestrates the application of adjustments and scoring.
- **`build_training_dataset`**: The main function in `src/features/lineup_dataset.py` that handles the heavy SQL joining and Python-side aggregation logic.
- **`_temporary_proxy_override`**: A context manager in `src/database/ingest.py` used to prevent broken environment variables from disrupting `nba_api` connections.

---

## 6. Data Layer

The database schema (`src/database/schema.py`) is structured as follows:

1. **`Player` (`players`)**: Core identity table (Name, Height, Weight, Position).
2. **`DefensivePlayType` (`defensive_play_types`)**: Granular metrics (e.g., Points Per Possession) for a specific player defending a specific play type (e.g., Isolation, Spot Up).
3. **`LineupMetric` (`lineup_metrics`)**: Represents a 5-man lineup. Stores historical outcomes like `minutes_played`, `possessions`, and the surrogate target `defensive_rating`.
4. **`LineupPlayer` (`lineup_players`)**: Association table connecting `LineupMetric` to exactly five `Player` records (distinguished by a `slot` column 1 through 5).

---

## 7. External Integrations

- **`nba_api`**: A Python client wrapping the undocumented `stats.nba.com` API. 
  - **Caution:** This API is aggressively rate-limited and heavily sensitive to connection issues.
- **SQLite**: Used for persistence; accessed exclusively via `sqlalchemy`.
- **SHAP**: Used to explain tree-based model decisions dynamically during recommendation.

---

## 8. Configuration & Environment

- Uses Python with a simple `requirements.txt`.
- Database URL defaults to `sqlite:///data/processed/nba_defense.sqlite`.
- Config classes (e.g., `IngestConfig`) manage the API fetching arguments and backoff parameters.
- Data outputs implicitly default to the `data/` folder relative to the project root.

---

## 9. Notable Patterns & Conventions

- **Safe Coercion:** Explicit helper functions like `_safe_float` and `_safe_int` are used to prevent API parsing crashes when endpoints return malformed strings or nulls.
- **Raw Storage Strategy:** The system intentionally caches the raw API output into JSON dumps. This avoids complete data loss if normalization logic breaks down the pipeline.
- **Dependency Injection:** Database connections are handled by passing around a `session_factory` instead of utilizing global or scoped sessions.
- **Swappable Architecture:** Explicit interfaces like `BaseSchemeModel` expect to handle future expansion from a simulator paradigm to true classification when better labels exist.
- **Proxy Workarounds:** Given the environment sensitivity of the API, `nba_api` calls are wrapped in a proxy-purging context manager to maintain stability.

---

## 10. Potential Pitfalls / Gotchas

- **Surrogate Target Limitation:** The model currently trains on the overall lineup defensive outcome (defensive rating) acting as a proxy. The simulation logic applies hardcoded heuristic feature adjustments to test schemes. It is a _simulator_, not a direct supervised classifier of real historical schemes.
- **NBA API Stability:** Requests will easily hang or block. Do not remove the `time.sleep()` calls and exponential backoff retry logic in `ingest.py`.
- **Enforcing 5 Players:** SQLite cannot elegantly enforce a strict "exactly 5 players per lineup" constraint dynamically. This validation is currently managed on the Python side (`build_synthetic_lineup_row` raises an error, and `parse_lineup_player_ids` checks length).
- **Silent Model Fallbacks:** Missing feature values in the testing pipeline are silently filled with means (or `0.0` as fallback). This can lead to silently "average" predictions if an underlying API endpoint goes blank.