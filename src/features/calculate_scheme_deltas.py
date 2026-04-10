"""Script to calculate data-driven defensive scheme adjustments."""

import json
import sys
from pathlib import Path

# Add project root to python path so this script can be run directly
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
from sqlalchemy import select

from src.database.connection import DEFAULT_DATABASE_URL, create_session_factory
from src.database.schema import DefensivePlayType, Player
from src.features.lineup_dataset import build_training_dataset


def _get_player_play_types(session, season: str) -> pd.DataFrame:
    """Fetch player play-type data."""
    query = (
        select(
            Player.nba_player_id, 
            Player.full_name, 
            DefensivePlayType.play_type, 
            DefensivePlayType.ppp_allowed, 
            DefensivePlayType.possessions
        )
        .join(DefensivePlayType.player)
        .where(DefensivePlayType.season == season)
        .where(DefensivePlayType.possessions >= 10)  # Basic volume filter
    )
    return pd.read_sql(query, session.bind)


def calculate_drop_deltas(session, season: str) -> dict[str, float]:
    """
    Calculate dynamic adjustments for Drop coverage.
    
    Why this proxy?
    In Drop coverage, the primary rim protector (typically a Center) sags back to defend the Roll Man.
    Since positional data isn't perfectly populated, we proxy "Centers" by finding the players
    who defend the highest volume of "Pick and Roll Roll Man" possessions (top 20%).
    
    Proxy logic:
    - pick_and_roll_roll_man_ppp_mean: Difference between the top 25% (best PPP) of these high-volume Roll Man defenders
      and the average PPP of that same cohort.
    - pick_and_roll_ball_handler_ppp_mean: Difference between the bottom 25% (worst PPP) of these same defenders
      and the average PPP (Drop bigs often concede pull-up jumpers to ball handlers).
    - spot_up_ppp_mean: Difference between the bottom 25% and average on Spot-Ups (Drop schemes often result in late contests on skips).
    """
    df = _get_player_play_types(session, season)
    
    # Proxy for Centers: Players with top 20% volume of Roll Man possessions
    roll_man_df = df[df['play_type'] == 'Pick and Roll Roll Man'].copy()
    if roll_man_df.empty:
        return {"pick_and_roll_roll_man_ppp_mean": 0.0, "pick_and_roll_ball_handler_ppp_mean": 0.0, "spot_up_ppp_mean": 0.0}
        
    threshold = roll_man_df['possessions'].quantile(0.80)
    big_ids = roll_man_df[roll_man_df['possessions'] >= threshold]['nba_player_id']
    
    bigs_df = df[df['nba_player_id'].isin(big_ids)]
    
    def get_delta(play_type: str, top: bool) -> float:
        pt_df = bigs_df[bigs_df['play_type'] == play_type]['ppp_allowed'].dropna()
        if len(pt_df) < 5:
            return 0.0
        avg_ppp = pt_df.mean()
        if top:
            cohort_ppp = pt_df[pt_df <= pt_df.quantile(0.25)].mean()
        else:
            cohort_ppp = pt_df[pt_df >= pt_df.quantile(0.75)].mean()
        return round(cohort_ppp - avg_ppp, 3)

    return {
        "pick_and_roll_roll_man_ppp_mean": get_delta("Pick and Roll Roll Man", top=True),
        "pick_and_roll_ball_handler_ppp_mean": get_delta("Pick and Roll Ball Handler", top=False),
        "spot_up_ppp_mean": get_delta("Spot Up", top=False),
    }


def calculate_switch_deltas(session, season: str) -> dict[str, float]:
    """
    Calculate dynamic adjustments for Switch coverage.
    
    Why this proxy?
    Switching defenses aim to keep bodies in front of Ball Handlers and eliminate Isolation advantages, 
    but often create physical mismatches (e.g., smaller guards stuck on big men rolling to the rim).
    
    Proxy logic:
    We proxy Guards/Wings by finding players with the highest volume (top 40%) of "Pick and Roll Ball Handler" possessions.
    - isolation_ppp_mean: The difference in Isolation PPP between the best 25% of these perimeter defenders and the average.
    - pick_and_roll_ball_handler_ppp_mean: The difference in Ball Handler PPP between the best 25% of these defenders and the average.
    - pick_and_roll_roll_man_ppp_mean: The difference in Roll Man PPP between the worst 25% of these perimeter defenders (representing a mismatch) and the average.
    """
    df = _get_player_play_types(session, season)
    
    # Proxy for Guards/Wings
    ball_handler_df = df[df['play_type'] == 'Pick and Roll Ball Handler'].copy()
    if ball_handler_df.empty:
        return {"isolation_ppp_mean": 0.0, "pick_and_roll_ball_handler_ppp_mean": 0.0, "pick_and_roll_roll_man_ppp_mean": 0.0}
        
    threshold = ball_handler_df['possessions'].quantile(0.60) # Top 40%
    perimeter_ids = ball_handler_df[ball_handler_df['possessions'] >= threshold]['nba_player_id']
    
    perimeter_df = df[df['nba_player_id'].isin(perimeter_ids)]
    
    def get_delta(play_type: str, top: bool) -> float:
        pt_df = perimeter_df[perimeter_df['play_type'] == play_type]['ppp_allowed'].dropna()
        if len(pt_df) < 5:
            return 0.0
        avg_ppp = pt_df.mean()
        if top:
            cohort_ppp = pt_df[pt_df <= pt_df.quantile(0.25)].mean()
        else:
            cohort_ppp = pt_df[pt_df >= pt_df.quantile(0.75)].mean()
        return round(cohort_ppp - avg_ppp, 3)

    return {
        "isolation_ppp_mean": get_delta("Isolation", top=True),
        "pick_and_roll_ball_handler_ppp_mean": get_delta("Pick and Roll Ball Handler", top=True),
        "pick_and_roll_roll_man_ppp_mean": get_delta("Pick and Roll Roll Man", top=False),
    }


def calculate_zone_deltas(session_factory, season: str) -> dict[str, float]:
    """
    Calculate dynamic adjustments for Zone coverage.
    
    Why this proxy?
    Zone defenses naturally wall off the paint and limit Isolation and Pick and Roll penetration, 
    but they intentionally concede Spot-Up three-pointers due to the defensive shell's geometry.
    
    Proxy logic:
    Since zone is a team-level concept, we look at lineup-level data. Teams facing a Zone 
    typically face a very high frequency of Spot-Up shots.
    - We isolate the lineups that face the highest volume (top 15%) of Spot-Up possessions.
    - spot_up_ppp_mean: The difference in Spot-Up PPP allowed by these "Zone-like" lineups vs the league average lineup.
    - isolation_ppp_mean: The difference in Isolation PPP allowed by these same lineups vs the league average lineup.
    - spot_up_percentile_mean: The difference in Spot-Up percentile for these lineups vs average.
    """
    df = build_training_dataset(session_factory, season=season, min_minutes=20.0)
    
    if df.empty or 'spot_up_possessions_mean' not in df.columns:
        return {"isolation_ppp_mean": -0.03, "spot_up_ppp_mean": 0.06, "spot_up_percentile_mean": -4.0}
        
    # High spot-up volume lineups are our proxy for zone
    threshold = df['spot_up_possessions_mean'].quantile(0.85)
    zone_proxies = df[df['spot_up_possessions_mean'] >= threshold]
    
    if zone_proxies.empty:
        return {"isolation_ppp_mean": -0.03, "spot_up_ppp_mean": 0.06, "spot_up_percentile_mean": -4.0}
    
    def get_lineup_delta(column: str) -> float:
        if column not in df.columns:
            return 0.0
        avg_val = df[column].mean()
        zone_val = zone_proxies[column].mean()
        return round(zone_val - avg_val, 3)

    return {
        "isolation_ppp_mean": get_lineup_delta("isolation_ppp_mean"),
        "spot_up_ppp_mean": get_lineup_delta("spot_up_ppp_mean"),
        "spot_up_percentile_mean": get_lineup_delta("spot_up_percentile_mean"),
    }


def main():
    season = "2024-25"
    session_factory = create_session_factory(DEFAULT_DATABASE_URL)
    session = session_factory()
    
    try:
        print("Calculating Drop deltas...")
        drop_deltas = calculate_drop_deltas(session, season)
        
        print("Calculating Switch deltas...")
        switch_deltas = calculate_switch_deltas(session, season)
        
        print("Calculating Zone deltas...")
        zone_deltas = calculate_zone_deltas(session_factory, season)
        
        profiles = {
            "Drop": drop_deltas,
            "Switch": switch_deltas,
            "Zone": zone_deltas,
        }
        
        output_path = Path("data/processed/dynamic_scheme_profiles.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(profiles, f, indent=4)
            
        print(f"\nSuccessfully wrote dynamic scheme profiles to {output_path}")
        print(json.dumps(profiles, indent=4))
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
