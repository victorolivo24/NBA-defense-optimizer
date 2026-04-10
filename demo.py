"""CLI entry point for a presentation-ready lineup recommendation demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.demo import (
    build_default_case_studies,
    format_demo_result,
    make_custom_lineup_demo_frame,
    run_demo_for_lineups,
    select_lineups,
    train_demo_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lineup-aware defensive scheme recommendation demo.",
    )
    parser.add_argument("--season", default="2024-25", help="Season string to load, for example 2024-25.")
    parser.add_argument(
        "--players",
        nargs="+",
        help='Five player names, for example --players "Jalen Brunson" "Josh Hart" "Mikal Bridges" "OG Anunoby" "Karl-Anthony Towns".',
    )
    parser.add_argument("--team", help="Filter to one team abbreviation, for example BOS.")
    parser.add_argument("--search", help="Filter lineup_name by a player or lineup substring.")
    parser.add_argument("--lineup-key", help="Select one exact lineup_key from the training dataset.")
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=10.0,
        help="Minimum lineup minutes required for default case studies or filters.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of matching lineups to display when filters are used.",
    )
    parser.add_argument(
        "--hardcoded",
        action="store_true",
        help="Force the use of original hardcoded scheme profiles instead of dynamic ones.",
    )
    return parser


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    args = build_parser().parse_args()
    
    dynamic_profiles = None
    if not args.hardcoded:
        try:
            path = Path("data/processed/dynamic_scheme_profiles.json")
            if path.exists():
                with open(path, "r") as f:
                    dynamic_profiles = json.load(f)
                print(f"Loaded dynamic scheme profiles from {path}.\n")
        except Exception as e:
            print(f"Warning: Failed to load dynamic profiles ({e}). Falling back to hardcoded defaults.\n")
            
    dataset, artifacts = train_demo_artifacts(season=args.season)

    case_label_column = None
    has_filters = any([args.lineup_key, args.team, args.search])

    if args.players:
        selected = make_custom_lineup_demo_frame(args.players, season=args.season)
        case_label_column = "case_label"
    elif has_filters:
        selected = select_lineups(
            dataset,
            lineup_key=args.lineup_key,
            team=args.team,
            search=args.search,
            min_minutes=args.min_minutes,
            limit=args.limit,
        )
    else:
        selected = build_default_case_studies(dataset, min_minutes=args.min_minutes)
        case_label_column = "case_label"
        if not args.hardcoded and dynamic_profiles:
            print("No player input supplied. Showing default case studies.\n")
        else:
            print("No player input supplied. Showing default case studies.\n")

    if selected.empty:
        selected = build_default_case_studies(dataset, min_minutes=args.min_minutes)
        case_label_column = "case_label"
        print("No matching lineups found. Falling back to default case studies.\n")

    for result in run_demo_for_lineups(selected, artifacts, case_label_column=case_label_column, scheme_profiles=dynamic_profiles):
        print(format_demo_result(result))
        print("\n" + "=" * 80 + "\n")
