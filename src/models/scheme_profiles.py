"""Explicit scheme simulator profiles used by the recommendation layer."""

from __future__ import annotations
# coaches adjust scheme profile

DEFAULT_SCHEME_PROFILES: dict[str, dict[str, float]] = {
    "Drop": {
        "pick_and_roll_roll_man_ppp_mean": -0.08,
        "pick_and_roll_ball_handler_ppp_mean": 0.03,
        "spot_up_ppp_mean": 0.02,
    },
    "Switch": {
        "isolation_ppp_mean": -0.05,
        "pick_and_roll_ball_handler_ppp_mean": -0.04,
        "pick_and_roll_roll_man_ppp_mean": 0.05,
    },
    "Zone": {
        "isolation_ppp_mean": -0.03,
        "spot_up_ppp_mean": 0.06,
        "spot_up_percentile_mean": -4.0,
    },
}
