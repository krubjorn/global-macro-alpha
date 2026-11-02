"""Carry (Koijen, Moskowitz, Pedersen & Vrugt 2018): return earned if prices do not move.

Inputs are built in clean.py: FX forward discount, commodity roll yield, bond yield minus funding
plus roll-down, equity dividend yield minus short rate. Ranked within asset class.
"""
from __future__ import annotations

import pandas as pd

from .base import xs_rank


def compute(panel, params: dict) -> pd.DataFrame:
    k = params["signals"]["carry"]["smooth_m"]
    raw = panel.carry.rolling(k, min_periods=1).mean() if k > 1 else panel.carry
    return xs_rank(raw, panel.meta, params["signals"]["clip_z"])
