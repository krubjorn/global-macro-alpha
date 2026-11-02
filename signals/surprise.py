"""Macro surprise momentum, point-in-time.

1. Each release: z = sign * (actual - survey) / sd of that ticker's PAST surprises
   (needs `min_releases` history; earlier releases are dropped, not back-filled).
2. Country x block score at each month-end = sum of z over releases already published,
   exponentially decayed by days since release (half-life ~3 months).
3. Map to assets:
     equity : + growth
     bond   : - growth - inflation
     FX     : + (growth_c - growth_US) + (inflation_c - inflation_US)   [hawkish-central-bank channel]
   Commodities get no macro score.
4. Time-series scaled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ts_scale


def standardise_releases(releases: pd.DataFrame, min_releases: int) -> pd.DataFrame:
    r = releases.dropna(subset=["release_dt", "actual", "survey"]).sort_values("release_dt").copy()
    r["surprise"] = r["actual"] - r["survey"]
    # past-only dispersion: expanding std up to the previous release of the same ticker
    sd = (r.groupby("ticker")["surprise"]
            .transform(lambda s: s.expanding(min_periods=min_releases).std().shift(1)))
    r["z"] = r["sign"] * r["surprise"] / sd
    return r.dropna(subset=["z"])


def scores(releases: pd.DataFrame, month_ends: pd.DatetimeIndex, halflife_days: float) -> pd.DataFrame:
    """Columns: MultiIndex (country, block). Values: decayed sum of z known at each month-end."""
    out = {}
    me = month_ends.values.astype("datetime64[D]").astype(np.int64)
    for (ctry, block), g in releases.groupby(["country", "block"]):
        t = g["release_dt"].values.astype("datetime64[D]").astype(np.int64)
        z = g["z"].to_numpy()
        age = me[:, None] - t[None, :]                        # days since release
        w = np.where(age >= 0, 0.5 ** (age / halflife_days), 0.0)
        out[(ctry, block)] = w @ z
    return pd.DataFrame(out, index=month_ends)


def compute(panel, params: dict, halflife_days: float | None = None) -> pd.DataFrame:
    cfg = params["signals"]["surprise"]
    hl = halflife_days or cfg["halflife_days"]
    month_ends = pd.date_range(panel.returns.index[0], panel.returns.index[-1], freq="ME")
    raw = pd.DataFrame(0.0, index=month_ends, columns=panel.meta.index)
    if panel.releases.empty:
        return raw * np.nan
    rel = standardise_releases(panel.releases, cfg["min_releases"])
    sc = scores(rel, month_ends, hl)

    def get(ctry, block):
        return sc[(ctry, block)] if (ctry, block) in sc else pd.Series(0.0, index=month_ends)

    for a, row in panel.meta.iterrows():
        c = row["country"]
        if row["cls"] == "equity":
            raw[a] = get(c, "growth")
        elif row["cls"] == "bond":
            raw[a] = -get(c, "growth") - get(c, "inflation")
        elif row["cls"] == "fx":
            raw[a] = (get(c, "growth") - get("US", "growth")
                      + get(c, "inflation") - get("US", "inflation"))
        else:
            raw[a] = np.nan
    first_live = sc.ne(0).idxmax().min() if not sc.empty else month_ends[0]
    raw = raw.loc[first_live:]
    return ts_scale(raw, params["signals"]["clip_z"])
