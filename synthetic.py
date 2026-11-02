"""Synthetic panel with KNOWN embedded effects, for testing the pipeline end-to-end.

What is planted (so you can check the pipeline finds it, and finds nothing where nothing is):
  * trend      : slow AR(1) drift in expected returns            -> TSMOM should work
  * carry      : expected return loads on published carry         -> carry should work
  * macro      : equity/bond/FX returns load on growth surprises  -> surprise signal should work
  * ENSO       : exposed assets load on last month's forecast revision -> climate sleeve should work
  * value, COT : pure noise                                       -> should NOT pass the gates

Never report numbers from this data. It exists to prove the plumbing, nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import Panel

ANN_VOL = {"equity": 0.18, "bond": 0.06, "fx_g10": 0.09, "fx_em": 0.12,
           "commodity_energy_metals": 0.28, "commodity_softs_ags": 0.25}


def make_panel(meta: pd.DataFrame, macro_spec: dict, start: str = "1995-01-02",
               end: str = "2026-09-30", seed: int = 11) -> Panel:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    months = pd.date_range(days[0], days[-1], freq="ME")
    n_m, assets = len(months), list(meta.index)
    vol_m = np.array([ANN_VOL.get(meta.at[a, "cost_bucket"], 0.15) for a in assets]) / np.sqrt(12)

    # ---- latent states (monthly) --------------------------------------------------------
    trend = _ar1(rng, n_m, len(assets), phi=0.95, sd=0.12)          # in units of monthly vol
    carry_true = _ar1(rng, n_m, len(assets), phi=0.97, sd=0.25)
    countries = sorted({c for c in meta["country"].dropna()} | set(macro_spec))
    growth = pd.DataFrame(_ar1(rng, n_m, len(countries), phi=0.8, sd=0.6), index=months, columns=countries)
    enso = np.zeros(n_m)
    enso_innov = rng.normal(0, 0.35, n_m)
    for t in range(1, n_m):
        enso[t] = 0.9 * enso[t - 1] + enso_innov[t]

    # ---- expected next-month returns from information at month-end t --------------------
    mu = np.zeros((n_m, len(assets)))
    cls = meta["cls"].to_numpy()
    ctry = meta["country"].to_numpy()
    prior = meta["enso_prior"].to_numpy()
    for t in range(1, n_m):
        mu[t] += 0.30 * trend[t - 1]                               # trend persistence
        mu[t] += 0.06 * carry_true[t - 1]                          # carry premium
        g = np.array([growth.iloc[t - 1].get(c, 0.0) for c in ctry])
        g_us = growth.iloc[t - 1].get("US", 0.0)
        mu[t] += np.where(cls == "equity", 0.08 * g, 0)
        mu[t] += np.where(cls == "bond", -0.08 * g, 0)
        mu[t] += np.where(cls == "fx", 0.06 * (g - g_us), 0)
        mu[t] += 0.30 * prior * enso_innov[t - 1]                  # slow pricing of ENSO news
    mu *= vol_m                                                    # convert to return units

    # ---- daily returns ------------------------------------------------------------------
    month_of_day = np.searchsorted(months.values, days.values)
    month_of_day = np.minimum(month_of_day, n_m - 1)
    n_d = len(days)
    # correlated shocks: one factor per asset class
    class_factor = {c: rng.standard_normal(n_d) for c in np.unique(cls)}
    eps = rng.standard_t(df=5, size=(n_d, len(assets))) / np.sqrt(5 / 3)
    shocks = np.column_stack([
        0.6 * class_factor[cls[j]] + 0.8 * eps[:, j] for j in range(len(assets))
    ])
    daily = mu[month_of_day] / 21 + shocks * vol_m / np.sqrt(21)
    returns = pd.DataFrame(daily, index=days, columns=assets)
    # staggered start dates, like real data (EM FX and some futures start later)
    for j, a in enumerate(assets):
        if meta.at[a, "cost_bucket"] == "fx_em":
            returns.loc[: "1999-12-31", a] = np.nan
        elif a in ("KM", "XM", "VG"):
            returns.loc[: "1997-12-31", a] = np.nan

    # ---- published inputs ---------------------------------------------------------------
    carry = pd.DataFrame(carry_true * 0.02 + rng.normal(0, 0.002, (n_m, len(assets))),
                         index=months, columns=assets)
    value = pd.DataFrame(_ar1(rng, n_m, len(assets), phi=0.98, sd=0.2), index=months, columns=assets)
    value.loc[:, meta.index[meta["cls"] == "commodity"]] = np.nan   # computed from prices instead

    releases = _releases(rng, growth, macro_spec)
    cot = _cot(rng, assets, days)
    plume = _plume(rng, months, enso)
    panel = Panel(returns, carry.where(returns.resample("ME").last().notna()),
                  value, releases, cot, plume, meta, qa=["SYNTHETIC DATA - plumbing test only"])
    return panel


def _ar1(rng, n, k, phi, sd):
    x = np.zeros((n, k))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sd, k)
    return x


def _releases(rng, growth: pd.DataFrame, macro_spec: dict) -> pd.DataFrame:
    rows = []
    for country, blocks in macro_spec.items():
        for block, series in blocks.items():
            for s in series:
                lag_days = int(rng.integers(3, 25))
                for ref, g in growth.get(country, pd.Series(0.0, index=growth.index)).items():
                    signal = g if block == "growth" else 0.3 * g
                    surprise = s["sign"] * (signal + rng.normal(0, 1.0))
                    survey = rng.normal(0, 1)
                    rows.append({"ticker": s["t"], "country": country, "block": block,
                                 "sign": s["sign"], "ref_date": ref,
                                 "release_dt": ref + pd.Timedelta(days=lag_days),
                                 "actual": survey + surprise, "survey": survey})
    return pd.DataFrame(rows)


def _cot(rng, assets, days) -> pd.DataFrame:
    tuesdays = days[days.weekday == 1]
    rows = []
    for a in assets:
        x = 0.0
        for d in tuesdays:
            x = 0.97 * x + rng.normal(0, 0.03)
            rows.append({"report_date": d, "asset": a, "net_spec_oi": x})
    return pd.DataFrame(rows)


def _plume(rng, months, enso) -> pd.DataFrame:
    rows = []
    for t, m in enumerate(months):
        issue = m.replace(day=19)
        for lead in range(1, 10):
            for model in ("dyn_a", "dyn_b", "stat_c"):
                f = enso[t] * 0.9 ** lead + rng.normal(0, 0.15)
                rows.append({"issue_date": issue, "lead_m": lead, "model": model, "nino34": f})
    return pd.DataFrame(rows)
