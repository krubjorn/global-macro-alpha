"""Climate sleeve: ENSO forecast revisions.

    F_t   = mean Nino-3.4 forecast over models and leads in `leads`, from the plume issued in month t
    dF_t  = F_t - F_{t-1}                       (the news, not the level)
    s_i,t = b_i * dF_t / rms(dF)_past

mode = "prior"     : b_i = enso_prior from tickers.yaml (no estimation, no look-ahead)
mode = "estimated" : b_i = tanh(t-stat / 2) of an expanding regression of r_{i,tau+1} on dF_tau,
                     using only pairs with tau + 1 <= t. Assets with a zero prior are left out in
                     both modes so the sleeve stays a focused, explainable bet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import monthly_returns


def revision_series(plume: pd.DataFrame, leads: list[int]) -> pd.Series:
    p = plume[plume["lead_m"].isin(leads)].copy()
    p["issue_date"] = pd.to_datetime(p["issue_date"])
    f = p.groupby("issue_date")["nino34"].mean().sort_index()
    f.index = f.index + pd.offsets.MonthEnd(0)              # usable at the issue month's end
    f = f.groupby(level=0).last()
    return f.diff().rename("dF")


def compute(panel, params: dict, leads: list[int] | None = None, mode: str | None = None) -> pd.DataFrame:
    cfg = params["signals"]["enso"]
    leads = leads or cfg["leads"]
    mode = mode or cfg["mode"]
    exposed = panel.meta.index[panel.meta["enso_prior"] != 0]
    out = pd.DataFrame(np.nan, index=pd.date_range(panel.returns.index[0], panel.returns.index[-1],
                                                    freq="ME"), columns=panel.meta.index)
    if panel.plume.empty:
        return out
    dF = revision_series(panel.plume, leads).reindex(out.index)
    rms = np.sqrt((dF ** 2).expanding(min_periods=24).mean().shift(1))
    dz = (dF / rms).clip(-3, 3)

    if mode == "prior":
        b = panel.meta.loc[exposed, "enso_prior"].astype(float)
        for a in exposed:
            out[a] = dz * b[a]
    else:
        nxt = monthly_returns(panel.returns[exposed]).reindex(out.index).shift(-1)
        n_min = cfg["min_months_beta"]
        for a in exposed:
            x, y = dz.to_numpy(), nxt[a].to_numpy()
            for i, t in enumerate(out.index):
                # pairs (dF_tau, r_{tau+1}) with tau+1 <= t  <=>  tau <= t-1  <=>  index < i
                xs, ys = x[:i], y[:i]
                ok = ~(np.isnan(xs) | np.isnan(ys))
                if ok.sum() < n_min:
                    continue
                xs, ys = xs[ok], ys[ok]
                beta = (xs @ ys) / (xs @ xs)
                resid = ys - beta * xs
                se = np.sqrt(resid.var(ddof=1) / (xs @ xs))
                out.at[t, a] = dz.iloc[i] * np.tanh(beta / se / 2)
    return out.clip(-params["signals"]["clip_z"], params["signals"]["clip_z"])
