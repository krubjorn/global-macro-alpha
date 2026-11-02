"""Signal -> positions -> daily P&L, with costs.

Timing (the main leakage risk, so spelled out):
  * signals are dated at calendar month-end t and use only data available by then;
  * weights are traded at the close `execution_lag_days` business days after the last trading
    day of month t, and earn returns from the following day.
Positions are held at constant notional between rebalances (no intra-month drift); this is the
standard research simplification and is conservative for trend, slightly flattering for reversal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..risk import ewma_vol, month_end_dates, shrunk_cov


@dataclass
class Backtest:
    name: str
    weights_m: pd.DataFrame      # notional weights decided at each month-end
    positions_d: pd.DataFrame    # daily positions actually held
    gross_d: pd.Series
    cost_d: pd.Series
    net_d: pd.Series

    @property
    def net_m(self) -> pd.Series:
        return (1 + self.net_d).resample("ME").prod() - 1

    @property
    def gross_m(self) -> pd.Series:
        return (1 + self.gross_d).resample("ME").prod() - 1

    def turnover_annual(self) -> float:
        dw = self.positions_d.diff().abs().sum(axis=1)
        years = len(self.positions_d) / 252
        return float(dw.sum() / years) if years > 0 else np.nan

    def slice(self, start=None, end=None) -> "Backtest":
        s = slice(start, end)
        return Backtest(self.name, self.weights_m.loc[s], self.positions_d.loc[s],
                        self.gross_d.loc[s], self.cost_d.loc[s], self.net_d.loc[s])


class RiskCache:
    """Vol and covariance computed once per panel and reused by every signal and trial."""

    def __init__(self, returns: pd.DataFrame, params: dict):
        rk = params["risk"]
        self.returns = returns
        self.trade_dates = month_end_dates(returns.index)
        self.vol = ewma_vol(returns, rk["vol_halflife_days"]).reindex(self.trade_dates)
        self.cov = {d: shrunk_cov(returns, d, rk["cov_window_days"]) for d in self.trade_dates}


def signal_to_weights(signal: pd.DataFrame, cache: RiskCache, params: dict) -> pd.DataFrame:
    """w_i = s_i / sigma_i, then the whole book scaled to `target_vol` ex ante."""
    pf = params["portfolio"]
    # signal at calendar month-end -> last trading day of that month
    sig = signal.copy()
    sig.index = pd.DatetimeIndex(sig.index) + pd.offsets.MonthEnd(0)
    td = pd.Series(cache.trade_dates, index=cache.trade_dates + pd.offsets.MonthEnd(0))
    sig = sig.reindex(td.index)
    sig.index = td.to_numpy()
    sig = sig.reindex(columns=cache.returns.columns)

    vol = cache.vol.clip(lower=pf["vol_floor"])
    raw = (sig / vol).fillna(0.0)
    w = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for d, row in raw.iterrows():
        if row.abs().sum() == 0:
            continue
        cov = cache.cov[d].fillna(0.0)
        x = row.to_numpy()
        ex_ante = float(np.sqrt(max(x @ cov.to_numpy() @ x, 0)))
        if ex_ante <= 0:
            continue
        w.loc[d] = (row * pf["target_vol"] / ex_ante).clip(-pf["max_notional"], pf["max_notional"])
    return w


def run(name: str, signal: pd.DataFrame, cache: RiskCache, meta: pd.DataFrame, params: dict,
        lag_days: int | None = None) -> Backtest:
    pf = params["portfolio"]
    lag = pf["execution_lag_days"] if lag_days is None else lag_days
    w_m = signal_to_weights(signal, cache, params)
    r = cache.returns
    pos = w_m.reindex(r.index).ffill().fillna(0.0).shift(lag + 1).fillna(0.0)
    # an asset with no return that day contributes nothing (and is not charged)
    gross = (pos * r.fillna(0.0)).sum(axis=1)

    hs = meta.loc[r.columns, "half_spread_bps"].to_numpy() / 1e4
    roll = meta.loc[r.columns, "roll_bps"].to_numpy() / 1e4 / 252
    trade_cost = (pos.diff().abs().fillna(pos.abs()) * hs).sum(axis=1)
    carry_cost = (pos.abs() * roll).sum(axis=1)
    cost = trade_cost + carry_cost
    return Backtest(name, w_m, pos, gross, cost, gross - cost)
