"""Performance and signal-quality statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..signals.base import monthly_returns


def nw_tstat(x: pd.Series, lags: int = 6) -> float:
    """Newey-West t-statistic of the mean."""
    x = x.dropna()
    if len(x) < 24:
        return np.nan
    res = sm.OLS(x.to_numpy(), np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def max_drawdown(r_m: pd.Series) -> float:
    nav = (1 + r_m.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def perf(r_m: pd.Series, lags: int = 6) -> dict:
    r = r_m.dropna()
    r = r.loc[r.ne(0).idxmax():] if r.ne(0).any() else r      # drop warm-up months of zeros
    if len(r) < 12:
        return {k: np.nan for k in ("ann_ret", "ann_vol", "sharpe", "t_nw", "max_dd", "hit", "skew",
                                    "kurt", "months")}
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "t_nw": nw_tstat(r, lags),
        "max_dd": max_drawdown(r),
        "hit": float((r > 0).mean()),
        "skew": float(stats.skew(r)),
        "kurt": float(stats.kurtosis(r, fisher=False)),
        "months": len(r),
    }


def information_coefficient(signal: pd.DataFrame, daily_returns: pd.DataFrame,
                            horizon: int = 1) -> pd.Series:
    """Cross-sectional Spearman IC between the month-t signal and the return in month t+horizon."""
    mret = monthly_returns(daily_returns)
    sig = signal.copy()
    sig.index = pd.DatetimeIndex(sig.index) + pd.offsets.MonthEnd(0)
    fwd = mret.shift(-horizon).reindex(sig.index)
    ics = {}
    for d in sig.index:
        s, f = sig.loc[d], fwd.loc[d]
        ok = s.notna() & f.notna() & (s != 0)
        if ok.sum() >= 5:
            ics[d] = stats.spearmanr(s[ok], f[ok]).statistic
    return pd.Series(ics, dtype=float)


def ic_decay(signal: pd.DataFrame, daily_returns: pd.DataFrame, horizons=range(1, 13)) -> pd.DataFrame:
    rows = []
    for h in horizons:
        ic = information_coefficient(signal, daily_returns, h)
        rows.append({"horizon_m": h, "mean_ic": ic.mean(),
                     "t_ic": ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 2 else np.nan})
    return pd.DataFrame(rows).set_index("horizon_m")


def breakeven_cost_bps(bt) -> float:
    """Average one-way cost (bps) at which net return would be zero."""
    years = len(bt.gross_d) / 252
    gross_ann = bt.gross_d.sum() / years
    turnover = bt.turnover_annual()
    return float(gross_ann / turnover * 1e4) if turnover > 0 else np.nan


def summary_row(bt, signal: pd.DataFrame, daily_returns: pd.DataFrame, lags: int = 6) -> dict:
    net, gross = perf(bt.net_m, lags), perf(bt.gross_m, lags)
    ic = information_coefficient(signal, daily_returns, 1)
    return {
        "signal": bt.name,
        "sharpe_gross": gross["sharpe"],
        "sharpe_net": net["sharpe"],
        "t_nw_net": net["t_nw"],
        "ann_ret_net": net["ann_ret"],
        "ann_vol": net["ann_vol"],
        "max_dd": net["max_dd"],
        "hit": net["hit"],
        "turnover_x": bt.turnover_annual(),
        "breakeven_bps": breakeven_cost_bps(bt),
        "mean_ic": ic.mean(),
        "months": net["months"],
    }
