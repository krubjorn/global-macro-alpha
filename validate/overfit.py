"""Multiple-testing and overfitting controls.

* deflated_sharpe : Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", JPM.
* pbo_cscv        : Bailey, Borwein, Lopez de Prado & Zhu (2017), Probability of Backtest
                    Overfitting via combinatorially symmetric cross-validation.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected maximum per-period Sharpe among n independent trials with true SR = 0."""
    if n_trials <= 1 or sr_var <= 0:
        return 0.0
    z = stats.norm.ppf
    return float(np.sqrt(sr_var) * ((1 - EULER) * z(1 - 1 / n_trials)
                                    + EULER * z(1 - 1 / (n_trials * np.e))))


def deflated_sharpe(r: pd.Series, n_trials: int, trial_sharpes: pd.Series | None = None) -> dict:
    """DSR = P(true SR > SR*), with SR* the expected best Sharpe from `n_trials` null trials.

    r              : the selected strategy's per-period (monthly) returns
    trial_sharpes  : per-period Sharpe of every trial tried (for the cross-trial variance);
                     if None, uses the asymptotic variance of a single SR estimate.
    """
    r = r.dropna()
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_var = float(np.var(trial_sharpes, ddof=1))
    else:
        sr_var = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (T - 1)
    sr_star = expected_max_sharpe(n_trials, sr_var)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr_star) * np.sqrt(T - 1) / denom
    return {"sr_monthly": sr, "sr_star_monthly": sr_star, "n_trials": n_trials,
            "dsr": float(stats.norm.cdf(z)), "months": T}


def pbo_cscv(trial_returns: pd.DataFrame, n_splits: int = 16) -> dict:
    """Probability that the in-sample best trial ranks below the median out of sample.

    trial_returns : T x N matrix of per-period returns, one column per trial (same dates).
    """
    m = trial_returns.dropna(how="any")
    T, N = m.shape
    if N < 2 or T < n_splits * 2:
        return {"pbo": np.nan, "n_trials": N, "n_combos": 0}
    blocks = np.array_split(np.arange(T), n_splits)
    x = m.to_numpy()
    logits = []
    for is_idx in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in is_idx])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_idx])
        sr_is = x[is_rows].mean(0) / x[is_rows].std(0, ddof=1)
        sr_oos = x[oos_rows].mean(0) / x[oos_rows].std(0, ddof=1)
        best = int(np.nanargmax(sr_is))
        rank = stats.rankdata(sr_oos)[best]                  # 1 = worst
        omega = rank / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "n_trials": N, "n_combos": len(logits),
            "median_logit": float(np.median(logits))}
