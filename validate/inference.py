"""Inference: spanning regressions, block-bootstrap Sharpe intervals, Hansen's SPA test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def spanning(y: pd.Series, X: pd.DataFrame, lags: int = 6) -> dict:
    """Regress a new sleeve on benchmark sleeves; alpha is the part they do not explain."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    d = d.loc[d["y"].ne(0).idxmax():]
    if len(d) < 36:
        return {"alpha_ann": np.nan, "t_alpha": np.nan, "r2": np.nan}
    res = sm.OLS(d["y"], sm.add_constant(d[X.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha_ann": res.params["const"] * 12, "t_alpha": res.tvalues["const"], "r2": res.rsquared,
           "months": len(d)}
    for c in X.columns:
        out[f"beta_{c}"] = res.params[c]
        out[f"corr_{c}"] = d["y"].corr(d[c])
    return out


def _stationary_indices(rng, T: int, mean_block: float, draws: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (draws x T)."""
    p = 1 / mean_block
    idx = np.empty((draws, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, draws)
    new_block = rng.random((draws, T)) < p
    fresh = rng.integers(0, T, (draws, T))
    for t in range(1, T):
        idx[:, t] = np.where(new_block[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_sharpe(r: pd.Series, block: int = 12, draws: int = 5000, seed: int = 7) -> dict:
    x = r.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(rng, len(x), block, draws)
    s = x[idx]
    sr = s.mean(1) / s.std(1, ddof=1) * np.sqrt(12)
    nav = np.cumprod(1 + s, axis=1)
    dd = (nav / np.maximum.accumulate(nav, axis=1) - 1).min(1)
    return {"sharpe_p05": np.percentile(sr, 5), "sharpe_p50": np.percentile(sr, 50),
            "sharpe_p95": np.percentile(sr, 95), "p_sharpe_le_0": float((sr <= 0).mean()),
            "maxdd_p05": np.percentile(dd, 5)}


def spa_test(benchmark: pd.Series, alternatives: pd.DataFrame, block: int = 12,
             draws: int = 2000, seed: int = 7) -> dict:
    """Hansen (2005) SPA, consistent p-value. H0: no alternative beats the benchmark.

    Performance = mean monthly return (sleeves are already scaled to the same target vol).
    """
    d = alternatives.sub(benchmark, axis=0).dropna()
    T, K = d.shape
    x = d.to_numpy()
    dbar = x.mean(0)
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(rng, T, block, draws)
    boot_means = np.stack([x[idx[b]].mean(0) for b in range(draws)])     # draws x K
    omega = np.sqrt(T * boot_means.var(0, ddof=1))
    stat = max(np.max(np.sqrt(T) * dbar / omega), 0.0)
    keep = np.sqrt(T) * dbar / omega >= -np.sqrt(2 * np.log(np.log(T)))
    g = np.where(keep, dbar, 0.0)
    boot_stat = np.maximum((np.sqrt(T) * (boot_means - g) / omega).max(1), 0.0)
    return {"spa_stat": float(stat), "p_value": float((boot_stat >= stat).mean()), "months": T,
            "best_alternative": d.columns[int(np.argmax(dbar))]}
