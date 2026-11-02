"""The research run: signals -> sleeves -> tests -> gates -> combination -> reports."""
from __future__ import annotations

import copy
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import report
from .backtest import combine as comb
from .backtest import metrics
from .backtest.portfolio import RiskCache, run
from .signals import BENCHMARK, FAMILIES, NEW
from .validate import inference, overfit
from .validate.trials import TrialLog

log = logging.getLogger(__name__)

# Parameter grids explored in the design sample. Every point is logged as a trial.
GRIDS = {
    "tsmom": {"lookback_m": [3, 6, 12], "skip_m": [0, 1]},
    "carry": {"smooth_m": [1, 3]},
    "value": {"commodity_reversal_m": [36, 60, 120]},  # FX/bond/equity value inputs are set in clean.py
    "surprise": {"halflife_days": [21, 63, 126]},
    "cot": {"z_window_weeks": [104, 156, 260]},
    "enso": {"leads": [[1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]], "mode": ["prior", "estimated"]},
}
FOCUSED = {"enso"}   # sleeves that trade a chosen subset of assets by design
CRISES = {"GFC 2008-09": ("2008-09-01", "2009-03-31"), "Covid 2020": ("2020-02-01", "2020-04-30"),
          "Inflation 2022": ("2022-01-01", "2022-12-31")}


def samples(params: dict, holdout_open: bool) -> dict:
    s = params["sample"]
    out = {"design": (s["start"], s["design_end"]),
           "validation": (pd.Timestamp(s["design_end"]) + pd.Timedelta(days=1), s["validation_end"]),
           "full_pre_holdout": (s["start"], s["validation_end"])}
    if holdout_open:
        out["holdout"] = (pd.Timestamp(s["validation_end"]) + pd.Timedelta(days=1), s["holdout_end"])
    return out


def with_override(params: dict, family: str, variant: dict) -> dict:
    p = copy.deepcopy(params)
    p["signals"][family].update(variant)
    return p


def grid_points(family: str):
    g = GRIDS[family]
    keys = list(g)
    for combo in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, combo))


def class_pnl_share(bt, returns: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """Share of total gross P&L earned in each asset class (gate 5: none above 50%)."""
    pnl = (bt.positions_d * returns.reindex(bt.positions_d.index).fillna(0.0)).sum()
    by_class = pnl.groupby(meta.loc[pnl.index, "cls"]).sum()
    total = by_class.sum()
    return by_class / total if total > 0 else by_class * np.nan


def run_research(panel, params: dict, out_dir: Path, holdout_open: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    val = params["validation"]
    lags = val["nw_lags"]
    smp = samples(params, holdout_open)
    end = params["sample"]["holdout_end"] if holdout_open else params["sample"]["validation_end"]
    panel = panel.truncate(end)
    log.info("risk model (%s assets, %s days)", panel.returns.shape[1], panel.returns.shape[0])
    cache = RiskCache(panel.returns, params)
    trials = TrialLog(out_dir)
    plot_splits = {**params["sample"], "holdout_end": str(panel.returns.index[-1].date())}

    # ---- 1. base signals and sleeves ---------------------------------------------------------
    signals, sleeves, rows = {}, {}, []
    for fam, fn in FAMILIES.items():
        log.info("signal %s", fam)
        sig = fn(panel, params)
        signals[fam] = sig
        bt = run(fam, sig, cache, panel.meta, params)
        sleeves[fam] = bt
        for sname, (a, b) in smp.items():
            row = metrics.summary_row(bt.slice(a, b), sig.loc[a:b], panel.returns, lags)
            rows.append({"sample": sname, **row})
        trials.record(fam, {k: params["signals"][fam][k] for k in GRIDS[fam]
                            if k in params["signals"][fam]},
                      "design", bt.slice(*smp["design"]).net_m,
                      metrics.summary_row(bt.slice(*smp["design"]), sig, panel.returns, lags))
        decay = metrics.ic_decay(sig.loc[: smp["full_pre_holdout"][1]], panel.returns)
        decay.to_csv(out_dir / f"ic_decay_{fam}.csv")
        report.tear_sheet(bt, decay, plot_splits, fig_dir / f"tearsheet_{fam}.png",
                          f"{fam}: single-signal backtest")
    perf = pd.DataFrame(rows).set_index(["signal", "sample"]).sort_index()
    perf.to_csv(out_dir / "signal_performance.csv")

    # ---- 2. parameter grids (design sample only) ----------------------------------------------
    grid_rows = []
    for fam in FAMILIES:
        for variant in grid_points(fam):
            p = with_override(params, fam, variant)
            sig = FAMILIES[fam](panel, p)
            bt = run(fam, sig, cache, panel.meta, p).slice(*smp["design"])
            st = metrics.summary_row(bt, sig, panel.returns, lags)
            trials.record(fam, variant, "design", bt.net_m, st)
            grid_rows.append({"family": fam, "variant": str(variant),
                              "sharpe_net": st["sharpe_net"], "t_nw_net": st["t_nw_net"]})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "parameter_grid.csv", index=False)

    # ---- 3. robustness -----------------------------------------------------------------------
    rob_rows = []
    pre = smp["full_pre_holdout"]
    for fam, sig in signals.items():
        base = sleeves[fam].slice(*pre)
        for lag in (0, 1, 2, 5):
            bt = run(fam, sig, cache, panel.meta, params, lag_days=lag).slice(*pre)
            rob_rows.append({"signal": fam, "test": f"execution lag {lag}d",
                             "sharpe_net": metrics.perf(bt.net_m)["sharpe"]})
        for cls in panel.meta["cls"].unique():
            s2 = sig.copy()
            s2.loc[:, panel.meta.index[panel.meta["cls"] == cls].intersection(s2.columns)] = 0.0
            bt = run(fam, s2, cache, panel.meta, params).slice(*pre)
            rob_rows.append({"signal": fam, "test": f"without {cls}",
                             "sharpe_net": metrics.perf(bt.net_m)["sharpe"]})
        for a, b in (("1995", "2004"), ("2005", "2014"), ("2015", "2023")):
            rob_rows.append({"signal": fam, "test": f"sub-period {a}-{b}",
                             "sharpe_net": metrics.perf(base.slice(a, b).net_m)["sharpe"]})
        for label, (a, b) in CRISES.items():
            r = sleeves[fam].slice(a, b).net_m
            rob_rows.append({"signal": fam, "test": f"crisis: {label} (cum. return)",
                             "sharpe_net": float((1 + r).prod() - 1) if len(r) else np.nan})
    robust = pd.DataFrame(rob_rows)
    robust.to_csv(out_dir / "robustness.csv", index=False)

    # ---- 4. gates ------------------------------------------------------------------------------
    sleeve_m = pd.DataFrame({f: bt.net_m for f, bt in sleeves.items()})
    gate_rows = []
    bench = sleeve_m[list(BENCHMARK)]
    for fam in FAMILIES:
        pre_bt = sleeves[fam].slice(*pre)
        st = metrics.perf(pre_bt.net_m, lags)
        t_need = 3.0 if fam in NEW else 2.0
        fam_sr = trials.trial_sharpes_monthly(fam)
        dsr = overfit.deflated_sharpe(pre_bt.net_m.loc[pre_bt.net_m.ne(0).idxmax():],
                                      max(trials.n_trials(fam), 1), fam_sr)
        # PBO needs enough genuinely different variants; with a handful of near-identical ones it
        # is noise, so per-family PBO is reported only from 6 trials and gated at book level.
        fam_ret = trials.design_returns(fam).loc[lambda d: d.ne(0).all(axis=1)]
        pbo = (overfit.pbo_cscv(fam_ret, val["pbo_splits"]) if fam_ret.shape[1] >= 6
               else {"pbo": np.nan})
        if fam in NEW:
            sp = inference.spanning(sleeve_m[fam].loc[: pre[1]], bench.loc[: pre[1]], lags)
            max_corr = max(abs(sp.get(f"corr_{b}", 0) or 0) for b in BENCHMARK)
            g4 = bool(sp["t_alpha"] > 2 and max_corr < 0.3)
        else:
            sp, max_corr, g4 = {"t_alpha": np.nan}, np.nan, None
        if fam in FOCUSED:   # a deliberately narrow sleeve: no single CONTRACT above 50% of P&L
            by_asset = panel.meta.assign(cls=panel.meta.index)
            shares = class_pnl_share(pre_bt, panel.returns, by_asset)
        else:                # a diversified signal: no single ASSET CLASS above 50% of P&L
            shares = class_pnl_share(pre_bt, panel.returns, panel.meta)
        fam_grid = grid[grid["family"] == fam]["sharpe_net"]
        g5 = bool((shares.max() <= 0.5 if shares.notna().all() else False)
                  and (np.sign(fam_grid).nunique() == 1))
        val_sr = perf.loc[(fam, "validation"), "sharpe_net"]
        des_sr = perf.loc[(fam, "design"), "sharpe_net"]
        row = {
            "signal": fam, "type": "new" if fam in NEW else "benchmark",
            "sharpe_design": des_sr, "sharpe_validation": val_sr,
            "t_nw": st["t_nw"], "gate2_significance": bool(st["t_nw"] > t_need),
            "dsr": dsr["dsr"], "pbo": pbo["pbo"], "n_trials": dsr["n_trials"],
            "gate3_overfit": bool(dsr["dsr"] > 0.95),
            "alpha_t_vs_benchmark": sp["t_alpha"], "max_corr_benchmark": max_corr,
            "gate4_incremental": g4,
            "max_pnl_share": shares.max(), "grid_sign_stable": bool(np.sign(fam_grid).nunique() == 1),
            "gate5_stability": g5,
        }
        if holdout_open:
            ho = perf.loc[(fam, "holdout"), "sharpe_net"]
            row["sharpe_holdout"] = ho
            row["gate1_holdout"] = bool(np.sign(ho) == np.sign(des_sr) and ho >= 0.5 * des_sr)
        gates_needed = ["gate2_significance", "gate3_overfit", "gate5_stability"]
        if fam in NEW:
            gates_needed.append("gate4_incremental")
        if holdout_open:
            gates_needed.append("gate1_holdout")
        row["passes_all"] = all(bool(row[g]) for g in gates_needed)
        gate_rows.append(row)
    gates = pd.DataFrame(gate_rows).set_index("signal")
    gates.to_csv(out_dir / "gates.csv")
    all_ret = trials.design_returns().loc[lambda d: d.ne(0).all(axis=1)]
    book_pbo = overfit.pbo_cscv(all_ret, val["pbo_splits"])

    # ---- 5. combination ----------------------------------------------------------------------
    combos = comb.combine(sleeve_m, params)
    combo_rows = []
    for name, r in combos.items():
        for sname, (a, b) in smp.items():
            combo_rows.append({"method": name, "sample": sname, **metrics.perf(r.loc[a:b], lags)})
    combo_perf = pd.DataFrame(combo_rows).set_index(["method", "sample"])
    combo_perf.to_csv(out_dir / "combined_performance.csv")

    best_single = perf.xs("design", level="sample")["sharpe_net"].idxmax()
    pre_slice = slice(None, pre[1])
    spa = inference.spa_test(sleeve_m[best_single].loc[pre_slice],
                             pd.DataFrame(combos).loc[pre_slice], val["bootstrap_block_m"],
                             val["spa_draws"], val["seed"])
    boot = {name: inference.bootstrap_sharpe(r.loc[pre_slice].loc[lambda s: s.ne(0).cummax()],
                                             val["bootstrap_block_m"], val["bootstrap_draws"],
                                             val["seed"])
            for name, r in combos.items()}
    total_trials = trials.n_trials() + len(combos)
    dsr_combo = {name: overfit.deflated_sharpe(r.loc[pre_slice].loc[lambda s: s.ne(0).cummax()],
                                               total_trials, trials.trial_sharpes_monthly())
                 for name, r in combos.items()}
    inference_tbl = pd.DataFrame({n: {**boot[n], "dsr": dsr_combo[n]["dsr"]} for n in combos}).T
    inference_tbl.to_csv(out_dir / "combined_inference.csv")
    report.combined_sheet(sleeve_m, combos, plot_splits, fig_dir / "combined.png")

    # ---- 6. summary markdown -----------------------------------------------------------------
    write_summary(out_dir, panel, perf, gates, combo_perf, inference_tbl, spa, best_single,
                  robust, holdout_open, total_trials, book_pbo)
    return {"perf": perf, "gates": gates, "combo": combo_perf, "spa": spa, "book_pbo": book_pbo,
            "inference": inference_tbl, "robust": robust, "grid": grid}


def write_summary(out_dir, panel, perf, gates, combo_perf, inference_tbl, spa, best_single,
                  robust, holdout_open, total_trials, book_pbo):
    cols = ["sharpe_gross", "sharpe_net", "t_nw_net", "ann_vol", "max_dd", "turnover_x",
            "breakeven_bps", "mean_ic"]
    lines = ["# Global macro alpha: research summary", ""]
    if any("SYNTHETIC" in q for q in panel.qa):
        lines += ["> **SYNTHETIC DATA.** These numbers test the plumbing only. Do not report them.", ""]
    lines += [f"Holdout {'OPEN' if holdout_open else 'sealed'}; trials logged: {total_trials}.", "",
              "## Single signals", ""]
    for s in perf.index.get_level_values("sample").unique():
        lines += [f"### {s}", "", report.to_markdown(perf.xs(s, level="sample")[cols]), ""]
    g = gates.copy()
    lines += ["## Gates", "", report.to_markdown(g), "",
              f"Book-level PBO across all {book_pbo['n_trials']} design-sample trials: "
              f"{book_pbo['pbo']:.2f} ({'PASS' if book_pbo['pbo'] < 0.2 else 'FAIL'}, threshold 0.20).",
              ""]
    lines += ["## Combined book", "", report.to_markdown(combo_perf[["ann_ret", "ann_vol", "sharpe",
                                                                     "t_nw", "max_dd"]]), "",
              report.to_markdown(inference_tbl), "",
              f"SPA test vs best single signal ({best_single}): p = {spa['p_value']:.3f} "
              f"(best blend: {spa['best_alternative']}). Low p = a blend beats the best single signal.",
              ""]
    lines += ["## Robustness", "", report.to_markdown(
        robust.pivot_table(index="test", columns="signal", values="sharpe_net")), ""]
    lines += ["## Data QA", "", *[f"- {q}" for q in panel.qa], ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
