"""Tear sheets (PNG) and summary tables (CSV + Markdown)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Reference categorical order (blue, orange, aqua) + recessive ink; see dataviz palette.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "lines.linewidth": 1.6,
    "legend.frameon": False,
})


def _shade_samples(ax, splits: dict):
    """Light band over the validation sample, darker over the holdout, with labels."""
    v0, v1 = pd.Timestamp(splits["design_end"]), pd.Timestamp(splits["validation_end"])
    ax.axvspan(v0, v1, color=INK2, alpha=0.06, lw=0)
    ax.axvspan(v1, pd.Timestamp(splits["holdout_end"]), color=INK2, alpha=0.12, lw=0)


def tear_sheet(bt, ic_decay: pd.DataFrame, splits: dict, out: Path, title: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    net, gross = bt.net_m, bt.gross_m
    start = net.ne(0).idxmax() if net.ne(0).any() else net.index[0]
    net, gross = net.loc[start:], gross.loc[start:]

    ax = axes[0, 0]
    ax.plot((1 + gross).cumprod(), color=C1, alpha=0.5, label="Gross")
    ax.plot((1 + net).cumprod(), color=C1, label="Net of costs")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale); shaded = out-of-sample")
    _shade_samples(ax, splits)
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    roll = net.rolling(36).mean() / net.rolling(36).std() * np.sqrt(12)
    ax.plot(roll, color=C1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("Rolling 36-month Sharpe (net)")
    _shade_samples(ax, splits)

    ax = axes[1, 0]
    if ic_decay is not None and not ic_decay.empty:
        ax.bar(ic_decay.index, ic_decay["mean_ic"], color=C1, width=0.6)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(ic_decay.index)
    ax.set_title("Information coefficient by horizon (months ahead)")

    ax = axes[1, 1]
    nav = (1 + net).cumprod()
    dd = nav / nav.cummax() - 1
    ax.fill_between(dd.index, dd, 0, color=C2, alpha=0.35, lw=0)
    ax.plot(dd, color=C2, lw=1)
    ax.set_title("Drawdown (net)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _shade_samples(ax, splits)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def combined_sheet(sleeves: pd.DataFrame, combos: dict, splits: dict, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    start = sleeves.ne(0).any(axis=1).idxmax()
    for i, (name, r) in enumerate(combos.items()):
        ax.plot((1 + r.loc[start:]).cumprod(), color=SERIES[i], label=name.replace("_", " "))
    ax.set_yscale("log")
    ax.set_title("Combined book, growth of $1 (net, 10% vol target)")
    _shade_samples(ax, splits)
    ax.legend(loc="upper left")

    ax = axes[1]
    corr = sleeves.loc[start:].replace(0, np.nan).corr()
    shown = corr.mask(np.eye(len(corr), dtype=bool))   # blank the diagonal
    im = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iat[i, j]
            if i == j:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Sleeve correlation (monthly net)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def to_markdown(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    header = "| " + " | ".join([d.index.name or ""] + list(map(str, d.columns))) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    label = (lambda i: " / ".join(map(str, i))) if isinstance(d.index, pd.MultiIndex) else str
    rows = ["| " + " | ".join([label(i)] + [str(v) for v in r]) + " |" for i, r in zip(d.index, d.values)]
    return "\n".join([header, sep, *rows])
