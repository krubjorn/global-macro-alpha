"""Raw pulls -> the canonical research panel.

The Panel is the only object the signal and backtest layers ever see, so swapping data vendors
(or the synthetic generator) never touches research code.

Conventions
-----------
* returns : daily *excess* returns, index = business days, columns = assets.
* carry   : annualised carry at month-end (return if prices do not move), columns = assets.
* value   : month-end value measure, higher = cheaper, NaN where the class has no external input
            (commodity value is computed from prices inside the signal).
* releases: one row per macro release with the point-in-time release_dt.
* cot     : weekly net speculative positions (% of open interest) by report date.
* plume   : ENSO forecasts by issue date, lead and model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Panel:
    returns: pd.DataFrame
    carry: pd.DataFrame
    value: pd.DataFrame
    releases: pd.DataFrame
    cot: pd.DataFrame
    plume: pd.DataFrame
    meta: pd.DataFrame
    qa: list = field(default_factory=list)

    def truncate(self, end: str) -> "Panel":
        """Drop everything after `end` (used to keep the holdout sealed)."""
        e = pd.Timestamp(end)
        return Panel(
            returns=self.returns.loc[:e],
            carry=self.carry.loc[:e],
            value=self.value.loc[:e],
            releases=self.releases[self.releases["release_dt"] <= e],
            cot=self.cot[self.cot["report_date"] <= e],
            plume=self.plume[self.plume["issue_date"] <= e],
            meta=self.meta,
            qa=list(self.qa),
        )

    def save(self, clean_dir: Path) -> None:
        clean_dir.mkdir(parents=True, exist_ok=True)
        for name in ("returns", "carry", "value", "releases", "cot", "plume", "meta"):
            getattr(self, name).to_parquet(clean_dir / f"{name}.parquet")

    @classmethod
    def load(cls, clean_dir: Path) -> "Panel":
        parts = {n: pd.read_parquet(clean_dir / f"{n}.parquet")
                 for n in ("returns", "carry", "value", "releases", "cot", "plume", "meta")}
        return cls(**parts)


# --------------------------------------------------------------------------- futures
def choose_contracts(px: pd.DataFrame, contracts: pd.DataFrame, rule: str = "max_oi",
                     days_before_fnd: int = 5) -> pd.Series:
    """Contract held on each date. Never rolls backwards to an earlier expiry.

    px        : long frame with date, contract, px_last, open_int
    contracts : contract, fut_notice_first, last_tradeable_dt
    """
    c = contracts.copy()
    # contracts with no first-notice date (cash-settled indices) roll off the last trade date
    c["deadline"] = c["fut_notice_first"].fillna(c["last_tradeable_dt"])
    c["deadline"] = c["deadline"] - pd.offsets.BDay(days_before_fnd)
    expiry_rank = c.sort_values("last_tradeable_dt")["contract"].reset_index(drop=True)
    rank = pd.Series(np.arange(len(expiry_rank)), index=expiry_rank.values)

    df = px.merge(c[["contract", "deadline"]], on="contract", how="left")
    df = df[(df["date"] < df["deadline"]) & df["px_last"].notna()]
    df["rank"] = df["contract"].map(rank)

    if rule == "max_oi":
        df = df.sort_values(["date", "open_int", "rank"], ascending=[True, False, True])
    else:  # nearest eligible contract
        df = df.sort_values(["date", "rank"])
    pick = df.groupby("date").first()["rank"]
    pick = pick.cummax()  # monotone: never step back to an older contract
    return pick.map(pd.Series(rank.index, index=rank.values)).rename("contract")


def continuous_returns(px: pd.DataFrame, held: pd.Series, winsor_sigma: float = 10) -> pd.Series:
    """Daily return of the contract held at t-1, measured on that same contract (no roll jumps)."""
    p = px.pivot_table(index="date", columns="contract", values="px_last").sort_index()
    held = held.reindex(p.index).ffill()
    prev_held = held.shift(1)
    rets = []
    for dt, c in prev_held.dropna().items():
        i = p.index.get_loc(dt)
        p0, p1 = p[c].iloc[i - 1], p[c].iloc[i]
        rets.append((dt, p1 / p0 - 1 if pd.notna(p0) and pd.notna(p1) and p0 != 0 else np.nan))
    r = pd.Series(dict(rets)).sort_index()
    return winsorise(r, winsor_sigma)


def roll_yield(px: pd.DataFrame, held: pd.Series, contracts: pd.DataFrame) -> pd.Series:
    """Annualised front-to-next carry: (F1 - F2) / F2 * 365 / days between expiries."""
    p = px.pivot_table(index="date", columns="contract", values="px_last").sort_index()
    order = contracts.sort_values("last_tradeable_dt").reset_index(drop=True)
    nxt = dict(zip(order["contract"], order["contract"].shift(-1)))
    ltd = dict(zip(order["contract"], order["last_tradeable_dt"]))
    out = {}
    for dt, c in held.reindex(p.index).ffill().dropna().items():
        c2 = nxt.get(c)
        if c2 is None or c2 not in p:
            continue
        f1, f2 = p.at[dt, c], p.at[dt, c2]
        days = (ltd[c2] - ltd[c]).days
        if pd.notna(f1) and pd.notna(f2) and f2 > 0 and days > 0:
            out[dt] = (f1 - f2) / f2 * 365 / days
    return pd.Series(out).sort_index()


def winsorise(r: pd.Series, n_sigma: float) -> pd.Series:
    sd = r.rolling(252, min_periods=60).std().bfill()
    return r.clip(lower=-n_sigma * sd, upper=n_sigma * sd)


# --------------------------------------------------------------------------- FX
def fx_panel(long: pd.DataFrame, scale: pd.Series,
             fx_specs: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily excess returns of a long-foreign 1M forward rolled daily, and annualised carry.

    All quotes converted to USD per 1 unit of foreign currency. Daily excess return:
        r_t = S_t / S_{t-1} - 1 + (S_{t-1} - F_{t-1}) / F_{t-1} / 21
    """
    w = long.pivot_table(index="date", columns="ticker", values="value").sort_index()
    rets, carry, spots = {}, {}, {}
    for ccy, s in fx_specs.items():
        if s["spot"] not in w or s["fwd"] not in w:
            continue
        spot = w[s["spot"]]
        fwd = spot + w[s["fwd"]] / 10 ** scale.get(s["fwd"], 4)  # FWD_SCALE = decimal places
        if s.get("invert"):
            spot, fwd = 1 / spot, 1 / fwd
        fwd_disc = (spot - fwd) / fwd
        rets[ccy] = spot.pct_change() + fwd_disc.shift(1) / 21
        carry[ccy] = fwd_disc * 12
        spots[ccy] = spot
    return pd.DataFrame(rets), pd.DataFrame(carry), pd.DataFrame(spots)


# --------------------------------------------------------------------------- carry for equity / bonds
def equity_carry(div_yield_pct: pd.Series, short_rate_pct: pd.Series) -> pd.Series:
    return (div_yield_pct - short_rate_pct) / 100


def bond_carry(yld_pct: pd.Series, short_rate_pct: pd.Series, tenor_y: float) -> pd.Series:
    """Yield minus funding plus linear-curve roll-down (approximation documented in the note).

    Roll-down per year ~ modified duration x slope, slope ~ (y_T - r) / T.
    """
    y, r = yld_pct / 100, short_rate_pct / 100
    duration = (1 - (1 + y) ** (-tenor_y)) / y.where(y.abs() > 1e-6, 1e-6)
    return (y - r) + duration * (y - r) / tenor_y


# --------------------------------------------------------------------------- value inputs
def fx_value(spot_usd_per_fx: pd.Series, cpi_foreign: pd.Series, cpi_us: pd.Series,
             months: int = 60) -> pd.Series:
    """Minus the 5-year change in the log real exchange rate (higher = currency has cheapened)."""
    m = pd.concat([spot_usd_per_fx, cpi_foreign, cpi_us], axis=1).resample("ME").last().ffill()
    rer = np.log(m.iloc[:, 0] * m.iloc[:, 1] / m.iloc[:, 2])
    return -(rer - rer.shift(months))


def bond_value(yld_pct: pd.Series, cpi_index: pd.Series) -> pd.Series:
    """Real yield: nominal minus trailing 12m CPI inflation (use breakevens when available)."""
    m = pd.concat([yld_pct, cpi_index], axis=1).resample("ME").last().ffill()
    infl = m.iloc[:, 1].pct_change(12) * 100
    return m.iloc[:, 0] - infl


def equity_value(pe: pd.Series, hist_m: int = 120) -> pd.Series:
    ey = (1 / pe.where(pe > 0)).resample("ME").last()
    return ey - ey.rolling(hist_m, min_periods=36).mean()


# --------------------------------------------------------------------------- assemble
def build_panel(raw_dir: Path, tickers: dict, params: dict, meta: pd.DataFrame) -> Panel:
    """Real-data path. Each block is wrapped so one missing source degrades rather than stops."""
    from .ingest import external

    dcfg = params["data"]
    bbg = raw_dir / "bbg"
    rets, carry, value, qa = {}, {}, {}, []

    short = _wide(bbg / "short_rates.parquet")
    cpi = _wide(bbg / "cpi.parquet")
    yields = _wide(bbg / "yields.parquet")
    cash = pd.read_parquet(bbg / "equity_cash.parquet") if (bbg / "equity_cash.parquet").exists() else None

    for cls in ("equity", "bond", "commodity"):
        for asset, spec in tickers[cls].items():
            f = bbg / "futures" / f"{asset}.parquet"
            if not f.exists():
                qa.append(f"missing futures file for {asset}")
                continue
            px = pd.read_parquet(f)
            contracts = pd.read_parquet(bbg / "futures" / f"{asset}_contracts.parquet")
            held = choose_contracts(px, contracts, dcfg["roll_rule"], dcfg["roll_days_before_fnd"])
            rets[asset] = continuous_returns(px, held, dcfg["winsor_sigma"])
            ctry = spec.get("country")
            rate = short.get(tickers["short_rates"].get(ctry, ""))
            if cls == "commodity":
                carry[asset] = roll_yield(px, held, contracts)
            elif cls == "equity" and cash is not None and rate is not None:
                c = cash[cash["ticker"] == spec["cash"]].pivot_table(index="date", columns="field",
                                                                     values="value")
                if "EQY_DVD_YLD_12M" in c:
                    carry[asset] = equity_carry(c["EQY_DVD_YLD_12M"], rate.reindex(c.index).ffill())
                if "PE_RATIO" in c:
                    value[asset] = equity_value(c["PE_RATIO"], params["signals"]["value"]["equity_hist_m"])
            elif cls == "bond" and spec["yld"] in yields and rate is not None:
                y = yields[spec["yld"]]
                carry[asset] = bond_carry(y, rate.reindex(y.index).ffill(), spec["tenor"])
                cpi_t = tickers["cpi"].get(ctry)
                if cpi_t in cpi:
                    value[asset] = bond_value(y, cpi[cpi_t])

    fx_file = bbg / "fx.parquet"
    if fx_file.exists():
        scale = pd.read_parquet(bbg / "fx_fwd_scale.parquet")["fwd_scale"]
        fx_r, fx_c, fx_s = fx_panel(pd.read_parquet(fx_file), scale, tickers["fx"])
        for ccy in fx_r:
            rets[ccy] = winsorise(fx_r[ccy].dropna(), dcfg["winsor_sigma"])
            carry[ccy] = fx_c[ccy]
            ctry = tickers["fx"][ccy]["country"]
            ct, us = tickers["cpi"].get(ctry), tickers["cpi"]["US"]
            if ct in cpi and us in cpi:
                value[ccy] = fx_value(fx_s[ccy], cpi[ct], cpi[us],
                                      params["signals"]["value"]["fx_reversal_m"])

    returns = pd.DataFrame(rets).sort_index()
    returns = returns.loc[:, [a for a in meta.index if a in returns]]
    carry_m = pd.DataFrame(carry).resample("ME").last().reindex(columns=returns.columns)
    value_m = pd.DataFrame(value).resample("ME").last().reindex(columns=returns.columns)

    releases = pd.read_parquet(bbg / "eco.parquet") if (bbg / "eco.parquet").exists() else _empty_releases()
    cot_dir = raw_dir / "cot"
    cot = external.load_cot_legacy(cot_dir) if cot_dir.exists() else _empty_cot()
    plume_f = raw_dir / "enso" / "plume.csv"
    plume = external.load_enso_plume(plume_f) if plume_f.exists() else _empty_plume()
    qa += quality_report(returns)
    return Panel(returns, carry_m, value_m, releases, cot, plume, meta.loc[returns.columns], qa)


def quality_report(returns: pd.DataFrame) -> list[str]:
    """Plain-language data checks that go into the QA report."""
    out = []
    for a, r in returns.items():
        first = r.first_valid_index()
        gaps = r.loc[first:].isna().mean() if first is not None else 1.0
        big = (r.abs() > 0.15).sum()
        out.append(f"{a}: starts {first.date() if first is not None else 'n/a'}, "
                   f"missing {gaps:.1%}, |daily ret|>15%: {big}")
    return out


def _wide(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).pivot_table(index="date", columns="ticker", values="value")


def _empty_releases() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "country", "block", "sign", "ref_date", "release_dt",
                                 "actual", "survey"])


def _empty_cot() -> pd.DataFrame:
    return pd.DataFrame(columns=["report_date", "asset", "net_spec_oi"])


def _empty_plume() -> pd.DataFrame:
    return pd.DataFrame(columns=["issue_date", "lead_m", "model", "nino34"])
