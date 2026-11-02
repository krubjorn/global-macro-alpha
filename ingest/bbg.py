"""Bloomberg pulls via xbbg (wraps blpapi). Run on a machine with a logged-in Terminal.

    pip install xbbg blpapi --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ (blpapi)

Everything lands in data/raw/bbg/ as Parquet, untouched. Cleaning happens in clean.py.
Pulls are idempotent: an existing file is skipped unless overwrite=True, so a run interrupted
by the daily data limit can simply be restarted.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _blp():
    try:
        from xbbg import blp  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xbbg/blpapi not installed; run on a Bloomberg Terminal machine") from exc
    return blp


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    log.info("saved %s rows -> %s", len(df), path)


def _bdh_long(tickers: list[str], fields: list[str], start: str, end: str, **ovr) -> pd.DataFrame:
    """bdh returned in long format: date, ticker, field, value."""
    blp = _blp()
    wide = blp.bdh(tickers=tickers, flds=fields, start_date=start, end_date=end, **ovr)
    if wide.empty:
        return pd.DataFrame(columns=["date", "ticker", "field", "value"])
    long = wide.stack(level=[0, 1], future_stack=True).rename("value").reset_index()
    long.columns = ["date", "ticker", "field", "value"]
    return long.dropna(subset=["value"])


# --------------------------------------------------------------------------- futures
def pull_futures_chain(asset: str, spec: dict, start: str, end: str, raw_dir: Path,
                       overwrite: bool = False, pause: float = 0.5) -> None:
    """All contracts (live + expired) for one root: daily px, OI, volume + contract dates."""
    out = raw_dir / "bbg" / "futures" / f"{asset}.parquet"
    meta_out = raw_dir / "bbg" / "futures" / f"{asset}_contracts.parquet"
    if out.exists() and not overwrite:
        return
    blp = _blp()
    generic = f"{spec['root']}1 {spec['yellow']}"
    chain = blp.bds(generic, "FUT_CHAIN", INCLUDE_EXPIRED_CONTRACTS="Y")
    contracts = chain.iloc[:, 0].astype(str).tolist()
    meta = blp.bdp(contracts, ["FUT_NOTICE_FIRST", "LAST_TRADEABLE_DT", "FUT_CONT_SIZE"])
    meta.index.name = "contract"
    meta = meta.reset_index()
    meta["fut_notice_first"] = pd.to_datetime(meta.get("fut_notice_first"))
    meta["last_tradeable_dt"] = pd.to_datetime(meta.get("last_tradeable_dt"))
    # only contracts that were alive inside the sample
    meta = meta[meta["last_tradeable_dt"] >= pd.Timestamp(start)]
    frames = []
    for chunk_start in range(0, len(meta), 25):  # chunk to stay under request limits
        chunk = meta["contract"].iloc[chunk_start:chunk_start + 25].tolist()
        frames.append(_bdh_long(chunk, ["PX_LAST", "OPEN_INT", "PX_VOLUME"], start, end))
        time.sleep(pause)
    px = pd.concat(frames, ignore_index=True)
    px = px.pivot_table(index=["date", "ticker"], columns="field", values="value").reset_index()
    px.columns = [c.lower() for c in px.columns]
    px = px.rename(columns={"ticker": "contract"})
    _save(px, out)
    _save(meta, meta_out)


# --------------------------------------------------------------------------- FX
def pull_fx(fx_specs: dict, start: str, end: str, raw_dir: Path, overwrite: bool = False) -> None:
    """Spot and 1M forward points, plus FWD_SCALE to convert points into outright forwards."""
    out = raw_dir / "bbg" / "fx.parquet"
    if out.exists() and not overwrite:
        return
    blp = _blp()
    spots = [s["spot"] for s in fx_specs.values()]
    fwds = [s["fwd"] for s in fx_specs.values()]
    long = _bdh_long(spots + fwds, ["PX_LAST"], start, end)
    scale = blp.bdp(fwds, "FWD_SCALE").iloc[:, 0].rename("fwd_scale")
    _save(long, out)
    _save(scale.to_frame(), raw_dir / "bbg" / "fx_fwd_scale.parquet")


# --------------------------------------------------------------------------- levels
def pull_levels(tickers: list[str], name: str, start: str, end: str, raw_dir: Path,
                fields: tuple[str, ...] = ("PX_LAST",), overwrite: bool = False) -> None:
    """Generic daily levels: yields, CPI indices, short rates, cash-index valuation fields."""
    out = raw_dir / "bbg" / f"{name}.parquet"
    if out.exists() and not overwrite:
        return
    _save(_bdh_long(list(tickers), list(fields), start, end), out)


# --------------------------------------------------------------------------- macro releases
def pull_eco(macro_spec: dict, start: str, end: str, raw_dir: Path, overwrite: bool = False) -> None:
    """Release history: actual, survey median and the release date (point-in-time stamp).

    ECO_RELEASE_DT arrives as a yyyymmdd float in history requests; it is parsed here so the
    clean layer only ever sees timestamps.
    """
    out = raw_dir / "bbg" / "eco.parquet"
    if out.exists() and not overwrite:
        return
    rows = []
    for country, blocks in macro_spec.items():
        for block, series in blocks.items():
            for s in series:
                rows.append({"ticker": s["t"], "country": country, "block": block, "sign": s["sign"]})
    spec = pd.DataFrame(rows)
    long = _bdh_long(spec["ticker"].tolist(),
                     ["ACTUAL_RELEASE", "BN_SURVEY_MEDIAN", "ECO_RELEASE_DT"], start, end)
    wide = long.pivot_table(index=["date", "ticker"], columns="field", values="value").reset_index()
    wide.columns = [c.lower() for c in wide.columns]
    rel = wide["eco_release_dt"]
    wide["release_dt"] = pd.to_datetime(rel.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    wide = wide.merge(spec, on="ticker", how="left")
    wide = wide.rename(columns={"actual_release": "actual", "bn_survey_median": "survey",
                                "date": "ref_date"})
    _save(wide[["ticker", "country", "block", "sign", "ref_date", "release_dt", "actual", "survey"]],
          out)


def pull_all(tickers: dict, params: dict, raw_dir: Path, overwrite: bool = False) -> None:
    start = params["sample"]["start"]
    end = params["sample"]["holdout_end"]
    for cls in ("equity", "bond", "commodity"):
        for asset, spec in tickers[cls].items():
            log.info("futures %s", asset)
            pull_futures_chain(asset, spec, start, end, raw_dir, overwrite)
    pull_fx(tickers["fx"], start, end, raw_dir, overwrite)
    pull_levels([s["yld"] for s in tickers["bond"].values()], "yields", start, end, raw_dir,
                overwrite=overwrite)
    pull_levels(list(tickers["cpi"].values()), "cpi", start, end, raw_dir, overwrite=overwrite)
    pull_levels(list(tickers["short_rates"].values()), "short_rates", start, end, raw_dir,
                overwrite=overwrite)
    pull_levels([s["cash"] for s in tickers["equity"].values()], "equity_cash", start, end, raw_dir,
                fields=("PX_LAST", "EQY_DVD_YLD_12M", "PE_RATIO"), overwrite=overwrite)
    pull_eco(tickers["macro"], start, end, raw_dir, overwrite)
