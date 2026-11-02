"""WRDS pulls via the `wrds` package (pip install wrds; first run prompts for credentials and
offers to create ~/.pgpass).

WRDS library and table names depend on your institution's subscription and occasionally get
renamed. Before the first pull run:

    python -m macro_alpha.ingest.wrds_pull --list crsp frb optionm

and adjust TABLES below if a name differs. Everything is saved untouched to data/raw/wrds/.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# library.table -> purpose. Edit to match `--list` output for your subscription.
TABLES = {
    "fama_bliss": "crsp.tfz_mth_fb",      # Fama-Bliss zero-coupon prices/yields (1-5y): bond carry/roll-down
    "riskfree": "crsp.tfz_mth_rf",        # 1M and 3M T-bill returns: excess-return cross-check
    "h15": "frb.rates_daily",             # FRB H.15 constant-maturity yields: yield cross-check
    "h10": "frb.fx_daily",                # FRB H.10 noon FX rates: spot cross-check
}


def connect():
    import wrds  # noqa: WPS433
    return wrds.Connection()


def list_tables(db, libraries: list[str]) -> None:
    for lib in libraries:
        try:
            print(lib, "->", db.list_tables(library=lib))
        except Exception as exc:  # noqa: BLE001
            print(lib, "-> not available:", exc)


def _pull(db, key: str, start: str, end: str, date_col: str) -> pd.DataFrame:
    lib_tab = TABLES[key]
    sql = f"select * from {lib_tab} where {date_col} between '{start}' and '{end}'"
    log.info("WRDS %s", sql)
    return db.raw_sql(sql, date_cols=[date_col])


def pull_all(params: dict, raw_dir: Path, overwrite: bool = False) -> None:
    out_dir = raw_dir / "wrds"
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = params["sample"]["start"], params["sample"]["holdout_end"]
    db = connect()
    plan = {"fama_bliss": "mcaldt", "riskfree": "caldt", "h15": "date", "h10": "date"}
    for key, date_col in plan.items():
        path = out_dir / f"{key}.parquet"
        if path.exists() and not overwrite:
            continue
        try:
            df = _pull(db, key, start, end, date_col)
            df.to_parquet(path)
            log.info("saved %s rows -> %s", len(df), path)
        except Exception as exc:  # noqa: BLE001  (missing subscription should not stop the run)
            log.warning("WRDS pull %s failed: %s", key, exc)
    db.close()


def cross_check(bbg: pd.Series, wrds: pd.Series, name: str, tol: float) -> pd.DataFrame:
    """Daily differences between two sources of the same series; rows above tol go in the QA report."""
    both = pd.concat({"bbg": bbg, "wrds": wrds}, axis=1).dropna()
    both["diff"] = both["bbg"] - both["wrds"]
    flagged = both[both["diff"].abs() > tol].copy()
    flagged["series"] = name
    return flagged


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", nargs="+", help="list tables in these WRDS libraries")
    args = ap.parse_args()
    if args.list:
        list_tables(connect(), args.list)
