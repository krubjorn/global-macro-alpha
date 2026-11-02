"""Public data that neither Bloomberg nor WRDS carries: ENSO forecasts and (fallback) CFTC COT.

ENSO forecasts
--------------
The signal needs the *forecast as issued* each month, not the realised index. Assemble the
IRI/CPC ENSO plume archive into one CSV at data/raw/enso/plume.csv with columns:

    issue_date (YYYY-MM-DD), lead_m (int, months ahead), model (str), nino34 (float, anomaly degC)

Use the issue date published with each plume (mid-month), not the first day of the month:
that is the point-in-time stamp. The realised ONI (for evaluation only) is parsed from the CPC
text file ``oni.ascii.txt`` saved to data/raw/enso/oni.ascii.txt.

CFTC Commitments of Traders
---------------------------
If Bloomberg COT tickers are not mapped, download the legacy futures-only annual files from
cftc.gov (one zip per year), unzip into data/raw/cot/, and map contract market codes to assets
in COT_CODES below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# CFTC contract market codes -> asset. Verify against the 'CFTC_Contract_Market_Code' column.
COT_CODES = {
    "13874A": "ES", "209742": "NQ", "042601": "TU", "044601": "FV", "043602": "TY", "020601": "US",
    "099741": "EUR", "097741": "JPY", "096742": "GBP", "092741": "CHF", "232741": "AUD",
    "112741": "NZD", "090741": "CAD", "095741": "MXN", "102741": "BRL", "122741": "ZAR",
    "067651": "CL", "023651": "NG", "022651": "HO", "111659": "XB", "088691": "GC", "084691": "SI",
    "085692": "HG", "002602": "C", "005602": "S", "001602": "W", "083731": "KC", "080732": "SB",
    "073732": "CC", "033661": "CT", "057642": "LC",
}


def load_enso_plume(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["issue_date"])
    need = {"issue_date", "lead_m", "model", "nino34"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"plume.csv missing columns {missing}")
    return df


def load_oni(path: Path) -> pd.Series:
    """CPC ONI text file: SEAS YR TOTAL ANOM. Indexed by the centre month of the season."""
    df = pd.read_csv(path, sep=r"\s+")
    centre = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
              "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    idx = pd.to_datetime({"year": df["YR"], "month": df["SEAS"].map(centre), "day": 1})
    idx = pd.DatetimeIndex(idx) + pd.offsets.MonthEnd(0)
    return pd.Series(df["ANOM"].to_numpy(), index=idx, name="oni")


def load_cot_legacy(cot_dir: Path) -> pd.DataFrame:
    """Legacy futures-only files -> report_date, asset, net non-commercial share of open interest."""
    frames = []
    for f in sorted(cot_dir.glob("*.txt")) + sorted(cot_dir.glob("*.csv")):
        frames.append(pd.read_csv(f, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"no COT files in {cot_dir}")
    raw = pd.concat(frames, ignore_index=True)
    code = raw["CFTC_Contract_Market_Code"].astype(str).str.strip()
    raw = raw[code.isin(COT_CODES)].copy()
    raw["asset"] = code[code.isin(COT_CODES)].map(COT_CODES)
    raw["report_date"] = pd.to_datetime(raw["As_of_Date_In_Form_YYMMDD"].astype(str), format="%y%m%d")
    longs = raw["NonComm_Positions_Long_All"].astype(float)
    shorts = raw["NonComm_Positions_Short_All"].astype(float)
    raw["net_spec_oi"] = (longs - shorts) / raw["Open_Interest_All"].astype(float)
    return raw[["report_date", "asset", "net_spec_oi"]].sort_values(["asset", "report_date"])
