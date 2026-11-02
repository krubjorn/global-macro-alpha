"""Trial log: every backtest variant ever run, so the multiple-testing adjustment is honest.

trials_log.csv          one row per (trial_id, sample)
trials/returns.parquet  monthly net returns, one column per trial_id (design sample)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class TrialLog:
    def __init__(self, root: Path):
        self.csv = root / "trials_log.csv"
        self.ret_path = root / "trials" / "returns.parquet"
        self.ret_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def trial_id(family: str, variant: dict) -> str:
        blob = json.dumps({"family": family, **variant}, sort_keys=True, default=str).encode()
        return f"{family}-{hashlib.sha1(blob).hexdigest()[:8]}"

    def record(self, family: str, variant: dict, sample: str, net_m: pd.Series, stats: dict) -> str:
        tid = self.trial_id(family, variant)
        row = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "trial_id": tid,
               "family": family, "variant": json.dumps(variant, sort_keys=True), "sample": sample,
               **{k: stats.get(k) for k in ("sharpe_net", "t_nw_net", "months")}}
        log = pd.read_csv(self.csv) if self.csv.exists() else pd.DataFrame()
        if not log.empty:
            log = log[~((log["trial_id"] == tid) & (log["sample"] == sample))]
        pd.concat([log, pd.DataFrame([row])], ignore_index=True).to_csv(self.csv, index=False)
        if sample == "design":
            rets = pd.read_parquet(self.ret_path) if self.ret_path.exists() else pd.DataFrame()
            rets = rets.drop(columns=[tid], errors="ignore")
            rets = pd.concat([rets, net_m.rename(tid)], axis=1)
            rets.to_parquet(self.ret_path)
        return tid

    def n_trials(self, family: str | None = None) -> int:
        if not self.csv.exists():
            return 0
        log = pd.read_csv(self.csv)
        log = log[log["sample"] == "design"]
        return int(log["trial_id"].nunique() if family is None
                   else log.loc[log["family"] == family, "trial_id"].nunique())

    def design_returns(self, family: str | None = None) -> pd.DataFrame:
        if not self.ret_path.exists():
            return pd.DataFrame()
        r = pd.read_parquet(self.ret_path)
        return r if family is None else r[[c for c in r.columns if c.startswith(f"{family}-")]]

    def trial_sharpes_monthly(self, family: str | None = None) -> pd.Series:
        r = self.design_returns(family)
        r = r.loc[:, r.ne(0).sum() > 24]
        return (r.mean() / r.std()).replace([np.inf, -np.inf], np.nan).dropna()
