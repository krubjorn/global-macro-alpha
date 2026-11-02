"""Load YAML configuration and build the asset metadata table."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
ASSET_CLASSES = ("equity", "bond", "fx", "commodity")


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


def load_all() -> tuple[dict, dict, dict]:
    """Return (tickers, params, costs)."""
    return load_yaml("tickers.yaml"), load_yaml("params.yaml"), load_yaml("costs.yaml")


def asset_meta(tickers: dict, costs: dict) -> pd.DataFrame:
    """One row per asset: class, country, ENSO prior and cost bucket."""
    rows = []
    for cls in ASSET_CLASSES:
        for name, spec in tickers[cls].items():
            if cls == "fx":
                bucket = "fx_em" if name in costs["em_fx"] else "fx_g10"
            elif cls == "commodity":
                bucket = ("commodity_softs_ags" if name in costs["softs_ags"]
                          else "commodity_energy_metals")
            else:
                bucket = cls
            rows.append({
                "asset": name,
                "cls": cls,
                "country": spec.get("country"),
                "enso_prior": spec.get("enso_prior", 0),
                "cost_bucket": bucket,
                "half_spread_bps": costs["half_spread_bps"][bucket],
                "roll_bps": costs["roll_bps_per_year"][bucket],
            })
    return pd.DataFrame(rows).set_index("asset")


def params_hash(params: dict) -> str:
    """Short stable hash of a parameter dict; identifies a trial in the trials log."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
