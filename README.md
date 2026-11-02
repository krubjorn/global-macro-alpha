# macro-alpha

Alpha research pipeline for ~55 global macro futures and FX forwards. It builds six signal families, tests each against five pre-registered gates, combines the survivors and writes tear sheets. 

## Quick start

```bash
pip install -r requirements.txt          # plus blpapi on the Terminal machine
pytest -q                                # 12 tests incl. look-ahead checks (~5 s)
python run_all.py --synthetic            # full run on planted synthetic data (~2 min)
```

The synthetic run proves the plumbing. Its data has known effects planted in it: trend, carry, macro surprise and ENSO are predictive, while value and COT are noise. The gates should pass carry and ENSO and fail value and COT, and they do. **Never report synthetic numbers.**

## Real-data run (maps to the 3-week schedule)

| Day | Command | What happens |
|---|---|---|
| 1 | edit `config/tickers.yaml` | Verify every root/ticker on the Terminal (`CT <GO>`, `ECO <GO>`) |
| 1 | `python -m macro_alpha.ingest.wrds_pull --list crsp frb` | Confirm WRDS table names; edit `TABLES` if needed |
| 2–3 | `python run_all.py --stage ingest` | Bloomberg + WRDS pulls to `data/raw/` (restartable; skips existing files) |
| 2–3 | manual | ENSO plume archive to `data/raw/enso/plume.csv`; CFTC COT files to `data/raw/cot/` (see `ingest/external.py`) |
| 4 | `python run_all.py --stage clean` | Continuous futures, FX forward returns, carry/value inputs, `reports/data_qa.md` |
| 5–12 | `python run_all.py --stage research` | Signals, grids, robustness, gates, combination, holdout **sealed** |
| 13 | `python run_all.py --stage research --open-holdout` | Gate 1. Run once, after freezing code; the run is time-stamped |

## Layout

```
config/            tickers.yaml (universe + Bloomberg tickers), params.yaml, costs.yaml
macro_alpha/
  ingest/          bbg.py (xbbg), wrds_pull.py, external.py (ENSO plume, ONI, CFTC)
  clean.py         raw -> Panel: returns, carry, value, releases, cot, plume (point-in-time)
  synthetic.py     planted-effect panel for testing
  risk.py          EWMA vol, Ledoit-Wolf covariance
  signals/         tsmom, carry, value, surprise, cot, enso (+ base helpers)
  backtest/        portfolio.py (weights, timing, costs), metrics.py, combine.py
  validate/        overfit.py (DSR, PBO-CSCV), inference.py (spanning, bootstrap, SPA), trials.py
  research.py      the full research run and gate logic
  report.py        tear sheets + Markdown summary
run_all.py         one command, raw pulls -> final figures
tests/             look-ahead, timing, vol-target, roll, DSR/PBO tests
```

## Outputs (`reports/`)

`summary.md` (all tables), `gates.csv`, `signal_performance.csv`, `parameter_grid.csv`, `robustness.csv`, `combined_performance.csv`, `combined_inference.csv`, `ic_decay_*.csv`, `figures/tearsheet_*.png`, `figures/combined.png`, plus `trials_log.csv` and `trials/returns.parquet` (every variant ever run, which feeds the DSR and PBO).
