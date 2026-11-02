"""One command from raw pulls to final figures.

    python run_all.py --synthetic                 # plumbing test, no data needed
    python run_all.py --stage ingest              # Bloomberg + WRDS pulls (Terminal machine)
    python run_all.py --stage clean               # raw -> data/clean panel + QA report
    python run_all.py --stage research            # signals, tests, gates, combination (holdout sealed)
    python run_all.py --stage research --open-holdout   # gate 3: run ONCE, after freezing code

Outputs land in reports/ (CSV, Markdown summary, PNG tear sheets) and trials_log.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
from pathlib import Path

from macro_alpha import config
from macro_alpha.clean import Panel, build_panel

ROOT = Path(__file__).resolve().parent
HOLDOUT_FLAG = ROOT / "reports" / ".holdout_opened"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["ingest", "clean", "research", "all"], default="all")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic panel (plumbing test)")
    ap.add_argument("--open-holdout", action="store_true", help="include the sealed holdout sample")
    ap.add_argument("--overwrite", action="store_true", help="re-pull raw files that already exist")
    ap.add_argument("--out", default=None, help="report directory (default reports/ or reports_synthetic/)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tickers, params, costs = config.load_all()
    meta = config.asset_meta(tickers, costs)
    raw_dir = ROOT / params["data"]["raw_dir"]
    clean_dir = ROOT / params["data"]["clean_dir"]

    if args.synthetic:
        from macro_alpha.synthetic import make_panel
        out = Path(args.out) if args.out else ROOT / "reports_synthetic"
        if out.exists():
            shutil.rmtree(out)                      # synthetic runs start clean, incl. trial log
        panel = make_panel(meta, tickers["macro"], end=params["sample"]["holdout_end"])
        _research(panel, params, out, args.open_holdout)
        return

    if args.stage in ("ingest", "all"):
        from macro_alpha.ingest import bbg, wrds_pull
        bbg.pull_all(tickers, params, raw_dir, args.overwrite)
        wrds_pull.pull_all(params, raw_dir, args.overwrite)

    if args.stage in ("clean", "all"):
        panel = build_panel(raw_dir, tickers, params, meta)
        panel.save(clean_dir)
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports" / "data_qa.md").write_text("# Data QA\n\n" + "\n".join(f"- {q}" for q in panel.qa))

    if args.stage in ("research", "all"):
        panel = Panel.load(clean_dir)
        out = Path(args.out) if args.out else ROOT / "reports"
        if args.open_holdout:
            if HOLDOUT_FLAG.exists():
                logging.warning("Holdout already opened on %s. Results are no longer a clean test; "
                                "this run is recorded.", HOLDOUT_FLAG.read_text().strip())
            HOLDOUT_FLAG.parent.mkdir(parents=True, exist_ok=True)
            with open(HOLDOUT_FLAG, "a") as fh:
                fh.write(dt.datetime.now().isoformat(timespec="seconds") + "\n")
        _research(panel, params, out, args.open_holdout)


def _research(panel, params, out: Path, holdout_open: bool) -> None:
    from macro_alpha.research import run_research
    res = run_research(panel, params, out, holdout_open)
    print("\nGates:\n", res["gates"][["sharpe_design", "sharpe_validation", "t_nw", "dsr", "pbo",
                                      "passes_all"]].round(2).to_string())
    print(f"\nSummary: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
