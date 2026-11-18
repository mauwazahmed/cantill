#!/usr/bin/env python3
"""One command: python run.py --config configs/A.yaml [--data-dir data] [--out outputs] [--name yourname]"""
import argparse
import json
import sys
import time
from pathlib import Path

from spendcube.pipeline import Pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description="Spend cube: ingest, clean, de-duplicate and classify purchase data")
    ap.add_argument("--config", required=True, nargs="+", help="company config(s) (YAML)")
    ap.add_argument("--data-dir", default=None, help="folder with the files to process; replaces {data_dir} in configs (default ./data)")
    ap.add_argument("--train-dir", default=None, help="folder with the labelled history (default ./data). When it differs from "
                    "--data-dir the history trains the classifier and golden keys and only --data-dir records are output")
    ap.add_argument("--file", action="append", default=[], metavar="CONFIG_PATH=ACTUAL_PATH",
                    help="override one configured file path without editing the config")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--name", default="mauwaz_ahmed", help="prefix for output files: <name>__<suffix>.csv")
    ap.add_argument("--suffix", default=None, help="output suffix (default: company code in the config)")
    ap.add_argument("--hide-labels", action="store_true", help="ignore labels (zero-label test)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cfg in a.config:
        t = time.time()
        ov = dict(x.split("=", 1) for x in a.file)
        p = Pipeline(cfg, a.data_dir, hide_labels=a.hide_labels, verbose=not a.quiet, train_dir=a.train_dir,
                     file_overrides=ov)
        out, _ = p.run()
        suffix = a.suffix or str(p.cfg.get("output_suffix", p.cfg.get("company")))
        f = out_dir / f"{a.name}__{suffix}.csv"
        out.to_csv(f, index=False)
        aux = out_dir / "_work"          # summaries + record-level working files (contain client text): git-ignored
        aux.mkdir(exist_ok=True)
        p.diagnostics().to_csv(aux / f"diagnostics__{suffix}.csv", index=False)
        p.review().to_csv(aux / f"review__{suffix}.csv", index=False)
        s = p.summary()
        (aux / f"summary__{suffix}.json").write_text(json.dumps(s, indent=2))
        print(json.dumps({k: s[k] for k in ["company", "records", "included", "excluded_by_reason", "total_spend",
                                            "suppliers", "items", "classification_mode"]}, indent=1))
        print(f"wrote {f} ({len(out)} rows) in {time.time() - t:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
