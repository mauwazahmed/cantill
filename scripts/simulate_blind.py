#!/usr/bin/env python3
"""Rehearse the blind test: split the kit into a 'train' part and a simulated A2/B2 (renamed files, no labels),
run the tool exactly as on the day, and score A2/B2 against the held-out labels.

  python scripts/simulate_blind.py --frac 0.2 --work /tmp/blind_sim
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluate import pair_f1  # noqa: E402


def split_A(src: Path, train: Path, blind: Path, frac: float, rs):
    (train / "company_A").mkdir(parents=True, exist_ok=True)
    blind.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(src / "erp_goods_receipts.csv", dtype=str)
    t = pd.read_csv(src / "invoice_tool_export.csv", dtype=str)
    lab = pd.read_csv(src / "labels_A.csv", dtype=str)
    he = rs.rand(len(e)) < frac
    # tool rows: hold out a random share, plus those whose PO belongs to a held-out ERP line
    po_held = set(e.loc[he, "po_number"])
    tpo = t.po_reference.str.extract(r"(4\d{9})")[0]
    ht = (rs.rand(len(t)) < frac) | (tpo.isin(po_held) & (rs.rand(len(t)) < 0.7))
    e[~he].to_csv(train / "company_A/erp_goods_receipts.csv", index=False)
    t[~ht].to_csv(train / "company_A/invoice_tool_export.csv", index=False)
    lab[~lab.record_id.str[4:].isin(e.loc[he, "line_id"])].to_csv(train / "company_A/labels_A.csv", index=False)
    shutil.copy(src / "taxonomy_A.csv", train / "company_A/taxonomy_A.csv")
    e[he].to_csv(blind / "erp_goods_receipts_A2.csv", index=False)
    t[ht].to_csv(blind / "invoice_tool_export_A2.csv", index=False)
    return lab[lab.record_id.str[4:].isin(e.loc[he, "line_id"])].set_index("record_id")


def split_B(src: Path, train: Path, blind: Path, frac: float, rs):
    (train / "company_B").mkdir(parents=True, exist_ok=True)
    blind.mkdir(parents=True, exist_ok=True)
    b = pd.read_csv(src / "invoice_lines.csv", dtype=str)
    s = pd.read_csv(src / "seed_labels.csv", dtype=str)
    hb = rs.rand(len(b)) < frac
    b[~hb].to_csv(train / "company_B/invoice_lines.csv", index=False)
    s[~s.line_id.isin(b.loc[hb, "line_id"])].to_csv(train / "company_B/seed_labels.csv", index=False)
    for f in ["taxonomy_B.csv", "fx_rates_usd.csv"]:
        shutil.copy(src / f, train / "company_B" / f)
    shutil.copy(src / "fx_rates_usd.csv", blind / "fx_rates_usd.csv")
    b[hb].to_csv(blind / "invoice_lines_B2.csv", index=False)
    return s[s.line_id.isin(b.loc[hb, "line_id"])].set_index("line_id")


def score(out_csv: Path, lab: pd.DataFrame, cols, keys=False):
    o = pd.read_csv(out_csv, dtype=str, keep_default_na=False).set_index("record_id")
    lab = lab[lab.index.isin(o.index)]
    if "include" in lab.columns:
        lab = lab[lab.include == "1"]
    o2 = o.loc[lab.index]
    ok = o2.include == "1"
    w = o2.spend.astype(float).abs()
    msg = []
    for k in range(1, len(cols) + 1):
        p = o2[[f"l{i}" for i in range(1, k + 1)]].apply(lambda r: "|".join(r), axis=1)
        t = lab[cols[:k]].apply(lambda r: "|".join(r), axis=1)
        hit = (p == t) & ok
        msg.append(f"L{k} {hit.mean():.3f} (spend-w {(hit * w).sum() / w.sum():.3f})")
    print("   classification:", "  ".join(msg))
    if keys:
        print("   supplier P/R/F1", pair_f1(lab.supplier_key, o2.supplier_key),
              " item P/R/F1", pair_f1(lab.item_key[lab.item_key != ""], o2.item_key[lab.item_key != ""]))
    print("   rows", len(o), "included", int((o.include == "1").sum()), "excluded",
          o.exclusion_reason.replace("", np.nan).value_counts().to_dict(), "total spend", round(o.spend.astype(float).sum(), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--work", default="/tmp/blind_sim")
    ap.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args()
    work = Path(a.work)
    if work.exists():
        shutil.rmtree(work)
    rs = np.random.RandomState(7)
    labA = split_A(Path(a.data) / "company_A", work / "train", work / "A2", a.frac, rs)
    labB = split_B(Path(a.data) / "company_B", work / "train", work / "B2", a.frac, rs)
    for comp, lab, keys in [("A", labA, True), ("B", labB, False)]:
        cmd = [sys.executable, str(ROOT / "run.py"), "--config", str(ROOT / f"configs/{comp}.yaml"),
               "--data-dir", str(work / f"{comp}2"), "--train-dir", str(work / "train"),
               "--out", str(work / "out"), "--suffix", f"{comp}2", "--quiet"]
        print(f"== {comp}2: {' '.join(cmd[1:])}")
        subprocess.run(cmd, check=True)
        score(work / f"out/mauwaz_ahmed__{comp}2.csv", lab, ["category_l1", "category_l2", "category_l3"], keys)


if __name__ == "__main__":
    main()
