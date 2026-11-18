#!/usr/bin/env python3
"""Profile a new company's invoice file + tree and write a draft config to review (for the blind-test C).

  python scripts/draft_config.py --data path/to/invoices.csv --tree path/to/tree.csv --company C \
      --currency USD --out configs/C.yaml

Prints a profile (null %, distinct values, samples, status values, guessed formats) and writes a YAML with the
best-guess mapping. Always read the profile and fix the YAML before running.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spendcube.ingest import read_table  # noqa: E402
from spendcube.parse import parse_number  # noqa: E402

PATTERNS = {
    "id": r"^(line_?id|record_?id|row_?id|id|line_?no|uid|transaction_?id|line_?key)$",
    "gross_amount": r"gross|incl|inc_?vat|total_?incl|with_?tax",
    "amount": r"amount|value|net|spend|line_?total|total|cost|extended|price_?total|usd|eur|zar|gbp",
    "tax_rate": r"(vat|tax|gst)_?(rate|pct|percent|%)",
    "tax_amount": r"(vat|tax|gst)_?(amount|amt|value)",
    "currency": r"^(currency|curr|ccy|currency_?code|doc_?currency|document_?currency|transaction_?currency)$",
    "rate": r"(fx|exchange|currency)_?rate|^rate$",
    "date": r"date|_dt$|^dt_|posted|period",
    "supplier_name": r"(supplier|vendor|payee|merchant|creditor)_?(name|desc|as_captured|legal)?",
    "supplier_id": r"(supplier|vendor|payee|creditor)_?(id|no|num|number|code|key)",
    "description": r"desc|text|narrat|memo|item_?name|product_?name|service|line_?item|comment|details|short_?text",
    "item_code": r"(material|item|sku|part|product|article)_?(no|num|number|code|id)?$",
    "status": r"status|state|approval|workflow",
    "invoice_number": r"invoice_?(no|num|number|id|ref)?$|^inv_?(no|num)|document_?(no|number)",
    "invoice_line": r"(invoice_?)?line_?(no|num|number|item)?$",
    "po": r"^po$|po_?(no|num|number|ref|reference)|purchase_?order",
    "quantity": r"^qty$|quantity",
    "account": r"gl|account|ledger|nominal|cost_?(centre|center|element)|expense|category|spend_?type|department|business_?unit|plant|site|entity|company_?code|region|type",
}


def guess(df: pd.DataFrame):
    cols = list(df.columns)
    lc = {c: re.sub(r"[^a-z0-9%]+", "_", c.lower()).strip("_") for c in cols}
    used, g = set(), {}

    def pick(key, pred=lambda c: True, multi=False):
        hits = [c for c in cols if c not in used and re.search(PATTERNS[key], lc[c]) and pred(c)]
        if not hits:
            return [] if multi else None
        if multi:
            used.update(hits)
            return hits
        used.add(hits[0])
        return hits[0]

    num = lambda c: parse_number(df[c].dropna().astype(str).head(500)).notna().mean() > 0.9 if df[c].notna().any() else False
    g["id"] = pick("id", lambda c: df[c].is_unique) or next((c for c in cols if df[c].is_unique), cols[0])
    used.add(g["id"])
    g["tax_rate"] = pick("tax_rate")
    g["tax_amount"] = pick("tax_amount", num)
    g["gross_amount"] = pick("gross_amount", num)
    g["amount"] = pick("amount", num)
    g["rate"] = pick("rate", num)
    g["currency"] = pick("currency", lambda c: df[c].dropna().astype(str).str.fullmatch(r"[A-Za-z]{3}").mean() > 0.8)
    g["quantity"] = pick("quantity", num)
    g["supplier_id"] = pick("supplier_id")
    g["supplier_name"] = pick("supplier_name", lambda c: not num(c))
    g["item_code"] = pick("item_code", lambda c: df[c].nunique() > 20)
    g["status"] = pick("status", lambda c: df[c].nunique() < 30)
    g["invoice_number"] = pick("invoice_number")
    g["invoice_line"] = pick("invoice_line")
    g["po"] = pick("po")
    g["dates"] = pick("date", multi=True)
    g["description"] = pick("description", lambda c: not num(c) and df[c].nunique() > 20
                            and not re.search(r"^(gl|account|ledger|nominal|cost|plant|vendor|supplier|status)", lc[c]), multi=True)
    g["features"] = [c for c in pick("account", multi=True) if df[c].nunique() < max(5000, len(df) // 5)]
    return g


def date_formats(s: pd.Series):
    s = s.dropna().astype(str).str.strip()
    shapes = s.str.replace(r"\d", "9", regex=True).str.replace(r"[A-Za-z]{3,9}", "Mon", regex=True).value_counts()
    fm = {"9999-99-99": "%Y-%m-%d", "99/99/9999": "%d/%m/%Y", "99-Mon-99": "%d-%b-%y", "99-Mon-9999": "%d-%b-%Y",
          "99.99.9999": "%d.%m.%Y", "99999999": "%Y%m%d", "9999/99/99": "%Y/%m/%d", "9999-99-99 99:99:99": "%Y-%m-%d %H:%M:%S",
          "9/99/9999": "%m/%d/%Y", "99-99-9999": "%d-%m-%Y", "9 Mon 9999": "%d %b %Y", "99 Mon 9999": "%d %b %Y"}
    out = []
    for shp in shapes.index[:6]:
        if shp in fm and fm[shp] not in out:
            out.append(fm[shp])
    # US vs EU day/month: if any first field > 12 it is day-first
    sl = s[s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")]
    if len(sl):
        a = sl.str.split("/").str[0].astype(int)
        b = sl.str.split("/").str[1].astype(int)
        out = [f for f in out if "/" not in f] + (["%m/%d/%Y"] if (b > 12).any() and not (a > 12).any() else ["%d/%m/%Y"])
    return out or ["%Y-%m-%d"], shapes.head(5).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--company", default="C")
    ap.add_argument("--currency", default="USD", help="reporting currency")
    ap.add_argument("--fx", default=None, help="optional FX table file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = read_table(a.data)
    print(f"== {a.data}: {len(df)} rows x {len(df.columns)} cols")
    prof = pd.DataFrame({"null%": (df.isna().mean() * 100).round(1), "distinct": df.nunique(),
                         "sample": [" | ".join(map(str, df[c].dropna().astype(str).unique()[:4]))[:90] for c in df.columns]})
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 90):
        print(prof)
    g = guess(df)
    print("\n== guessed mapping"); [print(f"  {k:15s} {v}") for k, v in g.items()]
    if g["status"]:
        print("\n== status values:", df[g["status"]].value_counts(dropna=False).to_dict())
    fmts, shapes = [], {}
    for c in g["dates"]:
        f, sh = date_formats(df[c])
        fmts += [x for x in f if x not in fmts]
        print(f"== date {c}: shapes {sh} -> {f}")
    tree = read_table(a.tree)
    print(f"\n== tree {a.tree}: columns {list(tree.columns)}  rows {len(tree)}")
    print(tree.head(8).to_string())

    amount_col = g["gross_amount"] or g["amount"]
    basis = "gross" if g["gross_amount"] and not g["amount"] else "net"
    src = {
        "file": str(Path(a.data).as_posix()) if Path(a.data).is_absolute() else "{data_dir}/" + Path(a.data).name,
        "id_col": g["id"],
        "amount": {"col": amount_col, "basis": basis},
        "dates": {re.sub(r"[^a-z0-9]+", "_", c.lower()).replace("_date", "").strip("_") or "date": c for c in g["dates"]},
        "date_formats": fmts or ["%Y-%m-%d"],
        "date_window": ["2015-01-01", "2027-12-31"],
        "description": g["description"],
        "features": g["features"],
    }
    if basis == "gross":
        if g["tax_rate"]:
            src["amount"].update({"tax_rate_col": g["tax_rate"], "default_tax_rate": 0.0})
        if g["tax_amount"]:
            src["amount"]["tax_amount_col"] = g["tax_amount"]
    if g["currency"]:
        src["currency"] = {"col": g["currency"], "default": a.currency}
        if g["rate"]:
            src["currency"]["rate_col"] = g["rate"]
    else:
        src["currency"] = {"fixed": a.currency}
    if g["supplier_name"] or g["supplier_id"]:
        src["supplier"] = {k: v for k, v in {"name": g["supplier_name"], "id": g["supplier_id"]}.items() if v}
    if g["item_code"]:
        src["item"] = {"code": g["item_code"]}
    for k in ["quantity", "invoice_line"]:
        if g[k]:
            src[k] = g[k]
    if g["invoice_number"]:
        src["invoice_number"] = {"col": g["invoice_number"]}
    if g["po"]:
        src["po"] = {"col": g["po"]}
    if g["status"]:
        src["status"] = {"col": g["status"]}
    if g["invoice_number"]:
        src["reversals"] = "cancel"
    # exact duplicates on every column except the id (safe default). Narrow with key_cols only after checking
    # that the dropped columns are noise (e.g. a load timestamp); without a line number, never narrow.
    src["dedup"] = {"exact": True}
    cfg = {"company": a.company, "reporting_currency": a.currency,
           "taxonomy": {"file": "{data_dir}/" + Path(a.tree).name},
           "sources": {a.company: src},
           "classification": {"meta_features": g["features"][:10], "zero_shot_extra_text": []}}
    if a.fx:
        cfg["fx"] = {"file": "{data_dir}/" + Path(a.fx).name, "currency_col": "currency", "month_col": "month",
                     "rate_col": "units_per_usd", "quote": "units_per_reporting", "row_rate_quote": "units_per_reporting"}
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120)
    print("\n== draft config\n" + text)
    if a.out:
        Path(a.out).write_text("# DRAFT written by scripts/draft_config.py - review every mapping before running\n" + text)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
