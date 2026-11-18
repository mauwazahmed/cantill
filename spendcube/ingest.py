"""Read the raw files described by a company config into one canonical table.

Canonical columns (one row per input record):
  record_id, src, src_priority,
  amount_raw, amount_ok, tax_basis, tax_rate, currency, rate_raw,
  date_<role> + date_<role>_flag for every configured date role,
  supplier_name, supplier_id, item_code, desc, qty, po, po_item, inv_no, inv_core, inv_line, status_raw,
  f_<col> for every configured feature column (kept as text for rules and the classifier).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .parse import parse_number, parse_rate, parse_date


def read_table(path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, sheet_name=cfg.get("sheet", 0))
    enc = cfg.get("encoding")
    sep = cfg.get("sep")
    if sep is None:
        with open(path, "r", encoding=enc or "utf-8", errors="replace") as fh:
            sample = fh.read(20000)
        try:
            sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            sep = ","
    for e in ([enc] if enc else ["utf-8", "utf-8-sig", "cp1252", "latin-1"]):
        try:
            return pd.read_csv(path, dtype=str, sep=sep, encoding=e, keep_default_na=True, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot read {path}")


def _col(df: pd.DataFrame, name: str | None, default=None) -> pd.Series:
    if name and name in df.columns:
        return df[name]
    if name and name not in df.columns:
        raise KeyError(f"column '{name}' not in file (have: {list(df.columns)[:40]})")
    return pd.Series(default, index=df.index, dtype=object)


def _join_cols(df: pd.DataFrame, cols) -> pd.Series:
    if not cols:
        return pd.Series("", index=df.index)
    if isinstance(cols, str):
        cols = [cols]
    parts = [df[c].fillna("").astype(str).str.strip() for c in cols if c in df.columns]
    if not parts:
        return pd.Series("", index=df.index)
    out = parts[0]
    for p in parts[1:]:
        # avoid repeating identical text (e.g. material text == service text)
        out = np.where((p != "") & (p != out), np.where(out != "", out + " ; " + p, p), out)
        out = pd.Series(out, index=df.index)
    return out.astype(str)


def ingest_source(name: str, sc: dict, resolve) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_table(resolve(sc["file"]), sc)
    raw.columns = [c.strip() for c in raw.columns]
    df = pd.DataFrame(index=raw.index)
    prefix = sc.get("id_prefix", "")
    df["record_id"] = prefix + _col(raw, sc["id_col"]).astype(str).str.strip()
    df["src"] = name
    df["src_priority"] = sc.get("priority", 1)

    # ---- amount & tax
    a = sc["amount"]
    df["amount_raw"] = parse_number(_col(raw, a["col"]))
    if a.get("negate_col"):  # optional debit/credit indicator
        neg = _col(raw, a["negate_col"]).astype(str).str.upper().isin([str(x).upper() for x in a.get("negate_values", ["C", "H", "CR"])])
        df.loc[neg, "amount_raw"] = -df.loc[neg, "amount_raw"].abs()
    df["amount_ok"] = df["amount_raw"].notna()
    df["tax_basis"] = a.get("basis", "net")
    if a.get("basis") == "gross":
        if a.get("tax_rate_col"):
            df["tax_rate"] = parse_rate(_col(raw, a["tax_rate_col"]), a.get("default_tax_rate", 0.0))
            df["tax_rate_given"] = _col(raw, a["tax_rate_col"]).notna()
        else:
            df["tax_rate"] = a.get("default_tax_rate", 0.0)
            df["tax_rate_given"] = False
        if a.get("tax_amount_col"):
            df["tax_amount"] = parse_number(_col(raw, a["tax_amount_col"]))
    else:
        df["tax_rate"] = 0.0
        df["tax_rate_given"] = True

    # ---- currency
    c = sc.get("currency", {})
    if c.get("col"):
        df["currency"] = _col(raw, c["col"]).fillna(c.get("default", "")).astype(str).str.strip().str.upper()
    else:
        df["currency"] = c.get("fixed", c.get("default", ""))
    df["rate_raw"] = parse_number(_col(raw, c.get("rate_col"))) if c.get("rate_col") else np.nan

    # ---- dates
    fmts = sc.get("date_formats")
    window = sc.get("date_window")
    for role, col in (sc.get("dates") or {}).items():
        d, f = parse_date(_col(raw, col), fmts, sc.get("dayfirst", True), tuple(window) if window else None)
        df[f"date_{role}"], df[f"date_{role}_flag"] = d, f

    # ---- parties, items, text
    s = sc.get("supplier", {})
    df["supplier_name"] = _col(raw, s.get("name")).astype("string").str.strip() if s.get("name") else pd.NA
    df["supplier_id"] = _col(raw, s.get("id")).astype("string").str.strip() if s.get("id") else pd.NA
    it = sc.get("item", {})
    df["item_code"] = _col(raw, it.get("code")).astype("string").str.strip() if it.get("code") else pd.NA
    df["desc"] = _join_cols(raw, sc.get("description"))
    df["qty"] = parse_number(_col(raw, sc.get("quantity"))) if sc.get("quantity") else np.nan

    po = sc.get("po", {})
    if po.get("col"):
        p = _col(raw, po["col"]).astype("string")
        if po.get("regex"):
            df["po"] = p.str.extract(po["regex"], expand=False)
        else:
            df["po"] = p.str.replace(r"\.0$", "", regex=True).str.strip()
        if po.get("item_col"):
            df["po_item"] = parse_number(_col(raw, po["item_col"]))
        elif po.get("item_regex"):
            df["po_item"] = parse_number(p.str.extract(po["item_regex"], expand=False))
        else:
            df["po_item"] = np.nan
    else:
        df["po"], df["po_item"] = pd.NA, np.nan

    inv = sc.get("invoice_number", {})
    if isinstance(inv, str):
        inv = {"col": inv}
    if inv.get("col"):
        n = _col(raw, inv["col"]).astype("string").str.strip()
        df["inv_no"] = n
        rx = inv.get("core_regex", r"(\d+)\D*$")
        df["inv_core"] = n.str.extract(rx, expand=False).str.lstrip("0")
    else:
        df["inv_no"], df["inv_core"] = pd.NA, pd.NA
    df["inv_line"] = _col(raw, sc.get("invoice_line")).astype("string") if sc.get("invoice_line") else pd.NA

    st = sc.get("status", {})
    if isinstance(st, str):
        st = {"col": st}
    df["status_raw"] = _col(raw, st.get("col")).astype("string").str.strip() if st.get("col") else pd.NA

    for f in sc.get("features", []) or []:
        df[f"f_{f}"] = _col(raw, f).astype("string").str.strip()
    return df, raw


def ingest(cfg: dict, resolvers: list[tuple[str, object]]) -> tuple[pd.DataFrame, dict]:
    """Ingest every source once per (tag, resolver). Tag 'ref' = labelled history used for training and
    cross-period de-duplication; tag 'cur' = the records to output. Rows are ordered ref first."""
    frames, raws = [], {}
    for name, sc in cfg["sources"].items():
        parts = []
        for tag, resolve in resolvers:
            df, raw = ingest_source(name, sc, resolve)
            df["is_ref"] = tag == "ref"
            df["orig_id"] = df["record_id"]
            if tag == "ref":
                df["record_id"] = "REF|" + df["record_id"]
            frames.append(df)
            parts.append(raw)
        raws[name] = pd.concat(parts, ignore_index=True, sort=False)
    data = pd.concat(frames, ignore_index=True, sort=False)
    # sources are concatenated source by source; keep raw frames aligned with data rows per source
    dup = data.record_id.duplicated(keep=False)
    if dup.any():
        # keep record ids unique (output needs one row per input record)
        data.loc[dup, "record_id"] = data.loc[dup, "record_id"] + "#" + data[dup].groupby("record_id").cumcount().astype(str)
    return data, raws
