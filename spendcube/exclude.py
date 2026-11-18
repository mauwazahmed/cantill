"""Exclusion logic: status, non-procurement rules, reversals and within-source duplicates.

Every excluded record gets one code from: CANCELLED, NOT_APPROVED, NON_PROCUREMENT, DUPLICATE.
Precedence when several apply: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .text import compact, supplier_norm

PRECEDENCE = {"CANCELLED": 0, "NOT_APPROVED": 1, "NON_PROCUREMENT": 2, "DUPLICATE": 3}


def set_code(df: pd.DataFrame, mask: pd.Series, code: str, why: str):
    """Assign an exclusion code unless a higher-precedence code is already set."""
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    cur = df["exclusion_reason"]
    better = cur.isna() | cur.map(lambda c: PRECEDENCE.get(c, 9) > PRECEDENCE[code])
    m = mask & better
    df.loc[m, "exclusion_reason"] = code
    df.loc[m, "exclusion_detail"] = why


def apply_status(df: pd.DataFrame, rules: list[dict], src_cfg: dict):
    st = df["status_raw"].fillna("").astype(str).str.strip().str.lower()
    has = st != ""
    if not has.any():
        return
    explicit = {}
    for code, vals in (src_cfg.get("status", {}) if isinstance(src_cfg.get("status"), dict) else {}).items():
        if code in PRECEDENCE or code == "INCLUDE":
            for v in vals:
                explicit[str(v).strip().lower()] = code
    done = pd.Series(False, index=df.index)
    for v, code in explicit.items():
        m = has & (st == v)
        if code != "INCLUDE":
            set_code(df, m, code, f"status={v}")
        done |= m
    for r in rules:
        m = has & ~done & st.str.contains(r["pattern"], regex=True, case=False)
        if r.get("code"):
            set_code(df, m, r["code"], f"status~{r['pattern']}")
        done |= m


ACCOUNT_LIKE = re.compile(r"gl|account|ledger|nominal|cost[_ ]?element|expense[_ ]?type|spend[_ ]?type|category", re.I)


def _expand_fields(df: pd.DataFrame, fields: list[str]) -> list[str]:
    cols = []
    for f in fields:
        if f == "@account":
            cols += [c for c in df.columns if c.startswith("f_") and ACCOUNT_LIKE.search(c[2:])]
        elif f == "@all":
            cols += ["desc", "supplier_name"] + [c for c in df.columns if c.startswith("f_")]
        elif f in df.columns:
            cols.append(f)
        elif f"f_{f}" in df.columns:
            cols.append(f"f_{f}")
    return list(dict.fromkeys(cols))


def apply_rules(df: pd.DataFrame, rules: list[dict]):
    """Regex rules over canonical fields (desc, supplier_name, f_<feature>, @account, @all)."""
    for r in rules:
        fields = r.get("fields", ["desc"])
        rx = re.compile(r["pattern"], re.I)
        hit = pd.Series(False, index=df.index)
        for col in _expand_fields(df, fields):
            s = df[col].fillna("").astype(str)
            u = s.unique()
            mm = {x: bool(rx.search(x)) for x in u}
            hit |= s.map(mm)
        if r.get("unless"):
            rx2 = re.compile(r["unless"], re.I)
            for col in _expand_fields(df, r.get("unless_fields", fields)):
                hit &= ~df[col].fillna("").astype(str).map(lambda x: bool(rx2.search(x)))
        if r.get("sources"):
            hit &= df["src"].isin(r["sources"])
        set_code(df, hit, r.get("code", "NON_PROCUREMENT"), r.get("name", r["pattern"]))


def apply_reversals(df: pd.DataFrame, src: str):
    """An invoice line posted and then reversed with the exact opposite amount is a cancelled line:
    both the original and the reversal are excluded as CANCELLED (net effect on spend is zero either way)."""
    d = df[(df.src == src) & df["amount_raw"].notna() & (df["amount_raw"] != 0)]
    party = d["supplier_id"].fillna(d["supplier_name"]).fillna("")
    key = pd.DataFrame({"p": party, "inv": d["inv_no"].fillna(""), "ln": d["inv_line"].fillna(""),
                        "cur": d["currency"].fillna(""), "abs": d["amount_raw"].abs().round(2)})
    key = key[key.inv != ""]
    sign = np.sign(d.loc[key.index, "amount_raw"])
    hits = []
    for _, g in key.assign(sign=sign).groupby(["p", "inv", "ln", "cur", "abs"], sort=False):
        if len(g) < 2:
            continue
        pos, neg = list(g.index[g.sign > 0]), list(g.index[g.sign < 0])
        n = min(len(pos), len(neg))
        hits += pos[:n] + neg[:n]
    set_code(df, df.index.isin(hits), "CANCELLED", "reversed by an equal and opposite posting")


def dedupe_exact(df: pd.DataFrame, src: str, raw: pd.DataFrame, key_cols: list[str] | None, ignore: list[str]):
    """Rows of one source identical on the key columns: keep the first, others DUPLICATE."""
    d = df[df.src == src]
    cols = key_cols or [c for c in raw.columns if c not in set(ignore)]
    k = raw.loc[:, cols].fillna("").astype(str).agg("\x1f".join, axis=1)
    k.index = d.index
    # keep a copy that is not already excluded for another reason
    order = df.loc[d.index, "exclusion_reason"].notna().astype(int).sort_values(kind="stable").index
    dup = k.loc[order].duplicated(keep="first").reindex(df.index, fill_value=False)
    set_code(df, dup, "DUPLICATE", "same record repeated in the source")


def dedupe_fuzzy_invoices(df: pd.DataFrame, src: str, text_threshold: float = 85.0):
    """Invoice lines exported twice by a tool: same invoice number (digits), same gross amount and quantity,
    same supplier and near-identical line text (OCR noise allowed). PO item numbers, when both present, must agree."""
    d = df[(df.src == src) & df["inv_core"].notna() & df["amount_raw"].notna()].copy()
    d["comp"] = d["desc"].map(compact)
    d["sup"] = d["supplier_name"].map(supplier_norm)
    dups = []
    for _, g in d.groupby(["inv_core", d["amount_raw"].round(2), d["qty"].fillna(-1).round(4)], sort=False):
        if len(g) < 2:
            continue
        g = g.assign(_x=g["exclusion_reason"].notna()).sort_values(["_x", "record_id"])
        kept = []
        for i, r in g.iterrows():
            is_dup = False
            for j in kept:
                k = g.loc[j]
                if pd.notna(r.po_item) and pd.notna(k.po_item) and r.po_item != k.po_item:
                    continue
                if pd.notna(r.po) and pd.notna(k.po) and r.po != k.po:
                    continue
                if fuzz.ratio(r.sup, k.sup) < 85:
                    continue
                if fuzz.ratio(r.comp, k.comp) >= text_threshold:
                    is_dup = True
                    break
            if is_dup:
                dups.append(i)
            else:
                kept.append(i)
    set_code(df, df.index.isin(dups), "DUPLICATE", "invoice line exported twice")
