"""Cross-system matching: find secondary-system records (e.g. AP-tool invoice lines) that describe the same
purchase as a primary-system record (e.g. ERP goods receipt).

Candidates share the net amount (to the cent, small tolerance), or the PO with an amount within 0.5 %.
Each candidate pair is scored on PO, PO item, quantity, supplier name and description similarity, and
accepted by explicit rules (see `accept`). Several secondary records may point at one primary record
(an invoice exported twice is still a duplicate of the same receipt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .text import clean_text, supplier_norm


def _cents(x: pd.Series) -> pd.Series:
    return (x * 100).round().astype("Int64")


def _first_date(d: pd.DataFrame) -> pd.Series:
    cols = [c for c in d.columns if c.startswith("date_") and not c.endswith("_flag")]
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    for c in cols:
        out = out.fillna(d[c])
    return out


def candidate_pairs(P: pd.DataFrame, S: pd.DataFrame, cent_tol: int = 2, po_rel_tol: float = 0.005) -> pd.DataFrame:
    p = pd.DataFrame({"pi": P.index, "pc": _cents(P["net"]), "ppo": P["po"]})
    alts = [pd.DataFrame({"si": S.index, "sc": _cents(S["net"]), "alt": "net"})]
    no_rate = S.index[~S["tax_rate_given"].astype(bool) & S["tax_basis"].eq("gross")]
    if len(no_rate):  # blank tax rate: also try "no tax" for matching purposes only
        alts.append(pd.DataFrame({"si": no_rate, "sc": _cents(S.loc[no_rate, "amount_raw"]), "alt": "gross"}))
    s = pd.concat(alts, ignore_index=True).dropna(subset=["sc"])
    out = []
    for d in range(-cent_tol, cent_tol + 1):
        out.append(s.assign(k=s["sc"] + d).merge(p.dropna(subset=["pc"]), left_on="k", right_on="pc")[["si", "pi", "alt"]])
    # same PO, amount within relative tolerance (VAT rounding at another rate, etc.)
    sp = S[S["po"].notna()]
    if len(sp):
        m = pd.DataFrame({"si": sp.index, "po": sp["po"].values, "sn": sp["net"].values}).merge(
            pd.DataFrame({"pi": P.index, "po": P["po"].values, "pn": P["net"].values}), on="po")
        m = m[(m.sn - m.pn).abs() <= po_rel_tol * m.pn.abs().clip(lower=1)]
        out.append(m[["si", "pi"]].assign(alt="po_amount"))
    c = pd.concat(out, ignore_index=True).drop_duplicates(["si", "pi"])
    # guard against huge candidate sets on tiny/common amounts: keep the 300 nearest by id per secondary row
    cnt = c.groupby("si").size()
    big = cnt[cnt > 300].index
    if len(big):
        c = pd.concat([c[~c.si.isin(big)], c[c.si.isin(big)].groupby("si").head(300)])
    return c


def score_pairs(c: pd.DataFrame, P: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    sp, pp = S.loc[c.si], P.loc[c.pi]
    c = c.copy()
    s_po, p_po = sp["po"].astype("string").values, pp["po"].astype("string").values
    c["po_eq"] = pd.notna(s_po) & pd.notna(p_po) & (s_po == p_po)
    c["po_conflict"] = pd.notna(s_po) & pd.notna(p_po) & (s_po != p_po)
    si_, pi_ = sp["po_item"].values, pp["po_item"].values
    c["poi_eq"] = pd.notna(si_) & pd.notna(pi_) & (si_ == pi_)
    c["poi_conflict"] = pd.notna(si_) & pd.notna(pi_) & (si_ != pi_)
    sq, pq = sp["qty"].values, pp["qty"].values
    c["qty_eq"] = pd.notna(sq) & pd.notna(pq) & (np.abs(sq - pq) <= 1e-6 + 1e-4 * np.abs(pq))
    st = sp["desc"].map(clean_text).values
    pt = pp["desc"].map(clean_text).values
    c["txt"] = [fuzz.token_set_ratio(a, b) if a and b else 0.0 for a, b in zip(st, pt)]
    c["txt2"] = [fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")) if a and b else 0.0 for a, b in zip(st, pt)]
    c["txt"] = np.maximum(c["txt"], c["txt2"])
    ss = sp["supplier_name"].map(supplier_norm).values
    ps = pp["supplier_name"].map(supplier_norm).values
    c["sup"] = [fuzz.ratio(a, b) if a and b else 0.0 for a, b in zip(ss, ps)]
    # date gap (days) between the two records, when both carry a date
    sd = _first_date(sp)
    pdt = _first_date(pp)
    c["dd"] = np.abs((sd.values - pdt.values) / np.timedelta64(1, "D"))
    c["score"] = (3 * c.po_eq - 4 * c.po_conflict + 1 * c.poi_eq - 1 * c.poi_conflict + 1.5 * c.qty_eq
                  + 2.5 * c.txt / 100 + 2 * c.sup / 100 + 0.5 * (c.alt == "net")
                  - 0.5 * (c.dd.fillna(0) > 60) - 1.0 * (c.dd.fillna(0) > 180))
    return c


def accept(c: pd.DataFrame, max_gap_days: float = 120) -> pd.Series:
    strong_text = c.txt >= 70
    good_sup = c.sup >= 80
    date_ok = c.dd.isna() | (c.dd <= max_gap_days)
    return (
        (c.po_eq & (c.qty_eq | strong_text) & ~c.poi_conflict)
        | (c.po_eq & c.poi_eq & good_sup)
        # no PO on one side: need amount + quantity + text + supplier, and dates (if any) close together
        | (~c.po_conflict & c.qty_eq & strong_text & good_sup & date_ok)
        | (~c.po_conflict & (c.txt >= 85) & good_sup & (c.alt != "po_amount") & date_ok)
    )


def match_sources(df: pd.DataFrame, primary: str, secondary: str, secondary_mask=None) -> pd.DataFrame:
    P = df[df.src == primary]
    S = df[(df.src == secondary) & (secondary_mask if secondary_mask is not None else True)]
    c = candidate_pairs(P, S)
    if c.empty:
        return pd.DataFrame(columns=["si", "pi", "score"])
    c = score_pairs(c, P, S)
    c = c[accept(c)].sort_values(["si", "score"], ascending=[True, False])
    best = c.drop_duplicates("si")
    return best[["si", "pi", "score", "po_eq", "qty_eq", "txt", "sup", "alt"]]
