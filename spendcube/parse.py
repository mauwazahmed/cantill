"""Parsing helpers: numbers, dates, rates. Everything is vectorised over pandas Series."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

_NUM_CLEAN = re.compile(r"[^\d,.\-()]")


def _one_number(s: str) -> float:
    """Parse one amount string in any common format.

    Handles: 1234.56 | 1,234.56 | 1.234,56 | 1 234,56 | (123.45) | 123.45- | R 1,234 | 15%.
    Returns NaN if nothing parseable.
    """
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "n/a"}:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1]
    if s.endswith("-") and not s.startswith("-"):
        neg, s = True, s[:-1]
    s = _NUM_CLEAN.sub("", s.replace(" ", "").replace(" ", ""))
    if s.startswith("-"):
        neg, s = (not neg), s[1:]
    if not s:
        return np.nan
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        # the right-most separator is the decimal one
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_c:
        parts = s.split(",")
        # 1,234 or 1,234,567 -> thousands; 12,5 or 1234,56 -> decimal
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif has_d and s.count(".") > 1:
        s = s.replace(".", "")
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_number(series: pd.Series) -> pd.Series:
    if series is None:
        return None
    s = series.astype("string")
    # fast path: already clean numbers
    fast = pd.to_numeric(s, errors="coerce")
    bad = fast.isna() & s.notna() & (s.str.strip() != "")
    if bad.any():
        uniq = s[bad].unique()
        m = {u: _one_number(u) for u in uniq}
        fast = fast.astype(float)
        fast[bad] = s[bad].map(m).astype(float)
    return fast.astype(float)


def parse_rate(series: pd.Series, default: float | None = None) -> pd.Series:
    """Tax rate as a fraction: '15', '15%', '0.15' -> 0.15. Blank -> default."""
    v = parse_number(series.astype("string").str.replace("%", "", regex=False))
    v = v.where(~(v > 1), v / 100.0)
    if default is not None:
        v = v.fillna(default)
    return v


def parse_date(series: pd.Series, formats: list[str] | None = None, dayfirst: bool = True,
               window: tuple[str, str] | None = None) -> tuple[pd.Series, pd.Series]:
    """Parse dates trying explicit formats first, then a tolerant fallback.

    Returns (dates, flag) where flag is '' ok, 'MISSING', 'UNPARSEABLE' or 'OUT_OF_RANGE'.
    Out-of-range dates are kept in a flag but set to NaT so callers fall back to another date.
    """
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    todo = s.notna() & (s != "")
    for fmt in formats or []:
        if not todo.any():
            break
        d = pd.to_datetime(s[todo], format=fmt, errors="coerce")
        hit = d.notna()
        out.loc[d.index[hit]] = d[hit]
        todo.loc[d.index[hit]] = False
    if todo.any():
        uniq = s[todo].unique()
        m = {}
        for u in uniq:
            try:
                m[u] = pd.to_datetime(u, dayfirst=dayfirst)
            except Exception:  # noqa: BLE001
                m[u] = pd.NaT
        out.loc[todo] = s[todo].map(m)
    flag = pd.Series("", index=s.index, dtype=object)
    flag[s.isna() | (s == "")] = "MISSING"
    flag[(flag == "") & out.isna()] = "UNPARSEABLE"
    if window:
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        oor = out.notna() & ((out < lo) | (out > hi))
        flag[oor] = "OUT_OF_RANGE"
        out[oor] = pd.NaT
    return out, flag
