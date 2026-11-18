"""Net-of-tax amounts and conversion to the reporting currency."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest import read_table


def net_amount(df: pd.DataFrame) -> pd.Series:
    net = df["amount_raw"].copy()
    gross = df["tax_basis"].eq("gross")
    if "tax_amount" in df.columns:
        has_tax = gross & df["tax_amount"].notna()
        net[has_tax] = df.loc[has_tax, "amount_raw"] - df.loc[has_tax, "tax_amount"]
        gross = gross & ~has_tax
    net[gross] = df.loc[gross, "amount_raw"] / (1.0 + df.loc[gross, "tax_rate"].astype(float))
    return net.round(2)


class FxTable:
    def __init__(self, fx_cfg: dict | None, resolve):
        self.ok = bool(fx_cfg and fx_cfg.get("file"))
        self.quote = (fx_cfg or {}).get("quote", "units_per_reporting")
        if not self.ok:
            return
        t = read_table(resolve(fx_cfg["file"]))
        t = pd.DataFrame({
            "cur": t[fx_cfg.get("currency_col", "currency")].str.strip().str.upper(),
            "month": pd.PeriodIndex(pd.to_datetime(t[fx_cfg.get("month_col", "month")], errors="coerce"), freq="M"),
            "rate": pd.to_numeric(t[fx_cfg.get("rate_col", "rate")], errors="coerce"),
        }).dropna()
        self.t = t
        self.by_cur = {c: g.set_index("month")["rate"].sort_index() for c, g in t.groupby("cur")}

    def lookup(self, cur: str, month) -> float:
        """Rate for the month, else the nearest month available for that currency."""
        if not self.ok or cur not in self.by_cur or month is None or pd.isna(month):
            if self.ok and cur in self.by_cur and (month is None or pd.isna(month)):
                return float(self.by_cur[cur].iloc[-1])
            return np.nan
        s = self.by_cur[cur]
        if month in s.index:
            return float(s.loc[month])
        ords = np.array([p.ordinal for p in s.index])
        return float(s.iloc[np.abs(ords - month.ordinal).argmin()])

    def band(self, cur: str, months: list) -> tuple[float, float]:
        if not self.ok or cur not in self.by_cur:
            return (np.nan, np.nan)
        s = self.by_cur[cur]
        vals = [self.lookup(cur, m) for m in months if m is not None and not pd.isna(m)]
        if not vals:
            return (float(s.min()), float(s.max()))
        return (min(vals), max(vals))


def to_reporting(df: pd.DataFrame, cfg: dict, resolve) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (amount in reporting currency, rate used, fx flag)."""
    rep = cfg["reporting_currency"].upper()
    fx = FxTable(cfg.get("fx"), resolve)
    fxc = cfg.get("fx") or {}
    roles = fxc.get("date_roles", [c[5:] for c in df.columns if c.startswith("date_") and not c.endswith("_flag")])
    tol = float(fxc.get("row_rate_tolerance", 1.25))
    row_quote = fxc.get("row_rate_quote", "units_per_reporting")

    net = df["net"]
    rate = pd.Series(np.nan, index=df.index)
    flag = pd.Series("", index=df.index, dtype=object)
    cur = df["currency"].fillna("").astype(str).str.upper()
    same = cur.eq(rep) | cur.eq("")
    rate[same] = 1.0
    flag[cur.eq("")] = "CURRENCY_MISSING_ASSUMED_REPORTING"

    months = {}
    for r in roles:
        col = f"date_{r}"
        if col in df.columns:
            months[r] = df[col].dt.to_period("M")
    todo = df.index[~same]
    rr = df["rate_raw"]
    if row_quote == "reporting_per_unit":
        rr = 1.0 / rr
    for i in todo:
        c = cur[i]
        ms = [months[r][i] for r in months]
        first = next((m for m in ms if m is not None and not pd.isna(m)), None)
        given = rr[i]
        if pd.notna(given) and given > 0:
            lo, hi = fx.band(c, ms)
            if np.isnan(lo) or (lo / tol <= given <= hi * tol):
                rate[i] = given
                continue
            flag[i] = "ROW_RATE_IMPLAUSIBLE_USED_TABLE"
        t = fx.lookup(c, first)
        if np.isnan(t):
            flag[i] = (flag[i] + ";" if flag[i] else "") + "NO_FX_RATE"
        else:
            rate[i] = t
            if not flag[i]:
                flag[i] = "TABLE_RATE"
    if fx.quote == "reporting_per_unit":
        # table stores reporting-per-unit; convert the table-derived rates to units-per-reporting
        tbl = flag.str.contains("TABLE")
        rate[tbl] = 1.0 / rate[tbl]
    amount = net / rate
    return amount.round(2), rate, flag
