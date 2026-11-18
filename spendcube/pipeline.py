"""End-to-end spend cube pipeline driven by one company config."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import exclude as X
from .classify import SpendClassifier
from .embed import load_embedder
from .entities import item_keys, supplier_keys
from .ingest import ingest, read_table
from .match import match_sources
from .money import net_amount, to_reporting
from .taxonomy import Taxonomy
from .text import Abbreviations, supplier_norm

ROOT = Path(__file__).resolve().parent.parent
EXCL_CONF = {"status": 0.99, "rule": 0.9, "reversal": 0.9, "dup_exact": 0.97, "dup_fuzzy": 0.9}


def load_yaml(p):
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _find_like(want: Path, roots: list[Path]) -> Path | None:
    """A file with the configured name was not found: look for the closest-named table under the roots
    (blind-test files may carry a suffix such as _A2)."""
    from rapidfuzz import fuzz
    stem = want.stem.lower()
    best, best_s = None, 0.0
    for r in roots:
        if not r.exists():
            continue
        for f in r.rglob("*"):
            if f.suffix.lower() not in {".csv", ".xlsx", ".xls", ".txt", ".tsv"}:
                continue
            fs = f.stem.lower()
            sc = 100.0 if (fs.startswith(stem) or stem.startswith(fs)) else fuzz.ratio(fs, stem)
            if sc > best_s:
                best, best_s = f, sc
    return best if best_s >= 75 else None


def make_resolver(cfg_path: Path, data_dir: str | None, fallback_dir: str | None = None, overrides: dict | None = None):
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    fallback = Path(fallback_dir) if fallback_dir else None
    overrides = overrides or {}

    def resolve(p: str) -> Path:
        """Order: explicit override -> exact path in the data dir -> closest-named file in the data dir ->
        (only then) the training/history dir, for shared files such as the tree, labels or FX table."""
        if p in overrides:
            return Path(overrides[p])

        def exact(d):
            q = Path(str(p).replace("{data_dir}", str(d)))
            for c in ([q] if q.is_absolute() else [Path.cwd() / q, ROOT / q, cfg_path.parent / q]):
                if c.exists():
                    return c
            return None

        hit = exact(data_dir)
        if hit is not None:
            return hit
        want = Path(str(p).replace("{data_dir}", str(data_dir)))
        alt = _find_like(want, [want.parent, data_dir])
        if alt is not None:
            print(f"  [resolve] {want.name} not found, using {alt}")
            return alt
        if fallback:
            hit = exact(fallback)
            if hit is not None:
                return hit
        raise FileNotFoundError(f"cannot find {p} (data dir {data_dir})")

    return resolve


class Pipeline:
    def __init__(self, config_path: str, data_dir: str | None = None, hide_labels: bool = False,
                 rules_path: str | None = None, verbose: bool = True, holdout: dict | None = None,
                 train_dir: str | None = None, file_overrides: dict | None = None):
        self.cfg_path = Path(config_path)
        self.cfg = load_yaml(self.cfg_path)
        data_dir = data_dir or str(ROOT / "data")
        train_dir = train_dir or str(ROOT / "data")
        # reference mode: the data to process differs from the labelled history the tool learns from
        self.use_ref = (Path(data_dir).resolve() != Path(train_dir).resolve()) and self.cfg.get("labels") is not None \
            and self.cfg.get("use_reference", True)
        self.resolve = make_resolver(self.cfg_path, data_dir, train_dir if self.use_ref else None, file_overrides)
        self.resolve_ref = make_resolver(self.cfg_path, train_dir) if self.use_ref else None
        self.rules = load_yaml(rules_path or ROOT / "rules" / "default_rules.yaml")
        abbr_file = ROOT / "rules" / "abbreviations.yaml"
        self.abbr = Abbreviations(load_yaml(abbr_file) if abbr_file.exists() else {})
        self.hide_labels = hide_labels
        self.holdout = holdout  # evaluation only: {"frac": 0.2, "mode": "tail"|"random", "seed": 0}
        self.holdout_ids = set()
        self.verbose = verbose
        self.timings = {}

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _t(self, name, t0):
        self.timings[name] = round(time.time() - t0, 1)
        self.log(f"  - {name}: {self.timings[name]}s")
        return time.time()

    # ------------------------------------------------------------------
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        t0 = time.time()
        T0 = t0
        self.log(f"[{cfg.get('company')}] ingest" + (" (+ reference history for training)" if self.use_ref else ""))
        res = ([("ref", self.resolve_ref)] if self.use_ref else []) + [("cur", self.resolve)]
        df, raws = ingest(cfg, res)
        t0 = self._t("ingest", t0)

        # ---- money
        df["net"] = net_amount(df)
        df["spend_rc"], df["fx_rate"], df["fx_flag"] = to_reporting(df, cfg, self.resolve)
        t0 = self._t("amounts+fx", t0)

        # ---- exclusions
        df["exclusion_reason"] = pd.Series([None] * len(df), dtype=object)
        df["exclusion_detail"] = ""
        status_rules = (cfg.get("status_rules") or []) + self.rules.get("status", [])
        np_rules = (cfg.get("non_procurement_rules") or []) + (
            [] if cfg.get("disable_default_rules") else self.rules.get("non_procurement", []))
        for name, sc in cfg["sources"].items():
            m = df.src == name
            sub = df[m].copy()
            X.apply_status(sub, status_rules, sc)
            df.loc[m, ["exclusion_reason", "exclusion_detail"]] = sub[["exclusion_reason", "exclusion_detail"]]
        X.apply_rules(df, np_rules)
        for name, sc in cfg["sources"].items():
            d = sc.get("dedup") or {}
            if sc.get("reversals") == "cancel":
                X.apply_reversals(df, name)
            if d.get("exact"):
                X.dedupe_exact(df, name, raws[name], d.get("key_cols"), [sc["id_col"]] + d.get("ignore_cols", []))
            if d.get("fuzzy_invoice"):
                X.dedupe_fuzzy_invoices(df, name, d.get("text_threshold", 85))
        t0 = self._t("status/rules/dedup", t0)

        # ---- cross-system matching
        df["matched_to"] = None
        df["match_score"] = np.nan
        mc = cfg.get("cross_source_match")
        if mc:
            mt = match_sources(df, mc["primary"], mc["secondary"], secondary_mask=~df["is_ref"])
            df.loc[mt.si, "matched_to"] = df.loc[mt.pi, "record_id"].values
            df.loc[mt.si, "match_score"] = mt.score.values
            X.set_code(df, df.index.isin(mt.si), "DUPLICATE", f"same purchase as a {mc['primary']} record")
            self.log(f"  matched {len(mt)} of {((df.src == mc['secondary']) & ~df.is_ref).sum()} {mc['secondary']} records to {mc['primary']}")
        t0 = self._t("cross-source match", t0)

        # ---- labels
        tax = Taxonomy(cfg["taxonomy"], self.resolve)
        lab_leaf, gold_sup, gold_item = self._labels(df, tax)
        t0 = self._t("labels", t0)

        # ---- suppliers & items
        df["supplier_key"], df["supplier_cluster"] = supplier_keys(df, gold_sup, cfg.get("supplier_threshold", 90))
        df["item_key"], df["item_cluster"] = item_keys(df, df["supplier_cluster"], gold_item,
                                                    merge_codes_by_description=cfg.get("merge_item_codes_by_description", False))
        if mc:  # matched secondary records describe the primary record's supplier and item
            m = df["matched_to"].notna()
            pk = df.set_index("record_id")
            for c in ["supplier_key", "supplier_cluster", "item_key", "item_cluster"]:
                df.loc[m, c] = pk.loc[df.loc[m, "matched_to"], c].values
        t0 = self._t("supplier/item keys", t0)

        # ---- spend & classification
        df["include"] = df["exclusion_reason"].isna().astype(int)
        df["spend"] = np.where(df["include"] == 1, df["spend_rc"].fillna(0.0), 0.0).round(2)
        inc = df["include"] == 1
        emb = load_embedder(self.resolve(cfg.get("embedding_model", "models/all-MiniLM-L6-v2")))
        clf = SpendClassifier(tax, cfg.get("classification"), emb, self.abbr, self.log)
        pred, conf, mode = clf.predict(df[inc], lab_leaf)
        df["leaf"] = -1
        df["confidence"] = 0.0
        df.loc[inc, "leaf"] = pred
        df.loc[inc, "confidence"] = conf
        self.mode = mode
        if lab_leaf is not None and (cfg.get("labels") or {}).get("use_as_truth", True):
            known = inc & lab_leaf.reindex(df.index).notna()
            df.loc[known, "leaf"] = lab_leaf[known].astype(int)
            df.loc[known, "confidence"] = 0.99
        # excluded records: confidence of the exclusion decision
        ex = ~inc
        det = df["exclusion_detail"].fillna("")
        econf = np.select(
            [det.str.startswith("status"), det.str.contains("reversed"), det.str.contains("repeated"),
             det.str.contains("exported twice"), det.str.contains("same purchase")],
            [EXCL_CONF["status"], EXCL_CONF["reversal"], EXCL_CONF["dup_exact"], EXCL_CONF["dup_fuzzy"],
             np.clip(0.6 + df["match_score"].fillna(0) / 25, 0.6, 0.98)], EXCL_CONF["rule"])
        df.loc[ex, "confidence"] = econf[ex.values]
        t0 = self._t("classification", t0)

        self.df_all = df
        df = df[~df["is_ref"]].copy()
        out = self._output(df, tax)
        self.timings["total"] = round(time.time() - T0, 1)
        self.df, self.tax = df, tax
        return out, df

    # ------------------------------------------------------------------
    def _labels(self, df, tax):
        lc = self.cfg.get("labels")
        if not lc or self.hide_labels:
            return None, None, None
        L = read_table((self.resolve_ref or self.resolve)(lc["file"]))
        rid = lc.get("id_prefix", "") + L[lc["id_col"]].astype(str).str.strip()
        L.index = rid
        pool = df[df["is_ref"]] if self.use_ref else df
        idx = pd.Series(pool.index, index=pool["orig_id"])
        idx = idx[~idx.index.duplicated()]
        common = L.index.intersection(idx.index)
        L = L.loc[common]
        if self.holdout:
            ids = L.index.to_numpy()
            n = int(len(ids) * self.holdout.get("frac", 0.2))
            if self.holdout.get("mode", "random") == "tail":
                held = ids[-n:]
            else:
                held = np.random.RandomState(self.holdout.get("seed", 0)).permutation(ids)[:n]
            self.holdout_ids = set(held)
            L = L.drop(index=held)
        rows = idx[L.index].values
        lab = pd.Series(np.nan, index=df.index)
        if lc.get("category_cols"):
            vals = L[lc["category_cols"]].values.tolist()
            leaf = [tax.match_label(v) for v in vals]
            lab.loc[rows] = [np.nan if x is None else x for x in leaf]
            if lc.get("include_col"):
                lab.loc[rows[L[lc["include_col"]].astype(str).str.strip().values != "1"]] = np.nan
            miss = sum(1 for v, l in zip(vals, leaf) if l is None and any(isinstance(x, str) and x for x in v))
            if miss:
                self.log(f"  [labels] {miss} labelled rows do not match the tree")
        gold_sup = gold_item = None
        if lc.get("supplier_key_col"):
            sn = df.loc[rows, "supplier_name"].map(lambda x: supplier_norm(x) if isinstance(x, str) else "")
            g = pd.DataFrame({"n": sn.values, "k": L[lc["supplier_key_col"]].values})
            g = g[(g.n != "") & g.k.notna()]
            gold_sup = g.groupby("n").k.agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        if lc.get("item_key_col"):
            code = df.loc[rows, "item_code"].astype("string").str.strip().str.upper().str.lstrip("0")
            g = pd.DataFrame({"c": code.values, "k": L[lc["item_key_col"]].values}).dropna()
            gold_item = g.groupby("c").k.agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
        self.log(f"  [labels] {int(lab.notna().sum())} category labels, golden suppliers={len(gold_sup or {})}, golden items={len(gold_item or {})}")
        return lab, gold_sup, gold_item

    @staticmethod
    def _public_key(k: str) -> str:
        """Golden keys pass through; our own text-derived keys are hashed so outputs carry no description text."""
        if isinstance(k, str) and (k.startswith("S:") or k.startswith("I:")):
            return k[0] + "-" + hashlib.md5(k.encode("utf-8")).hexdigest()[:10]
        return k

    def _output(self, df, tax):
        n_out = max(4, tax.depth)
        lv = np.full((len(df), n_out), "", dtype=object)
        inc = (df["include"] == 1).values
        for i in np.where(inc & (df["leaf"].values >= 0))[0]:
            lv[i] = tax.output_row(int(df["leaf"].values[i]), n_out)
        out = pd.DataFrame({
            "record_id": df["record_id"].values,
            "include": df["include"].values,
            "exclusion_reason": df["exclusion_reason"].fillna("").values,
            "supplier_key": df["supplier_key"].map(self._public_key).values,
            "item_key": df["item_key"].map(self._public_key).values,
            "spend": df["spend"].values,
        })
        for k in range(n_out):
            out[f"l{k + 1}"] = lv[:, k]
        out["confidence"] = df["confidence"].round(3).values
        return out

    def diagnostics(self) -> pd.DataFrame:
        cols = ["record_id", "src", "amount_raw", "net", "currency", "fx_rate", "fx_flag", "spend_rc",
                "exclusion_reason", "exclusion_detail", "matched_to", "match_score"]
        cols += [c for c in self.df.columns if c.startswith("date_") and c.endswith("_flag")]
        return self.df[[c for c in cols if c in self.df.columns]]

    def review(self, n: int = 400) -> pd.DataFrame:
        """Included lines where a wrong category costs most (spend x (1 - confidence)), grouped by description
        and supplier: the list to read when adding hints/rules for a new company."""
        d = self.df[self.df.include == 1].copy()
        if d.empty:
            return d
        d["path"] = [" > ".join(self.tax.paths[i]) if i >= 0 else "" for i in d["leaf"]]
        d["risk"] = d["spend"].abs() * (1 - d["confidence"])
        g = d.groupby(["desc", "supplier_cluster", "path"], dropna=False).agg(
            lines=("record_id", "size"), spend=("spend", "sum"), confidence=("confidence", "mean"), risk=("risk", "sum")).reset_index()
        return g.sort_values("risk", ascending=False).head(n)

    def summary(self) -> dict:
        df = self.df
        s = {
            "company": self.cfg.get("company"),
            "records": int(len(df)),
            "included": int(df.include.sum()),
            "excluded_by_reason": df.exclusion_reason.value_counts().to_dict(),
            "total_spend": round(float(df.spend.sum()), 2),
            "reporting_currency": self.cfg.get("reporting_currency"),
            "spend_by_source": df.groupby("src").spend.sum().round(2).to_dict(),
            "suppliers": int(df.loc[df.include == 1, "supplier_key"].nunique()),
            "items": int(df.loc[df.include == 1, "item_key"].nunique()),
            "classification_mode": self.mode,
            "fx_flags": df.fx_flag.replace("", np.nan).value_counts().to_dict(),
            "date_flags": {c: df[c].replace("", np.nan).value_counts().to_dict()
                           for c in df.columns if c.startswith("date_") and c.endswith("_flag")},
            "timings_s": self.timings,
        }
        return json.loads(json.dumps(s, default=str))
