"""Category tree loaded at run time. No category names live in code."""
from __future__ import annotations

import re

import pandas as pd

from .ingest import read_table


class Taxonomy:
    def __init__(self, cfg: dict | str, resolve):
        if isinstance(cfg, str):
            cfg = {"file": cfg}
        t = read_table(resolve(cfg["file"]))
        t.columns = [c.strip() for c in t.columns]
        level_cols = cfg.get("level_cols") or [c for c in t.columns if c not in set(cfg.get("ignore_cols", []))
                                               and not re.search(r"(code|id|desc|description|definition|example|keyword)s?$", c, re.I)]
        self.level_cols = level_cols
        self.depth = len(level_cols)
        t = t.fillna("")
        for c in level_cols:
            t[c] = t[c].astype(str).str.strip()
        rows = []
        for _, r in t.iterrows():
            p = []
            for c in level_cols:
                if r[c] == "":
                    break
                p.append(r[c])
            if p:
                rows.append(tuple(p))
        rows = list(dict.fromkeys(rows))
        # leaves: paths that are not a strict prefix of another path
        prefixes = {p[:k] for p in rows for k in range(1, len(p))}
        self.paths = [p for p in rows if p not in prefixes] or rows
        self.index = {p: i for i, p in enumerate(self.paths)}
        # optional free-text descriptions / keywords columns in the tree file
        dcols = cfg.get("description_cols") or [c for c in t.columns if c not in level_cols]
        self.extra = {}
        if dcols:
            for _, r in t.iterrows():
                p = tuple(x for x in (r[c] for c in level_cols) if x)
                if p in self.index:
                    self.extra[p] = " ".join(str(r[c]) for c in dcols if str(r[c]).strip())
        self.hints = {}
        for k, v in (cfg.get("hints") or {}).items():
            words = v if isinstance(v, str) else " ".join(v)
            for p in self.paths:
                if k == p[-1] or k == " > ".join(p) or k == "|".join(p):
                    self.hints[p] = self.hints.get(p, "") + " " + words

    def find(self, name: str) -> int | None:
        """Leaf by its name, or by its full path written 'L1 > L2 > L3' (or with '|')."""
        name = name.strip()
        parts = tuple(x.strip() for x in re.split(r"\s*[>|]\s*", name))
        if parts in self.index:
            return self.index[parts]
        hits = [i for p, i in self.index.items() if p[-1] == name or p[:len(parts)] == parts]
        return hits[0] if hits else None

    def __len__(self):
        return len(self.paths)

    def nodes(self, p) -> list[str]:
        """Path without consecutive repeats (trees often repeat a level name at the next level)."""
        out = []
        for x in p:
            if not out or x != out[-1]:
                out.append(x)
        return out

    def doc(self, p) -> str:
        parts = list(reversed(self.nodes(p)))
        extra = (self.extra.get(p, "") + " " + self.hints.get(p, "")).strip()
        return (" ".join(parts) + (" " + extra if extra else "")).strip()

    def match_label(self, values: list[str]) -> int | None:
        vals = [str(v).strip() for v in values if isinstance(v, str) and str(v).strip() and str(v).lower() != "nan"]
        if not vals:
            return None
        t = tuple(vals)
        if t in self.index:
            return self.index[t]
        for p, i in self.index.items():  # label given to a shorter depth than the leaf, or vice versa
            if p[:len(t)] == t or t[:len(p)] == p:
                return i
        return None

    def output_row(self, i: int, n_out: int = 4) -> list[str]:
        p = list(self.paths[i])
        return (p + [""] * n_out)[:n_out]
