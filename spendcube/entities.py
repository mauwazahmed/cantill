"""Supplier and item de-duplication.

Suppliers: normalise names (legal suffixes, spacing, punctuation, OCR digits), then union names that are
near-identical (rapidfuzz ratio >= threshold) or share a vendor id. Where the client's golden supplier key is
known for a name (labels), the cluster takes that key so our groups line up with the client's master data.

Items: one key per item code; codes whose cleaned descriptions are identical are merged; lines without a code
are keyed by their cleaned description, fuzzy-merged within the same supplier; secondary-system records that
matched a primary record inherit its item. Golden item keys from labels are used where known.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .text import compact, supplier_norm


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _fuzzy_links(names: list[str], threshold: float, min_len: int = 4) -> list[tuple[int, int]]:
    names = list(names)
    if len(names) < 2:
        return []
    links = []
    block = 4000
    for i in range(0, len(names), block):
        sc = process.cdist(names[i:i + block], names, scorer=fuzz.ratio, score_cutoff=threshold, workers=-1)
        r, cidx = np.nonzero(sc)
        for a, b in zip(r + i, cidx):
            if a < b and len(names[a]) >= min_len and len(names[b]) >= min_len:
                links.append((a, b))
    # prefix containment (e.g. 'oakentmining' vs 'oakentminingsouthafrica')
    srt = sorted(range(len(names)), key=lambda k: names[k])
    for x, y in zip(srt, srt[1:]):
        a, b = names[x], names[y]
        if len(a) >= 8 and b.startswith(a) and len(a) / len(b) >= 0.6:
            links.append((x, y))
    return links


def supplier_keys(df: pd.DataFrame, golden: dict[str, str] | None = None, threshold: float = 90.0,
                  forced: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Returns (output key, our own cluster key). The cluster key ignores golden keys and is used as a
    classifier feature: a client's golden key can lump many suppliers together (e.g. a catch-all bucket)."""
    sn = df["supplier_name"].map(lambda x: supplier_norm(x) if isinstance(x, str) else "")
    sid = df["supplier_id"].astype("string")
    node = np.where(sn != "", "n:" + sn, np.where(sid.notna(), "i:" + sid.fillna(""), "unknown"))
    node = pd.Series(node, index=df.index)
    uf = UnionFind()
    names = sorted({n for n in node if n.startswith("n:")})
    for a, b in _fuzzy_links([n[2:] for n in names], threshold):
        uf.union(names[a], names[b])
    # same vendor id (within a source) -> same supplier
    both = (sn != "") & sid.notna()
    for (_, v), g in df[both].assign(node=node[both]).groupby(["src", "supplier_id"]):
        nodes = g["node"].unique()
        for x in nodes[1:]:
            uf.union(nodes[0], x)
    root = node.map(uf.find)

    # golden keys: per name node
    key = pd.Series(index=df.index, dtype=object)
    if golden:
        gnode = node.map(lambda n: golden.get(n[2:]) if n.startswith("n:") else golden.get(n))
        # cluster majority of golden keys for nodes without one
        maj = gnode.dropna().groupby(root[gnode.notna()]).agg(lambda s: Counter(s).most_common(1)[0][0])
        key = gnode.fillna(root.map(maj))
    # readable fallback key: the most frequent spelling in the cluster
    rep = node.groupby(root).agg(lambda s: Counter(s).most_common(1)[0][0])
    key = key.fillna("S:" + root.map(rep).str[2:])
    key[node == "unknown"] = "S:UNKNOWN"
    cluster = "S:" + root.map(rep).str[2:]
    cluster[node == "unknown"] = "S:UNKNOWN"
    if forced is not None:
        key = forced.where(forced.notna(), key)
    return key, cluster


def _norm_code(x) -> str | None:
    if not isinstance(x, str) or not x.strip():
        return None
    c = x.strip().upper().lstrip("0")
    return c or None


def item_keys(df: pd.DataFrame, sup_key: pd.Series, golden: dict[str, str] | None = None,
              forced: pd.Series | None = None, threshold: float = 92.0,
              merge_codes_by_description: bool = False) -> tuple[pd.Series, pd.Series]:
    code = df["item_code"].map(_norm_code)
    comp = df["desc"].map(compact)
    node = pd.Series(np.where(code.notna(), "c:" + code.fillna(""), ""), index=df.index)
    uf = UnionFind()
    cd = pd.DataFrame({"code": code, "comp": comp})[code.notna() & (comp.str.len() >= 8)]
    if merge_codes_by_description:
        # optional: codes sharing an identical, specific description are treated as the same item.
        # Off by default: in A's golden data, distinct material numbers stay distinct items even with equal text.
        for _, g in cd.groupby("comp"):
            cs = g["code"].unique()
            for x in cs[1:]:
                uf.union("c:" + cs[0], "c:" + x)
    # code-less lines: attach to the most common coded item with the identical description, else key by
    # description within the supplier
    desc_to_code = cd.groupby("comp")["code"].agg(lambda s: Counter(s).most_common(1)[0][0]).to_dict()
    nocode = code.isna()
    dnode = []
    for i in df.index[nocode]:
        c = comp[i]
        if c in desc_to_code:
            dnode.append("c:" + desc_to_code[c])
        elif len(c) >= 3:
            dnode.append(f"d:{sup_key[i]}|{c}")
        else:
            dnode.append(f"d:{sup_key[i]}|<blank>")
    node[nocode] = dnode
    # fuzzy-merge descriptions within the same supplier (OCR / spacing variants)
    dn = pd.Series([n for n in node.unique() if n.startswith("d:")])
    if len(dn):
        sup = dn.str.split("|").str[0]
        for _, grp in dn.groupby(sup):
            vals = list(grp)
            if len(vals) < 2:
                continue
            for a, b in _fuzzy_links([v.split("|", 1)[1] for v in vals], threshold, min_len=6):
                uf.union(vals[a], vals[b])
    root = node.map(uf.find)
    key = pd.Series(index=df.index, dtype=object)
    if golden:
        g = code.map(lambda c: golden.get(c) if c else None)
        maj = g.dropna().groupby(root[g.notna()]).agg(lambda s: Counter(s).most_common(1)[0][0])
        key = g.fillna(root.map(maj))
    cluster = "I:" + root.str[2:]
    key = key.fillna(cluster)
    if forced is not None:
        key = forced.where(forced.notna(), key)
    return key, cluster
