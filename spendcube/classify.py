"""Classify spend lines into a category tree read at run time.

Evidence combined in log-probability space over the tree's leaves:
  1. zero-shot  - similarity of the line text to each category path: sentence embeddings (local ONNX model),
                  stemmed word overlap and category acronyms (e.g. 'CBS' -> 'Can Body Stock'). Needs no labels.
  2. supervised - LinearSVC on character/word n-grams + supplier/item/feature tokens, trained on whatever
                  labels exist (full labels for A, analyst seeds for B), temperature-calibrated by cross-validation.
  3. pooling    - suppliers and items are strongly single-category, so each line also receives the average
                  belief of its supplier and of its item (weighted by spend).
  4. self-training - when labels are scarce or absent, confident predictions become pseudo-labels for a
                  text classifier that learns the client's abbreviations, then the evidence is re-combined.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import LinearSVC

from .text import clean_text, is_blank_description

STOP = {"and", "of", "the", "for", "to", "in", "on", "with", "a", "an", "or", "other", "others", "misc",
        "miscellaneous", "general", "services", "service", "related", "unclassified"}


def _softmax(S: np.ndarray, T: float = 1.0) -> np.ndarray:
    Z = S / T
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


def _log_softmax(S: np.ndarray, T: float = 1.0) -> np.ndarray:
    Z = S / T
    Z = Z - Z.max(1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(1, keepdims=True))


def _acronym(name: str) -> str | None:
    words = [w for w in re.findall(r"[A-Za-z]+", re.sub(r"\(.*?\)", " ", name)) if w.lower() not in STOP]
    if len(words) < 2:
        return None
    return "".join(w[0] for w in words).lower()


def _stem(s: str) -> str:
    return " ".join(w[:5] for w in s.split() if len(w) > 1)


class SpendClassifier:
    def __init__(self, tax, cfg: dict | None, embedder=None, abbr=None, log=print):
        self.tax = tax
        self.cfg = cfg or {}
        self.emb = embedder
        self.abbr = abbr
        self.log = log
        self.K = len(tax)

    # ------------------------------------------------------------------ zero-shot evidence
    def zero_shot(self, texts: pd.Series, raw_texts: pd.Series) -> np.ndarray:
        tax, K = self.tax, self.K
        exp = self.abbr.expand_series(texts) if self.abbr else texts.map(clean_text)
        uniq = list(dict.fromkeys(exp.tolist()))
        pos = exp.map({u: i for i, u in enumerate(uniq)}).values
        docs = [tax.doc(p).lower() for p in tax.paths]
        S = np.zeros((len(uniq), K), np.float32)
        if self.emb is not None:
            U = self.emb.encode(uniq)
            nodes = sorted({n for p in tax.paths for n in tax.nodes(p)})
            N = self.emb.encode([n.lower() for n in nodes])
            nid = {n: i for i, n in enumerate(nodes)}
            SN = U @ N.T
            Sp = np.zeros_like(S)
            for j, p in enumerate(tax.paths):
                ns = tax.nodes(p)
                w = np.array([0.5 ** (len(ns) - 1 - i) for i in range(len(ns))])
                Sp[:, j] = SN[:, [nid[x] for x in ns]] @ (w / w.sum())
            F = self.emb.encode(docs)
            S += 0.5 * Sp + 0.5 * (U @ F.T)
        # stemmed word overlap
        tv = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True).fit([_stem(x) for x in uniq + docs])
        Sw = (tv.transform([_stem(x) for x in uniq]) @ tv.transform([_stem(d) for d in docs]).T).toarray()
        S += (0.3 if self.emb is not None else 1.0) * Sw
        if self.emb is None:
            cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True).fit(uniq + docs)
            S += 0.5 * (cv.transform(uniq) @ cv.transform(docs).T).toarray()
        # acronyms of category names appearing as tokens in the raw text
        acr = {}
        for j, p in enumerate(tax.paths):
            ns = tax.nodes(p)
            for k, x in enumerate(ns):
                a = _acronym(x)
                if a and len(a) >= 2:
                    acr.setdefault(a, []).append((j, 1.0 if k == len(ns) - 1 else 0.5))
        if acr:
            rt = raw_texts.fillna("").astype(str)
            ru = list(dict.fromkeys(rt.tolist()))
            A = np.zeros((len(ru), K), np.float32)
            for i, r in enumerate(ru):
                for t in set(re.findall(r"[a-z]+", r.lower())):
                    for j, w in acr.get(t, []):
                        A[i, j] = max(A[i, j], w)
            rpos = rt.map({u: i for i, u in enumerate(ru)}).values
            Sline = S[pos] + 0.25 * A[rpos]
        else:
            Sline = S[pos]
        L = _log_softmax(Sline, self.cfg.get("zero_shot_temperature", 0.05))
        blank = texts.map(is_blank_description).values
        L[blank] = -np.log(K)  # no evidence from an empty description
        return L

    # ------------------------------------------------------------------ features for trainable models
    def _features(self, df: pd.DataFrame, texts: pd.Series, meta_cols: list[str]):
        ct = texts.map(clean_text)
        et = self.abbr.expand_series(texts) if self.abbr else ct
        meta = pd.Series("", index=df.index)
        for c in meta_cols:
            if c in df.columns:
                v = df[c].fillna("na").astype(str).str.replace(r"\s+", "_", regex=True)
                meta = meta + " " + c + "=" + v
        self.v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
        self.v2 = TfidfVectorizer(token_pattern=r"[a-z]{2,}", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.v3 = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=1)
        X = hstack([self.v1.fit_transform(ct), self.v2.fit_transform(et), self.v3.fit_transform(meta)]).tocsr()
        return X

    @staticmethod
    def _svm_logp(X, y, Xall, K, C=0.5, w=None):
        m = LinearSVC(C=C).fit(X, y, sample_weight=w)
        D = m.decision_function(Xall)
        if D.ndim == 1:
            D = np.vstack([-D, D]).T
        full = np.full((Xall.shape[0], K), np.nan, np.float32)
        full[:, m.classes_] = D
        return full, m.classes_

    def _calibrate_T(self, X, y, K, C):
        """Out-of-fold decision values -> temperature minimising NLL."""
        n = len(y)
        if n < 100 or len(np.unique(y)) < 2:
            return 0.3
        counts = pd.Series(y).value_counts()
        folds = StratifiedKFold(3, shuffle=True, random_state=0) if counts.min() >= 3 else KFold(3, shuffle=True, random_state=0)
        D = np.full((n, K), np.nan, np.float32)
        for tr, te in folds.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            m = LinearSVC(C=C).fit(X[tr], y[tr])
            d = m.decision_function(X[te])
            if d.ndim == 1:
                d = np.vstack([-d, d]).T
            D[np.ix_(te, m.classes_)] = d
        D = np.nan_to_num(D, nan=-5.0)
        best, bestT = 1e9, 0.3
        for T in [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
            lp = _log_softmax(D, T)
            nll = -lp[np.arange(n), y].mean()
            if nll < best:
                best, bestT = nll, T
        acc = (D.argmax(1) == y).mean()
        self.log(f"[classify] supervised CV accuracy {acc:.3f} (temperature {bestT})")
        self.cv_accuracy = acc
        return bestT

    # ------------------------------------------------------------------ pooling over suppliers / items
    @staticmethod
    def _pool(P: np.ndarray, keys: pd.Series, w: np.ndarray) -> np.ndarray:
        k = keys.fillna("").astype(str).values
        valid = (k != "") & ~pd.Series(k).str.contains("UNKNOWN|<blank>", regex=True).values
        G = np.full_like(P, 1.0 / P.shape[1])
        if valid.any():
            df = pd.DataFrame(P[valid] * w[valid, None])
            sm = df.groupby(k[valid]).transform("sum").values
            ws = pd.Series(w[valid]).groupby(k[valid]).transform("sum").values[:, None]
            G[valid] = sm / ws
        return G

    def _decode_hier(self, P: np.ndarray) -> np.ndarray:
        """Pick the level-1 node with the most probability mass, then level 2 inside it, ... down to a leaf.
        Scores are weighted by level (spend-weighted accuracy at every level counts), so a confident
        parent beats a slightly higher but isolated leaf elsewhere."""
        paths = self.tax.paths
        depth = max(len(p) for p in paths)
        cand = np.ones((P.shape[0], len(paths)), bool)
        for lvl in range(depth):
            keys = [p[:lvl + 1] if len(p) > lvl else p for p in paths]
            uk = list(dict.fromkeys(keys))
            M = np.zeros((len(uk), len(paths)), np.float32)
            kid = {k: i for i, k in enumerate(uk)}
            for j, k in enumerate(keys):
                M[kid[k], j] = 1.0
            mass = (P * cand) @ M.T          # probability mass of every node at this level
            best = mass.argmax(1)
            cand &= M[best].astype(bool)
        return np.where(cand.any(1), (P * cand).argmax(1), P.argmax(1))

    # ------------------------------------------------------------------ main entry
    def predict(self, df: pd.DataFrame, labels: pd.Series | None = None):
        cfg = self.cfg
        K = self.K
        texts = df["desc"].fillna("").astype(str)
        zs_extra = cfg.get("zero_shot_extra_text", [])
        zs_texts = texts.copy()
        for c in zs_extra:
            col = c if c in df.columns else f"f_{c}"
            if col in df.columns:
                add = df[col].fillna("").astype(str)
                zs_texts = np.where(texts.map(is_blank_description), add, zs_texts + " " + add)
                zs_texts = pd.Series(zs_texts, index=df.index)
        Z = self.zero_shot(zs_texts, texts)
        wsn = cfg.get("supplier_name_weight", 0.3)
        if wsn and "supplier_name" in df.columns and df["supplier_name"].notna().any():
            sn = df["supplier_name"].fillna("").astype(str)
            Zs = self.zero_shot(sn, sn)
            Z = Z + wsn * Zs

        # optional config rules: regex on a field -> strong prior for one category (written per company, never in code)
        for r in cfg.get("rules", []) or []:
            j = self.tax.find(r["category"])
            if j is None:
                self.log(f"[classify] rule category not in tree: {r['category']}")
                continue
            hit = np.zeros(len(df), bool)
            for f in r.get("fields", ["desc"]):
                col = f if f in df.columns else f"f_{f}"
                if col in df.columns:
                    hit |= df[col].fillna("").astype(str).str.contains(r["pattern"], case=False, regex=True).values
            Z[hit, j] += float(r.get("strength", 4.0))
            self.log(f"[classify] rule '{r['pattern']}' -> {r['category']}: {int(hit.sum())} lines")

        meta_cols = ["supplier_cluster", "item_key", "item_cluster"] + [f"f_{c}" for c in cfg.get("meta_features", [])]
        X = self._features(df, texts, meta_cols)
        y = labels.reindex(df.index) if labels is not None else pd.Series(np.nan, index=df.index)
        has = y.notna().values
        n_lab = int(has.sum())
        C = cfg.get("svm_C", 0.5)
        w_amt = np.log1p(df["spend"].abs().fillna(0).values) + 1.0

        if n_lab >= 20 and y[has].nunique() >= 2:
            yl = y[has].astype(int).values
            T = self._calibrate_T(X[has], yl, K, C)
            D, cls = self._svm_logp(X[has], yl, X, K, C)
            Ls = _log_softmax(np.nan_to_num(D, nan=np.nanmin(D) - 1.0), T)
            # leaves never seen in labels keep a floor so zero-shot can still pick them
            unseen = np.setdiff1d(np.arange(K), cls)
            Ls[:, unseen] = np.minimum(Ls[:, unseen], np.log(1e-3))
            coverage = n_lab / len(df)
            wz = cfg.get("zero_shot_weight_with_labels", 0.15 if coverage > 0.5 else 0.35)
            L = Ls + wz * Z
            mode = "supervised" if coverage > 0.5 else "semi-supervised"
        else:
            L = Z.copy()
            mode = "zero-shot"
        self.log(f"[classify] mode={mode} labelled={n_lab} rows={len(df)} leaves={K} embedder={'yes' if self.emb else 'no'}")

        a_sup = cfg.get("supplier_pool_weight", 1.0 if mode == "zero-shot" else 0.5)
        a_item = cfg.get("item_pool_weight", 0.3 if mode == "zero-shot" else 0.5)

        cpow = cfg.get("pool_conf_power", 2.0)

        def pooled(L):
            P = _softmax(L)
            # confident lines speak for their supplier/item; uncertain ones barely count
            w = w_amt * (P.max(1) ** cpow) if cpow else w_amt
            out = L.copy()
            if a_sup:
                out += a_sup * np.log(self._pool(P, df["supplier_cluster"], w) + 1e-6)
            if a_item:
                out += a_item * np.log(self._pool(P, df["item_cluster"], w) + 1e-6)
            return out

        F = pooled(L)
        # self-training when labels do not cover most rows
        if mode != "supervised":
            frac = cfg.get("self_train_fraction", 0.5)
            for it in range(cfg.get("self_train_iters", 0)):
                lab = F.argmax(1)
                srt = np.sort(F, 1)
                margin = srt[:, -1] - srt[:, -2]
                sel = np.zeros(len(df), bool)
                for c in np.unique(lab):
                    ii = np.where(lab == c)[0]
                    k = max(2, int(len(ii) * frac))
                    sel[ii[np.argsort(-margin[ii])[:k]]] = True
                yy = lab.copy()
                if n_lab:
                    sel |= has
                    yy[has] = y[has].astype(int).values
                if len(np.unique(yy[sel])) < 2:
                    break
                D, cls = self._svm_logp(X[sel], yy[sel], X, K, cfg.get("self_train_C", 0.3))
                Lst = _log_softmax(np.nan_to_num(D, nan=-5.0), 0.3)
                F = pooled(L + cfg.get("self_train_weight", 0.7) * Lst)

        P = _softmax(F, cfg.get("final_temperature", 1.0))
        if cfg.get("hierarchical_decoding", True):
            pred = self._decode_hier(P)
        else:
            pred = P.argmax(1)
        conf = P[np.arange(len(pred)), pred]
        if mode == "zero-shot":
            conf = conf * cfg.get("zero_shot_conf_scale", 0.6)
        self.mode = mode
        return pred, conf, mode
