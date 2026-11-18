#!/usr/bin/env python3
"""Measure the tool against the labels we have.

  python evaluate.py zero-shot-A      # classify A from its tree alone (labels hidden)
  python evaluate.py holdout-A        # train on first 80% of A's labels, test on the last 20% (time-ordered)
  python evaluate.py entities-A       # supplier / item grouping vs the client's golden keys, golden maps hidden
  python evaluate.py seeds-B          # B exclusions vs analyst seeds + 5-fold CV of categories on seeds
  python evaluate.py zero-shot-B      # B from its tree alone, scored on the seeds
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from spendcube.ingest import read_table
from spendcube.pipeline import Pipeline


def cat_report(df, out, lab, cols, name, ids=None):
    o = out.set_index("record_id")
    lab = lab.copy()
    if ids is not None:
        lab = lab[lab.index.isin(ids)]
    lab = lab[lab.index.isin(o.index[o.include == 1])]
    o = o.loc[lab.index]
    w = df.set_index("record_id").loc[lab.index, "spend_rc"].abs().fillna(0).values
    res = {}
    for k in range(1, len(cols) + 1):
        pred = o[[f"l{i}" for i in range(1, k + 1)]].fillna("").agg("|".join, axis=1)
        true = lab[cols[:k]].fillna("").agg("|".join, axis=1)
        ok = (pred.values == true.values)
        res[f"L{k}"] = (round(ok.mean(), 3), round((ok * w).sum() / w.sum(), 3))
    conf = o.confidence.values
    print(f"{name}: n={len(lab)}  " + "  ".join(f"{k} acc={a} spend-w={b}" for k, (a, b) in res.items())
          + f"  mean-conf={conf.mean():.3f}")
    return res


def pair_f1(true, pred):
    """Pairwise precision/recall/F1 of a clustering (counts pairs inside clusters)."""
    t = pd.Series(true).astype(str).values
    p = pd.Series(pred).astype(str).values
    tp = pd.crosstab(t, p).values.astype(float)
    pairs = lambda x: (x * (x - 1) / 2).sum()
    same_both = pairs(tp)
    same_p = pairs(tp.sum(0))
    same_t = pairs(tp.sum(1))
    prec = same_both / same_p if same_p else 1.0
    rec = same_both / same_t if same_t else 1.0
    return round(prec, 3), round(rec, 3), round(2 * prec * rec / (prec + rec + 1e-12), 3)


def main(task):
    t = time.time()
    if task in ("zero-shot-A", "holdout-A", "entities-A"):
        lab = read_table("data/company_A/labels_A.csv").set_index("record_id")
        cols = ["category_l1", "category_l2", "category_l3"]
        if task == "zero-shot-A":
            p = Pipeline("configs/A.yaml", hide_labels=True, verbose=False)
            out, df = p.run()
            cat_report(df, out, lab, cols, "A zero-label (ERP lines)")
        elif task == "holdout-A":
            p = Pipeline("configs/A.yaml", verbose=False, holdout={"frac": 0.2, "mode": "tail"})
            out, df = p.run()
            cat_report(df, out, lab, cols, "A supervised, last 20% held out", ids=p.holdout_ids)
        else:
            p = Pipeline("configs/A.yaml", verbose=False, hide_labels=True)
            out, df = p.run()
            o = out.set_index("record_id").loc[lab.index]
            print("supplier grouping (ERP lines) pairwise P/R/F1:", pair_f1(lab.supplier_key, o.supplier_key),
                  "ARI", round(adjusted_rand_score(lab.supplier_key, o.supplier_key), 3))
            m = lab.item_key.notna()
            print("item grouping (lines with a golden item) pairwise P/R/F1:", pair_f1(lab.item_key[m], o.item_key[m]),
                  "ARI", round(adjusted_rand_score(lab.item_key[m], o.item_key[m]), 3))
            # tool records matched to ERP: do they share the ERP record's keys? (by construction yes)
    elif task in ("seeds-B", "zero-shot-B"):
        seeds = read_table("data/company_B/seed_labels.csv")
        seeds.index = seeds.line_id
        cols = ["category_l1", "category_l2", "category_l3"]
        if task == "zero-shot-B":
            p = Pipeline("configs/B.yaml", hide_labels=True, verbose=False)
            out, df = p.run()
            cat_report(df, out, seeds[seeds.include == "1"], cols, "B zero-label (seed lines)")
        else:
            p = Pipeline("configs/B.yaml", verbose=False, holdout={"frac": 0.2, "mode": "random", "seed": 1})
            out, df = p.run()
            o = out.set_index("record_id").loc[seeds.index]
            ct = pd.crosstab(seeds.include.map({"1": "seed include", "0": "seed exclude"}),
                             o.include.map({1: "we include", 0: "we exclude"}).values)
            print(ct)
            dis = seeds[(seeds.include == "1") & (o.include.values == 0)]
            print("seed-included lines we exclude, by our reason:", o.loc[dis.index, "exclusion_reason"].value_counts().to_dict())
            dis2 = seeds[(seeds.include == "0") & (o.include.values == 1)]
            print("seed-excluded lines we include:", len(dis2), dis2.index[:10].tolist())
            cat_report(df, out, seeds[seeds.include == "1"], cols, "B semi-supervised, 20% of seeds held out", ids=p.holdout_ids)
    print(f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
