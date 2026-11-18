# Approach

**Pipeline (`spendcube/pipeline.py`).** A YAML config maps each source file's columns to a canonical record:
id, amount and its tax basis, currency and rate, dates, supplier, item code, description, status, PO, invoice
number, and extra feature columns. Steps, in order:

1. **Ingest.** Read every source.
2. **Clean money.** Net of tax (gross / (1 + rate); a blank rate uses the configured default). Then the
   reporting currency, using the line rate when it sits inside the FX table's band, else the table month.
3. **Exclude.** Status, then generic and company regex rules, then reversals, then duplicates.
4. **Match across systems.**
5. **Key suppliers and items.**
6. **Classify.**
7. **Write output.** One row per input record.

Precedence when several codes apply: CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE. Blind-test runs pass
`--train-dir`. The labelled A/B history is then processed alongside the new files, giving training labels,
golden keys and cross-period duplicate checks, and only the new records are written.

**Matching and de-duplication.**
- **Cross-system matching** (`match.py`). Candidates share the net amount to the cent (±2 c), or the same PO
  with the amount within 0.5%. Each pair is scored on PO and PO item, quantity, OCR-tolerant description
  similarity, supplier-name similarity and date gap, and accepted by explicit rules. A conflicting PO always
  rejects the pair. Such pairs sit about 100 days apart versus about 11 for true matches, so they are repeat
  purchases. Several tool lines may point at one receipt.
- **Exported-twice invoices.** Same invoice digits, gross amount and quantity, plus near-identical text,
  compatible PO and same supplier.
- **Exact duplicates.** Identical on the configured key columns.
- **Reversals.** +X and −X on the same invoice line become CANCELLED.
- **Suppliers.** Names are normalised (legal suffixes, spacing, OCR digits), then unioned at a fuzzy ratio
  ≥ 90 or when they share a vendor id. The client's golden key is used where the name is known.
- **Items.** One key per code. Code-less lines take the code with the identical description, else a
  description key within the supplier, fuzzy-merged.

**Classifying into an unseen tree (`classify.py`).** The tree is read at run time, so the code contains no
category names. Evidence is combined in log-probability space over the leaves:
1. **Zero-shot.** Cosine similarity of the expanded description with each category path. It uses a local ONNX
   sentence model, with the leaf weighted most and parents less. Added to that are stemmed word overlap,
   category acronyms ("CBS" → Can Body Stock) and the supplier name's similarity.
2. **Supervised**, whenever labels exist. A LinearSVC on character and word n-grams plus supplier, item and
   feature tokens, temperature-calibrated on out-of-fold scores.
3. **Pooling.** Each line also gets the confidence-weighted mean belief of its supplier and of its item, since
   suppliers are about 97–99% single-L1.
4. **Hierarchical decoding.** Pick the L1 with the most mass, then L2 inside it, and so on down to a leaf.
   Confidence is the probability of the chosen leaf.

For a new company, the review file ranks lines by spend × (1 − confidence). Keyword `hints` per category and
regex `rules` can then be added to that company's config (never to code). On a synthetic C built from B, one
rule and two hints raised L3 accuracy from 38% to 65%.

**Measured accuracy** (`python evaluate.py …`)

| Test | L1 (spend-w) | L2 (spend-w) | L3 (spend-w) |
|---|---|---|---|
| **A, labels hidden (tree only)**, 37,029 ERP lines | 51% (51%) | 30% (41%) | 21% (28%) |
| B, labels hidden, scored on 2,537 seeds | 77% (88%) | 74% (82%) | 42% (31%) |
| A supervised, trained on first 80%, tested on last 20% | 96% (99.5%) | 95% (99%) | 93% (94%) |
| B, 20% of seeds held out | 99% (99.8%) | 98% (99.8%) | 98% (99.8%) |
| Rehearsal A2 (20% held out, renamed files, `--train-dir`) | 99.9% | 99.4% | 98.6% |
| Rehearsal B2 | 97.8% | 97.5% | 96.9% |

A's tree-only score is low because its texts are SAP short-texts ("ROD ANCHR;DIA:20MM…") and its tree names
mining-specific assemblies. B-style generic trees do much better.

- **Exclusions.** B's rules reproduce all 450 analyst NON_PROCUREMENT seeds with no false exclusions. The only
  disagreement is 13 reversed lines, which I mark CANCELLED.
- **Grouping vs golden keys, golden maps hidden.** Suppliers outside the client's catch-all bucket:
  precision/recall 1.00/1.00. Items: pairwise precision 1.00.
- **Rehearsal grouping.** Supplier F1 0.998, item F1 0.95.

**Runtime and cost.** About 60 s for A (65.5k records) and 40 s for B (41k) on 2 CPU cores. Classification is
most of that, and 25,000 lines run in under 1 minute. There are no API calls, so the LLM cost is 0 per 1,000
lines. The embedding model (90 MB) is downloaded once. Without it the tool still runs, with lexical zero-shot
only.
