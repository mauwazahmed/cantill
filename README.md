# Spend cube builder — Cantill.ai Data & ML task

Takes a company's raw purchase data and its category tree, combines and cleans the data, decides what counts as
spend, de-duplicates suppliers and items, and classifies every included line into the tree with a confidence.
Each company is described by a YAML config. A new company needs a new config, not new code. The tool calls no
paid API and runs on CPU only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.10+
pip install -r requirements.txt
python scripts/download_model.py                         # local sentence-embedding model (~90 MB, once)
```

Put the task data in `data/` (git-ignored, never committed):

```
data/company_A/{erp_goods_receipts.csv, invoice_tool_export.csv, labels_A.csv, taxonomy_A.csv}
data/company_B/{invoice_lines.csv, seed_labels.csv, taxonomy_B.csv, fx_rates_usd.csv}
```

## The one command

```bash
python run.py --config configs/A.yaml configs/B.yaml
```

This writes `outputs/mauwaz_ahmed__A.csv` and `outputs/mauwaz_ahmed__B.csv` (one row per input record,
including excluded ones). It also writes `outputs/_work/` (git-ignored), which holds a run summary, a diagnostics
file per record (FX and date flags, exclusion detail, the matched ERP record) and a review list of risky lines.
Runtime on a 2-core laptop: about 60 s for A (65.5k records) and 40 s for B (41k records). The 60-minute budget
for 25,000 lines is met with a wide margin.

## Blind-test day (A2, B2, C)

```bash
# A2 / B2: existing configs. --train-dir holds the labelled A/B history the tool learns from
# (classifier, golden supplier/item keys, cross-period duplicates). Only --data-dir records are output.
python run.py --config configs/A.yaml --data-dir blind/A2 --train-dir data --suffix A2 --out submission
python run.py --config configs/B.yaml --data-dir blind/B2 --train-dir data --suffix B2 --out submission

# C: draft a config from the file, review it, run, read the review list, add hints/rules to the config, rerun
python scripts/draft_config.py --data blind/C/<invoices>.csv --tree blind/C/<tree>.csv --company C \
       --currency USD --out configs/C.yaml
python run.py --config configs/C.yaml --data-dir blind/C --suffix C --out submission
#   -> open submission/_work/review__C.csv (lines where a wrong category costs most)
#   -> add `taxonomy.hints` / `classification.rules` to configs/C.yaml, rerun
```

If the blind files are named differently from the configs, the tool picks the closest-named file in `--data-dir`
and prints which one it used. A path can also be forced without editing a config:
`--file "{data_dir}/company_A/erp_goods_receipts.csv=blind/A2/erp.csv"`. Only push `configs/C.yaml` and the three
CSVs. `submission/_work/` is git-ignored.

## Measuring

```bash
python evaluate.py zero-shot-A   # A classified from its tree alone (labels hidden)
python evaluate.py holdout-A     # train on the first 80% of A's labels, test on the last 20%
python evaluate.py entities-A    # supplier/item grouping vs the client's golden keys (golden maps hidden)
python evaluate.py seeds-B       # B exclusions vs the analyst seeds + categories on 20% held-out seeds
python evaluate.py zero-shot-B   # B from its tree alone, scored on the seeds
python scripts/simulate_blind.py # full rehearsal: split A/B into history + simulated A2/B2, run, score
```

## Layout

```
run.py                     one command (CLI)
evaluate.py                accuracy / grouping measurements
configs/A.yaml, B.yaml     company configs (files, joins, columns, formats, rules)
rules/default_rules.yaml   generic status mapping + non-procurement rules (regex), shared by all companies
rules/abbreviations.yaml   generic MRO/SAP abbreviations used to expand short texts
spendcube/
  ingest.py      read files into one canonical table (ids, amounts, tax, currency, dates, supplier, item, text)
  parse.py       numbers in any format, tax rates (15 / 0.15 / 15%), multi-format dates with range checks
  money.py       net of tax, FX to the reporting currency with a plausibility check of line rates
  exclude.py     status, non-procurement rules, reversals, exact and fuzzy (exported-twice) duplicates
  match.py       cross-system matching (ERP receipt <-> AP-tool invoice line)
  entities.py    supplier and item de-duplication (normalisation + fuzzy union-find, golden keys when known)
  taxonomy.py    tree loaded at run time (any depth; no category names in code)
  embed.py       local ONNX sentence embeddings (all-MiniLM-L6-v2), optional
  classify.py    zero-shot + supervised + supplier/item pooling, calibrated confidence
  pipeline.py    orchestration, outputs, diagnostics
scripts/         download_model.py, draft_config.py (new-company config drafter), simulate_blind.py
outputs/         mauwaz_ahmed__A.csv, mauwaz_ahmed__B.csv
```

## Tools used

- Python, pandas, scikit-learn (LinearSVC, TF-IDF), rapidfuzz, ONNX Runtime with the open
  `sentence-transformers/all-MiniLM-L6-v2` model (Apache-2.0) run locally.
- No LLM API inside the tool, so the LLM cost per 1,000 lines is 0.
- AI assistance: Claude (Anthropic) was used as a coding assistant. It explored the kit's data with me to
  inform the design decisions, and it helped write and test the code and docs. Every output file is produced
  by this tool running unattended.
