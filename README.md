# Spend cube builder - Cantill.ai Data & ML task

Takes a company's raw purchase data and its category tree, combines and cleans the data, decides what counts as
spend, de-duplicates suppliers and items, and classifies every included line into the tree with a confidence.
Each company is described by a YAML config. A new company needs a new config, not new code. The tool calls no
paid API and runs on CPU only.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate        # Python 3.10+
pip install -r requirements.txt
python scripts/download_model.py                         # local sentence-embedding model (~90 MB, once)
```

Put the task data in `data/` (git-ignored, never committed):

```
data/company_A/{files}
data/company_B/{files}
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


