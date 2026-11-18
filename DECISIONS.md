# What I found in the data and what I decided

## General rules (all companies)
- **Exclusion precedence:** CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE. When a duplicate group has an
  excluded copy, the copy kept as spend is one that is not excluded for another reason.
- **Unknown status values:** paid, approved, posted and similar count as spend. Cancel, void and reversal values
  → CANCELLED. Anything else → NOT_APPROVED (`rules/default_rules.yaml`). A source without a status column is
  treated as posted AP data, so it counts.
- **Non-procurement:** generic regex rules on description, supplier and account-like columns. They cover:
  description that is *only* a tax label; tax authority as supplier; intercompany (text or IC account codes);
  interest, bank and FX items; royalties; revenue share and partner billings; payroll and garnishments; donations
  and fines. Tax *services*, payroll *services* and customs *clearance* are left in, because they are purchased
  services. Customs duty is left in because B's tree has a category for it.
- **Zero, negative and unparseable amounts:** zero-value lines stay included with spend 0. Negative lines
  without a matching original are credit notes or returns and stay as negative spend. Unparseable amounts are
  flagged in diagnostics.
- **Dates:** the formats are tried in the configured order, then a tolerant parser. A date outside the
  configured window, or unparseable, is flagged (`_work/diagnostics__*.csv`) and set empty, so FX falls back to
  the next date.
- **Blank or meaningless descriptions** ("NO OBJECT", "Product", "Parts"): the text gives no zero-shot evidence,
  so the line is classified by its supplier, item, account features and pooling.
- **Keys:** golden keys pass through. Our own keys are hashed (`S-…`, `I-…`), so the public outputs carry no
  description text.
- **Excluded rows:** l1–l4 are blank and spend is 0. Confidence is the confidence of the exclusion decision.

## Company A (ZAR)

### ERP goods receipts
37,029 lines. Ids run A000002–A045000 with 17.7% gaps, so I assume A2 holds the missing ERP lines. For that
reason A2 runs match new tool lines against the A history as well.
- `value_zar` is already net and in ZAR, so `document_currency` (EUR/CHF/MZN on 16 lines) is ignored.
- 66% of lines have no posting date, vendor id, site or UoM; it looks like a second extract.
- There is no status, so every ERP line counts.
- 434 negative lines (returns or reversed receipts) stay as negative spend. The client labels categorise them.

### Invoice tool
28,520 lines.
- **Amounts:** all gross, some with thousands separators ("5,266.63").
- **VAT rate:** written as `15`, `0.15`, `15%`, blank (23%) or `0` (5.5%). Blank → 15% per the brief. `0` is
  zero-rated, so net = gross.
- **Dates:** 65% missing, in three formats.
- **Invoice numbers:** five formats (`INV/2023/…`, `SI-2023-…`, `TAX INV …`, `IN0…`, bare digits). Only the
  trailing digits are compared.
- **PO references:** `4510…`, `PO 4510…`, `PO#4510…`, `4510…/10.0`, or missing (18%). The PO and the PO item
  are extracted by regex. 5,701 lines cite a PO that is not in the ERP file.
- **OCR noise:** digits inside words (0/O, 1/I, 5/S, 8/B) and spaces inside names ("COR ORA", "YARF IELD").
  These are normalised before every comparison.
- **Status:** Paid/Approved count. Cancelled → CANCELLED (188). On Hold/Rejected → NOT_APPROVED (315). Status
  wins over DUPLICATE.

### Exported twice and cross-system matching
- **Exported twice:** copies have adjacent tool ids, the same invoice digits, gross amount and quantity, and OCR
  variants of the text. Their PO is reformatted or dropped and their dates differ. Rule: same invoice digits +
  amount + quantity + supplier, text ratio ≥ 85, and PO/PO item not contradicting → DUPLICATE (2,229). Same
  invoice and amount with clearly different text ("RIGHTHAND" vs "LEFT HND KEEPER PLATE") stays separate.
- **Matched to ERP:** 20,993 tool lines (19,428 of them excluded on that ground). Candidates have the same
  cent-level net amount, or the same PO within 0.5%. Accepted when:
  - the PO matches and the quantity or text agrees; or
  - there is no PO conflict, quantity + text + supplier all agree, and the dates are within 120 days.
- **PO conflicts:** about 3,100 tool lines have a best ERP candidate that agrees on the cent-level amount and
  mostly on quantity, text and supplier, but carries a different PO. Their date gap is about 100 days (median), versus about 11 days for PO-confirmed
  matches. I treat them as repeat purchases, not duplicates, and they stay as spend.
- **Tool-only spend:** 6,360 lines, ZAR 449m (15% of A's ZAR 3.0bn), counted once.

### Golden keys
- **Suppliers:** one golden key (SG-f85711) covers 122 different supplier names (20,891 lines, ZAR 1.18bn).
  It looks like a client catch-all. Output keys reproduce the client's key for every known name, including tool
  names after normalisation. The classifier uses our own supplier clusters instead, because the bucket erased
  the supplier signal (time-split L3 accuracy 83% → 93%). Outside the bucket, our clustering matches the golden
  keys exactly.
- **Items:** golden item keys are 1:1 with material numbers, apart from buckets. Two material numbers with the
  same text ("POWDER;STNEDST") are *different* golden items, so I do not merge codes by description (a config
  switch can). Lines without a material number have no golden item, so I key them by description within the
  supplier, fuzzy-merged.

### Tree
A's tree repeats the L2 name at L3 on half its rows. The output copies the tree row exactly. Labelled ERP lines
output their label; tool-only lines are classified.

## Company B (USD)

### Records
41,028 lines from 8 ERPs.
- There is no status column. `paid_date` is blank on 25%, but an unpaid posted invoice is still approved spend,
  so it is not used to exclude.
- **Duplicates:** 1,066 lines are identical except `line_id`. Others differ only by a blank GL description. Key =
  system, invoice, line, vendor, amount, currency, description, material, quantity, PO → 597 DUPLICATE.
- **Reversals:** 311 invoice lines are posted and reversed with the exact opposite amount. Both sides become
  CANCELLED (686 lines; spend-neutral). The analyst seeds had 13 of these as included; I follow the brief's
  definition of a cancelled line instead.

### Non-procurement
Account descriptions IC06, IC05, IC67…, "IC Payables", "I/C Payable" and "Intercompany - CZ" are intercompany.
Descriptions that are only a tax label ("Sales Tax Expense -", "State & Local Taxes -", "GST Input Tax -") are
taxes. Together these reproduce all 450 seed exclusions, with none wrong among the seed inclusions.

### Currency
- `currency_rate` is filled on 51% of lines. On USD lines it holds an MXN-like 17–19 (B2/B3/B4), so it is
  ignored: USD stays USD.
- 60 lines, mostly EUR/CHF in B3, carry about 18.5. That is an MXN rate stamped on the wrong line, and it is
  replaced from the table.
- ARS/TRY rates that differ from the invoice month do match the posting month, so they are kept. The rule: a
  line rate is used if it falls within ×/÷1.25 of the table's rates for the invoice and posting months.
- Missing rates use the invoice month, then the posting month, then the nearest month in the table.

### Dates and descriptions
- All dates are ISO, but 78 invoice dates fall in 2018–2022/2027, outside a 2024 file. They are flagged and
  FX uses the posting month.
- 16.7% of descriptions are blank, and languages vary. Classification leans on vendor, GL, material and plant,
  plus supplier pooling.

### Labels and keys
- The seeds (2,550 included) train the classifier and are output as labelled.
- There are no supplier names, only vendor ids, so supplier key = vendor id. One vendor id under different ERPs
  cannot be merged without names.
- Item key = material number. Lines without one are keyed by description within the vendor.

## Assumptions to revisit
- Which copy of a duplicate is "first" follows file order (and history before new records).
- Reversal pairs → CANCELLED, not included.
- Repeat purchases with a conflicting PO are separate spend.
