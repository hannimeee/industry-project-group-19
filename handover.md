# BNPL project handover

Updated 9 September 2026. Based on this working conversation, current notebook source, saved notebook outputs and Parquet metadata inspected on this date. Counts describe the supplied release, not permanent constraints. Paths below are relative to this repository.

## 1. Purpose, scope and working preferences

MAST30034 Applied Data Science, Project 2: Buy Now, Pay Later merchant ranking. The eventual objective is to rank merchants and recommend the best 100, with justified features/heuristics and insights. See the [BNPL assignment specification](../MAST30034%20Project%202%20Specs/MAST30034_Project_2_Buy_Now_Pay_Later_Spec.pdf).

This work began for **Week 6**: preliminary analysis, a generalised ETL draft and at least one useful external dataset. Week 7 concerns outliers, missingness and insights; Week 8 expects automated ETL and incorporation of a fraud-transaction delta; later weeks concern ranking, 3–5 segments, top-10 results within segments, presentation and final reproducibility. The spec recommends SA2 Census data; it does not prescribe our exact layer names or require this particular income calculation.

The user supplied curated/analysis diagrams. The implemented scope is **landing → raw → curated**. The analysis diagram is future intent, not implemented functionality: historical spending features, fraud modelling once labels exist, merchant aggregates, revenue forecasts and ranking. No `is_fraud` ground-truth field is currently created. Do not treat supplied probabilities as observed fraud labels.

Working preferences: compact, readable Python; Markdown sections between script-like notebook stages; useful comments; every function has a 1–2-line docstring. Put exploratory evidence in `00_preliminary_analysis.ipynb`; keep ETL focused, although the user has added diagnostics in its section 10. Use PySpark for transactions/joins because there are 14.2 million orders. Preserve supplied landing files. Current writers explicitly use **uncompressed Parquet** at the user's request. Do not silently reinstate reverted changes or convert hypotheses into data definitions.

## 2. Files and execution

- `code/00_preliminary_analysis.ipynb`: read-only exploration; inventory, schemas, tags, join coverage, transaction scans, ABS candidate and zero-income investigation. Uses pandas/NumPy/PyArrow. Full transaction scans are performed file by file. Section 7's added ABS checks require the extracts downloaded by ETL section 2; they do not require curated outputs.
- `code/01_etl.ipynb`: paths/helpers → ABS acquisition → source reads → raw writes → reference cleaning → income mapping → consumer enrichment → transaction enrichment → row preservation → diagnostics.
- `tables/landing/`: supplied source files and ABS extracts.
- `tables/raw/`: six supplied datasets plus four ABS datasets, each in its own directory.
- `tables/curated/`: consumers, merchants, consumer fraud, merchant fraud, postcode/SA2 overlaps, postcode income, transactions. **No curated users table.**

Run from the repository root or `code/`; notebooks find an ancestor containing `tables/landing`. The existing `mast30034` environment has pandas, NumPy, PyArrow, requests, openpyxl and PySpark. Local Python used during checks: `/Users/hannahluo/miniconda3/envs/mast30034/bin/python`. Java 17 was available; inspected Spark outputs identify version 4.1.1. Spark uses `local[4]` (four local worker threads), 32 shuffle partitions and Australia/Melbourne timezone. In an agent sandbox Spark may require permission to bind its local Java socket.

ETL writes use overwrite, not append. Run in order with fresh upstream DataFrames. Read a **whole table directory**, e.g. `spark.read.parquet(str(RAW / "transactions"))`, not an individual part file. Month partition columns are recovered from directories and may be absent from physical file schemas. File counts/names are not stable interfaces.

## 3. Supplied landing inputs and raw contracts

Notation: `str` = string; `long` = 64-bit integer; `double` = numeric measure; `date` = calendar date; `bool` = boolean. Raw names and types below are the expected Spark table schemas; column order is not significant.

Part 1 is `tables/landing/project-2-bnpl-tables-part1/`.

| Raw directory | Landing source | Grain/key | Rows | Raw columns |
|---|---|---|---:|---|
| `consumers` | `tbl_consumer.csv`, pipe-delimited | `consumer_id` | 499,999 | `name:str, address:str, state:str, postcode:str, gender:str, consumer_id:long` |
| `users` | `consumer_user_details.parquet` | `user_id`; `consumer_id` also unique | 499,999 | `user_id:long, consumer_id:long` |
| `merchants` | `tbl_merchants.parquet` | `merchant_abn` | 4,026 | `name:str, tags:str, merchant_abn:long` |
| `consumer_fraud` | `consumer_fraud_probability.csv`, comma-delimited | Intended `(user_id, order_datetime)`; duplicates retained in raw | 34,864 | `user_id:long, order_datetime:date, fraud_probability:double` |
| `merchant_fraud` | `merchant_fraud_probability.csv`, comma-delimited | `(merchant_abn, order_datetime)` | 114 | `merchant_abn:long, order_datetime:date, fraud_probability:double` |
| `transactions` | Snapshot directories in parts 2–4 | `order_id` | 14,195,505 | `user_id:long, merchant_abn:long, dollar_value:double, order_id:str, order_datetime:date, source_snapshot:str, order_month:str` |

Merchant ABN is a named pandas index, but Spark reads it as a physical column; do not add another index column. CSV schemas are explicit and fail on malformed records. Postcodes remain strings. Transactions are discovered using `project-2-bnpl-tables-part*/transactions*_snapshot`; each snapshot root is read with partition discovery, then combined by column name. Output `order_month` is `yyyy-MM`, used for partitioning raw and curated transactions.

| Snapshot directory | Actual date range | Rows |
|---|---|---:|
| `transactions_20210228_20210827_snapshot` | 2021-02-28 to 2021-08-27 | 3,643,266 |
| `transactions_20210828_20220227_snapshot` | 2021-08-28 to 2022-02-27 | 4,508,106 |
| `transactions_20220228_20220828_snapshot` | 2022-02-28 to **2022-10-26** | 6,044,133 |

Retain actual partitions, including 1,634,128 orders after the final folder's advertised end date. Exploration found no overlapping/missing dates across snapshots. `order_datetime` has no time-of-day. No automatic transaction deduplication or outlier removal is applied. Future overlapping releases require a separately agreed precedence rule.

## 4. External ABS inputs

All sources use 2021 Census/geography. G02 means **General Community Profile table 02, “Selected medians and averages”**; SA2 is the chosen geography. The selected income is median weekly personal income, dollars, based on people aged 15+ and usual income before tax from all sources. It is not salary-only or household income.

Downloads are temporary; retained source extracts are Parquet with original column names, under `tables/landing/abs_2021/<name>/part-00000.parquet`. Existing extracts are reused. There is **no active manifest/hash tracking**. pandas handles the ZIP/Excel import; Spark processes the extracted tables.

| Extract | Rows | Landing columns, all strings | Raw output columns |
|---|---:|---|---|
| `sa2_income` | 2,472 | `SA2_CODE_2021`, plus eight G02 measures listed below | `sa2_code:str, median_personal_income_weekly:double` |
| `mb_sa2` | 368,286 | `MB_CODE_2021, SA2_CODE_2021, STATE_CODE_2021` | `mb_code:str, sa2_code:str, state_code:str` |
| `mb_poa` | 368,286 | `MB_CODE_2021, POA_CODE_2021` | `mb_code:str, postcode:str` |
| `mb_population` | 368,285 | `MB_CODE_2021, Person` | `mb_code:str, population:long` |

G02 retained measures: `Median_age_persons`, `Median_mortgage_repay_monthly`, `Median_tot_prsnl_inc_weekly`, `Median_rent_weekly`, `Median_tot_fam_inc_weekly`, `Average_num_psns_per_bedroom`, `Median_tot_hhd_inc_weekly`, `Average_household_size`. Only `Median_tot_prsnl_inc_weekly` feeds raw income.

Sources (exact download URLs are also in ETL section 2):

- [G02 SA2 Australia DataPack ZIP](https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip): extract `G02_AUST_SA2.csv` by suffix.
- [Allocation files](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/access-and-downloads/allocation-files): `MB_2021_AUST.xlsx` and `POA_2021_AUST.xlsx`.
- [Mesh-block population counts](https://www.abs.gov.au/census/guide-census-data/mesh-block-counts/2021): `Mesh Block Counts, 2021.xlsx`; concatenate `Table ` sheets, skip six heading rows, select `MB_CODE_2021`/`Person`, retain 11-digit mesh-block codes. Allocation files include a nonnumeric outside-Australia code, explaining the one-row difference.
- [General Community Profile templates](https://www.abs.gov.au/census/guide-census-data/about-census-tools/download-product-templates-and-data-item-lists): XLSX G02 layout/notes.
- [2021 Census dictionary](https://www.abs.gov.au/census/guide-census-data/census-dictionary/2021), [personal income INCP](https://www.abs.gov.au/census/guide-census-data/census-dictionary/2021/variables-topic/income-and-work/total-personal-income-weekly-incp), [median-income glossary](https://www.abs.gov.au/census/guide-census-data/census-dictionary/2021/glossary/m).

## 5. Curated contracts and joins

| Directory | Grain/key | Rows currently saved | Columns |
|---|---|---:|---|
| `consumers` | `user_id` | 499,999 | `user_id:long, name:str, address:str, state:str, postcode:str, gender:str, consumer_record_matched:bool, income_sa2_count:long, district_income:double, income_coverage:double, income_matched:bool` |
| `merchants` | `merchant_abn` | 4,026 | Raw merchant columns plus `category:str, letter_group:str, take_rate:double` |
| `consumer_fraud` | `(user_id, order_datetime)` | 34,765 | Raw score columns with score renamed `consumer_fraud_probability` |
| `merchant_fraud` | `(merchant_abn, order_datetime)` | 114 | Raw score columns with score renamed `merchant_fraud_probability` |
| `postcode_sa2_population` | `(state, postcode, sa2_code)` | 5,464 | `state:str, postcode:str, sa2_code:str, population:long` |
| `postcode_income` | `(state, postcode)` | 2,655 | `state:str, postcode:str, income_sa2_count:long, area_population:long, income_population:long, district_income:double, income_coverage:double` |
| `transactions` | `order_id` | 14,195,505 | `order_id:str, order_datetime:date, order_month:str, dollar_value:double, user_id:long, merchant_abn:long, consumer_fraud_probability:double, merchant_fraud_probability:double, consumer_postcode:str, consumer_gender:str, consumer_district_income:double, income_matched:bool, consumer_record_matched:bool, merchant_record_matched:bool` |

Processing rules:

1. Left-join users to consumers on `consumer_id`, then drop `consumer_id`. Keep the mapping in raw only. Uppercase/trim state; pad plausible 3–4-digit postcodes to four characters, leaving other source values for review. A format check is not proof of a valid geographic postcode.
2. Parse merchant tags from the final letter/rate fields so commas within categories are preserved. Brackets vary between `()` and `[]`. Keep original tags. Stop on failed take-rate parsing.
3. Remove **exact duplicate fraud rows only**; conflicting entity/date scores fail uniqueness checks. The 99 extra consumer-fraud rows are exact duplicates. Keep score/take-rate units as supplied; do not divide by 100 or invent thresholds without an agreed interpretation.
4. Build income lookup as described below; left-join to consumers by `state + postcode`.
5. Left-join transactions to consumer attributes by `user_id`; attach merchant existence by ABN; attach consumer fraud on `user_id + order_datetime`, merchant fraud on `merchant_abn + order_datetime`. Fraud matches do not require a merchant-master match. Missing scores remain NULL, never zero. Only match flags are filled with False.
6. Keep merchant names/categories/rates in merchants, not repeated in transactions. Raw transaction provenance remains available by order ID.

**Current schema versus older draft:** curated transactions omit `source_snapshot`, `income_coverage` and `income_year`. Consumers retain `income_coverage`; current income tables do not create `income_year`. Section 8 Markdown still mentions coverage/year, but the final code selection and saved schema above are authoritative. The comment describing False flags as “unknown score” is imprecise: these flags describe master-record/income matches, not fraud-score availability.

## 6. Income mapping and exclusions

Join `mb_sa2`, `mb_poa`, `mb_population` on `mb_code`. State codes 1–9 map to NSW, VIC, QLD, SA, WA, TAS, NT, ACT, OT. Each mesh block supplies both a POA and SA2 assignment; POAs can intersect several SA2s. Aggregate positive recorded population to `(state, postcode, sa2_code)` before joining SA2 income.

Exclude non-four-digit POAs, special codes `9494` (no usual address), `9797` (migratory/offshore/shipping), and nonpositive/missing mesh-block populations. `ZZZZ` fails the format check. For each state/POA:

`district_income = sum(overlap_population × SA2_income for usable SA2s) / sum(overlap_population for usable SA2s)`

Usable currently means **SA2 income > 0**. If the usable denominator is zero, leave income NULL. Areas removed before grouping have no lookup row and also yield NULL on the consumer left join.

- `income_sa2_count`: distinct SA2s in positive-population overlaps, including those without usable income.
- `area_population`: sum of all observed positive overlap population.
- `income_population`: population in overlaps whose SA2 income is usable.
- `income_coverage`: `income_population / area_population`, a numeric fraction. It is not consumer match rate and does not measure missing population records.

This is an **average of SA2 medians**, not the POA median, an individual income or a weighted median of people. All-age population is a geographic approximation for an income measure covering ages 15+. Customer locations must not be labelled merchant locations. Census data released from June 2022 support retrospective enrichment, not availability before release in historical forecasting.

### Zero-income evidence

The original G02 personal-income field has 2,472 rows: 0 NULL/blank/nonnumeric values, 42 zeros, 2,430 positives. Four zero-income SA2s enter positive-population postcode overlaps:

| SA2 | Name | State/postcode | Recorded overlap population |
|---|---|---|---:|
| 208031192 | Moorabbin Airport | VIC 3194 | 26 |
| 506011111 | Ashendon – Lesley | WA 6111 | 3 |
| 801101145 | Molonglo – East | ACT 2611 | 4 |
| 506021121 | Perth Airport | WA 6105 | 20 |

Total: 53 residents across four pairs. ABS QuickStats for each states that information cannot be provided because population is absent/very small. Sources and executable checks are in preliminary section 7.1–7.3; example [Moorabbin Airport](https://www.abs.gov.au/census/find-census-data/quickstats/2021/208031192).

The user considered including genuine zeros. Evidence did not establish these as genuine $0 medians, so the positive-income rule was retained. This is a conservative project rule, **not an ABS definition that every zero means NULL**. Keep source zeros; exclude their weighted contribution, not consumers/orders. A confirmed genuine zero would belong in the numerator as zero and in the denominator with its population.

### Missing consumer income: full follow-up audit

83,998 / 499,999 consumers lack income; 532 / 3,167 distinct consumer postcodes are involved. An earlier side discussion checked only 20 examples; the later full check found:

| Cause | Postcodes | Consumers |
|---|---:|---:|
| Absent from original ABS mesh-block-to-POA allocation | 527 | 83,181 |
| State/territory mismatch | 3 | 484 |
| POA exists but has zero recorded population | 2 | 333 |

Mismatches: consumers use NSW for 2899 (175 people), WA for 6798 (177) and 6799 (132); ABS assigns all three to OT. Existing OT income estimates are available, but reconciliation has **not been implemented**. Zero-population POAs: VIC 3062 (174 consumers), WA 6182 (159).

A direct ABS POA median was explored only for feasibility. It would simplify the statistic but would not recover the 527 absent codes because it uses the same POA geography. ABS excludes non-street-delivery codes such as PO boxes; some other postcodes receive no mesh blocks. This explains possible causes, not a verified classification of every missing code. See [ABS POA documentation](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/edition-3-july-2021-june-2026/non-abs-structures/postal-areas). No direct-POA replacement or income imputation was implemented.

## 7. Merchant categories and letter codes

Initial lowercase/trim produced 126 labels, inflated by formatting variants. Current identical pandas/Spark rules: lowercase; trim; collapse repeated whitespace; no space before commas/colons and one after; one space around hyphens. Preserve punctuation/words; no fuzzy merges or spelling repair. Result: **25 categories**, all in multiple letters, 21 in all five. All 4,026 merchant results were checked for pandas/Spark agreement. Original tags remain unchanged.

Example antique-shop variants collapse to counts `a=37, b=54, c=30, d=7, e=1`. Exploratory output includes the full category-by-letter table and rate summaries. `letter_code` in exploration corresponds to `letter_group` in ETL.

| Letter | Merchants | Observed take-rate range |
|---|---:|---:|
| a | 1,602 | 5.50–7.00 |
| b | 1,351 | 3.10–5.10 |
| c | 922 | 1.40–3.10 |
| d | 98 | 0.52–1.40 |
| e | 53 | 0.10–0.48 |

Letters correlate strongly with take-rate tiers, not exclusive industries. Their business meaning remains unconfirmed; do not equate them with risk grades or infer exact boundaries from rounded values. The spec describes a percentage revenue share, but source units/letter semantics have not been formally confirmed.

## 8. Validation and observed missingness

- Original consumers/users are one-to-one with no unmatched records in either direction. Transactions use 24,081 distinct users, while the consumer master has 499,999.
- Transactions reference 4,422 merchant ABNs. Of these, 396 lack master records, affecting 580,830 orders (4.09%). Nineteen merchant-fraud rows also refer to absent master merchants. Retain/flag them; no ranking eligibility rule has been applied.
- `require_key` checks non-null uniqueness for reference join keys and raw order IDs. CSV reads fail fast. Final ETL verifies raw/curated transaction row-count equality. Row-count equality alone is not a full content comparison.
- A separate prior Spark audit found 14,195,505 rows and distinct order IDs, with no null IDs, in both raw and curated. Current Parquet metadata still totals 14,195,505 each; this handover did not rerun the full ETL or distinct-ID audit.
- All source transaction amounts were positive/finite; unusually large amounts were retained for later outlier analysis.

Current saved ETL diagnostic outputs (not independently rescanned for this document):

| Transaction diagnostic | Count |
|---|---:|
| Missing consumer fraud score | 14,115,157 (99.4340%) |
| Missing merchant fraud score | 14,191,446 (99.9714%) |
| Missing either score | 14,195,100 (99.9971%) |
| Missing both scores | 14,111,503 (99.4082%) |
| Missing consumer district income | 2,368,565 |

Only 405 orders have both scores. Scores end before the final transaction period; do not forward-fill or treat missing scores as no fraud without an explicit decision. Saved diagnostics show no NULL order IDs/dates/months, values, user IDs, ABNs, consumer postcodes or genders; available transaction income ranges from 276 to 3,250. Consumer-master missingness and transaction missingness have different denominators.

## 9. Revision history and operational caveats

1. User had supplied landing data, preliminary exploration and an ETL start. A curated-layer draft was developed around the supplied diagram; analysis-layer work remained deferred.
2. Added brief docstrings/comments; switched all writers from Snappy to uncompressed Parquet on request. Compression changes do not make binary Parquet human-readable; existing files change only on a write.
3. Removed the ABS manifest/hash code and documentation on request. Do not reintroduce it by assuming it is missing functionality.
4. Built an interactive POA → mesh blocks → SA2 illustration from actual local ABS data. POA 3000 spans five populated SA2 overlaps and has an estimated income of about $776.16/week. The illustration lives outside the repo in the conversation's visualization directory; it is not a pipeline dependency.
5. Investigated 18 `order_month=... 2` raw folders: they were empty, not duplicated data. Cause was not proven. Normal multiple `part-*.parquet` files within months are expected. Prior duplicate-ID audit passed; no cleanup/rebuild was necessary.
6. Investigated zero incomes; added reproducible evidence in preliminary section 7. Retained the positive-income rule.
7. Explored direct POA lookup coverage without implementing it. Found a temporarily missing section-6 write; **current code again saves `postcode_income` before section 7 reads it**.
8. A `FAILED_READ_FILE.FILE_NOT_EXIST` error occurred after lookup overwrite/repeated cell execution. Old lazy DataFrames can retain references to replaced files even after dropping income columns. A refresh/rebuild fix was added and tested twice, but the user **reverted it after a rerun fixed the issue**. Current section 7 drops old income columns and reuses `consumers`; the refresh/rebuild fix is not active. Do not represent the reverted fix as current or silently restore it. On recurrence, recreate upstream DataFrames by running in order; avoid concurrent writers or using old downstream DataFrames after overwrite. The repository is in iCloud, but synchronization was not established as the cause.
9. Added category-by-letter exploration, then fixed formatting duplicates in both notebooks. Earlier 126-category statistics are superseded by 25 categories. A full pipeline rerun after the latest category edit was not performed by the assistant; validate saved merchant contents if consuming outputs independently of notebook source.

Some preliminary Markdown still describes completed decisions as future proposals (e.g. geography selection or order-ID checks). Treat this as historical narrative; distinguish code, saved outputs and dated conversation checks. User edits have also removed earlier provenance/coverage fields from curated transactions. Verify current schema before downstream code changes.

## 10. Unimplemented decisions / next-work boundary

No automatic imputation for missing incomes/scores, territory reconciliation, future-snapshot overlap resolution, outlier deletion, fraud delta integration, trained model, merchant ranking, analysis-layer aggregates or revenue formula has been implemented. The analysis diagram suggested historical spending/deviation and merchant revenue/customer stability features; these require separate definitions, temporal leakage controls and confirmed score/take-rate units. It did not settle how fraud probabilities affect expected revenue.

For future agents: use current notebook source as the implementation reference, this document for rationale/history, and the assignment for deliverables. Preserve the original data and uncertainty flags. Separate confirmed observations from inferred meanings, especially zeros and letter codes. This handover is documentation only; it did not modify notebook logic or regenerate tables.
