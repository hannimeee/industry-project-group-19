# MAST30034 Project 2 — Group 19

Merchant ranking and portfolio selection for the Buy Now, Pay Later industry project. This README is the single repository guide; model explanations and assessed findings belong in the notebooks, while generated experiment reports remain with their data under `tables/analysis/`.

## Start here

- **Client-facing findings:** [13_non_technical_summary.ipynb](code/notebooks/13_non_technical_summary.ipynb).
- **Technical model and complete saved ranking views:** [12_summary.ipynb](code/notebooks/12_summary.ipynb).
- **Current results:** [fraud-adjusted final exports](handoff/fraud_adjusted/final/) and their [data dictionary](handoff/fraud_adjusted/final/data_dictionary.csv).
- **Interactive presentation:** open [dashboard/dist/index.html](dashboard/dist/index.html) locally, keeping its entire folder together.
- **Data preparation and quality:** [01_etl.ipynb](code/notebooks/01_etl.ipynb), then [02_analysis.ipynb](code/notebooks/02_analysis.ipynb).

The reorganisation separates notebooks, executable helpers, tests and reference mappings. It preserves the existing forecasting, fraud, selection and segmentation logic. The delivered portfolio files remain the reference results; reorganising code or refreshing explanatory tables does not establish a new model result.

## Repository layout

```text
README.md                         Single project/reproduction guide
code/
  notebooks/                      Explanations, analysis and executed outputs
  scripts/                        Python helpers and command-line entry points
  tests/                          Research regression tests
  requirements-notebooks.txt      Current notebook execution environment
  requirements-*.txt              Historical experiment-specific environments
tables/
  reference/                      Tracked category correspondence CSVs
  landing/archives/               Original four BNPL ZIPs, unchanged
  landing/                        Other original/external source files
  raw/                            Parsed source tables
  curated/                        Joined/enriched analytical tables
  analysis/                       Experiment results, audits and caches
  execution/                      Notebook execution logs, statuses and failed attempts
handoff/
  portfolio_optimizer/            Reusable aggregate-input optimizer
  optimize.py                     Portable CLI
  input_snapshot/, reference/     Original gross-fee inputs and selections
  final/                          Original gross-fee outputs and relationship matrices
  fraud_adjusted/                 Current adjusted inputs, final outputs and provenance
  tests/, requirements.txt        Portable package checks and historical pins
dashboard/
  refresh_final.py                Validate exports and rebuild local dashboard data
  dist/                          Offline HTML, JavaScript, CSS, data and fonts
```

Raw, intermediate and generated `tables/` layers are Git-ignored. The small `tables/reference/` mappings are tracked. Frozen handoff datasets retain their existing portable-package paths. This remains an exception to the specification's instruction to put all data under `tables/`; the requested reorganisation does not relocate the handoff package. Keep the repository private as instructed by the project specification. Do not commit credentials or change supplied dataset bytes.

### Reading and execution order

| Order | Notebook or entry point | Responsibility |
|---|---|---|
| 00 | `00_preliminary_analysis.ipynb` | Inspect supplied sources and initial quality |
| 01 | `01_etl.ipynb` | Ingestion, joins, external income, score completion and merchant aggregates |
| 02 | `02_analysis.ipynb` | Missingness/treatment, geography, outliers and fraud diagnostics |
| 02–03 | `02_abs_momentum_comparison.ipynb`, `03_external_data_comparison.ipynb` | Controlled external-feature experiments |
| 04–06 | Cash-flow variability, lookback and risk-comparison notebooks | Define and compare risk/history choices |
| 07–09 | Risk-aware selection, diversification and customer-concentration notebooks | Portfolio research and stress diagnostics |
| 10 | `10_categories.ipynb` | Existing category exploration and segmentation code |
| 11 | `code/scripts/11_category_ranking.py` | Optional category portfolio regeneration; this is a script |
| 12 | `12_summary.ipynb` | Technical definitions and checks of frozen selections |
| 13 | `13_non_technical_summary.ipynb` | Stakeholder findings and required ranking views |

This is the narrative order, not a claim that every research experiment must run before viewing final results. The two `02` notebooks have different purposes; the runner places transaction analysis first. The existing positional segmentation logic and relationship calculations in notebook 10 are preserved. Its rerun relationship copies now go to `tables/analysis/merchant_similarity/export/`, without republishing the frozen inputs under `handoff/final/`. Summary notebooks describe the delivered segment membership, so regenerating an exploratory mapping does not relabel frozen results.

## Execution and reproducibility

Run commands from the repository root. The current local notebook environment uses **Python 3.13.5 and Java 17**; its package versions are recorded in [code/requirements-notebooks.txt](code/requirements-notebooks.txt). Historical experiments and the portable optimizer originally used Python 3.12 with their separate pinned requirements. Use isolated environments when reproducing those historical configurations rather than combining conflicting pins.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-notebooks.txt
python -m ipykernel install --prefix .venv --name bnpl-project --display-name "BNPL project"
```

Java must be available to PySpark; check `java -version`. On Windows activate the corresponding `Scripts` environment. The runner executes each notebook in a fresh kernel from its notebook directory. **For a complete research rebuild, use the phased recipe below rather than running the runner without notebook names.** Numeric notebook order alone does not build the prerequisite studies or respect their immutable-cache boundaries.

To refresh just the two summary views, use the following command **before the research freezes, or in an isolated rebuild**:

```sh
python code/scripts/run_notebooks.py 12_summary.ipynb 13_non_technical_summary.ipynb --kernel bnpl-project
python -m pytest -p no:capture code/tests handoff/tests -q
```

The summaries can write presentation tables into `tables/analysis/`; even a display-oriented rerun can invalidate a later study's frozen inventory. Opening the already executed notebook outputs requires no execution and makes no such changes.

**All 14 notebooks completed successfully in fresh kernels using the phased rebuild below.** The final concentration study and notebook 09 finished on 13 October 2026 (Melbourne time). The final audit on 17 October checked all execution records and found **141 executed non-empty code cells and no error outputs**. Per-notebook logs and JSON status files are written to `tables/execution/`; `latest_run.json` describes only its particular runner invocation, while `reorganisation_validation.json` covers all 14 notebooks. Failed attempts are saved separately as `*.failed.ipynb` and do not overwrite the last successful notebook output. Execution records sit outside `tables/analysis/` because notebooks 08/09 fingerprint that whole analytical layer; writing runner logs inside it would invalidate their checks. The default timeout is 1,800 seconds per cell and can be changed with `--timeout`. Use these execution records rather than assuming that old displayed outputs prove a successful rerun.

### Inputs and dependencies

The four unchanged `project-2-bnpl-tables-part1.zip` through `part4.zip` archives have been copied to `tables/landing/archives/`. Shared research helpers resolve an explicit `--source-dir` where supported, then `BNPL_SOURCE_DIR`, then a complete legacy sibling `../project 2 materials` directory, then the repository landing folder. Set the environment variable to choose a source consistently:

```sh
export BNPL_SOURCE_DIR="$PWD/tables/landing/archives"
```

The ETL also expects extracted files directly under `tables/landing/project-2-bnpl-tables-partN/`, for each part 1–4. Each ZIP contains an internal `tables/` prefix: extract the **contents of that internal directory** into its part directory, without leaving an extra `tables/` level. For example, part 1's archive member `tables/consumer_user_details.parquet` belongs at `tables/landing/project-2-bnpl-tables-part1/consumer_user_details.parquet`. Preserve all deeper dataset/partition directories and keep the original ZIPs unchanged. The research helpers that read ZIPs directly still use `tables/landing/archives/`.

The ETL and external preparation stages may download official ABS files on first use. Offline options require the complete saved cache. Large transaction scans, UMAP, placebo refits and portfolio searches can take substantial time and disk space. Do not mistake a saved-results display for a fresh model fit.

A complete raw-data rerun has more dependencies than the portable optimizer: external preparation builds the audited monthly panel; `timesfm_comparison.py prepare` builds its reconciled daily panel; adjusted preparation checks that daily audit and the original handoff snapshot. Rebuild missing inputs in that order and inspect failures. Source/protocol fingerprints deliberately reject incompatible caches. A copied file with different timestamps can fail a historical size/timestamp check even when its bytes match; do not remove these checks or manufacture a matching audit.

### Complete research rebuild: preserve the cache order

The recipe below rebuilds the research and notebook evidence while using the existing frozen handoff portfolios for the final summaries. It does not regenerate those delivered portfolios. Finish the environment and input setup above first. Use a **fresh `tables/analysis/` layer**: move the entire existing layer to a unique directory outside it, retaining its contents and freeze records. Do not delete individual manifests, modify recorded hashes, or place the backup inside the layer being inventoried. The first four commands below do this in a POSIX shell; on Windows perform the equivalent directory move before running the Python commands.

Run each command to successful completion before starting the next. The default online preparation commands may fetch external sources; `--offline` is appropriate only when the required landing caches are complete. Notebook 02 ABS builds its study when its saved decision is absent. Notebooks 04–07 run their own existing studies, whereas 03, 08 and 09 need the explicit prerequisite commands shown here.

```sh
mkdir -p tables/execution
bnpl_analysis_backup=$(mktemp -d "$PWD/tables/execution/analysis-before-rebuild.XXXXXX")
if [ -d tables/analysis ]; then mv tables/analysis "$bnpl_analysis_backup/analysis"; fi
mkdir -p tables/analysis

# Complete all early analyses and summary exports before later studies freeze their inputs.
python code/scripts/run_notebooks.py 00_preliminary_analysis.ipynb 01_etl.ipynb 02_analysis.ipynb 02_abs_momentum_comparison.ipynb 10_categories.ipynb 12_summary.ipynb 13_non_technical_summary.ipynb --kernel bnpl-project
python code/scripts/external_data_comparison.py prepare
python code/scripts/external_data_comparison.py backtest --jobs 4
python code/scripts/external_data_comparison.py freeze
python code/scripts/external_data_comparison.py confirm
python code/scripts/external_data_comparison.py report
python code/scripts/run_notebooks.py 03_external_data_comparison.ipynb --kernel bnpl-project
python code/scripts/forecast_comparison.py --jobs 4
python code/scripts/timesfm_comparison.py prepare
python code/scripts/simple_return_comparison.py
python code/scripts/run_notebooks.py 04_cash_flow_variability.ipynb 05_cash_flow_lookback_comparison.ipynb 06_cash_flow_risk_comparison.ipynb 07_risk_aware_selection.ipynb --kernel bnpl-project

# Freeze and finish diversification before starting the concentration study.
python code/scripts/diversification_frontier.py all
python code/scripts/run_notebooks.py 08_diversification_frontier.ipynb --kernel bnpl-project
python code/scripts/customer_concentration.py all
python code/scripts/run_notebooks.py 09_customer_concentration.ipynb --kernel bnpl-project
python -m pytest -p no:capture code/tests handoff/tests -q
```

`timesfm_comparison.py prepare` builds the audited daily panel on CPU; it does not load or run the optional GPU forecasting model. The concentration command above is the original study entry point used on macOS/Linux and can parallelize dates. On Windows, use `customer_concentration_serial.py all` for the historical resource-limited route described under [Customer-concentration and risk studies](#customer-concentration-and-risk-studies); the two launchers do not provide identical resource guarantees.

**Earlier analytical outputs become immutable inputs to later studies.** In particular, notebooks 06 and 07 write research results; rerunning them after a later freeze can change bytes or modification times even when displayed numbers look identical. Do not rerun ETL, summary exporters, earlier study/report commands, or copy replacement files into that analytical layer after the dependent study has frozen it. Notebook 08 checks the wider analysis inventory, and the subsequent concentration study adds its own files. Consequently, after completing 09, the earlier 08 inventory is naturally no longer the current whole-layer inventory. Its successful execution record remains evidence of its completed phase; a plain “run every notebook again” against the populated final cache is not supported. To execute that earlier phase again, use a fresh rebuild or an exact preserved snapshot from that phase, keeping all original provenance checks intact.

## Ranking and deliverables

The specification asks for all-merchant rankings, an overall top 100, the top 10 within three to five segments, justified insights, automated ETL and readable, reproducible notebooks. The summary notebooks distinguish three different outputs:

1. **Full-pool baseline order:** the existing expected-fee criterion orders the available merchant pool, with eligibility and exclusion reasons disclosed. This is a fee ranking, not a new independently fitted merchant-risk score.
2. **Selected portfolio:** the existing joint optimizer selects 100 eligible merchants at a specified return-retention floor. Covariance and shared customers matter jointly. Exported `rank_by_expected_fees` orders merchants *within that already selected portfolio*.
3. **Segment top-10 view:** ranks 1–10 from each delivered segment's existing 100-merchant list. This does not claim a separately optimized ten-merchant portfolio.

The five segments are Technology/Digital/Communications; Home/Garden/General Supplies; Fashion/Jewellery/Personal Care; Arts/Crafts/Gifts/Entertainment; and Automotive/Outdoor/Recreational Equipment. Their exact delivered category labels remain fixed.

The technical summary retains its 100% baseline default and also displays the existing 99% dashboard view. The dashboard initially opens 99%. These are alternative saved preferences; their default labels do not establish that a setting is universally optimal. Use the stated retention when quoting a list or result.

The final summary should explain the problem, data and ETL, missingness/outlier treatment, external evidence, ranking rationale, required lists, interesting merchant differences, limitations and practical recommendations. The ZIP specifications also require presentation and group-process deliverables; code organisation alone cannot demonstrate their completion or guarantee a grade. The Project 2 Week 8 specification refers to a transaction delta “to be treated as fraud”, allowing removal or an `is_fraud` field. Its relationship to the supplied probability tables has not been established from the inspected files; no filename or matching keys are specified. This is an unresolved checkpoint instruction, not a demonstrated missing final-submission item. Supplied probabilities must not be represented as verified transaction outcomes.

## Model and fee definitions

The frozen snapshot is **30 September 2022**. Expected monthly fees average **19 complete months, March 2021–September 2022**, including genuine zero months after coverage checks. Weekly risk uses **26 complete Monday–Sunday weeks, 28 March–25 September 2022**. Amounts are AUD. Merchant eligibility uses valid metadata/take rates, positive sales in both preceding three-month windows, and positive expected/risk-window fees.

Current transaction fees are:

```text
adjusted fee = dollar_value × take_rate / 100 × (1 − consumer_fraud_probability / 100)
```

Both rates are percentage points. Merchant fraud is not multiplied in again. This is a modelled fee-earning proxy; it excludes principal losses, recoveries, funding/operating costs and settlement timing, and is not net profit.

For selection `z` of exactly `K` merchants, expected fees are `sum(mu_i × z_i)`. The eligible top `K` by expected fees is the baseline; numeric ABN breaks fee ties. The optimizer minimizes:

```text
weekly CV = sqrt(z' Sigma z) / sum(historical_mean_weekly_fees_i × z_i)
customer exposure A_ci = mu_i × customer_window_fees_ci / merchant_window_fees_i
HHI = sum_c (sum_i A_ci × z_i / sum_i mu_i × z_i)^2
score = 100 × [w × CV / baseline_CV + (1 − w) × HHI / baseline_HHI]
```

Selection must retain at least `retention × baseline expected fees`; all eligible merchants remain candidates. The delivered `w=0.5` expresses equal preference, not an empirically learned optimum. Covariance uses sample denominator `T−1` and Ledoit–Wolf shrinkage fitted on standardized nonconstant histories, preserving individual sample variances. Shared customers are combined before HHI is calculated. HHI lies in `[0,1]`; CV is not bounded by one. The mixed score has baseline 100 and is not a probability or a monetary saving. Each scope has its own baseline, so normalized scores are not a common absolute scale across segments.

## Fraud adjustment and findings

The current [adjusted final directory](handoff/fraud_adjusted/final/) contains six scopes × five retentions (100%, 99.5%, 99%, 98%, 95%): **30 portfolios and 3,000 merchant rows**. Global results are in `portfolio_dashboard.csv`/`.parquet`; segment files use prefixes `tech`, `home`, `fashion`, `arts`, `auto`.

Use `merchant_expected_monthly_adjusted_fees`, `portfolio_expected_monthly_adjusted_fees` and `fee_basis=consumer_fraud_adjusted`. Legacy `*_gross_fees` columns in this directory are compatibility aliases containing **adjusted** values. In `handoff/final/` they retain their original **unadjusted** meaning. Never mix the two bases.

| Global retention | Adjusted monthly fees | Weekly CV | Customer HHI | Risk index | Certified at requested tolerance |
|---|---:|---:|---:|---:|---|
| 100% | A$1,551,710.02 | 3.138% | 0.00004583 | 100.000 | Yes |
| 99.5% | A$1,544,100.83 | 3.102% | 0.00004548 | 99.061 | Yes |
| 99% | A$1,536,392.77 | 3.088% | 0.00004527 | 98.601 | Yes |
| 98% | A$1,520,697.99 | 3.032% | 0.00004537 | 97.817 | Yes |
| 95% | A$1,474,508.54 | 2.934% | 0.00004530 | 96.168 | No |

These are frozen historical results, not a newly confirmed backtest. The original preparation audit covers 13,440,766 orders across 579 days: 12,890,819 have merchant metadata and 549,947 lack a take rate and remain outside fee-based selection. It preserves supplied consumer scores on 80,308 orders and imputes 13,360,458 using the ETL's existing OLS specification. Only one imputed prediction was clipped. Across all metadata merchants, the average monthly fee total changes from A$4,802,322.77 gross to A$4,446,173.87 adjusted, a 7.42% modelled discount.

The OLS uses complete same-day spending, average order value, squared spending and squared order count. The labels are entity/date probability scores ending in February 2022. **96.36% of the adjusted run's orders are flagged outside observed predictor support.** That flag is not a fraud count or an error estimate. Validation on observed scores does not establish accuracy on the missing-score population. These are retrospective reconstructions, not purchase-time predictions; the separate earlier-history estimates are not silently substituted.

Sixteen of 30 portfolios meet the requested 0.1% optimization-gap tolerance; fourteen are explicitly best found with bounds. The global 95% gap is approximately 0.712%. Certification concerns numerical optimization, not future loss protection. [comparison.csv](handoff/fraud_adjusted/comparison.csv) compares membership and measurement changes; [old_and_new_on_adjusted_basis.csv](handoff/fraud_adjusted/old_and_new_on_adjusted_basis.csv) evaluates both selections on the **same** adjusted fee/risk basis.

### Regenerating adjusted portfolios

This is an expensive, optional regeneration workflow after the audited monthly/daily inputs and ETL definitions exist:

```sh
python code/scripts/fraud_adjusted_portfolios.py prepare
python code/scripts/fraud_adjusted_portfolios.py select
python code/scripts/fraud_adjusted_portfolios.py report
python code/scripts/fraud_adjusted_validation.py
```

`all` combines preparation, selection and report, but validation is separate. `select --scope global` restricts a run; other scopes are `tech`, `home`, `fashion`, `arts`, `auto`. After preparation and global selection, `python code/scripts/11_category_ranking.py --fraud-adjusted` runs/verifies category scopes and exports all scopes. Without the flag it follows the original gross-only workflow. The adjusted script uses the fixed explicit category lookup; it does not infer a new mapping from exploratory notebook output.

Reusing checkpoints requires identical input/settings fingerprints. Preparation and search records belong in `tables/analysis/fraud_adjusted_portfolios/`. Path/code changes can invalidate old protocols; preserve old results and use a separately audited rebuild rather than editing recorded hashes.

## Portable optimizer

The portable package consumes cleaned aggregate tables, so it can be used without rerunning the source archives or research notebooks. For a new adjusted-input experiment:

```sh
python handoff/optimize.py --help
python handoff/optimize.py optimize --input handoff/fraud_adjusted/input_snapshot --output handoff/runs/adjusted_custom --retentions 100 99 95 --count 100 --weight 0.5
```

Install `handoff/requirements.txt` in its own historical Python 3.12 environment when exact legacy dependency versions are needed. CLI retention values are **percentages** (`97.5`); exported retentions are **fractions** (`0.975`). Generic optimizer output retains legacy fee field names even when its input is adjusted. Keep input metadata and fee basis with every result.

`python handoff/optimize.py export` verifies/re-exports the **original five global gross-fee reference portfolios**; it is not the adjusted-results refresh command. Original global 100%/99.5% searches met tolerance; 99%/98%/95% remained uncertified, with approximately 0.115%/0.159%/0.912% gaps. The original 95% run stopped after stagnation. Preserve those original bounds when discussing that study.

### Input contract and reuse

| Input | Required fields | Grain |
|---|---|---|
| merchants | `merchant_abn`, `name`, `expected_monthly_fees`, `eligible` | One row per metadata merchant |
| weekly_fees | `merchant_abn`, `week_end`, `gross_fees` | One merchant/week, including genuine zeros |
| customer_fees | `consumer_id`, `merchant_abn`, `gross_fees` | One customer/merchant over the declared risk window |

Category is optional metadata. IDs must be nonnegative integers/digit strings with consistent identities across tables; floating IDs, duplicate keys, missing weeks, unknown merchant references and negative/nonfinite fees are rejected. Fees must share currency and treatment. Declare `as_of`, `weeks`, `window_start`, `window_end`, `expected_return_start`, `expected_return_end`, `currency` and `complete_coverage` in metadata. The latter is an upstream audit declaration, not permission to treat missing files as zero activity. Customer totals must reconcile with the weekly window; aggregates alone cannot prove their underlying dates.

```sh
python handoff/optimize.py prepare --merchants new_data/merchants.parquet --weekly-fees new_data/weekly_fees.parquet --customer-fees new_data/customer_fees.parquet --metadata new_data/metadata.json --output new_snapshot
python handoff/optimize.py optimize --input new_snapshot --output handoff/runs/new_dataset --retentions 100 99 97.5 95
```

CSV is also supported. Optional `--monthly-sales` derives expected gross fees/eligibility from a complete merchant/month sales panel and percentage take rates; do not use that gross-only derivation as an undocumented replacement for transaction-level fraud adjustment. At least six complete months are needed. New input snapshots need new/empty directories. A zero-variability baseline is rejected because normalization is undefined.

The solver combines one-for-one swaps, tangent lower bounds and MILP branch-and-bound. Defaults are one numerical/BLAS worker, 30 iterations, 200 nodes and 30 seconds per solver call, a 300-second soft per-portfolio search budget, and eight rounds of stagnation. Input loading, initial bound calculations and completion of an active step can exceed that soft budget. Windows additionally applies a 4 GiB committed-memory cap, 20% CPU cap and lower priority; other systems retain thread limits without those native caps. Use `--help` for explicit larger budgets.

`--gap .001` requests a 0.1% relative gap; the alternative absolute tolerance is 0.0001 score points. `--resume` requires matching data, code, dependency, settings and seed hashes. Compatible historical seeds are rescored; future-dated seeds are rejected and old bounds are not imported. Different machines/budgets may produce different uncertified solutions. Serve frozen results for exact list reproduction and precompute dashboard choices rather than optimizing on every slider change.

## Dashboard

Open [dashboard/dist/index.html](dashboard/dist/index.html) offline. Keep all JavaScript, CSS, fonts and images beside it. To refresh its data from current adjusted exports and curated descriptive attributes:

```sh
python -m pip install -r dashboard/requirements.txt
python dashboard/refresh_final.py
```

Optional `--source` and `--merchants` override those locations. Refresh validates hashes, membership/ranks, totals, fee basis, score arithmetic, scope compatibility and relationship diagnostics before replacing the bundle. It reads saved selections rather than running optimization. No online publication is part of this workflow.

The data grain is one merchant per portfolio, keyed by `portfolio_id, merchant_abn`. **Sum merchant fees within one portfolio; take FIRST for repeated `portfolio_*` fields.** Never sum all five settings together. Display HHI with sufficient decimals and CV as a percentage. Use the explicit adjusted monthly amount; curated `expected_revenue` is a different all-history total. The displayed results badge and the 30 September 2022 selection cutoff are separate dates.

The four descriptive distributions cover district income, average consumer score, transaction count and average transaction value. Their curated history can extend beyond the selection cutoff and differs from the reconstructed fraud inputs. District income is a population-weighted average of SA2 median weekly personal incomes, not personal income or an additional ranking criterion. Counts, means and shares must retain their stated units and windows.

Relationship matrices come from the original `handoff/final/` outputs: mean pairwise monthly transaction-revenue Pearson correlations and customer-set Jaccard similarities. They describe the full data, not the current portfolio or its risk window. Self-pairs are excluded; diagonal cells compare different merchants in the same segment. Undefined correlations are excluded. Colour ranges are local to each matrix; read numeric values for comparison. Strategy labels describe retention preferences, not thresholds on risk scores.

## Historical experiments

Research experiments remain evidence, including negative results; they are not additional inputs to the frozen final score. Generated `protocol.json`, model records, audit tables and `findings.md` remain in each experiment's `tables/analysis/` directory. Preserve them when interpreting old results. Repeated analysis of already inspected outcomes never creates a new holdout. Take rates/categories are assumed stable; synthetic sales approximate hypothetical selection outcomes; overlapping future windows are dependent and must not be summed as separate economic earnings.

### ABS momentum comparison

Notebook: [02_abs_momentum_comparison.ipynb](code/notebooks/02_abs_momentum_comparison.ipynb). Historical environment: `code/requirements-abs.txt` (Python 3.12).

```sh
python code/scripts/abs_momentum_comparison.py --source-dir tables/landing/archives
python code/scripts/abs_momentum_comparison.py --prepare-only
python code/scripts/abs_momentum_comparison.py --offline --reuse-panel
```

The last command trusts an already audited panel and complete ABS cache; omit `--reuse-panel` for a fresh audit. Outputs: `tables/analysis/abs_momentum_comparison/`; source cache: `tables/landing/abs_retail_trade/`. The original exact order-ID audit uses an on-disk SQLite index and needs substantial free space. Notebook `RUN_PIPELINE` controls a full rerun versus displaying a completed report.

This fixed comparison adds industry spending momentum to log recent sales and log own-growth ratio. Tweedie settings are power 1.5, log link, alpha 0.1, 2,000 iterations and tolerance 1e-6. Training-only scaling and matured three-month labels are used. Momentum uses six observations from one latest historically available Table 12 vintage. The [25-category mapping](tables/reference/abs_category_mapping.csv) contains 13 mapped and 12 unmapped labels; unmatched merchants receive identical internal-only forecasts in both arms. Ranking predicts fees, with numeric-ABN ties and exactly 100 merchants.

Development origins are December 2021–March 2022; the final June 2022 selection is evaluated on July–September. April 2022 was the latest Table 12 reference month available at June 30. The frozen retention rule required at least 1% final selected-fee improvement and positive uplift in three of four earlier origins. Negative results are valid evidence, subject to the later [selection-headroom caveat](#forecast-comparison). Publication dates, model fits, coverage, selections and independent realized-fee checks are saved. Cached dated releases are currently accessible versions, not independent captures of publication-day bytes.

### External data comparison

Notebook: [03_external_data_comparison.ipynb](code/notebooks/03_external_data_comparison.ipynb). Historical environment: `code/requirements-external.txt` (Python 3.12).

```sh
python code/scripts/external_data_comparison.py prepare
python code/scripts/external_data_comparison.py backtest --offline --jobs 4
python code/scripts/external_data_comparison.py freeze --offline
python code/scripts/external_data_comparison.py confirm --offline
python code/scripts/external_data_comparison.py report --offline
```

Outputs: `tables/analysis/external_data_comparison/`. Preparation audits original orders and builds the monthly/first-26-day panels. Cached external sources live under `tables/landing/external_data_comparison/`. The [reference mapping](tables/reference/external_category_mapping.csv) declares all 25 categories, proxies and source queries.

Features are same-vintage CPI year-on-year log change, Australian Google search momentum over two 13-week periods, and national industry business exits divided by starting businesses. Actual publication dates determine historical availability. Missing sources route predictions to the matching smaller subset, not zero-valued artificial momentum. The category-controlled baseline and every arm share eligibility, preprocessing and source-missingness flags. Return/downside model families are selected separately using matured baseline validation only; fixed Tweedie and shallow Poisson boosting are the candidate families. Return targets three-month sales; downside targets `max(0, 0.5 × recent-three-month sales − future-three-month sales)`.

Google Trends requires three official exports per request, Australian query headers, 26 complete weeks, at least 20 nonzero weeks and no more than 0.20 log-point spread. The historical run encountered HTTP 429/incomplete exports, so Trends was **not evaluable** and its arms were disabled. A future cache is not evidence that the earlier experiment had coverage.

The frozen decision uses at least four informative origins and two disjoint outcome windows; gains must exceed economic thresholds, be positive on at least 75% of informative origins, beat the 95th percentile of 99 fixed-seed best-arm placebos, and survive source-contribution checks. Return requires at least 1% uplift against both category baseline and repeat sales; downside requires 1% deviance and economic-utility improvement with at most 1% lost fees. Placebos shuffle category source histories with availability preserved; they are diagnostics, not causal tests. Strict mappings, concentration sensitivity and disjoint-window schedules are reported.

`freeze` precedes the separate October 1–26 check; `confirm` saves forecasts/selections before reading October. A confirmation marker blocks development refitting in that directory. October cannot rescue a rejected external source, does not establish three-month performance, and is already observed for subsequent reruns. Preserve freeze markers and use a new output directory for a new retrospective reproduction.

### Forecast comparison

Historical environment: `code/requirements-forecast.txt` (Python 3.12). Uses the external experiment's audited monthly panel and source fingerprints, with no new external downloads.

```sh
python code/scripts/forecast_comparison.py --jobs 4
python code/scripts/selection_headroom.py
```

Eight November 2021–June 2022 origins compare repeat sales, the original internal Tweedie regression, pooled six-month-lag Poisson boosting, and independent nonseasonal damped Holt forecasts. Eligibility, percentage take rates, three-month horizons and deterministic top-100 rules remain fixed; forecasts are saved before outcomes. Holt retains its recorded convergence/fallback/clipping rules. Outputs and frozen protocols are under `tables/analysis/forecast_comparison/`; hindsight ceilings are under `tables/analysis/selection_headroom/`.

Historical mean selected-fee uplifts against repeat were −0.325% for pooled lags, −0.270% for damped trend and −0.017% for the original regression. Better full-pool WAPE did not imply a better selected 100. A later hindsight check showed repeat already captured 99.82% of maximum attainable selected fees on average, with only 0.048%–0.397% available across folds. Thus the original 1% success screen was unattainable here; failure of that screen is not strong evidence against forecasting. Hindsight selections are diagnostic ceilings, never candidate forecasts. These results are retrospective, not untouched confirmation.

### TimesFM comparison

The optional TimesFM study is separate from the current notebook CPU environment. Historical pins are in `code/requirements-timesfm.txt`; the original isolated Windows environment used CUDA-enabled PyTorch 2.6.0 and an RTX 3050 Laptop GPU with 4 GiB VRAM.

```sh
python code/scripts/timesfm_comparison.py prepare
python code/scripts/timesfm_comparison.py smoke
python code/scripts/timesfm_comparison.py run
```

`prepare` reconciles the daily and monthly audited panels and does not require running the forecasting model. `smoke`/`run` require the configured CUDA environment and official local checkpoint; there is no silent CPU/cloud fallback. The original checkpoint is `google/timesfm-3.0-pytorch` revision `43046b85ec22d584a13f8098c2ed39c889e129c2`, under `tables/landing/timesfm3/checkpoint/` with `config.json`, `model.safetensors`, license and upstream README. Preserve upstream data-package/license documents under `tables/`; the repository documentation consolidation does not remove them. Verify the model's license before use beyond the academic experiment.

The primary return proxy integrates sorted nonnegative quantiles 0.1–0.9, using weights 0.15 on endpoints and 0.1 internally, with constant-tail assumptions. Native daily-median sums and weekly seasonal naive are diagnostics; daily quantiles are not joint portfolio samples. Settings use float32, batch 32, seed 2026, no fine-tuning/covariates and all historical daily sales through each origin. Outputs/audits are under `tables/analysis/timesfm_comparison/`.

The model dates from 2026 and its pretraining includes post-2022 data; this is a retrospective transfer experiment, not technology available at the simulated decision dates. Pretraining independence from synthetic source data is unknown. The 1% uplift screen is subject to the same hindsight-headroom limitation.

### Customer-concentration and risk studies

Notebooks 04–09 and their matching `code/scripts/` helpers retain the existing variability, history-window, diversification and shared-customer comparisons. The historical resource-limited concentration launcher is **Windows-only**:

```sh
python code/scripts/customer_concentration_serial.py all
```

It raises an error on macOS/Linux because its Windows resource ceilings have no unlimited fallback. On those platforms, the original study entry point is:

```sh
python code/scripts/customer_concentration.py prepare
python code/scripts/customer_concentration.py select
python code/scripts/customer_concentration.py evaluate
python code/scripts/customer_concentration.py oracle
python code/scripts/customer_concentration.py report
```

Stages are `prepare`, `select`, `evaluate`, `oracle`, `report`; `all` runs the sequence, and independent checks are `verify-select` and `verify-oracle`. Run one command at a time. The original entry point retains its existing model and search budgets but can parallelize dates; it does not supply the Windows launcher's resource guarantees or its recorded serial recovery settings. The Windows launcher preserves historical BLAS reduction settings and the two recovered May/June 95% hindsight cases. Respect any recorded local resource-stop marker and review its cause before restarting. `oracle` is hindsight, not an attainable forecast. Outputs are generated under `tables/analysis/customer_concentration/`; historical result caches are not assumed to exist in this checkout.

## Validation and provenance

The full test run of the reorganised code completed on 12 October 2026 with **233 passed and 8 skipped in 15.66 seconds** (`python -m pytest -p no:capture code/tests handoff/tests -q`). Separately, all 14 notebooks passed in fresh kernels using the documented phase order. The 17 October audit checked their records, current executable content, cell execution counts and absence of error outputs.

Notebook 13 received a Markdown-only clarification of the Week 8 transaction-delta wording after execution. Its code and outputs are unchanged; `13_non_technical_summary.markdown_update.json` records the original/current source hashes and identical code hashes. It was not rerun solely for that wording change, which would rewrite earlier presentation exports after the research freezes.

The concentration study completed all five stages: preparation, selection, evaluation, hindsight analysis and reporting. Its local execution wrapper initially capped independent date workers, then resumed at the original six-worker limit using verified checkpoints. May/June hindsight dates were queued first. The original per-date functions, retention order, search budgets, seeds and final assembly order were unchanged; wrapper versions, stage records and checkpoint checks are retained under `tables/execution/customer_concentration/`.

The final audit verified that all **58 tracked delivery files** across the final portfolios, input snapshots and dashboard match their Git HEAD bytes. Earlier checks against the pre-reorganisation snapshot also found those files unchanged. Both relocated reference mappings and all four source archives retain their original bytes. These checks concern saved data and selections; the historical packaging manifest's source/document hashes remain historical as explained below.

Current execution records belong in `tables/execution/`. Historical evidence remains in [handoff/VALIDATION.json](handoff/VALIDATION.json), [adjusted acceptance checks](handoff/fraud_adjusted/acceptance_checks.json) and [adjusted verification](handoff/fraud_adjusted/provenance/verification.json). The earlier records document 230-test gross and 241-test adjusted validation runs; they do **not** assert that the reorganised revision passed those runs.

Seven BLAS thread-control test cases require a backend detectable by `threadpoolctl`. They explicitly skip when none is detected, including the current Apple Accelerate environment; mathematical and selection tests remain enabled. This skip means thread limits/restoration cannot be verified or enforced through that API on this backend, not that historical Windows reduction ordering has been reproduced. The separate native Windows resource-enforcement test retains its platform guard.

`handoff/BUILD_MANIFEST.json` is a **historical packaging audit**. Moving source files and consolidating/deleting the old README files invalidates its old package/source path hashes. Preserve it as history rather than rewriting it to claim the old audit covered new files. Frozen data-file checksums in each final manifest can still verify the delivered selections independently. New execution or validation must record its own current sources and outcomes.

The organisation and presentation edits retain the selection/model definitions. A missing input, unavailable external source, failed execution or uncertified optimization must remain visible. Review the actual runner records and generated findings before submitting or claiming full end-to-end reproducibility.
