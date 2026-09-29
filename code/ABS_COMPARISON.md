# ABS momentum controlled comparison

This experiment tests one addition to **recent merchant sales + own growth**: published ABS industry spending momentum. Both methods choose 100 merchants from the same eligible pool and hold that selection for three complete months. Gross fee revenue, rather than net profit or a risk-adjusted objective, is the outcome.

## Reproduce

Use Python 3.12 and install `code/requirements-abs.txt` in your project environment. The first run requires internet access to the official ABS website. From the repository root:

```text
python -m pip install -r code/requirements-abs.txt
python code/abs_momentum_comparison.py --source-dir "../project 2 materials"
python -m pytest -p no:capture code/tests/test_abs_momentum_comparison.py -q
```

The input folder must contain `project-2-bnpl-tables-part1.zip` through `project-2-bnpl-tables-part4.zip`. Source archives are left unchanged. The default source location is the sibling `project 2 materials` directory, regardless of the current working directory. No Spark, Census extracts or consumer enrichment is required. The exact order-ID audit uses a temporary SQLite index on disk and takes several minutes; allow approximately 2 GB of free disk space plus retained outputs.

Open `code/02_abs_momentum_comparison.ipynb` for the explanatory workflow and results. On a fresh checkout, it runs the full workflow. If a completed report exists, its default is to display the saved results; set `RUN_PIPELINE = True` to rerun. A rerun reproduces the same experiment; it is not a new unseen holdout.

The test command disables pytest's capture plugin to avoid an obsolete `pyreadline` package in the original Windows environment. This does not disable tests or assertions. A fresh isolated environment can use ordinary `pytest`.

## Staged execution and cached data

```text
python code/abs_momentum_comparison.py --prepare-only
python code/abs_momentum_comparison.py --offline --reuse-panel
```

Preparation verifies historical releases, scans all original orders, audits exact global order-ID uniqueness, and constructs features. It fits no model and scores no outcomes. `--offline` requires every needed ABS file to be cached. `--reuse-panel` explicitly trusts the previously audited merchant-month panel; use it only while the original archives and metadata remain unchanged. Omit that flag for a fresh transaction audit. `--output-dir` can redirect analytical results; the default is `tables/analysis/abs_momentum_comparison/`.

Historical HTML and original `.xls`/`.xlsx` workbooks are cached in `tables/landing/abs_retail_trade/`. The original dated release pages provide detailed publication dates. November 2021 was a single combined release on 11 January 2022; the parser validates this documented exception rather than assuming a fixed lag. It fails on unverified publication evidence or inconsistent workbook metadata. Cached files are the currently accessible dated ABS releases, not independently archived captures of their exact original publication bytes.

## Frozen definitions

- **Recent sales:** sales over the latest three complete months divided by three.
- **Own growth:** latest three-month sales / preceding three-month sales minus one.
- **ABS momentum:** the same three-versus-three ratio from the latest Table 12 vintage actually published at that historical origin. All six ABS observations come from that one vintage.
- **Model inputs:** log recent monthly sales and log(1 + own growth); the enhanced model adds raw ABS momentum. Standardization uses training rows only.
- **Target:** total sales during the next three complete months, in A$1,000 for fitting. Multiply predictions by 1,000 and the merchant's take rate / 100 to obtain gross fees.
- **Model:** Tweedie GLM, power 1.5, log link, alpha 0.1, intercept, L-BFGS, 2,000 maximum iterations, tolerance 1e-6. Equal row weights; no tuning or feature interactions.
- **Eligibility:** valid percentage take rate and positive sales in both preceding three-month windows. Months with no transactions count as zero only after complete source coverage is verified. A merchant's future inactivity is retained as a zero outcome.
- **Ranking:** predicted fees descending, ABN ascending for ties, exactly 100 distinct merchants.

For each origin, two models fit the exact same matched-merchant rows, with and without ABS momentum. One separate internal-only model supplies identical forecasts for unmatched merchants to both selection policies. Category labels route this fixed coverage split; they are not extra regression predictors. `code/abs_category_mapping.csv` explicitly specifies all 25 labels, including the 12 unmapped ones and assumptions behind the 13 mapped labels.

The earliest feature origin is August 2021. Development selections occur at December 2021 through March 2022 month ends, with three-month outcomes. The final June 30, 2022 selection is scored on July-September. Training rows enter only after all three target months have ended. Final training origins are August 2021-March 2022; the first development fit has only two distinct origins. **April 2022 is the latest Table 12 reference month available at June 30.**

`protocol.json` freezes the mapping, features, model parameters, dates and retention rule before evaluation. Reusing its output directory with changed settings fails. `fitted_models.json` records the fitted coefficients, scaling values, iterations and training origins. `runtime_versions.json` records the actual dependency versions. These are experiment records, not a replacement for the existing ingestion pipeline or a source-hash tracking system.

## Outputs

Analytical tables are saved as uncompressed Parquet plus CSV under the ignored output directory:

| Output | Purpose |
|---|---|
| `merchant_month` | Complete monthly sales and order counts for metadata merchants |
| `source_partition_audit`, `data_audit.json` | Counts, sales reconciliation, source dates and global ID audit |
| `unknown_merchants_monthly` | Complete-month transactions excluded because merchant metadata is absent |
| `abs_releases`, `abs_vintage_values`, `abs_features` | Publication evidence, six observations per vintage/series and historical momentum |
| `category_mapping`, `merchant_features`, `eligibility`, `coverage` | Mapping assumptions, past-only inputs and exclusion/coverage summaries |
| `unscored_predictions/` | Forecasts and fixed selections saved before outcome attachment |
| `predictions`, `final_selected_merchants` | Forecasts, realized fees, ranks and selection flags |
| `fold_metrics`, `forecast_metrics` | Top-100 revenue and forecast MAE/WAPE/signed bias |
| `selection_changes`, `category_composition` | Entrants/removals and mapping/category concentration |
| `decision.json`, `findings.md` | Prespecified decision and concise interpretation |

All money metrics are Australian dollars. `own_growth` and `abs_momentum` are fractions (0.10 means 10%); exported metrics ending `_pct` are percentage points (1.0 means 1%). `bias_dollars` is sum(predicted fees - actual fees). Forecast metrics include all eligible merchants or the matched subset, not only selected merchants. Each method's top 100 is scored over the full three-month period without monthly reselection.

## Interpretation and validation

Retain ABS momentum for the next stage only if final top-100 fees improve by **at least 1%** and **at least three of four** earlier uplifts are positive. Forecast accuracy, category attribution and the repeat-latest-sales benchmark provide context without changing this rule. A negative result is a valid completed experiment.

Tests cover hand-calculated features/targets/fees, missing history, future zero sales, late ABS releases, later revisions, workbook geography and adjustment metadata, historical label maturity, future-data perturbations, common unmatched forecasts, exact selection size, deterministic ties, independent realized-fee recomputation, protocol changes and duplicate order IDs across archives. Runtime checks audit the real sources and fail on model nonconvergence or nonfinite predictions.

Assumptions: categories and percentage take rates are static and known throughout the study; historical sales approximate outcomes under hypothetical selection. Fee revenue excludes defaults and costs. ABS is nominal and seasonally adjusted, with broad category proxies. The data are synthetic, the final test is one quarter, and earlier windows overlap. Do not interpret merchant or transaction counts as independent evidence about macroeconomic sensitivity, or claim causality/statistical significance from this pilot.

Sources: [ABS April 2022 release](https://www.abs.gov.au/statistics/industry/retail-and-wholesale-trade/retail-trade-australia/apr-2022), [ABS May 2022 release and July publication date](https://www.abs.gov.au/statistics/industry/retail-and-wholesale-trade/retail-trade-australia/may-2022), [ABS retail methodology](https://www.abs.gov.au/methodologies/retail-trade-australia-methodology/aug-2022), [scikit-learn TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html).
