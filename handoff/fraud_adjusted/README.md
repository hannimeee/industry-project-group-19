# Fraud-adjusted portfolio rerun

This rerun discounts each transaction's fee using the **consumer** fraud score, matching section 17 of the team's `code/01_etl.ipynb`:

```text
adjusted fee = dollar_value * take_rate / 100 * (1 - consumer_fraud_probability / 100)
```

Both supplied fields are percentages. Merchant fraud is not multiplied in again: doing so would introduce a different, unsupported joint-probability assumption.

## Outputs

`final/` contains the global portfolio and all five category portfolios, at 100%, 99.5%, 99%, 98% and 95% retention: **30 portfolios, 3,000 merchant rows**. Each scope has its own recomputed fraud-adjusted revenue baseline. `comparison.csv` compares these selections with the original gross-fee selections.

The original `handoff/final/` files remain unchanged. In this new directory, the legacy `*_gross_fees` columns are compatibility aliases for adjusted fees. Prefer the explicit `merchant_expected_monthly_adjusted_fees` and `portfolio_expected_monthly_adjusted_fees` columns. Every row declares `fee_basis=consumer_fraud_adjusted`; portfolio identifiers include the scope and basis. Portfolio-level columns repeat on merchant rows: never sum them.

Expected return is still the mean over 19 complete months, March 2021–September 2022. Weekly covariance and customer exposures are also rebuilt from transaction-level adjusted fees over March 28–September 25, 2022. Risk remains a 50/50 combination of baseline-relative weekly CV and customer HHI. Selection date is September 30, 2022; exactly 100 eligible merchants per portfolio.

## Missing-score reconstruction

The team's completed transaction exports and fitted models are Git-ignored and were not available in this checkout. The rerun therefore loads the **exact existing model definitions** from the ETL notebook, fits its consumer OLS on the supplied entity/date labels and fills only missing scores. It does not invent a new regression or tune features. Supplied scores, including 0 and 100, remain unchanged. Predictions are clipped to 0–100 exactly as in the existing notebook.

The model uses complete same-day customer spending, average order value, squared spending and squared order count, across all merchants including unknown merchant IDs. Customer identity uses the original one-to-one user/consumer bridge. Generated scores, model coefficients, support/clipping flags, and daily coverage are saved under `tables/analysis/fraud_adjusted_portfolios/`.

This is retrospective reconstruction of supplied scores, not verified fraud outcomes or a new point-in-time backtest. Labels end in February 2022; later activity is extrapolated and flagged when outside training support. The discount is a user-specified fee-earning proxy; it does not establish BNPL liability, principal losses, recoveries or net profit.

## Reproduce

Use the existing pinned `handoff/requirements.txt` dependencies and the supplied source archives under `project 2 materials/` beside the repository. Existing audited daily panels are required for reconciliation.

```text
python code/fraud_adjusted_portfolios.py prepare
python code/fraud_adjusted_portfolios.py select
python code/fraud_adjusted_portfolios.py report
python code/fraud_adjusted_validation.py
python -m pytest -p no:capture code/tests handoff/tests -q
```

`all` runs preparation, selection and export; run the separate validation command afterwards. `select --scope global` runs only one scope; other options are `tech`, `home`, `fashion`, `arts`, `auto`. Completed selection checkpoints can be resumed with the same command and unchanged inputs/settings. Preparation verifies original source signatures and records SHA-256 hashes. Changed inputs require a new preparation directory; existing results are not silently overwritten.

### Category-ranking entry point

The team's existing category script also supports the adjusted run:

```text
python code/10_category_ranking.py --fraud-adjusted
```

Run this after preparation and the global selection above. It runs all five category groups sequentially, verifying and reusing completed checkpoints with unchanged inputs/settings, and re-exports the global and category files. Each group contains five 100-merchant portfolios (500 rows). The exact 25-category lookup matches the team's category notebook and is included in `code/fraud_adjusted_portfolios.py`; this mode does not require the ignored `category_segments.parquet` file. Omitting the flag retains the original gross-only workflow and its original output location.

| Category group | New CSV (Parquet with the same stem is also included) |
|---|---|
| Technology, Digital & Communications | `final/tech_portfolio_dashboard.csv` |
| Home, Garden & General Supplies | `final/home_portfolio_dashboard.csv` |
| Fashion, Jewellery & Personal Care | `final/fashion_portfolio_dashboard.csv` |
| Arts, Crafts, Gifts & Entertainment | `final/arts_portfolio_dashboard.csv` |
| Automotive, Outdoor & Recreational Equipment | `final/auto_portfolio_dashboard.csv` |

Numerical work is sequential with one BLAS thread. Windows retains the 4 GiB committed-memory and 20% CPU limits. Each portfolio uses the portable optimizer's original default search budget, warm-started with the original same-scope selections (rescored against adjusted inputs). Certification flags, lower bounds and gaps remain explicit; a completed rerun need not prove every portfolio globally optimal.

This completed run passed all 241 tests and independent checks of all 30 portfolios. Sixteen meet the 0.1% optimization-gap tolerance; fourteen remain explicitly uncertified. See `FINDINGS.md`, `acceptance_checks.json` and `provenance/verification.json` for results and validation evidence.

## Portable reuse without source archives

The rebuilt aggregate inputs are included in `input_snapshot/`. From the repository root:

```text
python handoff/optimize.py optimize --input handoff/fraud_adjusted/input_snapshot --output handoff/runs/adjusted_custom --retentions 100 99 95 --weight 0.5
```

This command uses fraud-adjusted amounts even though the generic optimizer preserves its legacy gross-fee column names. The new final exports add explicit adjusted-fee aliases. Do not mix gross-only and adjusted portfolios under the same identifier.
