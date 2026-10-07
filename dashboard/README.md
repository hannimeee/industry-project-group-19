# Current local dashboard — consumer-fraud-adjusted results

The dashboard now lives in this repository's `dashboard/` directory. Deploy the complete `dashboard/dist/` folder. No online deployment was performed for this update.

Run `python dashboard/refresh_final.py` from the repository root with the dependencies in `dashboard/requirements.txt`. The default source is `handoff/fraud_adjusted/final/`; explicit adjusted-return columns and `fee_basis` are validated. Source hashes, merchant totals, retention and risk scores are checked before writing the bundle. Curated reference attributes are joined by ABN. Internal `expected_monthly_fees` keys remain for compatibility but hold adjusted monthly return.

Return, weekly volatility and consumer exposures now include take rate and transaction-level consumer-fraud adjustment. The original category relationship matrices still come from `handoff/final/`, are full-dataset descriptive measures, and are labelled separately. Curated fraud averages in merchant details are descriptive full-history attributes, not necessarily the reconstructed scores used in the adjusted optimisation.

Alpha strategy labels: 100% High Growth; 99.5% Growth; 99% Balanced; 98% Conservative Balanced; 95% Conservative. Labels are alpha preferences, not risk-score thresholds.

The tutorial allows interaction inside the highlighted section, merchant and histogram popups, pagination and relationship switches. Histogram-to-ranking navigation moves the tour to its merchant step. Skip, Back, Next, Escape and keyboard traversal remain available.

`code/11_summary.ipynb` was rerun on the matching adjusted exports and input snapshot. It contains a comparison with original selections and their evaluation on the same adjusted basis. Original gross-only handoff exports and the separate demo dashboard remain unchanged.

---
## Historical implementation notes (superseded where the basis or path differs)

# Final merchant dashboard — Group 19

Open **dist/index.html** locally. Keep the entire `dist/` folder, including fonts, together. The page works offline. The original demonstration dashboard at the course-folder level remains unchanged.

## Included data

The source directory is `../MAST30034 Project 2 Repository/handoff/final/`:

- `portfolio_dashboard.csv`: global results.
- `tech_portfolio_dashboard.csv`, `home_portfolio_dashboard.csv`, `fashion_portfolio_dashboard.csv`, `arts_portfolio_dashboard.csv`, `auto_portfolio_dashboard.csv`: segment-specific results.
- `correlation_revenue.csv`, `correlation_customers.csv`, and their pair diagnostics: category relationship matrices.

Each of six scopes has five return-retention targets (95%, 98%, 99%, 99.5%, 100%) and 100 merchants per portfolio: 30 portfolios / 3,000 ranking rows. Initial selection: Global, α = 99%. Scope changes preserve alpha and page; alpha changes preserve page. The table retains the supplied monthly-gross-fee ordering, with direct page buttons 1–10.

Source portfolio IDs repeat between scope files. The bundle uses `scope::source_portfolio_id` as its unique selection ID and preserves `source_id` for provenance. Each scope's risk scores and achieved retention are relative to its own 100% baseline; risk scores are not on a shared absolute scale across scopes.

## Return retention, risk and certification

Return retention (α) is the minimum expected-fee retention target used to express risk tolerance. A lower floor lets the model sacrifice more expected return to reduce risk. It is a revenue constraint, not the risk score or a direct CV/HHI weight.

Expected return is mean monthly gross fees in AUD, not annual revenue or net profit. The score combines baseline-normalised weekly CV and shared-customer HHI with supplied weights (currently 50/50). CV is displayed as a percentage, HHI to eight decimal places, and the combined risk index has baseline 100. The graph displays only the five results for the chosen scope, with alpha labels and risk-score coordinates. Lines are visual guides, not a certified efficient frontier.

Each portfolio preserves its supplied solver certification, gap and status; none are inferred from its alpha level. The metric-card **?** buttons explain definitions and selection quality. Gross fees exclude fraud losses, unpaid principal, funding/operating costs and settlement timing.

The results badge shows 03 October 2022 as requested; this is kept separate from the underlying selection snapshot of 30 September 2022. Expected fees use March 2021–September 2022; portfolio risk uses the 26 complete weeks 28 March–25 September 2022.

## District income reference

`avg_consumer_district_income` is joined by merchant ABN from `../MAST30034 Project 2 Repository/tables/curated/merchants/` (Parquet dataset). The current table contains 4,026 merchants. All 3,000 selected rows match an ABN and have non-null income.

The UI shows this as **Avg. district income · AUD/week**, preserving the source numeric value in the bundle and formatting to cents. This is contextual area-income data, not an individual customer's income or another ranking criterion. The ETL averages consumer district estimates over merchant transactions. Those district estimates derive from population-weighted averages of 2021 ABS SA2 median weekly personal incomes; the pipeline includes imputation for missing district estimates. It is not the median personal income of the merchant's customers. Merchant ranking order is unchanged. Any future null value displays an em dash; missing merchant keys fail refresh instead of silently dropping ranked merchants.

## Category relationships

The matrix definitions come from `code/04_categories.ipynb`:

- **Revenue correlation:** average pairwise Pearson correlation of monthly merchant transaction revenue (−1 to +1).
- **Customer overlap:** average pairwise Jaccard similarity of unique customer sets (0 to 1), despite the filename `correlation_customers.csv`.

Self-pairs are excluded. Diagonal cells compare different merchants within one segment; they are neither variances nor self-correlations. Undefined revenue correlations are excluded, not treated as zero. Exact values and valid/excluded pair counts are available on hover. Colour shading now uses each matrix’s observed minimum and maximum for contrast. The displayed legend states that range; compare numeric values rather than colours across measures.

These are full-dataset descriptive matrices, independent of the currently selected portfolio and alpha. Their underlying transaction history is separate from the portfolio's frozen 26-week risk window. No portfolio-specific covariance is fabricated.

## Refresh

The page itself needs no dependencies. Refreshing from Parquet needs PyArrow:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python refresh_final.py
```

Optional alternative locations:

```sh
.venv/bin/python refresh_final.py --source /path/to/final --merchants /path/to/curated/merchants
```

The refresh reads saved selections, never runs the optimizer, and never modifies the repository. It validates ranks, counts, repeated portfolio metrics, fee sums, risk-score arithmetic and retention against the appropriate scope baseline, plus matrix labels/ranges/symmetry and diagnostic totals. Portfolio metrics are taken once per portfolio, never summed across merchant rows.

The original manifest covers only the global files and dictionary, and those hashes are verified. New segment and relationship inputs are validated structurally and mathematically; their SHA-256 hashes, and the curated income Parquet hashes, are recorded in the bundle. They are not falsely represented as covered by the old manifest. The output is replaced only after the complete import validates.

All scopes must share snapshot, risk window, currency, weights and size. A future dataset with new periods or definitions requires updating the explanatory text as well as refreshing data.

## Verification and guide

Checks reconciled 3,000 ranking rows and all income joins to source data, validated all 30 selections and 300 table pages, checked scope/alpha/page persistence and both matrix controls, and confirmed the original demo's file hashes are unchanged. UI logic checks use a DOM stub; visual browser inspection of local-file URLs is unavailable in this environment.

The top-right **?** replays the guide. First-visit state uses `merchant-dashboard-real.guide.v1`. Metric popups support close, Escape and focus restoration. Space Grotesk is bundled locally with its licence; no external font or data requests are made.


## Merchant distributions and revenue naming

The collapsible Merchant distributions section sits above Category relationships and displays five histograms for the selected 100 merchants: average district income, average consumer fraud score, total transactions, average transaction value, and monthly expected revenue.

Monthly expected revenue is the existing final-table `merchant_expected_monthly_gross_fees` value, labelled more concisely. It already includes the take rate: monthly transaction sales × take rate / 100. Do not multiply gross fees by the take rate again. It excludes fraud losses. The curated `expected_revenue` is a different, fraud-adjusted total over stored history and is deliberately not substituted.

The refresh now also joins `avg_consumer_fraud_prob`, `total_transactions` and `avg_value_per_transaction` from curated merchants by ABN. Fraud scores are already in percentage points (e.g. 4.8 means 4.8%, not 480%). Counts and transaction-value averages cover the curated history, not necessarily the portfolio's 26-week risk window; this is noted on the graphs.

Income and fraud-score histograms use linear scales. The other metrics use log10 when all values are positive, max/min exceeds 30 and max/median exceeds 8 in the current scope's supplied portfolio rows; otherwise they use linear scales. The scale is printed under each chart. Ten equal-width bins in the chosen transformed scale use a common domain across all five alpha selections within a scope. All values, including extreme values, are retained. Null attributes are excluded and the available count is shown. Original units are used on ticks and merchant lists even on logarithmic axes. Both all-equal and empty inputs are handled.

Hover, focus or click a bar to inspect its merchant list. Clicking pins keyboard focus in the list; use the close button or Escape to dismiss. Each merchant entry shows its rank and exact value and jumps to its ranking page with a highlighted row. The list scrolls independently. Changing scope or alpha closes stale lists and updates the distributions. Colour contrast changes in relationship matrices are explicitly local to each matrix.

Integrity checks covered all 150 metric/portfolio distributions, their 100-member bin totals and fixed domains across alpha, plus popup lists and rank-to-page navigation. Visual browser inspection remains unavailable.


## Merchant detail hover

The table reference column now displays the average consumer fraud score (stored percentage points), instead of district income. Hover, focus, or click a merchant row for a detail panel showing its full name/category and five source attributes: district income, fraud score, transaction count, average transaction value and curated `expected_revenue`. The latter is the take-rate- and consumer-fraud-adjusted total over stored history, not a monthly estimate. Both the raw historical value and the existing monthly gross-fee value are preserved in the bundle.

There are now four distribution charts; the monthly revenue histogram has been removed. The metric-card monthly revenue definition still accurately describes the supplied monthly gross fees; a decision about displaying historical fraud-adjusted totals in the ranking is pending.

Global portfolio comparison now includes selected-merchant percentages across the five segments, updated with alpha and hidden in per-segment views. Shares use merchant counts, not return weights. The refresh derives category-to-segment membership from the supplied per-segment results, rejects conflicting/unmapped categories, and exports a segment ID for each selected merchant.
