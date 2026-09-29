# External features for merchant return and downside

Experiment 03 tests category CPI inflation, Google search momentum and industry business exit rates. The original ABS Retail Trade experiment remains separate and unchanged. A completed experiment can reject all external features.

Open `03_external_data_comparison.ipynb` for the executed explanation and results. The current findings are generated in `tables/analysis/external_data_comparison/findings.md`. Generated tables, cached workbooks and audit indexes are ignored by Git; Parquet files are uncompressed.

## Reproduce in stages

Use Python 3.12. Run from the repository root:

```text
python -m pip install -r code/requirements-external.txt
python code/external_data_comparison.py prepare
python code/external_data_comparison.py backtest --offline --jobs 4
python code/external_data_comparison.py freeze --offline
python code/external_data_comparison.py confirm --offline
python code/external_data_comparison.py report --offline
python -m pytest -p no:capture code/tests -q
```

Preparation requires the four original `project-2-bnpl-tables-part*.zip` archives in the sibling `../project 2 materials` folder. Use `--source-dir` to change that location. Historical ABS pages and workbooks require internet access on the first preparation; `--offline` requires an existing complete cache. All analytical stages work offline after preparation. `all` runs the same stages in order, including the final October check; use the separate commands when reviewing the freeze boundary.

The source audit scans daily Parquet partitions and aggregates immediately. It retains an exact SQLite order-ID index to check October against development transactions. Allow several GB of free disk space. First preparation and 99 placebo refits can take substantial time; four ordinary worker processes run the placebos with one numerical-computation thread per fit. Completed placebo repetitions are cached and reused after an interrupted backtest. Prepared input fingerprints prevent checkpoints from being reused with different merchant panels or external features.

The audit deliberately skips October partition payloads. Previously audited data from experiment 02 lacked verifiable source-change fingerprints, so experiment 03 rebuilt the panel. Subsequent reuse requires identical archive names, sizes and modification timestamps and identical parsed metadata. This is a local cache check, not a cryptographic archival guarantee. Changed sources require a fresh audit directory.

The `freeze` stage saves the chosen return and downside subsets before October access. The `confirm` stage saves all September forecasts and top-100 selections before reading October. Once `confirmation_read.json` exists, development refitting in that output directory is rejected. Repeating `confirm` returns the saved decision. Use a different `--output-dir` for a complete reproduction, while acknowledging that October is now observed: rerunning cannot create a new holdout. Interrupted October reads require inspection rather than silently resetting the holdout marker.

## Files and responsibilities

- `external_data_sources.py`: source audit, explicit ABS publication evidence, historical workbook parsing and Google CSV quality gates.
- `external_data_models.py`: past-only merchant features, common preprocessing, fixed models, missing-source fallback and economic scoring.
- `external_data_comparison.py`: staged execution, chronological family selection, all subset arms, placebos, candidate freezing, confirmation and reporting.
- `external_category_mapping.csv`: all 25 exact normalized merchant labels, identifiers, queries, proxy flags and assumptions.

The implementation reuses the previous experiment's metadata parser, fixed Tweedie settings, file-writing conventions and internal baseline. It does not invoke the unrelated consumer enrichment pipeline or require Spark.

## Sources and fixed features

| Source | Feature | Publication treatment |
|---|---|---|
| CPI Table 7 | `log(latest quarterly index / same-quarter prior-year index)` | Both indexes from the latest detailed release available at the origin |
| Google Trends | `log(sum of latest 13 complete weeks / sum of preceding 13 weeks)` | A separate Australian Web Search window ending before the origin month; reconstructed history |
| CABEE Data Cube 2 | National industry exits divided by beginning-of-year businesses | Detailed industry-class publication date, not the earlier headline date |

ABS cache: `tables/landing/external_data_comparison/cpi` and `cabee`. The parser validates CPI series identifiers, descriptions, geography, units, adjustment and reference period, and CABEE codes and national industry totals. Conflicting category correspondence becomes unmatched; the code does not invent a replacement. Telecom combines the three published national numerator and denominator counts before division. State rows are not added to national totals.

CPI releases used are June 2021 through June 2022. At June 30, 2022 the available quarter is March, published April 27; at September 30 it is June. The June 2021 page now shows an October 19 housing-graph update: its explicit original July 28 release date and update explanation are checked and retained. No future quarterly release enters an earlier origin.

CABEE's 2019–20 industry workbook was published February 16, 2021; its 2020–21 successor was published December 16, 2021. The dates are verified in the workbook headers. The older sheet has a trailing space in its name, which is normalized when finding the annual employment table. The published all-employment national totals supply the rate. This annual series has little variation during the study.

These are currently accessible dated ABS release workbooks, not independent captures of their exact bytes on publication day. The historical release, reference period and publication evidence are recorded; the June 2021 page update is disclosed.

Google export requests and exact filenames are saved in `trends_export_requests.csv`. Place official CSVs under `tables/landing/external_data_comparison/trends`, using each request's identifier followed by `_1.csv`, `_2.csv` and `_3.csv`. The importer checks the Australian query header, 26 complete Sunday–Saturday weeks, positive totals in both halves, at least 20 nonzero weeks, and a maximum 0.20 log-point range across three exports. Daily CSV observations are aggregated to weekly means before taking the ratio. Insufficient exports, unstable estimates and low volume produce missing values, never zero momentum. Identical replicates are flagged and are not claimed to be independent samples. No query tuning against merchant outcomes is permitted.

In the present run the public endpoint returned HTTP 429. Historical browser charts loaded and one shoes CSV was obtained, but repeated subsequent export attempts delivered no files. Trends is therefore not evaluable under the three-export gate; its dependent arms are disabled. The code implements all eight arms and the official CSV ingestion route without bypassing access restrictions. A future run using new exports must remain explicitly retrospective.

## Merchant features, outcomes and controls

There are 19 complete months, March 2021–September 2022. Each eligible merchant has a unique metadata record, a valid percentage take rate, and positive sales in each of the two preceding three-month windows. Complete audited coverage allows transaction-free months to be zero; future inactivity is retained. Unknown merchants are reconciled separately and excluded from fee-based selection. Valid large transactions and original take rates are preserved.

Internal features are log recent monthly sales and the log ratio of latest to preceding three-month sales. Take rate is not a sales predictor. Models predict sales in A$1,000 and convert predictions to gross fees using take rate divided by 100.

The return target is total sales in the following three complete months. The downside target is `max(0, 0.5 * recent-three-month sales - future-three-month sales)`. Predicted downside is clipped to zero through half the reference sales before conversion to fees. All-zero downside training targets produce zero forecasts. Zero predicted means use a small positive value only inside deviance calculations.

Comparators are repeat sales, the prior Minimal + own growth method, and B, the category-controlled baseline. Minimal reproduces experiment 02's fixed matched/unmatched cohort split using internal predictors only; ABS Retail Trade momentum is never included. B adds category indicators and the same three source-missingness flags used in every arm. The eight definitions are B, C, G, E, CG, CE, GE and CGE; only B, C, E and CE are evaluable with the current source access.

Every paired arm uses identical eligible training rows and equal row weights. A single fitted training-only numeric imputer and standardizer is shared across arms at each cutoff. Category indicators remain unscaled. Numeric source features with no overall training variation are disabled. If a source is missing for a candidate, its prediction is copied exactly from the corresponding smaller source subset, including combination arms. Runtime assertions check this equality.

Tweedie settings remain power 1.5, log link, alpha 0.1, fitted intercept, L-BFGS, 2,000 iterations and tolerance 1e-6. Gradient boosting uses Poisson loss, 200 iterations, learning rate 0.05, seven leaves, minimum 100 rows per leaf, L2 penalty 1, no early stopping and seed 2026. Return and downside families are chosen independently using equal-origin average fee Tweedie deviance from already matured B validation forecasts. Fewer than two available validation origins, or a tie, selects Tweedie. Every source arm uses the same selected family; there is no source-specific tuning.

Return policies rank predicted fees. Risk policies hold B return forecasts fixed and rank predicted fees minus predicted shortfall. The primary shortfall penalty is 1; 0.5 and 2 are reported only as sensitivities. All selections contain exactly 100 eligible, distinct merchants, with ascending numeric ABN breaking ties. Selections remain fixed throughout their outcome period.

## Chronological evidence and selection

Three-month development origins are November 2021 through June 2022, monthly. Merchant feature history starts in August 2021. Only rows whose entire outcome is completed enter training. This gives one origin of training at the first evaluation and eight at the final June evaluation. Current outcomes are attached after predictions and ranks have been saved.

An informative fold requires each enabled source to have within-category variation in training and usable observations at prediction time. Other folds are still reported as static industry-context comparisons. This prevents an external numeric industry label from being mistaken for evidence about changing economic conditions. Candidate averages use their own informative origins, paired against both controls on those same dates. They should not be compared as if every candidate had identical dates.

The frozen decision requires at least four informative origins and two disjoint outcome windows. Return needs mean fee uplift of at least 1% against B and repeat, positive uplift in at least 75% of informative origins, and a gain above the 95th percentile of the best-arm placebo distribution. Downside needs at least 1% deviance improvement and utility improvement worth 1% of B risk-policy fees, with no more than a 1% fee reduction, at least 75% positive utility origins and the corresponding placebo gate.

Each of 99 seed-fixed placebos shuffles complete source histories between categories with identical availability patterns, independently per source. Time ordering and missingness are preserved. Every enabled subset is refitted, and the best economic result per repetition forms the comparison distribution. These are diagnostics for accidental selection gains, not causal significance tests.

Every included source must have positive economic benefit in a leave-one-source-out comparison over the candidate's informative origins. Among qualifying arms within 0.25 percentage points of the best result, fewer sources win; remaining ties follow CPI, CABEE, Trends. A positive total contribution that reverses after removing the largest positive merchant or industry contribution is flagged fragile and is not automatically retained. Strict mappings remove every starred proxy and refit the comparisons without changing eligibility. Three disjoint-window schedules and alternate risk penalties are reported without retuning the primary rule.

## October confirmation

July–September has already been examined, so development is retrospective. October 1–26 was reserved for one shorter check. At the September 30 decision, the target is the first 26 days of the following month. Its reference is the average first-26-day sales from the preceding three months. Internal and external feature definitions remain the same. Training uses completed first-26-day outcomes through September, and family selection uses only prior B losses for that horizon.

Return and downside source subsets are frozen before the shorter-horizon fit; only the internal-baseline family-selection rule operates during that fit. Their forecasts are combined once for the selected risk policy. Predictions, model settings and selections are written before October transactions are read. The final policies must satisfy the same 1% economic thresholds and revenue guardrails; downside also requires 1% deviance improvement. If no external candidate qualified in development, B is frozen and October compares internal policies and benchmarks only. October cannot rescue a rejected source.

Results are classified as supported for a further pilot, promising but unconfirmed, no demonstrated contribution, or not evaluable. Reconstructed Google history remains qualified. No October result establishes independent three-month performance.

## Outputs and validation

| Artifact | Contents |
|---|---|
| `protocol.json`, `runtime_versions.json` | Frozen mappings, thresholds, models, operational interpretations and installed versions |
| `merchant_month`, `merchant_features`, `eligibility` | Audited full-month/first-26-day panels, past-only features and exclusions |
| `merchant_audit.json`, `source_partition_audit`, `unknown_merchants_monthly` | Source signatures, exact ID audit, count/sales reconciliation and excluded merchants |
| `external_releases`, `external_vintage_values`, `external_features` | ABS dates, source URLs, same-vintage inputs and category/origin features |
| `category_mapping`, `coverage`, `trends_quality`, `source_status.json` | Proxy assumptions, count/fee coverage, query gates and disabled-source explanations |
| `development/` | Unscored and scored predictions, fold and forecast metrics, model records, calibration, category composition, entrants/removals |
| `strict_mapping/`, `strict_mapping_summary`, `nonoverlapping_schedules`, `source_variation` | Proxy and time-window sensitivity, static/within-category input variance; alternate risk penalties are in fold metrics |
| `placebos/`, `placebo_distribution` | All 99 assignments, fold results and best-arm null gains |
| `development_summary`, `candidate_decisions`, `frozen_candidates.json` | Informative-date summaries, all retention gates and frozen selections |
| `confirmation_development/`, `confirmation/` | Shorter-horizon baseline validation, predictions saved before scoring, October audit and results |
| `final_selected_merchants`, `findings.md`, `decision.json` | Final selections and the research conclusion |

Money metrics are Australian dollars; names ending `_pct` are percentages, where 1 means 1%. Source feature values are log changes or fractional rates. Prediction metrics cover the full pool and source-matched merchants, not just selected merchants. `selection_changes` reconciles signed entrant/removal contributions with independently recomputed selected fees. Risk calibration compares predicted and actual dollar shortfall by prediction decile.

Tests cover hand calculations, incomplete windows, zero outcomes, vintage dates, national totals, query quality, common preprocessing, missing-source routing for both model families, future-data perturbations, maturity-based family selection, deterministic ranking, independent fees, placebo availability, protocol freezing and the October guard. Runtime checks validate real-data source coverage, counts, sales, IDs, convergence and finite predictions. Additional actual-data perturbation and reproducibility evidence is saved beside the results.

Categories and take rates are assumed stable and historically available. Recorded merchant sales approximate outcomes under a hypothetical selection. Fees exclude defaults, funding and operating costs. CPI measures consumer prices, not margins or elasticity; business exits are not default probabilities. Synthetic merchant data, overlapping outcomes, shared industry signals and short history limit conclusions to a pilot decision.

## Primary references

- [ABS March 2022 CPI](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/mar-2022) and [Table 7 workbook](https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/mar-2022/640105.xlsx).
- [ABS CABEE publication timetable](https://www.abs.gov.au/articles/publication-changes-and-draft-roadmap-counts-australian-businesses-entries-and-exits) and [2017–2021 methodology](https://www.abs.gov.au/methodologies/counts-australian-businesses-including-entries-and-exits-methodology/jul2017-jun2021).
- [ANZSIC class codes and titles](https://www.abs.gov.au/statistics/classifications/australian-and-new-zealand-standard-industrial-classification-anzsic/2006-revision-2-0/numbering-system-and-titles/division-subdivision-group-and-class-codes-and-titles).
- [Google Trends data explanation](https://support.google.com/trends/answer/4365533?hl=en) and [official API access](https://developers.google.com/search/apis/trends).
- [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) and [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
