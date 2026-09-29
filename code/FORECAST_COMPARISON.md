# Comparing expected-fee forecasts

This experiment compares four ways of forecasting the next three complete months of merchant sales. It changes only the forecasting method; merchant eligibility, take rates, horizon and top-100 ranking rules stay fixed. It does not estimate portfolio covariance or impose risk penalties.

The completed run found no improvement over repeat sales: mean selected-fee uplift was -0.325% for pooled lags, -0.270% for damped trend, and -0.017% for the original regression. The two new methods won in 1/8 and 2/8 origins, respectively. Damped trend improved July–September full-pool fee WAPE to 11.29% versus 13.11% for repeat sales, but still earned 0.020% less selected fees in that period. Keep repeat sales as the current expected-fee benchmark. Both existing baselines reproduced exactly across 62,084 paired forecast rows; all 31,042 Holt fits converged, with 201 negative monthly forecasts clipped under the frozen rule. These findings are retrospective, not independent confirmation.

Subsequent interpretation correction: hindsight analysis found that repeat sales already captures 99.82% of the maximum attainable selected fees on average. Even perfect future knowledge improves it by only 0.048%–0.397% across the eight folds. The original 1% practical-success screen was therefore unattainable for this comparison. Preserve the executed results, but do not interpret failure of that screen as strong evidence against forecasting or features. Run `python code/selection_headroom.py` for independent recomputation and the report in `tables/analysis/selection_headroom/findings.md`. These future-informed selections are diagnostic ceilings, never candidate forecasts.

Run from the repository root with Python 3.12:

```text
python -m pip install -r code/requirements-forecast.txt
python code/forecast_comparison.py --jobs 4
python -m pytest -p no:capture code/tests -q
```

The command reuses the audited merchant panel in `tables/analysis/external_data_comparison`. It requires the original four archives in `../project 2 materials` and checks their saved sizes/timestamps, reparsed merchant metadata, the panel fingerprint, and sales/order reconciliation before reuse. The earlier experiment's preparation command is documented in `EXTERNAL_DATA_COMPARISON.md`. No external data are downloaded or used by this experiment. Existing experiment outputs are read without modification.

Use `--source-dir`, `--prepared-dir` and `--output-dir` to change paths. The comparison always refits its models; it does not reuse forecast checkpoints. An existing output directory must have an identical frozen protocol and input fingerprints. Parameters are saved before fitting; changing them requires a different output directory and a new, explicitly retrospective experiment.

## Fixed methods

| Method | Sales forecast |
|---|---|
| `repeat` | Repeat the preceding three-month sales total. |
| `minimal` | Reproduce the earlier log recent sales + log growth-ratio Tweedie regression, including its original fixed category coverage split. No external values enter its predictions. |
| `pooled_lags` | One gradient-boosting model fitted across merchants and matured historical origins, using six individual monthly sales values to predict the next three-month sales total directly. |
| `damped_trend` | Fit independent nonseasonal additive damped Holt forecasts to each merchant's entire available monthly history and sum the next three monthly predictions. |

The pooled model uses Poisson loss to estimate a nonnegative mean, 200 iterations, learning rate 0.05, seven leaves, at least 100 observations per leaf, L2 penalty 1, no early stopping, and seed 2026. Sales inputs and targets are in A$1,000. There are no category indicators, calendar features, extra customer features, external predictors or hyperparameter searches. Fixed tree settings match the earlier experiment's shallow model; the six lag inputs are new. Since the model family and inputs both change relative to `minimal`, this comparison does not attribute any difference exclusively to either change.

Holt uses all complete merchant months from March 2021 through each cutoff, estimated initial states, estimated smoothing and damping parameters, SLSQP optimization, no brute-force initialization, and a maximum of 1,000 optimizer iterations. Each series is scaled by its past mean for numerical conditioning. Negative monthly forecasts are clipped to zero. A constant history gets its exact constant forecast. Nonconverged/nonfinite forecasts use repeat sales as the predeclared fallback; convergence statuses, warnings, fitted parameters and clipping counts are exported.

## Chronology and comparison

There are eight selection dates: November 30, 2021 through June 30, 2022. All methods use the same eligible merchants, requiring positive sales in both preceding three-month windows and a valid take rate. Historical feature origins begin in August 2021. The entire target must have finished before a training row can enter a pooled fit: one training origin is available for the November test and eight for the June test. Holt has 9–16 monthly observations per merchant across these cutoffs.

Each sales forecast is multiplied by take rate / 100 to obtain expected gross fee dollars. Merchants are ranked by predicted fees descending, with ties broken by ascending numeric ABN. Exactly 100 are held for the following three months. Predictions, ranks and selections are saved before attaching that fold's future outcomes. No separate daily forecasts are needed.

The primary comparison is the equal-origin mean percentage difference in actual selected fees versus `repeat`. Also report mean dollar differences, positive origins, each origin separately, the three possible non-overlapping schedules, selection overlap and changes, category contributions, full-pool fee MAE/WAPE/bias/deviance, and sensitivity to removing the largest positive merchant/category contribution. WAPE is the sum of absolute fee prediction errors divided by total actual fees; it is not classification accuracy. Better full-pool forecast accuracy need not produce a better top 100. Deviance retains the previous implementation's 1e-12 floor for zero predictions expressed in fee thousands; positive outcomes following Holt's clipped zero forecasts can produce very large deviance. The floor never alters forecast dollars or selection.

A fixed descriptive screen flags candidates averaging at least 1% more selected fees and winning in at least 6/8 origins for further testing. This is not statistical significance. Results are retrospective: July–September and October have already been examined elsewhere, and none is an untouched holdout for this experiment. October is not included because it is a partial month and a different horizon. Overlapping outcome windows are dependent, so their fee totals must not be summed as distinct revenue. With synthetic transactions and only 19 complete months, no result establishes general performance over economic cycles.

## Outputs and checks

Generated files remain ignored under `tables/analysis/forecast_comparison`, with uncompressed Parquet and CSV tables:

- `findings.md`, `summary.csv`, `fold_metrics.csv`: concise results and forecast accuracy.
- `protocol.json`, `run_metadata.json`, `forecast_freeze.json`: parameters, versions, input fingerprints and saved-prediction boundary.
- `features`, `eligibility`: the actual past-only model inputs and candidate exclusions.
- `unscored/`, `predictions`, `final_selected_merchants`: all forecasts, ranks, selections and subsequently attached realized fees.
- `selection_changes`, `category_contributions`, `nonoverlapping_schedules`: economic attribution and sensitivity diagnostics.
- `pooled_fit_details`, `damped_fit_details`, `acceptance_checks.json`: training chronology, numerical checks and convergence records.

Checks cover hand-calculated lags and targets, rate conversion, deterministic ties, future inactivity, incomplete history/outcomes, duplicated keys, exact baseline reproduction, identical pooled forecasts after future-data perturbation, sampled Holt reruns after the same perturbation, and independent recomputation of selected fees from monthly rows. Unit tests also perturb a middle decision date to check earlier-fold causality.

Rates and categories are assumed stable and historically known. Historical recorded sales approximate outcomes under hypothetical selection. Gross fees exclude defaults, funding and operating costs.

Method references: [scikit-learn lagged-feature forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html), [global forecasting research](https://arxiv.org/abs/2008.00444), and [statsmodels Holt implementation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.Holt.html).
