# Fraud-adjusted portfolio findings

Recalculated all 30 final portfolios: the global merchant pool and five category groups, each at 100%, 99.5%, 99%, 98% and 95% expected-revenue retention. The date, 100-merchant size and 50/50 risk weights are unchanged. Both expected fees and risk inputs use transaction-level consumer-fraud-adjusted fees.

Across all metadata merchants, historical mean monthly fees decrease from A$4,802,322.77 to A$4,446,173.87, a 7.42% discount. This is the requested fee proxy, not measured fraud losses or net profit.

## Global results

| Revenue floor | Adjusted monthly fees (AUD) | Weekly CV | Customer HHI | Risk score | Replaced vs original | Gap | Certified |
|---|---:|---:|---:|---:|---:|---:|---|
| 100.0% | 1,551,710.02 | 3.138% | 0.00004583 | 100.000 | 1 | 0.000% | True |
| 99.5% | 1,544,100.83 | 3.102% | 0.00004548 | 99.061 | 2 | 0.043% | True |
| 99.0% | 1,536,392.77 | 3.088% | 0.00004527 | 98.601 | 2 | 0.048% | True |
| 98.0% | 1,520,697.99 | 3.032% | 0.00004537 | 97.817 | 3 | 0.096% | True |
| 95.0% | 1,474,508.54 | 2.934% | 0.00004530 | 96.168 | 5 | 0.712% | False |

## Validation and limitations

- All 13,440,766 transactions across 579 complete-history days were processed. 12,890,819 have merchant metadata; 549,947 remain outside fee-based selection. No take rate was invented.
- 80,308 orders retain supplied scores; 13,360,458 use the exact existing consumer OLS specification. There are no missing completed scores. The reconstructed temporal and random-split benchmark metrics match the teammate notebook to printed precision.
- 12,951,733 orders (96.36%) receive the model's low-support flag, meaning at least one activity predictor lies outside the observed labelled range. This is not a count of confirmed fraud or a prediction-error estimate. Good validation on observed scores does not establish calibration on these orders.
- Only 1 order(s) use a clipped missing-score prediction. Supplied scores are never clipped or overwritten.
- 16 of 30 portfolios meet the 0.1% optimization-gap criterion; 14 are best-found feasible portfolios with explicit bounds. No claim of exact global optimality is made for them.
- Old and new selections are also recomputed on the SAME adjusted fee/CV/HHI basis in `old_and_new_on_adjusted_basis.csv`. A former gross-fee portfolio may not satisfy the new adjusted-revenue floor.
- Covariance is re-estimated and customer exposures are rebuilt after discounting. Retention is relative to each scope's adjusted-revenue-only baseline, not its old gross-fee baseline. Lower historical score does not establish future downside protection.
- No new forecasting model, merchant-fraud multiplier, net-loss assumption or untouched backtest was introduced. The full original result files remain preserved under `handoff/final/`.

Use `final/` in this folder for the new files. They include explicit adjusted-fee aliases and `fee_basis`; legacy gross-fee field names remain compatibility aliases only in this directory. The old offline dashboard bundle has not been silently relabelled or overwritten.

The category entry point (`python code/10_category_ranking.py --fraud-adjusted`) reverified all 25 completed category portfolios and regenerated their exports. All 241 regression tests passed afterwards. See `provenance/verification.json`.
