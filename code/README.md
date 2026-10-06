# Historical analysis and original ETL

**For final dashboard data and reusable portfolio selection, start at
[../handoff/README.md](../handoff/README.md).** You do not need to run these experiments to use the final data.

This directory retains the research record and its original paths/imports. Moving or rewriting those
files would break notebook links, experiment fingerprints and reproducibility. Earlier unsuccessful
experiments remain evidence; they are not dependencies of the portable optimizer.

| Entry | Purpose |
|---|---|
| `00_preliminary_analysis.ipynb`, `01_etl.ipynb` | Supplied-data inspection and the team's original ETL |
| `02_abs_momentum_comparison.ipynb`, `03_external_data_comparison.ipynb` | External-feature controlled comparisons |
| `04_cash_flow_variability.ipynb` through `06_cash_flow_risk_comparison.ipynb` | Risk definitions, history windows and downside comparisons |
| `07_risk_aware_selection.ipynb`, `08_diversification_frontier.ipynb` | Variability portfolios, revenue floors and hindsight diagnostics |
| `09_customer_concentration.ipynb` | Shared-customer exposure, concentration portfolios and stress scenarios |
| `joint_risk_portfolios.py` | Frozen original five-level 50/50 study; preserve its source for recorded hashes |
| `export_portfolio_handoff.py` | Repository-only adapter: verify original outputs and rebuild the portable handoff |
| `fraud_adjusted_portfolios.py` | Reconstruct the team's consumer fraud scores and rerun all 30 final portfolios using adjusted fees |
| `fraud_adjusted_validation.py` | Independently recompute adjusted portfolio metrics and produce the rerun findings |
| `10_category_ranking.py --fraud-adjusted` | Rerun or verify all 25 category portfolios on adjusted fees, exporting to `handoff/fraud_adjusted/final/` |
| `tests/` | Original experiment regression tests |

Generated historical datasets remain in ignored `tables/analysis/`. The handoff has a separate,
small set of inputs and no runtime dependency on these experiments or their cached outputs.
