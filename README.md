# mast30034-project-2
MAST30034 Project 2 (Group): Buy Now, Pay Later Merchant Ranking

## Final dashboard and reusable optimizer — start here

The final handoff is in **[handoff/](handoff/README.md)**. It is self-contained and can be copied to another computer.

- **Dashboard data:** [one merged Parquet table](handoff/final/portfolio_dashboard.parquet), or [CSV](handoff/final/portfolio_dashboard.csv): five portfolios, 100 merchants each.
- **Field meanings and aggregation:** [data dictionary](handoff/final/data_dictionary.csv). Portfolio metrics repeat on merchant rows: use FIRST per portfolio, never SUM.
- **Reusable entry point:** `python handoff/optimize.py --help`. Change revenue retention levels, portfolio size, weights or same-schema input data.
- **Immediate reproduction:** `python handoff/optimize.py export` verifies and re-exports the saved five portfolios without rerunning optimization.
- **Tests:** `python -m pytest -p no:capture code/tests handoff/tests -q`.

The score combines baseline-normalized weekly CV and shared-customer HHI at 50/50 weights.
It is an index with baseline 100, not a loss probability. Results retain their solver bounds and certification flags;
three portfolios remain uncertified at the requested 0.1% tolerance. See the handoff guide for assumptions and exact definitions.

## Historical research record

Original notebooks, helpers and paths are retained under [code/](code/README.md); they are not required by the portable handoff.
For current ETL implementation details, see [the team's ETL notebook](code/01_etl.ipynb).

The [ABS momentum comparison notebook](code/02_abs_momentum_comparison.ipynb) tests
whether one external spending feature improves the selection of 100 merchants.
See the [reproduction guide](code/ABS_COMPARISON.md) for inputs, execution, validation
and the frozen evaluation protocol. Generated data stay in the ignored `tables/` layers.

The [external features notebook](code/03_external_data_comparison.ipynb) tests CPI,
Google Trends and business exit rates for merchant return and revenue downside,
controlling for the existing category labels. Its [reproduction guide](code/EXTERNAL_DATA_COMPARISON.md)
documents staged preparation, 99 placebo trials, candidate freezing and the separate
October 1–26 check. Unavailable sources are reported explicitly.

The [forecast comparison guide](code/FORECAST_COMPARISON.md) documents a separate
retrospective comparison of repeat sales, the original regression, pooled monthly-lag
forecasting and damped-trend forecasting for expected fees from 100 selected merchants.

The [customer concentration notebook](code/09_customer_concentration.ipynb) compares
merchant-level concentration with portfolio exposure to shared customers. It reports
observed fee revenue, concentration, downside, hypothetical customer-loss scenarios
and separately labelled hindsight opportunities.

Run this study with `python code/customer_concentration_serial.py all`, one command
at a time. The serial entry point retains the scientific settings and solver budgets
while using one numerical worker and lower Windows process priority. The optimizer
retains its original single-threaded loop; surrounding BLAS reductions use the
original 20-thread setting so their rounding order and solver traces reproduce.
The May/June 2022 hindsight cases at the 95% fee floor retain the single-thread
setting used when those two original cases were completed during recovery.
The hard CPU and memory ceilings still apply. Individual stages are `prepare`, `select`, `evaluate`, `oracle`
and `report`; `verify-select` and `verify-oracle` independently reproduce the searches.
The original module can use parallel date workers, so the serial entry point is
recommended for a laptop. Outputs are saved under the ignored
`tables/analysis/customer_concentration/` directory.

The Windows serial launcher also enforces a 4 GiB committed-memory limit for the
worker and its descendants, with a 20% CPU-rate ceiling. Independent verification
saves each completed MILP step using an exact input fingerprint, so interrupted
runs can replay verified solver steps without changing the search budget. The
local recovery controller records memory usage and stops when system memory is
critically low. A resource stop preserves completed checkpoints and blocks another
automatic start until its cause has been reviewed.
