# Final portfolio handoff / 最终交接入口

**Dashboard 直接读取 `final/portfolio_dashboard.parquet` 或同名 CSV。**
两者内容相同：5 个档位 × 100 家商户 = 500 行。组合指标在每家商户行中重复，不能相加。
这个目录可以整体复制到其他电脑；不依赖原项目的 `code/` 或 `tables/`。

`VALIDATION.json` records the completed 230-test validation and final-data checks.
`BUILD_MANIFEST.json` lists checksums of the shipped files; generated `runs/` and caches are excluded.

## What to give the dashboard developer

- `final/portfolio_dashboard.parquet` — the final merged table, uncompressed Parquet.
- `final/portfolio_dashboard.csv` — equivalent CSV, for tools without Parquet support.
- `final/data_dictionary.csv` — every column's meaning, unit and aggregation rule.
- `final/manifest.json` — row counts, primary key and checksums.

Grain: **one selected merchant per portfolio**. Primary key: `portfolio_id, merchant_abn`.
The five `retention` values are **1, .995, .99, .98, .95**. These are minimum expected-fee
retention fractions; actual expected retention may be slightly higher. Every portfolio has 100 merchants.
The snapshot is **30 September 2022**, using 26 complete weeks from **28 March–25 September 2022**.
Expected fees are the mean monthly gross fees from the 19 complete months, March 2021–September 2022.
HHI is small (around 0.000047 for the baseline): display enough decimals or scientific notation,
not two decimal places that would misleadingly show zero. CV can be formatted as a percentage.

### Dashboard integration

```python
import pandas as pd

data = pd.read_parquet("final/portfolio_dashboard.parquet")
# CSV alternative: pd.read_csv(..., dtype={"merchant_abn": "string"})
chosen = data.loc[data["retention"].eq(0.98)]  # current file has one snapshot and one weight
overall = chosen.iloc[0]                    # FIRST, never SUM portfolio_* fields
merchants = chosen.sort_values("rank_by_expected_fees")
overview_for_all_five = data.drop_duplicates("portfolio_id")
```

If you combine different snapshots, weights or portfolio sizes later, filter **portfolio_id** instead.
Read merchant ABNs as identifiers, not amounts. All currency amounts in this supplied snapshot are AUD.
`merchant_expected_monthly_gross_fees` may be summed within one portfolio to reproduce its expected fees.
Never sum the repeated `portfolio_expected_monthly_gross_fees` or any other `portfolio_*` field.
Do not add up merchants across all five portfolios: the same merchant can occur in multiple portfolios.

There is **no individual merchant risk score**. Rows are ordered by expected fees within the selected
portfolio, not by a universal independent merchant ranking. Selection accounts for relationships
between merchants: covariance and shared customers.

## Risk definitions and interpretation

For eligible merchant i, let mu_i be its historical mean monthly gross fees. Weekly fees use the
same take rate as expected fees. Fees are transaction value × take-rate percentage / 100.
They exclude fraud losses, unpaid principal, funding/operating costs and settlement timing.

For a binary selection z of exactly K merchants:

```text
expected fees = sum(mu_i * z_i)
weekly CV = sqrt(z' Sigma z) / sum(historical_mean_weekly_fees_i * z_i)
customer exposure A_ci = mu_i * historical_customer_fees_ci / historical_merchant_fees_i
HHI = sum_c (sum_i A_ci*z_i / sum_i mu_i*z_i)^2
score = 100 * [weight * CV / baseline_CV + (1-weight) * HHI / baseline_HHI]
```

The baseline is the K eligible merchants with highest expected fees; ties use ascending numeric ABN.
The optimizer minimizes the score subject to exactly K merchants and expected fees at least
`retention × baseline expected fees`. Every eligible merchant remains a candidate; there is no arbitrary shortlist.

Sigma is `(1-alpha)*sample_covariance + alpha*diag(sample_variances)`, with sample denominator T-1.
Alpha is Ledoit–Wolf shrinkage fitted on standardized nonconstant eligible weekly histories.
This preserves individual sample variances. Shared customers are combined before computing HHI.
The implementation uses weekly factors and a sparse customer–merchant matrix, not a customer×customer matrix.

**HHI is naturally in [0,1]. CV has no inherent upper bound of 1. The mixed score is an index with
baseline 100, not a bounded probability.** No min-max rescaling or fake individual risk score is added.
Fifty-fifty weights express a preference; they were not learned as an empirically optimal weight.
Smaller estimated risk does not demonstrate future downside protection.

## Existing results: export immediately, without re-optimizing

Tested with **Python 3.12**. From this directory:

```text
python -m pip install -r requirements.txt
python optimize.py export
```

This verifies the supplied input snapshot, independently recomputes the five reference portfolios,
and rewrites the merged dashboard table. It does **not** run the expensive selection searches.
The underlying reference selections and original solver uncertainty are preserved.
`reference/provenance.json` records source hashes and the early stop of the original 95% search.

The original 100% and 99.5% portfolios meet the 0.1% optimization-gap criterion. The 99%, 98% and
95% portfolios are explicitly **best found, not certified to that tolerance**. Their gap upper bounds
are approximately 0.115%, 0.159% and 0.912%. The 95% search stopped after prolonged stagnation,
before exhausting its second-tier maximum. No future outcomes were used to choose these portfolios.

## Change the revenue floor, count or preference

```text
python optimize.py optimize --input input_snapshot --output runs/custom --retentions 100 99.5 97.5 95 --count 100 --weight 0.5
```

**CLI `--retentions` takes percentages** (97.5 means 97.5%). Output `retention` is a fraction (.975).
You are not restricted to the original four percentages. `--count` sets portfolio size; `--weight`
sets the CV weight in [0,1], with HHI weight 1-weight. Changing K recomputes the revenue-only baseline.
Each new run writes the same merged dashboard schema in its own output folder.

Optional historical seed portfolios accelerate the search but are rescored against current inputs:

```text
python optimize.py optimize --input input_snapshot --output runs/seeded --retentions 99 97.5 --seed-portfolios reference/selections.parquet
```

Only compatible-size seeds are used. Future-dated seeds are rejected. Seeds do not import old objective bounds.

### Algorithm, limits and truthful optimality

The search combines exact one-for-one swap improvements, tangent-plane lower bounds and MILP
branch-and-bound. CV and HHI have different denominators; exact binary–continuous normalization
keeps them separate. An improved portfolio is validated against the actual score, not just its relaxation.
Completed cases and interrupted incumbents are fingerprinted and rechecked before reuse.

Defaults are deliberately bounded for a laptop: one numerical worker, one BLAS thread, 30 iterations,
200 MILP nodes per call, 30 seconds per solver call, a 300-second soft search budget per portfolio,
and stop after eight consecutive rounds without a meaningful incumbent/bound improvement.
The budget excludes input loading and initial component-bound calculations. A running swap pass or
solver call must finish; this is not a hard real-time latency guarantee. On Windows there is also a
4 GiB committed-memory cap, 20% CPU-rate cap and below-normal priority. Other systems retain the
serial/thread limits but do not have the Windows native memory/CPU caps.

To invest more computation explicitly:

```text
python optimize.py optimize --input input_snapshot --output runs/deeper --retentions 97.5 --iterations 100 --nodes 2000 --time-limit 60 --max-seconds 3600 --patience 100 --gap 0.001 --bound-iterations 40
```

`--gap .001` requests a 0.1% relative objective gap. The alternative absolute tolerance is 0.0001 score
points (one millionth of the baseline objective). `portfolio_certified` indicates whether either
tolerance was met; otherwise the feasible portfolio and valid lower bound are still returned and
explicitly labelled uncertified. Finite budgets may find different approximate solutions on different
machines; use the frozen reference table to reproduce the exact delivered five lists.

To resume, repeat **the same optimize command** with `--resume`. Inputs, settings, code, dependencies
and seed hashes must match. Completed cases are reused; interrupted cases reuse the saved incumbent
and bound, but rebuild cuts with a fresh per-case search budget. Progress is saved after every solver
step in one atomic checkpoint plus an append-only progress log. Use a new output folder for changed
settings; nothing silently replaces a different run. This is a batch optimizer, not an instantaneous
dashboard-slider backend. Precompute desired levels and serve their saved tables.

## Reuse with another dataset

No merchant name, ABN, calendar cutoff, merchant count, currency amount or archived analysis path is
hardcoded in the portable optimizer. It consumes **already cleaned aggregate inputs**, not the original
archive/consumer-enrichment pipeline. Prepare these same-schema tables:

| Input | Required columns | Grain |
|---|---|---|
| merchants | merchant_abn, name, expected_monthly_fees, eligible | one row per metadata merchant |
| weekly_fees | merchant_abn, week_end, gross_fees | one row per merchant per complete week, including genuine zeros |
| customer_fees | consumer_id, merchant_abn, gross_fees | one row per customer–merchant pair, aggregated over exactly the declared risk window |

Category is optional display metadata. IDs must be nonnegative integers or digit strings; floating-point
IDs, duplicate keys, unknown merchants, missing weeks, negative/nonfinite fees and inconsistent customer
totals are rejected. All three tables must use the same currency and historical take-rate treatment.
`consumer_id` must identify the same person across merchants; do not use a merchant-specific customer ID.
An eligible merchant requires positive expected fees and positive mean fees over the chosen risk window.
A zero-variability baseline makes CV normalization undefined and is rejected explicitly.

**Expected revenue and eligibility:** preserve the project's definition: mean monthly fees across all
complete historical months, valid metadata/take rates, and positive sales in both preceding three-month
windows. You may provide those audited features directly. Alternatively provide metadata with
`merchant_abn, name, take_rate` (percentage points) and a complete `merchant_abn, month, sales` panel
via `--monthly-sales`; the helper calculates exactly that mean and eligibility. Include explicit genuine
zero months for every metadata merchant. At least six complete months are required. Supplied take rates
must be valid in (0,100]. Invalid metadata must be resolved upstream rather than guessed.

Create `metadata.json` (adjust dates and currency for your data):

```json
{
  "as_of": "2022-09-30",
  "weeks": 26,
  "window_start": "2022-03-28",
  "window_end": "2022-09-25",
  "expected_return_start": "2021-03-01",
  "expected_return_end": "2022-09-30",
  "currency": "AUD",
  "complete_coverage": true
}
```

`complete_coverage` is an upstream audit declaration, not something inferred from absent transactions.
Missing source coverage is never converted to zero. Customer aggregates must cover the declared dates;
reconciliation to weekly totals catches many errors but cannot independently prove the dates of an
already aggregated customer table. Stable customer identities, categories and take rates remain assumptions.

```text
python optimize.py prepare --merchants new_data/merchants.parquet --weekly-fees new_data/weekly_fees.parquet --customer-fees new_data/customer_fees.parquet --metadata new_data/metadata.json --output new_snapshot
python optimize.py optimize --input new_snapshot --output runs/new_dataset --retentions 100 99 97.5 95
```

CSV works for any table argument too. To derive features from the monthly panel, add
`--monthly-sales new_data/merchant_month.parquet` to `prepare`. Weekly input can contain extra history;
`prepare` retains only the declared complete weeks. It does not trim an already aggregated customer
table: prepare that customer window upstream. `prepare` creates checksummed, uncompressed Parquet
inputs and a `snapshot.json`. New snapshots require a new/empty directory.

## Files and tests

```text
handoff/
  README.md                   # this guide
  optimize.py                 # one CLI entry point
  requirements.txt            # pinned CPU-only dependencies
  portfolio_optimizer/        # pure mathematics, input validation, export, CLI
  final/                      # ONLY these files are needed by the dashboard
  input_snapshot/             # complete portable inputs for current merchant pool
  reference/                  # frozen five selections, metrics, provenance
  tests/                      # independent arithmetic, brute-force and reuse tests
```

```text
python optimize.py --help
python -m pytest -p no:capture tests -q
```

Historical research code stays in the repository's `code/` directory for reproducibility; it is not
needed in this package. The repository-only adapter `code/export_portfolio_handoff.py` rebuilds this
handoff from the original audited outputs. Previous experiments and supplied data are not deleted.

The test command disables pytest's capture plugin to avoid the older `pyreadline` installation on
the original Windows machine; it also works in a clean environment.
