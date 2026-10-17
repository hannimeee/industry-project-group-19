"""Freeze March 2021 fees and top 100, then evaluate every later observed month."""

import hashlib
import json
from pathlib import Path
from source_paths import source_directory

import numpy as np
import pandas as pd

from abs_momentum_comparison import require, write_table
from external_data_sources import archive_signature, merchants_and_mapping, save_json

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "tables/analysis/external_data_comparison"
OUT = ROOT / "tables/analysis/first_month_selection"
FIRST = pd.Timestamp("2021-03-01")


def freeze_selection(panel, merchants, top_n=100):
    """Only March outcomes and historically assumed-known metadata affect selection."""
    require(merchants.merchant_abn.is_unique, "Duplicate metadata")
    march = panel.loc[panel.month.eq(FIRST), ["merchant_abn", "sales"]]
    require(march.merchant_abn.is_unique, "Duplicate March merchant")
    base = merchants.loc[merchants.valid_take_rate,
                         ["merchant_abn", "name", "category", "take_rate"]].merge(
        march, on="merchant_abn", how="left", validate="one_to_one")
    require(base.sales.notna().all() and np.isfinite(base.sales).all()
            and base.sales.ge(0).all(), "Missing or invalid March coverage")
    base = base.rename(columns={"sales": "march_sales"})
    base["march_fees"] = base.march_sales * base.take_rate / 100
    base = base.sort_values(["march_fees", "merchant_abn"], ascending=[False, True]).reset_index(drop=True)
    base["march_rank"] = np.arange(1, len(base) + 1)
    base["selected"] = base.march_rank.le(top_n)
    require(base.selected.sum() == top_n and base.loc[base.selected, "march_fees"].gt(0).all(),
            "Insufficient positive-fee merchants")
    return base


def score_month(frozen, actual, forecast_scale=1.0, top_n=100):
    """Score a fixed pool; preserve zero outcomes and deterministic numeric-ABN ties."""
    f = frozen.merge(actual[["merchant_abn", "sales"]], on="merchant_abn", how="left", validate="one_to_one")
    require(f.sales.notna().all() and f.sales.ge(0).all(), "Incomplete or negative outcomes")
    f["actual_fees"] = f.sales * f.take_rate / 100
    f["predicted_fees"] = f.march_fees * forecast_scale
    f = f.sort_values(["actual_fees", "merchant_abn"], ascending=[False, True]).reset_index(drop=True)
    f["actual_rank"] = np.arange(1, len(f) + 1)
    f["hindsight_selected"] = f.actual_rank.le(top_n)
    chosen = f.loc[f.selected]
    obtained = float(chosen.actual_fees.sum())
    best = float(f.loc[f.hindsight_selected, "actual_fees"].sum())
    require(best >= obtained - 1e-6 and best > 0, "Invalid hindsight ceiling")
    error = f.predicted_fees - f.actual_fees
    metrics = dict(selected_actual_fees=obtained, hindsight_best_fees=best,
                   capture_pct=100 * obtained / best, fee_gap=best - obtained,
                   overlap=int((f.selected & f.hindsight_selected).sum()),
                   selected_predicted_fees=float(chosen.predicted_fees.sum()),
                   fee_wape_pct=float(100 * error.abs().sum() / f.actual_fees.sum()),
                   fee_bias_pct=float(100 * error.sum() / f.actual_fees.sum()),
                   selected_fee_wape_pct=float(100 * (chosen.predicted_fees - chosen.actual_fees).abs().sum() / obtained),
                   fee_rank_spearman=float(f.march_fees.corr(f.actual_fees, method="spearman")))
    require(np.isfinite(list(metrics.values())).all(), "Nonfinite results")
    return f, metrics


def run():
    source = source_directory()
    audit = json.loads((PREPARED / "merchant_audit.json").read_text())
    require(archive_signature(source) == audit["archives"], "Source archives changed")
    merchants = pd.read_parquet(PREPARED / "merchants.parquet")
    reparsed, _ = merchants_and_mapping(source)
    pd.testing.assert_frame_equal(merchants, reparsed)
    panel = pd.read_parquet(PREPARED / "merchant_month.parquet")
    unknown = pd.read_parquet(PREPARED / "unknown_merchants_monthly.parquet")
    days = pd.read_parquet(PREPARED / "source_partition_audit.parquet")
    months = pd.date_range(FIRST, "2022-09-01", freq="MS")
    require(set(panel.month) == set(months) and not panel.duplicated(["merchant_abn", "month"]).any(),
            "Invalid merchant-month keys")
    require(panel.groupby("merchant_abn").size().eq(19).all()
            and set(panel.merchant_abn) == set(merchants.merchant_abn), "Incomplete monthly panel")
    require(np.isfinite(panel.sales).all() and panel.sales.ge(0).all(), "Invalid monthly sales")
    require(set(days.loc[days.day.ge(FIRST), "day"]) == set(pd.date_range(FIRST, "2022-09-30")),
            "Incomplete daily source coverage")
    combined = pd.concat([panel, unknown]).groupby("month")[["orders", "sales"]].sum()
    source_months = days.loc[days.day.ge(FIRST)].assign(month=lambda x: x.day.dt.to_period("M").dt.to_timestamp())
    source_months = source_months.groupby("month")[["orders", "sales"]].sum()
    pd.testing.assert_frame_equal(combined, source_months, check_dtype=False, rtol=1e-12, atol=1e-7)

    frozen = freeze_selection(panel, merchants)
    changed = panel.copy()
    changed.loc[changed.month.gt(FIRST), "sales"] *= 1000
    pd.testing.assert_frame_equal(frozen, freeze_selection(changed, merchants), check_exact=True)
    write_table(frozen, OUT, "march_frozen_forecasts")
    save_json(OUT / "protocol.json", dict(selection_date="2021-03-31", top_n=100,
        history="March 2021 only; February has only one observed day", update_forecasts=False,
        forecast="Repeat March monthly sales; multiply by take rate / 100",
        october="October 1-26; forecast multiplied by 26/31; selection unchanged",
        pool="All unique metadata merchants with valid take rate, including zero March sales",
        no_future_activity_filter=True, six_month_eligibility_not_applicable=True,
        retrospective=True, source_archives=audit["archives"],
        input_sha256={name: hashlib.sha256((PREPARED / name).read_bytes()).hexdigest()
                      for name in ["merchants.parquet", "merchant_month.parquet", "confirmation/october_panel.parquet"]}))

    october = pd.read_parquet(PREPARED / "confirmation/october_panel.parquet").rename(
        columns={"sales_26": "sales", "orders_26": "orders"})
    october_unknown = pd.read_parquet(PREPARED / "confirmation/unknown_october_merchants.parquet")
    october_days = pd.read_parquet(PREPARED / "confirmation/october_audit.parquet")
    require(set(october_days.day) == set(pd.date_range("2022-10-01", "2022-10-26")), "Incomplete October")
    require(october.orders.sum() + october_unknown.orders_26.sum() == october_days.orders.sum(), "October count mismatch")
    require(np.isclose(october.sales.sum() + october_unknown.sales_26.sum(), october_days.sales.sum(), rtol=1e-12),
            "October sales mismatch")
    outcomes = pd.concat([panel.loc[panel.month.gt(FIRST), ["merchant_abn", "month", "sales"]],
                          october[["merchant_abn", "month", "sales"]]], ignore_index=True)
    scores, records = [], []
    for month, actual in outcomes.groupby("month"):
        partial = month == pd.Timestamp("2022-10-01")
        f, metrics = score_month(frozen, actual, 26 / 31 if partial else 1)
        f["month"], f["partial_month"] = month, partial
        # Recompute selection fees directly from the panel, independently of scored rows.
        selected = actual.loc[actual.merchant_abn.isin(frozen.loc[frozen.selected, "merchant_abn"])]
        direct = selected.merge(merchants[["merchant_abn", "take_rate"]], on="merchant_abn", validate="one_to_one")
        require(np.isclose((direct.sales * direct.take_rate / 100).sum(), metrics["selected_actual_fees"], rtol=1e-12),
                "Independent selected-fee check failed")
        scores.append(f)
        records.append(dict(month=month, partial_month=partial, **metrics))
    predictions, monthly = pd.concat(scores, ignore_index=True), pd.DataFrame(records)
    full = monthly.loc[~monthly.partial_month]
    full_predictions = predictions.loc[~predictions.partial_month]
    totals = outcomes.loc[outcomes.month.lt("2022-10-01")].groupby("merchant_abn", as_index=False).sales.sum()
    cumulative, cumulative_metrics = score_month(frozen, totals, forecast_scale=18)
    summary = dict(pool_size=len(frozen), march_active_merchants=int(frozen.march_sales.gt(0).sum()),
                   full_evaluation_months=len(full), mean_monthly_capture_pct=float(full.capture_pct.mean()),
                   minimum_monthly_capture_pct=float(full.capture_pct.min()), maximum_monthly_capture_pct=float(full.capture_pct.max()),
                   mean_monthly_overlap=float(full.overlap.mean()), minimum_overlap=int(full.overlap.min()),
                   maximum_overlap=int(full.overlap.max()),
                   pooled_fee_wape_pct=float(100 * (full_predictions.predicted_fees - full_predictions.actual_fees).abs().sum()
                                             / full_predictions.actual_fees.sum()),
                   pooled_fee_bias_pct=float(100 * (full_predictions.predicted_fees - full_predictions.actual_fees).sum()
                                             / full_predictions.actual_fees.sum()),
                   cumulative_fixed_hindsight=cumulative_metrics)
    for frame, name in [(monthly, "monthly_metrics"), (predictions, "predictions"),
                         (cumulative, "cumulative_apr2021_sep2022")]:
        write_table(frame, OUT, name)
    save_json(OUT / "summary.json", summary)
    save_json(OUT / "acceptance_checks.json", dict(source_archives_unchanged=True, metadata_reparsed=True,
        all_monthly_counts_and_sales_reconciled=True, october_counts_and_sales_reconciled=True,
        complete_source_dates=True, unique_complete_merchant_months=True,
        future_perturbation_leaves_forecasts_and_selection_identical=True,
        exactly_100_unique_merchants_fixed_for_all_months=True, independent_fee_recomputation=True))
    lines = ["# Select once using March 2021, then hold through October 2022", "",
        "March is the first complete month; February contains only 28 February. Rank March gross fees "
        "(sales x take rate / 100), choose 100, and never update their membership or monthly forecasts. "
        "For October 1-26 only, scale the March forecast by 26/31. This scaling cannot change selection.", "",
        f"The fixed pool includes all {len(frozen):,} valid metadata merchants; {summary['march_active_merchants']:,} "
        "had positive March sales. Future inactivity is retained as zero; merchants with no March sales remain "
        "eligible for the hindsight comparison. The earlier six-month eligibility rule cannot apply with only one month. "
        "Unknown-metadata transactions are excluded. Categories and take rates are assumed stable and historically known.", "",
        f"Across 18 complete future months, mean capture of each month's hindsight maximum was "
        f"**{summary['mean_monthly_capture_pct']:.3f}%**, range **{summary['minimum_monthly_capture_pct']:.3f}-"
        f"{summary['maximum_monthly_capture_pct']:.3f}%**. Mean overlap was **{summary['mean_monthly_overlap']:.2f}/100**, "
        f"range **{summary['minimum_overlap']}-{summary['maximum_overlap']}**.", "",
        f"Repeating March fees nevertheless has pooled merchant-month WAPE **{summary['pooled_fee_wape_pct']:.2f}%** "
        f"and aggregate bias **{summary['pooled_fee_bias_pct']:+.2f}%**. WAPE is sum absolute fee error divided by sum actual fees. "
        "Capture and overlap measure ranking quality, not accuracy of predicted revenue amounts.", "",
        f"Over April 2021-September 2022 together, these same 100 earned **A${cumulative_metrics['selected_actual_fees']:,.2f}**, "
        f"versus **A${cumulative_metrics['hindsight_best_fees']:,.2f}** from the best single fixed set of 100 for that whole period "
        f"(**{cumulative_metrics['capture_pct']:.3f}%** captured; **{cumulative_metrics['overlap']}/100** overlap). "
        "This cumulative comparator differs from switching to a new hindsight best 100 every month.", "",
        "| Outcome month | Actual fees of March 100 | Hindsight best 100 fees | Captured | Overlap | Full-pool fee WAPE |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in monthly.itertuples():
        label = row.month.strftime("%Y-%m") + (" (1-26 only)" if row.partial_month else "")
        lines.append(f"| {label} | A${row.selected_actual_fees:,.2f} | A${row.hindsight_best_fees:,.2f} | "
                     f"{row.capture_pct:.3f}% | {row.overlap}/100 | {row.fee_wape_pct:.2f}% |")
    lines += ["", "Historical descriptive evidence on synthetic data, with already examined outcomes. "
        "No new model is fitted, no future sales influence the selection, and the result does not guarantee future stability. "
        "Gross fees exclude fraud/default losses and other costs. Checks cover original archive signatures, reparsed metadata, "
        "monthly source reconciliation, future-data perturbation and independently recalculated selected fees.", "",
        "Reproduce from the repository: `python code/scripts/first_month_selection.py` (uses the existing audited panels "
        "and installed experiment dependencies). Generated tables are CSV and uncompressed Parquet in "
        "`tables/analysis/first_month_selection/`."]
    (OUT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(monthly[["month", "capture_pct", "overlap", "fee_wape_pct"]].round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
