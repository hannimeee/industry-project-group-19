"""Retrospective, past-only comparison of merchant sales forecasting methods."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import warnings
from pathlib import Path
from source_paths import source_directory

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import Holt
from threadpoolctl import threadpool_limits

import external_data_models as mdl
import external_data_sources as src
from abs_momentum_comparison import require, write_table

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "tables/analysis/external_data_comparison"
OUT = ROOT / "tables/analysis/forecast_comparison"
ORIGINS = pd.date_range("2021-11-30", "2022-06-30", freq="ME")
LAGS = [f"sales_lag_{i}" for i in range(6)]  # lag 0 is the just-completed month
METHODS = ["repeat", "minimal", "pooled_lags", "damped_trend"]
TREE_PARAMS = dict(loss="poisson", max_iter=200, learning_rate=0.05,
                   max_leaf_nodes=7, min_samples_leaf=100, l2_regularization=1,
                   early_stopping=False, random_state=2026)


def digest(frame):
    h = hashlib.sha256()
    h.update(str(list(zip(frame.columns, frame.dtypes.astype(str)))).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return h.hexdigest()


def build_features(panel, merchants, through=ORIGINS[-1]):
    """Reuse audited eligibility/internal features and add six past monthly sales."""
    frames = []
    for origin in pd.date_range("2021-08-31", through, freq="ME"):
        past = panel.loc[panel.month.le(origin)]
        labels = merchants[["category"]].drop_duplicates().assign(origin=origin)
        f = mdl.features_at(past, merchants, labels, origin)
        wide = past.pivot(index="merchant_abn", columns="month", values="sales")
        months = pd.date_range(end=origin.to_period("M").start_time, periods=6, freq="MS")
        for col, month in zip(LAGS, reversed(months)):
            f[col] = wide[month].reindex(f.merchant_abn).to_numpy()
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def pooled_prediction(train, candidates):
    """Estimate a nonnegative conditional mean for the whole three-month target."""
    require(np.isfinite(train[LAGS]).all().all() and np.isfinite(candidates[LAGS]).all().all(),
            "Nonfinite lag input")
    model = HistGradientBoostingRegressor(**TREE_PARAMS)
    with threadpool_limits(limits=1):
        model.fit(train[LAGS].to_numpy() / 1000, train.target_sales.to_numpy() / 1000)
        prediction = model.predict(candidates[LAGS].to_numpy() / 1000) * 1000
    require(np.isfinite(prediction).all() and (prediction >= 0).all(), "Invalid pooled forecast")
    return prediction, dict(iterations=int(model.n_iter_), training_rows=len(train),
                           training_origins=sorted(train.origin.dt.strftime("%Y-%m-%d").unique().tolist()))


def damped_one(values):
    """Fit nonseasonal additive damped Holt using only one merchant's past sales."""
    values = np.asarray(values, float)
    require(len(values) >= 6 and np.isfinite(values).all() and (values >= 0).all(),
            "Invalid damped-trend history")
    fallback = float(values[-3:].sum())
    if np.ptp(values) == 0:
        return dict(predicted_sales=fallback, status="constant_history", clipped_months=0,
                    history_months=len(values), warnings="")
    # Scaling by a past-only mean improves conditioning without changing units.
    scale = float(values.mean())
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = Holt(values / scale, damped_trend=True, initialization_method="estimated").fit(
            use_brute=False, method="SLSQP", minimize_kwargs={"options": {"maxiter": 1000}})
    forecast = np.asarray(fit.forecast(3)) * scale
    success = bool(fit.mle_retvals.success) and np.isfinite(forecast).all()
    return dict(predicted_sales=float(np.maximum(forecast, 0).sum()) if success else fallback,
                status="converged" if success else "fallback_repeat_nonconvergence",
                clipped_months=int((forecast < 0).sum()) if success else 0,
                history_months=len(values), warnings="; ".join(str(w.message) for w in caught),
                smoothing_level=float(fit.params["smoothing_level"]),
                smoothing_trend=float(fit.params["smoothing_trend"]),
                damping_trend=float(fit.params["damping_trend"]))


def damped_predictions(panel, candidates, origin, jobs):
    past = panel.loc[panel.month.le(origin)]
    months = pd.date_range("2021-03-01", origin, freq="MS")
    wide = past.pivot(index="merchant_abn", columns="month", values="sales").reindex(
        index=candidates.merchant_abn, columns=months)
    require(wide.notna().all().all(), "Incomplete damped-trend history")
    results = Parallel(n_jobs=jobs, batch_size=64)(delayed(damped_one)(v) for v in wide.to_numpy())
    details = pd.DataFrame(results).assign(merchant_abn=candidates.merchant_abn.to_numpy(), origin=origin)
    return details.predicted_sales.to_numpy(), details


def make_predictions(candidates, forecasts):
    frames = []
    for method, sales in forecasts.items():
        f = candidates[["merchant_abn", "name", "category", "origin", "take_rate"]].copy()
        f["method"], f["predicted_sales"] = method, sales
        f["predicted_fees"] = f.predicted_sales * f.take_rate / 100
        require(np.isfinite(f.predicted_fees).all() and f.predicted_fees.ge(0).all(), "Invalid fees")
        f["rank"] = mdl.ranks(f.merchant_abn, f.predicted_fees)
        f["selected"] = f["rank"].le(100)
        require(f.loc[f.selected, "merchant_abn"].nunique() == 100, "Selection must contain 100")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def attach_outcomes(predictions, panel):
    """Score only after predictions and selections have been saved."""
    scored = predictions.copy()
    for origin, group in scored.groupby("origin"):
        ids = group.merchant_abn.drop_duplicates()
        outcomes = pd.Series(mdl.target_at(panel, ids, origin, "3m"), index=ids)
        scored.loc[group.index, "actual_sales"] = group.merchant_abn.map(outcomes).to_numpy()
    scored["actual_fees"] = scored.actual_sales * scored.take_rate / 100
    return scored


def summarize(scored):
    metrics, changes = [], []
    for origin, group in scored.groupby("origin"):
        base = group.loc[group.method.eq("repeat")].set_index("merchant_abn")
        base_ids = set(base.index[base.selected])
        base_fees = float(base.loc[base.selected, "actual_fees"].sum())
        old = group.loc[group.method.eq("minimal") & group.selected, "actual_fees"].sum()
        for method, f in group.groupby("method", sort=False):
            chosen = f.loc[f.selected]
            fees = float(chosen.actual_fees.sum())
            current = set(chosen.merchant_abn)
            change = f.loc[f.merchant_abn.isin(current ^ base_ids),
                           ["merchant_abn", "category", "actual_fees"]].copy()
            change["direction"] = np.where(change.merchant_abn.isin(current), "entrant", "removal")
            change["fee_contribution"] = np.where(change.direction.eq("entrant"),
                                                   change.actual_fees, -change.actual_fees)
            change["origin"], change["method"] = origin, method
            require(np.isclose(change.fee_contribution.sum(), fees - base_fees, atol=1e-6),
                    "Changed selections do not reconcile")
            changes.append(change)
            metrics.append(dict(origin=origin, method=method, selected_fees=fees,
                                delta_vs_repeat=fees - base_fees, uplift_vs_repeat_pct=(fees / base_fees - 1) * 100,
                                uplift_vs_minimal_pct=(fees / old - 1) * 100,
                                overlap_with_repeat=len(current & base_ids),
                                selected_categories=chosen.category.nunique(),
                                **mdl.forecast_metrics(f.actual_fees, f.predicted_fees)))
    return pd.DataFrame(metrics), pd.concat(changes, ignore_index=True)


def verify_reproduction(predictions, existing):
    old = existing.loc[existing.policy.eq("return") & existing.arm.isin(["minimal", "repeat"])].rename(
        columns={"arm": "method"})
    cols = ["origin", "method", "merchant_abn", "predicted_fees", "rank", "selected"]
    pair = predictions.loc[predictions.method.isin(["minimal", "repeat"]), cols].merge(
        old[cols], on=cols[:3], how="outer", validate="one_to_one", suffixes=("_new", "_old"), indicator=True)
    require(pair._merge.eq("both").all(), "Benchmark eligibility changed")
    require(np.allclose(pair.predicted_fees_new, pair.predicted_fees_old, rtol=1e-10, atol=1e-7),
            "Historical forecast mismatch")
    require(pair["rank_new"].eq(pair["rank_old"]).all() and pair.selected_new.eq(pair.selected_old).all(),
            "Historical ranking mismatch")
    return dict(paired_rows=len(pair), max_fee_difference=float(
        np.abs(pair.predicted_fees_new - pair.predicted_fees_old).max()))


def write_report(out, metrics, changes, details):
    rows = []
    for method, f in metrics.groupby("method", sort=False):
        contributions = changes.loc[changes.method.eq(method)]
        delta = f.delta_vs_repeat.sum()
        merchant = contributions.groupby("merchant_abn").fee_contribution.sum()
        category = contributions.groupby("category").fee_contribution.sum()
        rows.append(dict(method=method, mean_selected_fees=float(f.selected_fees.mean()),
                         mean_delta_dollars=float(f.delta_vs_repeat.mean()),
                         mean_uplift_pct=float(f.uplift_vs_repeat_pct.mean()),
                         positive_origins=int(f.uplift_vs_repeat_pct.gt(0).sum()),
                         mean_wape_pct=float(f.wape_pct.mean()),
                         mean_deviance=float(f.deviance.mean()),
                         mean_overlap=float(f.overlap_with_repeat.mean()),
                         delta_without_largest_merchant=float(delta - max(0, merchant.max())) if len(merchant) else 0,
                         delta_without_largest_category=float(delta - max(0, category.max())) if len(category) else 0))
    summary = pd.DataFrame(rows)
    summary["pilot_screen_pass"] = summary.mean_uplift_pct.ge(1) & summary.positive_origins.ge(6)
    summary["fragile"] = summary.mean_uplift_pct.gt(0) & (
        summary.delta_without_largest_merchant.le(0) | summary.delta_without_largest_category.le(0))
    write_table(summary, out, "summary")
    schedules = metrics.assign(schedule=(metrics.origin.dt.year * 12 + metrics.origin.dt.month) % 3)
    schedules = schedules.groupby(["schedule", "method"], as_index=False).agg(
        origins=("origin", "size"), mean_uplift_pct=("uplift_vs_repeat_pct", "mean"),
        mean_delta_dollars=("delta_vs_repeat", "mean"))
    write_table(schedules, out, "nonoverlapping_schedules")
    write_table(changes.groupby(["method", "category"], as_index=False).fee_contribution.sum(), out, "category_contributions")
    final = metrics.loc[metrics.origin.eq(ORIGINS[-1])]
    lines = ["# Merchant forecasting comparison", "",
             "Eight retrospective three-month tests; selection dates November 2021 through June 2022. "
             "Every evaluation period has already been examined in earlier research. No untouched test remains.", "",
             "Primary comparison: actual gross fees from 100 merchants selected by predicted fees. "
             "Means below weight each origin equally. Overlapping windows must not be summed as distinct revenue.", "",
             "| Method | Mean selected fees | Mean uplift vs repeat | Mean dollar difference | Positive origins | Mean fee WAPE |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | A${r.mean_selected_fees:,.2f} | {r.mean_uplift_pct:+.3f}% | "
                     f"A${r.mean_delta_dollars:+,.2f} | {r.positive_origins}/8 | {r.mean_wape_pct:.2f}% |")
    lines += ["", "## Each historical selection date", "",
              "| Selection date | Minimal uplift | Pooled-lags uplift | Damped-trend uplift |",
              "|---|---:|---:|---:|"]
    pivot = metrics.pivot(index="origin", columns="method", values="uplift_vs_repeat_pct")
    for date, r in pivot.iterrows():
        lines.append(f"| {date:%Y-%m-%d} | {r['minimal']:+.3f}% | {r['pooled_lags']:+.3f}% | {r['damped_trend']:+.3f}% |")
    lines += ["", "## July-September 2022 outcomes", "",
              "| Method | Selected fees | Uplift vs repeat | Fee WAPE | Shared selections with repeat |",
              "|---|---:|---:|---:|---:|"]
    for r in final.itertuples():
        lines.append(f"| {r.method} | A${r.selected_fees:,.2f} | {r.uplift_vs_repeat_pct:+.3f}% | "
                     f"{r.wape_pct:.2f}% | {r.overlap_with_repeat}/100 |")
    lines += ["", "## Interpretation", "",
              "The fixed pilot screen requires mean uplift of at least 1% and positive uplift in at least 6/8 origins. "
              "It is a descriptive screen for further testing, not statistical significance or independent confirmation.", ""]
    for r in summary.loc[summary.method.isin(["pooled_lags", "damped_trend"])].itertuples():
        lines.append(f"- {r.method}: {'passes' if r.pilot_screen_pass else 'fails'} pilot screen; "
                     f"positive aggregate selection gain fragile to largest merchant/category removal: {r.fragile}.")
    lines += ["", "## Methods and limitations", "",
              "- repeat: next-three-month sales equal the previous three-month sales.",
              "- minimal: unchanged two-input Tweedie regression, including its original fixed category coverage split. "
              "No external values enter this model; its split is preserved for exact reproduction.",
              "- pooled_lags: one shallow Poisson gradient-boosting model across all eligible training rows; "
              "six individual monthly sales inputs, target is total next-three-month sales. Fixed settings, no tuning, "
              "no category, calendar or external predictors. This changes both model family and inputs, so does not isolate either change.",
              "- damped_trend: independent nonseasonal additive damped Holt per merchant, using all complete months available "
              "at its selection date. Sum three monthly forecasts, clipping any negative monthly sales to zero. "
              "Estimated initial states and smoothing parameters; unsuccessful optimization falls back to repeat sales and is reported.",
              "- Pooled models use feature origins from August 2021 whose three-month targets finish by each selection date. "
              "November has one matured origin; June has eight. Holt uses 9-16 observations per merchant. "
              "No random splitting or future features; future inactivity remains a zero outcome.",
              "- Same eligibility, take rates, forecast horizon and numeric ABN tie-break across all methods. "
              "Select 100 and hold them for three months. Rates are assumed stable and historically known.",
              "- Gross fees are sales times take rate / 100, excluding defaults, funding and operating costs. "
              "Historical synthetic sales stand in for hypothetical selection outcomes. This experiment does not fit portfolio risk.",
              "- Fee WAPE is sum absolute forecast error divided by sum actual fees across the eligible pool. "
              "Tweedie deviance uses the previous experiment's 1e-12 floor on zero predicted fees in thousands; "
              "Holt's clipped zero forecasts can incur very large deviance when actual sales are positive. "
              "The floor only affects this diagnostic, never ranks or fee predictions.",
              "- The many merchants share few economic periods. Non-overlapping schedules and selection-change contributions "
              "are exported; neither is an independent new validation dataset.",
              f"- Holt status counts: {details.status.value_counts().to_dict()}; "
              f"negative monthly forecasts clipped: {int(details.clipped_months.sum())}.",
              "", "See protocol.json, acceptance_checks.json, predictions.parquet, fold_metrics.csv and selection_changes.csv "
              "for reproducibility and diagnostics."]
    (out / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(source_dir, prepared, out, jobs):
    require((prepared / "merchant_audit.json").exists(), "Prepare the audited merchant panel first")
    merchants, _ = src.merchants_and_mapping(source_dir)
    panel = src.aggregate_development(source_dir, merchants, prepared)
    old_signature = json.loads((prepared / "backtest_inputs.json").read_text(encoding="utf-8"))
    require(digest(panel) == old_signature["panel"], "Audited panel fingerprint changed")
    audit = json.loads((prepared / "merchant_audit.json").read_text(encoding="utf-8"))
    require(panel.orders.sum() + audit["unknown_orders"] == audit["complete_month_orders"], "Order reconciliation")
    require(np.isclose(panel.sales.sum() + audit["unknown_sales"], audit["complete_month_sales"], rtol=1e-12),
            "Sales reconciliation")
    require(merchants.merchant_abn.is_unique and not panel.duplicated(["merchant_abn", "month"]).any(), "Duplicate keys")
    features = build_features(panel, merchants)
    protocol = dict(name="Merchant forecasting comparison v1", retrospective=True, unused_holdout=False,
                    origins=ORIGINS.strftime("%Y-%m-%d").tolist(), target="next three complete months sales",
                    methods=METHODS, tree_params=TREE_PARAMS, tree_features=LAGS, target_and_input_scale=1000,
                    tree_training="expanding, matured merchant-origin rows, equal weights; no hyperparameter search",
                    damped=dict(seasonal=False, additive=True, estimated_initialization=True,
                                fit="SLSQP", maxiter=1000, use_brute=False, scaling="past mean",
                                negative_forecast="clip each month to zero", nonconvergence="repeat sales"),
                    rank="predicted sales * take rate / 100 descending, then numeric ABN ascending",
                    top_n=100, primary_metric="equal-origin mean selected-fee uplift vs repeat",
                    pilot_screen=dict(minimum_mean_uplift_pct=1, minimum_positive_origins=6),
                    inputs=dict(panel=digest(panel), merchants=digest(merchants), features=digest(features)),
                    model_versions={p: importlib.metadata.version(p) for p in
                                    ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pyarrow"]})
    path = out / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == protocol, "Changed protocol: use a new output directory")
    src.save_json(path, protocol)
    src.save_json(out / "run_metadata.json", dict(started_at=pd.Timestamp.now(tz="UTC"), python=sys.version,
                                                 source_archives=audit["archives"]))
    write_table(features, out, "features")
    write_table(features.groupby(["origin", "eligibility_reason"], as_index=False).size(), out, "eligibility")
    frames, diagnostics, fits = [], [], []
    for origin in ORIGINS:
        train = mdl.training_at(panel, features, origin, "3m")
        test = features.loc[features.origin.eq(origin) & features.eligible].sort_values("merchant_abn").reset_index(drop=True)
        require(train.origin.max() + pd.offsets.MonthEnd(3) <= origin, "Immature training outcome")
        pooled, info = pooled_prediction(train, test)
        damped, detail = damped_predictions(panel, test, origin, jobs)
        forecasts = dict(repeat=test.recent_three_month_sales.to_numpy(),
                         minimal=mdl.minimal_predictions(train, test), pooled_lags=pooled, damped_trend=damped)
        prediction = make_predictions(test, forecasts)
        write_table(prediction, out / "unscored", origin.strftime("%Y-%m-%d"))
        frames.append(prediction)
        diagnostics.append(detail)
        fits.append(dict(origin=origin, **info, candidates=len(test), history_months=detail.history_months.iloc[0]))
        print(f"Forecasts saved for {origin.date()}: {len(test):,} eligible, {len(train):,} training rows", flush=True)
    predictions, details = pd.concat(frames, ignore_index=True), pd.concat(diagnostics, ignore_index=True)
    src.save_json(out / "forecast_freeze.json", dict(saved_at=pd.Timestamp.now(tz="UTC"),
                                                    predictions_digest=digest(predictions)))
    existing = pd.read_parquet(prepared / "development/predictions.parquet")
    checks = dict(benchmark_reproduction=verify_reproduction(predictions, existing),
                  source_and_panel_reconciled=True, complete_windows=True, matured_training=True)
    # A real-data perturbation refits the final pooled model and compares its full ranking.
    cutoff = ORIGINS[-1]
    changed = panel.copy()
    changed.loc[changed.month.gt(cutoff), ["sales", "sales_26"]] *= 1000
    changed_features = build_features(changed, merchants)
    pd.testing.assert_frame_equal(features, changed_features, check_exact=True)
    train = mdl.training_at(panel, features, cutoff, "3m")
    altered_train = mdl.training_at(changed, changed_features, cutoff, "3m")
    pd.testing.assert_frame_equal(train, altered_train, check_exact=True)
    test = features.loc[features.origin.eq(cutoff) & features.eligible].sort_values("merchant_abn").reset_index(drop=True)
    again, _ = pooled_prediction(altered_train, test)
    saved = predictions.loc[predictions.origin.eq(cutoff) & predictions.method.eq("pooled_lags")]
    require(np.array_equal(again, saved.predicted_sales.to_numpy()), "Pooled forecasts changed after future perturbation")
    require(np.array_equal(mdl.ranks(test.merchant_abn, again * test.take_rate / 100), saved["rank"]), "Ranks changed")
    # Holt consumes precisely the same past panel; check a deterministic sample's refit too.
    ids = test.iloc[np.linspace(0, len(test) - 1, 20, dtype=int)].reset_index(drop=True)
    repeated_holt, _ = damped_predictions(changed, ids, cutoff, jobs=1)
    saved_holt = predictions.loc[predictions.origin.eq(cutoff) & predictions.method.eq("damped_trend")].set_index("merchant_abn")
    require(np.array_equal(repeated_holt, saved_holt.loc[ids.merchant_abn, "predicted_sales"].to_numpy()), "Holt forecasts changed")
    checks.update(real_future_perturbation_passed=True, exact_pooled_rerun=True, exact_holt_sample_rerun=20)
    scored = attach_outcomes(predictions, panel)
    metrics, changes = summarize(scored)
    for (origin, method), f in scored.loc[scored.selected].groupby(["origin", "method"]):
        months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        raw = panel.loc[panel.month.isin(months) & panel.merchant_abn.isin(f.merchant_abn)]
        independent = raw.merge(merchants[["merchant_abn", "take_rate"]], on="merchant_abn", validate="many_to_one")
        require(np.isclose((independent.sales * independent.take_rate / 100).sum(), f.actual_fees.sum(), rtol=1e-12),
                "Independent realized fee check failed")
    checks.update(selections_checked=len(metrics), each_selection_unique_100=True,
                  independent_realized_fees=True, finite_nonnegative_predictions=True,
                  holt_status_counts=details.status.value_counts().to_dict(),
                  clipped_holt_months=int(details.clipped_months.sum()))
    write_table(scored, out, "predictions")
    write_table(metrics, out, "fold_metrics")
    write_table(changes, out, "selection_changes")
    write_table(details, out, "damped_fit_details")
    write_table(pd.DataFrame(fits), out, "pooled_fit_details")
    write_table(scored.loc[scored.origin.eq(cutoff) & scored.selected], out, "final_selected_merchants")
    src.save_json(out / "acceptance_checks.json", checks)
    summary = write_report(out, metrics, changes, details)
    print(summary.to_string(index=False), flush=True)
    print(f"Report: {out / 'findings.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=source_directory())
    parser.add_argument("--prepared-dir", type=Path, default=PREPARED)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    run(args.source_dir, args.prepared_dir, args.output_dir, args.jobs)
