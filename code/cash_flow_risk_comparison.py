"""Distinguish weekly fee variability, linear trend and downside; fixed portfolios only."""

import importlib.metadata
import json

import numpy as np
import pandas as pd

import cash_flow_variability as cf
from cash_flow_lookback_comparison import WINDOWS, history_dates
from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from forecast_comparison import ORIGINS, digest

OUT = cf.OUT.parent / "cash_flow_risk_comparison"


def downside(values, reference):
    """Lower partial moment of order two: divide by ALL weeks, including no-shortfall weeks."""
    x, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    require(np.isfinite(x).all() and np.isfinite(reference).all(), "Nonfinite downside inputs")
    losses = np.maximum(reference - x, 0)
    return np.sqrt(np.mean(losses ** 2, axis=0))


def decompose(values):
    """OLS on a centered weekly index; descriptive decomposition and df-adjusted residual SD."""
    x = np.asarray(values, dtype=float)
    require(x.ndim == 2 and x.shape[0] >= 3 and np.isfinite(x).all(), "Need three finite weekly observations")
    n = len(x)
    time = np.arange(n, dtype=float) - (n - 1) / 2
    mean = x.mean(axis=0)
    centered = x - mean
    slope = time @ centered / (time @ time)
    fitted = mean + np.outer(time, slope)
    residual = x - fitted
    # Remove only numerical dust (not small real fluctuations) for perfectly linear histories.
    tolerance = 1e-12 * np.maximum(1, np.abs(x).max(axis=0))
    residual[np.abs(residual) < tolerance] = 0
    raw_variance = np.sum(centered ** 2, axis=0) / (n - 1)
    trend_variance = slope ** 2 * np.sum(time ** 2) / (n - 1)
    residual_variance = np.sum(residual ** 2, axis=0) / (n - 1)
    np.testing.assert_allclose(raw_variance, trend_variance + residual_variance, rtol=1e-10, atol=1e-7)
    fraction = np.divide(trend_variance, raw_variance, out=np.full_like(mean, np.nan), where=raw_variance > 0)
    stats = pd.DataFrame(dict(weekly_mean=mean, raw_sd=np.sqrt(raw_variance),
        trend_slope_per_week=slope, trend_variance=trend_variance,
        trend_variance_share=fraction, residual_descriptive_sd=np.sqrt(residual_variance),
        residual_sd=np.sqrt(residual_variance * (n - 1) / (n - 2)),
        downside_mean=downside(x, mean), downside_trend=downside(x, fitted),
        fitted_negative_weeks=(fitted < 0).sum(axis=0)))
    return stats, fitted, residual


def history_matrix(weekly, dates, ids):
    frame = weekly.loc[weekly.week_end.isin(dates)]
    require(not frame.duplicated(["merchant_abn", "week_end"]).any(), "Duplicate merchant-week")
    wide = frame.pivot(index="week_end", columns="merchant_abn", values="gross_fees").reindex(index=dates, columns=ids)
    require(wide.notna().all().all(), "Incomplete merchant-week history")
    return wide.to_numpy()


def estimate_at(weekly, base_features, selected, as_of, window):
    """Construct all measures with exactly the same history, eligibility and selected IDs."""
    dates = history_dates(weekly, as_of, window)
    require(dates is not None, "Unavailable lookback")
    base = base_features.sort_values("merchant_abn").reset_index(drop=True)
    ids = base.merchant_abn.to_numpy()
    x = history_matrix(weekly, dates, ids)
    stats, fitted, residual = decompose(x)
    columns = ["merchant_abn", "name", "category", "take_rate", "eligible", "eligibility_reason", "expected_monthly_fees"]
    features = pd.concat([base[columns], stats], axis=1)
    features["as_of"], features["window"] = pd.Timestamp(as_of), window
    features["history_start"], features["history_end"], features["history_weeks"] = dates[0]-pd.Timedelta(days=6), dates[-1], len(dates)
    chosen = selected.merchant_abn.to_numpy()
    positions = pd.Index(ids).get_indexer(chosen)
    require((positions >= 0).all() and len(set(chosen)) == len(chosen), "Invalid selected IDs")
    raw_model = cf.estimate_covariance(x, ids, base.eligible)
    # cf's sample denominator is n-1; rescaling gives residual covariance with n-2 df.
    residual_model = cf.estimate_covariance(residual * np.sqrt((len(x)-1)/(len(x)-2)), ids, base.eligible)
    total = x[:, positions].sum(axis=1)
    portfolio, portfolio_fit, _ = decompose(total[:, None])
    row = portfolio.iloc[0].to_dict()
    row.update(as_of=pd.Timestamp(as_of), window=window, history_start=features.history_start.iloc[0],
        history_end=dates[-1], history_weeks=len(dates), selected_merchants=len(chosen),
        selected_ids_digest=digest(selected[["merchant_abn"]].reset_index(drop=True)),
        expected_monthly_fees=float(selected.expected_monthly_fees.sum()),
        raw_shrunk_sd=cf.portfolio_volatility(raw_model, chosen, "shrunk"), raw_alpha=raw_model.alpha,
        residual_shrunk_sd=cf.portfolio_volatility(residual_model, chosen, "shrunk"), residual_alpha=residual_model.alpha)
    for model, metric in [(raw_model, "raw_sd"), (residual_model, "residual_sd")]:
        np.testing.assert_allclose(cf.portfolio_volatility(model, chosen, "sample"), row[metric], rtol=1e-10, atol=1e-7)
        for estimator in ["sample", "shrunk"]:
            sub = cf.covariance_submatrix(model, chosen, estimator).to_numpy()
            np.testing.assert_allclose(np.diag(sub), model.variances[positions], rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(sub.sum(), cf.portfolio_volatility(model, chosen, estimator)**2, rtol=1e-10, atol=1e-5)
            require(np.linalg.eigvalsh(sub).min() >= -1e-8*max(1, np.diag(sub).max()), "Non-PSD covariance")
    np.testing.assert_allclose(portfolio_fit[:, 0], fitted[:, positions].sum(axis=1), rtol=1e-10, atol=1e-7)
    history = pd.DataFrame(dict(as_of=as_of, window=window, phase="history", week_end=dates,
        gross_fees=total, reference_mean=row["weekly_mean"], reference_trend=portfolio_fit[:, 0]))
    return features, row, history, residual_model


def save_residual_covariance(model, dates, directory):
    """Use the existing covariance loader's format; centered factors include n-2 correction."""
    directory.mkdir(parents=True, exist_ok=True)
    factors = pd.DataFrame(model.centered.T, columns=[f"centered_{w:02}" for w in range(len(dates))])
    factors.insert(0, "merchant_abn", model.ids)
    factors["sample_variance"] = model.variances
    factors.to_parquet(directory / "factors.parquet", index=False, compression=None)
    save_json(directory / "settings.json", dict(alpha=model.alpha, week_ends=dates.strftime("%Y-%m-%d").tolist(),
        statistic="OLS residual covariance with n-2 degrees of freedom",
        factor_scaling="residual * sqrt((n-1)/(n-2)); existing helper divides factor products by n-1",
        shrinkage="Ledoit-Wolf on standardized eligible nonconstant residual histories; diagonal preserved"))


def future_dates(as_of, coverage_end):
    end = pd.Timestamp(as_of) + pd.offsets.MonthEnd(3)
    if end > coverage_end:
        return None
    dates = pd.date_range(pd.Timestamp(as_of)+pd.Timedelta(days=1), end, freq="W-SUN")
    return dates[dates-pd.Timedelta(days=6) > as_of]


def realized_measures(values, dates, past):
    """Past references stay frozen; a future refit is used ONLY as a descriptive outcome."""
    x = np.asarray(values, dtype=float)
    future_stats, _, _ = decompose(x)
    elapsed = np.asarray((dates-pd.Timestamp(past.history_end.iloc[0])).days/7, dtype=float)
    centered_future_time = elapsed + (int(past.history_weeks.iloc[0])-1)/2
    line = past.weekly_mean.to_numpy() + np.outer(centered_future_time, past.trend_slope_per_week.to_numpy())
    negative = (line < 0).sum(axis=0)
    line = np.maximum(line, 0)  # Nonnegative revenue forecast; never change historical OLS decomposition.
    result = pd.DataFrame(dict(future_weeks=len(dates), future_weekly_mean=x.mean(axis=0),
        future_raw_sd=future_stats.raw_sd, future_residual_sd=future_stats.residual_sd,
        future_slope_per_week=future_stats.trend_slope_per_week,
        future_downside_mean=downside(x, past.weekly_mean.to_numpy()),
        future_downside_trend=downside(x, line), future_trend_error_rms=np.sqrt(np.mean((x-line)**2, axis=0)),
        future_below_mean_fraction=np.mean(x < past.weekly_mean.to_numpy(), axis=0),
        future_full_week_fees=x.sum(axis=0), forecast_clipped_weeks=negative))
    return result, line


def evaluate_future(weekly, features, portfolio, selected, coverage_end):
    """Sum fees first for portfolio downside: individual shortfalls are not additive."""
    as_of, window = portfolio["as_of"], portfolio["window"]
    dates = future_dates(as_of, coverage_end)
    if dates is None:
        return dict(as_of=as_of, window=window, status="unavailable", future_weeks=0), pd.DataFrame(), pd.DataFrame()
    x = history_matrix(weekly, dates, features.merchant_abn.to_numpy())
    per_merchant, _ = realized_measures(x, dates, features)
    per_merchant = pd.concat([features[["merchant_abn", "as_of", "window"]].reset_index(drop=True), per_merchant], axis=1)
    positions = pd.Index(features.merchant_abn).get_indexer(selected.merchant_abn)
    total = x[:, positions].sum(axis=1)
    stats, line = realized_measures(total[:, None], dates, pd.DataFrame([portfolio]))
    result = stats.iloc[0].to_dict()
    result.update(as_of=as_of, window=window, status="observed")
    series = pd.DataFrame(dict(as_of=as_of, window=window, phase="future", week_end=dates,
        gross_fees=total, reference_mean=portfolio["weekly_mean"], reference_trend=line[:, 0]))
    return result, per_merchant, series


# Targets differ. Do not choose a winner by comparing MAE across these rows.
METRICS = {
    "raw_sd": "future_raw_sd", "raw_shrunk_sd": "future_raw_sd",
    "residual_sd": "future_residual_sd", "residual_shrunk_sd": "future_residual_sd",
    "downside_mean": "future_downside_mean", "downside_trend": "future_downside_trend",
}


def score_portfolios(estimates, outcomes):
    joined = estimates.merge(outcomes, on=["as_of", "window"], validate="one_to_one")
    rows = []
    for method, target in METRICS.items():
        part = joined[["as_of", "window", "status", "future_weeks"]].copy()
        part["measure"], part["target"] = method, target
        part["estimate"], part["actual"] = joined[method], joined[target]
        part["signed_error"] = part.estimate-part.actual
        part["absolute_error"] = part.signed_error.abs()
        rows.append(part)
    return joined, pd.concat(rows, ignore_index=True)


def summarize_errors(scored, dates, windows, comparison):
    data = scored.loc[scored.as_of.isin(dates) & scored.window.isin(windows)]
    require(data.status.eq("observed").all(), "Unobserved outcome in summary")
    rows = []
    for (window, measure, target), g in data.groupby(["window", "measure", "target"]):
        require(set(g.as_of) == set(dates), "Unmatched summary dates")
        rows.append(dict(comparison=comparison, window=window, measure=measure, target=target, origins=len(g),
            mean_estimate=g.estimate.mean(), mean_actual=g.actual.mean(),
            mean_absolute_error=g.absolute_error.mean(), mean_signed_error=g.signed_error.mean(),
            underestimate_count=int(g.signed_error.lt(0).sum())))
    return pd.DataFrame(rows)


def merchant_diagnostics(features, outcomes):
    """Equal-origin rank associations with the SAME later relative downside, avoiding size alone."""
    joined = features.merge(outcomes, on=["as_of", "window", "merchant_abn"], validate="one_to_one")
    rows = []
    for (as_of, window), group in joined.groupby(["as_of", "window"]):
        group = group.loc[group.eligible & group.weekly_mean.gt(0)]
        target = group.future_downside_mean/group.weekly_mean
        predictors = {name: group[name]/group.weekly_mean for name in ["raw_sd", "residual_sd", "downside_mean", "downside_trend"]}
        predictors["negative_trend"] = (-group.trend_slope_per_week).clip(lower=0)/group.weekly_mean
        for name, values in predictors.items():
            correlation = values.corr(target, method="spearman") if values.nunique() > 1 and target.nunique() > 1 else np.nan
            rows.append(dict(as_of=as_of, window=window, measure=name, merchants=len(group),
                common_target="future_downside_below_frozen_mean / historical_mean",
                spearman=correlation, target_nonzero_merchants=int(target.gt(0).sum())))
    return pd.DataFrame(rows)


def toy_scenarios():
    t = np.arange(26)
    scenarios = {"constant": np.full(26, 162.5), "steady_growth": 100+5*t,
        "steady_decline": 225-5*t, "alternating": 162.5+30*np.where(t%2, 1, -1),
        "midpoint_drop": np.where(t < 13, 225, 100)}
    result, _, _ = decompose(np.column_stack(list(scenarios.values())))
    result.insert(0, "scenario", list(scenarios))
    return result


def write_report(estimates, joined, summary, associations, features):
    primary = joined.loc[joined.window.eq("26w") & joined.status.eq("observed")]
    latest_windows = estimates.loc[estimates.as_of.eq(cf.CUTOFFS[-1])].set_index("window")
    latest = latest_windows.loc["26w"]
    lines = ["# Raw variability, trend fluctuations and weekly downside", "",
        "## Frozen comparison", "",
        "The revenue-only top 100 and historical-mean expected monthly fees are unchanged. Primary: 26 weeks, "
        "eight November 2021-June 2022 origins. Sensitivity: 26/52/all history on the same five February-June "
        "origins. September 2022 is descriptive only. No portfolio is optimized or recommended from these results.", "",
        "Raw SD measures all weekly fee variation (n-1 denominator). A straight line with intercept is fitted "
        "to past weekly fees. Residual SD uses sqrt(SSE/(n-2)); descriptive residual variance uses SSE/(n-1), "
        "so raw variance = linear-trend variance + descriptive residual variance exactly. The n-2 correction can "
        "make residual SD slightly larger than raw SD when the fitted trend explains very little. Report slope direction "
        "alongside residual risk. R-squared is a descriptive variance share, not causal evidence or a growth forecast.", "",
        "Weekly downside is sqrt(mean(max(reference-fees,0)^2)), dividing by ALL weeks. The primary reference "
        "is the historical weekly mean, frozen for later scoring. A trend-relative downside sensitivity uses the "
        "past fitted line, extrapolated to each future week and floored at zero. These thresholds are explicit "
        "analyst choices, not contractual minimum income or business loss. Neither is a standard deviation. "
        "Sum merchant fees BEFORE calculating portfolio shortfalls; merchant downside is not additive.", "",
        "## Primary portfolio decomposition", "",
        "| Cutoff | Weekly trend (A$/week) | Trend share of raw variance | Raw SD | Residual SD | Downside below mean | Downside below trend |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in primary.itertuples():
        lines.append(f"| {r.as_of:%Y-%m-%d} | {r.trend_slope_per_week:+,.0f} | {100*r.trend_variance_share:.1f}% | "
                     f"{r.raw_sd:,.0f} | {r.residual_sd:,.0f} | {r.downside_mean:,.0f} | {r.downside_trend:,.0f} |")
    growing = primary.loc[primary.trend_slope_per_week.gt(0)]
    lines += ["", f"{len(growing)}/{len(primary)} historical portfolios had a positive fitted trend. Across those "
        f"origins, the median share of raw variance accounted for by the linear trend was {100*growing.trend_variance_share.median():.1f}%. "
        "This quantifies how much a raw-variance penalty would count sustained growth as variability. It does not "
        "prove that all growth was predictable or that penalizing it is always inappropriate.", "",
        "## Later outcomes: compare like with like", "",
        "| Measure | Later target | Mean estimate | Mean target | MAE (A$) | Underestimates |",
        "|---|---|---:|---:|---:|---:|"]
    for r in summary.loc[summary.comparison.eq("primary_26w_eight")].itertuples():
        lines.append(f"| {r.measure} | {r.target} | {r.mean_estimate:,.0f} | {r.mean_actual:,.0f} | {r.mean_absolute_error:,.0f} | {r.underestimate_count}/{r.origins} |")
    lines += ["", "Do not rank raw SD, residual SD and downside by the MAEs above: they measure different targets. "
        "Future residual SD is computed by fitting a line within the future quarter only to describe its realized "
        "fluctuations. It is NOT a forecast available at selection. The separate frozen-line error below captures "
        "failure of the historical trend to continue.", "",
        "| Cutoff | Later raw SD | Later residual SD | RMS error vs frozen trend | Downside vs frozen mean | Downside vs frozen trend |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in primary.itertuples():
        lines.append(f"| {r.as_of:%Y-%m-%d} | {r.future_raw_sd:,.0f} | {r.future_residual_sd:,.0f} | "
            f"{r.future_trend_error_rms:,.0f} | {r.future_downside_mean:,.0f} | {r.future_downside_trend:,.0f} |")
    correlations = associations.loc[associations.window.eq("26w")].groupby("measure").spearman.mean()
    lines += ["", "## Merchant-level common-outcome diagnostic", "",
        "All measures are divided by the same historical mean and compared with future downside below that "
        "frozen mean, also divided by it. These are descriptive within-origin Spearman associations; normalization "
        "reduces the obvious dollar-size effect but ratios share a denominator. They are not calibrated loss "
        "predictions or independent merchant-level evidence of macroeconomic generalization.", "",
        "| Measure | Equal-origin mean rank correlation |", "|---|---:|"]
    for name, value in correlations.items():
        lines.append(f"| {name} | {value:.3f} |")
    latest_features = features.loc[features.window.eq("26w") & features.as_of.eq(cf.CUTOFFS[-1]) & features.eligible]
    lines += ["", "## Latest descriptive snapshot", "",
        f"At 30 September 2022, the fixed top 100 had raw weekly SD A${latest.raw_sd:,.0f}, residual SD "
        f"A${latest.residual_sd:,.0f}, and a fitted weekly trend of A${latest.trend_slope_per_week:+,.0f}. "
        f"The linear trend accounted for {100*latest.trend_variance_share:.1f}% of raw variance. Of "
        f"{len(latest_features):,} eligible merchants, {latest_features.trend_slope_per_week.gt(0).sum():,} had a positive "
        f"and {latest_features.trend_slope_per_week.lt(0).sum():,} a negative fitted slope. No future quarter is available.", "",
        f"For the typical eligible merchant, the median trend share was only "
        f"{100*latest_features.trend_variance_share.median():.1f}%. A combined portfolio can have a much higher "
        "trend share because individual fluctuations partly offset while shared trends remain. This result "
        f"is specific to the latest 26-week portfolio window: its 52-week trend share is "
        f"{100*latest_windows.loc['52w', 'trend_variance_share']:.2f}%, and its all-history share is "
        f"{100*latest_windows.loc['all', 'trend_variance_share']:.1f}%. A single straight line does not describe all history equally well.", "",
        "## Decision implications", "",
        "Keep raw volatility as a description of total fee variability. Use detrended covariance as a candidate "
        "measure of fluctuations around a trend, always accompanied by the trend direction and frozen-reference "
        "downside. Do not silently replace the current risk feature or expected return. A perfectly smooth decline "
        "has zero residual risk, just like perfectly smooth growth; the synthetic examples demonstrate this. "
        "Downside below a sample mean also penalizes the early weeks of steady growth. Detrended downside avoids "
        "that mechanically, but can hide a decline that follows its fitted path. A persistent-income floor needs "
        "a business justification before optimization.", "",
        "Regression removes only a straight-line trend. Seasonal patterns, structural breaks and nonlinear trends "
        "can remain. Fitting a line cannot increase the residual sum of squares; this alone is not better risk prediction. "
        "OLS variance correction and Ledoit-Wolf shrinkage do not establish independent, stationary weekly errors. "
        "Residual shrinkage preserves each residual variance and captures co-movement, but is only an estimator. "
        "Downside is computed directly, not by summing individual downside values or treating clipped residuals as ordinary covariance.", "",
        "All lookback sensitivities, every origin, three non-overlapping schedules and signed errors are exported. "
        "The short synthetic history, noisy 11-13-week outcomes and overlapping quarters do not establish "
        "statistical significance. There is no untouched holdout or evidence yet that a risk-aware selection improves "
        "the return-risk trade-off. Categories/take rates are assumed stable; gross fees exclude fraud losses, "
        "funding/operating costs and settlement timing. Zero volatility does not confer eligibility.", "",
        "## Reproduction", "",
        "Run `python code/cash_flow_risk_comparison.py`, then "
        "`python -m pytest -p no:capture code/tests/test_cash_flow_risk_comparison.py -q`. "
        "Use the existing pinned requirements-forecast.txt environment. Notebook 06_cash_flow_risk_comparison.ipynb "
        "contains the explanation, results and plots. Generated data stay in the ignored "
        "tables/analysis/cash_flow_risk_comparison directory as uncompressed Parquet with CSV summaries. "
        "Existing ABS, return and cash-flow experiments are preserved."]
    (OUT / "findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    """Audit, freeze past-only estimates, then attach future diagnostics; never optimize selections."""
    previous = json.loads((OUT/"acceptance_checks.json").read_text()) if (OUT/"acceptance_checks.json").exists() else None
    daily, monthly, merchants, audit = cf.load_audited_inputs()
    weekly = cf.weekly_panel(daily, merchants)
    protocol = dict(cutoffs=cf.CUTOFFS.strftime("%Y-%m-%d").tolist(), primary="26 weeks on all eight historical origins",
        sensitivity="26/52/all on the same five February-June origins; no window optimization", windows=WINDOWS,
        raw_sd="sqrt(sum((fees-mean)^2)/(n-1))", trend="Past-only OLS intercept and linear weekly index",
        residual_sd="sqrt(sum(residual^2)/(n-2)); decomposition uses n-1 separately",
        downside="sqrt(mean(max(reference-fees,0)^2)) over ALL weeks",
        references=["historical weekly mean, frozen", "past fitted trend; future extrapolation floored at zero"],
        future_residual="Future refit is outcome description only; separately evaluate frozen-line prediction error",
        selection="Unchanged eligible expanding-monthly-mean top 100, numeric ascending ABN ties",
        shrinkage="Same diagonal-preserving standardized Ledoit-Wolf, fitted separately on past raw/residual histories",
        primary_question="How much raw variability is explained by rising versus falling trends?",
        evaluation="Next three calendar months, whole Monday-Sunday weeks; fixed selections, like-for-like targets",
        merchant_association="Risk/mean versus later frozen-mean downside/mean; descriptive Spearman, no p-values",
        portfolio_optimization=False, changes_to_expected_return=False, untouched_holdout=False, retrospective=True,
        inputs=dict(daily=audit["daily_sha256"], monthly=digest(monthly), merchants=digest(merchants)),
        versions={p:importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]})
    if (OUT/"protocol.json").exists():
        require(json.loads((OUT/"protocol.json").read_text()) == protocol, "Changed risk-measure protocol or inputs")
    save_json(OUT/"protocol.json", protocol)
    old_selection = pd.read_parquet(cf.OUT/"baseline_selections.parquet")
    old_estimates = pd.read_parquet(cf.OUT/"portfolio_diagnostics.parquet").set_index("as_of")
    frames, estimates, histories, selections, checks, availability = [], [], [], [], [], []
    for as_of in cf.CUTOFFS:
        print(f"Estimating risk definitions at {as_of.date()}", flush=True)
        base, _ = cf.features_at_cutoff(weekly, monthly, merchants, as_of)
        selected = cf.baseline_selection(base)
        pd.testing.assert_frame_equal(selected, old_selection.loc[old_selection.as_of.eq(as_of)].reset_index(drop=True), check_exact=True)
        selections.append(selected)
        # Rebuild from changed future daily transactions, not merely from saved feature values.
        if as_of in ORIGINS:
            changed = daily.copy()
            changed.loc[changed.day.gt(as_of), "sales"] *= 1000
            changed_monthly = monthly.copy()
            changed_monthly.loc[changed_monthly.month.gt(as_of), "sales"] *= 1000
            modified = cf.weekly_panel(changed, merchants)
            new_base, _ = cf.features_at_cutoff(modified, changed_monthly, merchants, as_of)
            new_selection = cf.baseline_selection(new_base)
            pd.testing.assert_frame_equal(base, new_base, check_exact=True)
            pd.testing.assert_frame_equal(selected, new_selection, check_exact=True)
        for window in WINDOWS:
            dates = history_dates(weekly, as_of, window)
            availability.append(dict(as_of=as_of, window=window, available=dates is not None))
            if dates is None:
                continue
            f, p, h, model = estimate_at(weekly, base, selected, as_of, window)
            if as_of in ORIGINS:
                f2, p2, h2, model2 = estimate_at(modified, new_base, new_selection, as_of, window)
                for a, b in [(f,f2), (pd.DataFrame([p]),pd.DataFrame([p2])), (h,h2)]:
                    pd.testing.assert_frame_equal(a, b, check_exact=True)
                np.testing.assert_array_equal(model.centered, model2.centered)
                require(model.alpha == model2.alpha, "Future residual shrinkage leakage")
            if window == "26w":
                np.testing.assert_allclose([p["raw_sd"], p["raw_shrunk_sd"]],
                    old_estimates.loc[as_of, ["history_weekly_sd", "regularized_weekly_sd"]].to_numpy(dtype=float), rtol=1e-12, atol=1e-7)
            directory = OUT/"residual_covariance"/as_of.strftime("%Y-%m-%d")/window
            save_residual_covariance(model, dates, directory)
            restored = cf.load_covariance(directory)
            np.testing.assert_array_equal(model.centered, restored.centered)
            np.testing.assert_array_equal(model.ids, restored.ids)
            np.testing.assert_array_equal(model.variances, restored.variances)
            require(model.alpha == restored.alpha, "Covariance roundtrip changed")
            frames.append(f)
            estimates.append(p)
            histories.append(h)
            checks.append(dict(as_of=as_of, window=window, decomposition=True, covariance_reconciled=True,
                covariance_roundtrip=True, unchanged_baseline=True, future_daily_perturbation=as_of in ORIGINS))
    features = pd.concat(frames, ignore_index=True)
    estimates, selection = pd.DataFrame(estimates), pd.concat(selections, ignore_index=True)
    require(not features.duplicated(["merchant_abn", "as_of", "window"]).any(), "Duplicate merchant features")
    write_table(features, OUT, "merchant_features")
    write_table(estimates, OUT, "unscored_portfolios")
    write_table(selection, OUT, "baseline_selections")
    write_table(pd.DataFrame(availability), OUT, "availability")
    save_json(OUT/"feature_freeze.json", dict(features=digest(features), estimates=digest(estimates), selections=digest(selection),
        saved_at=pd.Timestamp.now(tz="UTC"), stage="Estimates, references and selections saved before future outcomes"))
    outcomes, merchant_outcomes = [], []
    for p in estimates.to_dict("records"):
        f = features.loc[features.as_of.eq(p["as_of"]) & features.window.eq(p["window"])].reset_index(drop=True)
        s = selection.loc[selection.as_of.eq(p["as_of"])]
        o, m, h = evaluate_future(weekly, f, p, s, daily.day.max())
        outcomes.append(o)
        if len(m):
            merchant_outcomes.append(m)
            histories.append(h)
            # Independent transaction-day summation validates every future portfolio, not just its SD.
            raw = daily.loc[daily.merchant_abn.isin(s.merchant_abn)].copy()
            raw["week_end"] = raw.day.dt.to_period("W-SUN").dt.end_time.dt.normalize()
            rates = merchants.set_index("merchant_abn").take_rate/100
            direct = (raw.sales * raw.merchant_abn.map(rates)).groupby(raw.week_end).sum().reindex(h.week_end)
            np.testing.assert_allclose(direct.to_numpy(), h.gross_fees.to_numpy(), rtol=1e-12, atol=1e-7)
            np.testing.assert_allclose(o["future_raw_sd"], old_estimates.loc[p["as_of"], "future_weekly_sd"], rtol=1e-12)
    outcomes, merchant_outcomes = pd.DataFrame(outcomes), pd.concat(merchant_outcomes, ignore_index=True)
    joined, scored = score_portfolios(estimates, outcomes)
    common = pd.date_range("2022-02-28", "2022-06-30", freq="ME")
    summary = pd.concat([summarize_errors(scored, ORIGINS, ["26w"], "primary_26w_eight"),
        summarize_errors(scored, common, WINDOWS, "sensitivity_common_five")], ignore_index=True)
    schedules = pd.concat([summarize_errors(scored, ORIGINS[k::3], ["26w"], f"schedule_{k}").assign(
        cutoffs=", ".join(ORIGINS[k::3].strftime("%Y-%m-%d"))) for k in range(3)], ignore_index=True)
    associations = merchant_diagnostics(features, merchant_outcomes)
    histories = pd.concat(histories, ignore_index=True)
    signatures = dict(features=digest(features), estimates=digest(estimates), selections=digest(selection),
        outcomes=digest(outcomes), merchant_outcomes=digest(merchant_outcomes), summary=digest(summary), associations=digest(associations))
    if previous:
        require(previous["digests"] == signatures, "Risk-measure rerun changed results")
    for name, frame in [("portfolio_diagnostics", joined), ("portfolio_outcomes", outcomes),
        ("merchant_outcomes", merchant_outcomes), ("fold_metrics", scored), ("summary", summary),
        ("nonoverlapping_schedules", schedules), ("merchant_rank_associations", associations),
        ("portfolio_weekly_fees", histories), ("toy_scenarios", toy_scenarios()), ("cutoff_checks", pd.DataFrame(checks))]:
        write_table(frame, OUT, name)
    save_json(OUT/"acceptance_checks.json", dict(source_signatures_and_daily_checksum=True,
        daily_monthly_and_weekly_sales_orders_reconciled=True, unique_feature_keys=True,
        exactly_100_eligible_unique_merchants_per_cutoff=True, baseline_reproductions=len(cf.CUTOFFS),
        future_daily_perturbations=int(sum(c["future_daily_perturbation"] for c in checks)),
        covariance_and_decomposition_checks=len(checks), independent_daily_future_portfolios=len(merchant_outcomes.groupby(["as_of","window"])),
        references_frozen_before_outcomes=True, september_unscored=True,
        reproducible_rerun=previous is not None, digests=signatures))
    write_report(estimates, joined, summary, associations, features)
    print(summary.loc[summary.comparison.eq("primary_26w_eight")].round(3).to_string(index=False), flush=True)
    return features, joined, summary


if __name__ == "__main__":
    run()
