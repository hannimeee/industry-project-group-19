"""Weekly merchant fee volatility and covariance; no portfolio optimization."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf_shrinkage
from threadpoolctl import threadpool_limits

from abs_momentum_comparison import require, write_table
from external_data_models import features_at
from external_data_sources import save_json
from forecast_comparison import ORIGINS, digest
from simple_return_comparison import OUT as SIMPLE_OUT, sales_forecasts
from timesfm_comparison import OUT as DAILY_OUT, validated_inputs

OUT = DAILY_OUT.parent / "cash_flow_variability"
CUTOFFS = ORIGINS.union(pd.DatetimeIndex(["2022-09-30"]))
LOOKBACK = 26


def validate_daily(daily, merchants, start, end):
    """Require a complete audited merchant-calendar grid before using any zeros."""
    require(merchants.merchant_abn.is_unique, "Duplicate merchant metadata")
    require(merchants.take_rate.between(0, 100, inclusive="right").all(), "Invalid take rate")
    require(not daily.duplicated(["merchant_abn", "day"]).any(), "Duplicate merchant-day")
    require(daily[["merchant_abn", "day", "sales", "orders"]].notna().all().all(), "Missing daily values")
    require(np.isfinite(daily.sales).all() and daily.sales.ge(0).all(), "Invalid daily sales")
    require(np.isfinite(daily.orders).all() and daily.orders.ge(0).all()
            and daily.orders.eq(np.floor(daily.orders)).all(), "Invalid daily order counts")
    dates = pd.date_range(start, end)
    require(set(daily.merchant_abn) == set(merchants.merchant_abn), "Merchant universe mismatch")
    require(set(daily.day) == set(dates), "Incomplete daily coverage")
    require(daily.groupby("merchant_abn").size().eq(len(dates)).all(), "Incomplete merchant calendar")


def weekly_panel(daily, merchants):
    """Aggregate only full Monday-Sunday weeks, rejecting gaps instead of filling them."""
    validate_daily(daily, merchants, daily.day.min(), daily.day.max())
    frame = daily.copy()
    frame["week_end"] = frame.day.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    grouped = frame.groupby(["merchant_abn", "week_end"], as_index=False).agg(
        sales=("sales", "sum"), orders=("orders", "sum"), days=("day", "size"))
    grouped["week_start"] = grouped.week_end - pd.Timedelta(days=6)
    full = grouped.loc[grouped.week_start.ge(daily.day.min()) & grouped.week_end.le(daily.day.max())].copy()
    require(full.days.eq(7).all(), "Incomplete interior week")
    full = full.drop(columns="days").merge(merchants[["merchant_abn", "take_rate"]],
                                           on="merchant_abn", validate="many_to_one")
    full["gross_fees"] = full.sales * full.take_rate / 100
    full = full.sort_values(["week_end", "merchant_abn"]).reset_index(drop=True)
    require(len(full) > 0, "No complete weeks")
    covered = daily.loc[daily.day.between(full.week_start.min(), full.week_end.max())]
    expected = covered.groupby("merchant_abn")[["sales", "orders"]].sum()
    actual = full.groupby("merchant_abn")[["sales", "orders"]].sum()
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-7)
    return full


def risk_window(as_of):
    """Include a Sunday cutoff itself; omit any unfinished calendar week."""
    as_of = pd.Timestamp(as_of).normalize()
    end = as_of - pd.Timedelta(days=(as_of.weekday() + 1) % 7)
    return pd.date_range(end=end, periods=LOOKBACK, freq="W-SUN")


@dataclass
class Covariance:
    """Low-rank sample covariance plus a diagonal shrinkage target, in fee dollars."""
    ids: np.ndarray
    centered: np.ndarray  # weeks by merchants, in the same ABN order as ids
    variances: np.ndarray
    alpha: float


def estimate_covariance(values, ids, eligible):
    """Shrink standardized eligible histories, preserving every sample variance."""
    x = np.asarray(values, dtype=float)
    ids, eligible = np.asarray(ids), np.asarray(eligible, dtype=bool)
    require(x.ndim == 2 and x.shape[0] >= 2 and x.shape[1] == len(ids)
            and len(eligible) == len(ids) and pd.Index(ids).is_unique, "Invalid covariance dimensions")
    require(np.isfinite(x).all(), "Nonfinite weekly fees")
    centered = x - x.mean(axis=0)
    centered[:, np.ptp(x, axis=0) == 0] = 0
    variances = np.sum(centered ** 2, axis=0) / (len(x) - 1)
    fitted = eligible & (variances > 0)
    alpha = 0.0
    if fitted.sum() > 1:
        standardized = centered[:, fitted] / np.sqrt(variances[fitted])
        with threadpool_limits(limits=1):
            alpha = float(ledoit_wolf_shrinkage(standardized, assume_centered=True, block_size=256))
    require(np.isfinite(alpha) and 0 <= alpha <= 1, "Invalid shrinkage coefficient")
    return Covariance(ids, centered, variances, alpha)


def subset_positions(model, ids, estimator):
    """Resolve unique requested ABNs in caller order, never by positional coincidence."""
    require(estimator in {"sample", "shrunk"}, "Estimator must be sample or shrunk")
    require(len(ids) > 0 and pd.Index(ids).is_unique, "Portfolio must contain unique merchants")
    positions = pd.Index(model.ids).get_indexer(ids)
    require((positions >= 0).all(), "Unknown merchant in covariance request")
    return positions


def covariance_submatrix(model, ids, estimator):
    """Materialize only the requested sample or regularized covariance submatrix."""
    positions = subset_positions(model, ids, estimator)
    c = model.centered[:, positions]
    matrix = c.T @ c / (len(c) - 1)
    if estimator == "shrunk":
        matrix = (1 - model.alpha) * matrix + model.alpha * np.diag(model.variances[positions])
    return pd.DataFrame(matrix, index=pd.Index(ids, name="merchant_abn"), columns=ids)


def portfolio_volatility(model, ids, estimator):
    """Calculate the standard deviation of summed fees without a dense covariance matrix."""
    positions = subset_positions(model, ids, estimator)
    totals = model.centered[:, positions].sum(axis=1)
    variance = float(totals @ totals / (len(totals) - 1))
    if estimator == "shrunk":
        variance = (1 - model.alpha) * variance + model.alpha * float(model.variances[positions].sum())
    require(np.isfinite(variance) and variance >= 0, "Invalid portfolio variance")
    return float(np.sqrt(variance))


def features_at_cutoff(weekly, monthly, merchants, as_of):
    """Build past-only risk features and the unchanged expected monthly fee estimate."""
    as_of = pd.Timestamp(as_of)
    require(as_of == as_of + pd.offsets.MonthEnd(0), "Use a complete-month cutoff")
    dates = risk_window(as_of)
    ids = merchants.merchant_abn.sort_values().to_numpy()
    history = weekly.loc[weekly.week_end.isin(dates)]
    require(not history.duplicated(["merchant_abn", "week_end"]).any(), "Duplicate merchant-week")
    wide = history.pivot(index="week_end", columns="merchant_abn", values="gross_fees").reindex(index=dates, columns=ids)
    require(wide.notna().all().all(), "Incomplete 26-week history")
    labels = merchants[["category"]].drop_duplicates().assign(origin=as_of)
    eligibility = features_at(monthly.loc[monthly.month.le(as_of)], merchants, labels, as_of)
    f = eligibility.set_index("merchant_abn").loc[ids, ["name", "category", "take_rate", "eligible", "eligibility_reason"]].reset_index()
    f["as_of"], f["window_start"], f["window_end"] = as_of, dates[0] - pd.Timedelta(days=6), dates[-1]
    f["n_weeks"] = len(dates)
    x = wide.to_numpy()
    model = estimate_covariance(x, ids, f.eligible.to_numpy())
    f["weekly_mean_fees"] = x.mean(axis=0)
    f["weekly_variance"] = model.variances
    f["weekly_volatility"] = np.sqrt(model.variances)
    f["active_weeks"] = (x > 0).sum(axis=0)
    monthly_sales = sales_forecasts(monthly, ids, as_of)["expanding_mean"] / 3
    f["expected_monthly_fees"] = monthly_sales * f.take_rate / 100
    return f, model


def baseline_selection(features):
    """Select the revenue-only top 100 with the existing eligibility and ABN tie rule."""
    ranked = features.loc[features.eligible].sort_values(
        ["expected_monthly_fees", "merchant_abn"], ascending=[False, True]).copy()
    require(len(ranked) >= 100 and ranked.merchant_abn.is_unique, "Need 100 unique eligible merchants")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.head(100).reset_index(drop=True)


def save_covariance(model, directory, as_of):
    """Persist labeled factors in uncompressed Parquet and their scalar settings in JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    factors = pd.DataFrame(model.centered.T, columns=[f"centered_{w:02}" for w in range(model.centered.shape[0])])
    factors.insert(0, "merchant_abn", model.ids)
    factors["sample_variance"] = model.variances
    factors.to_parquet(directory / "factors.parquet", index=False, compression=None)
    save_json(directory / "settings.json", dict(as_of=str(as_of.date()), alpha=model.alpha,
        week_ends=risk_window(as_of).strftime("%Y-%m-%d").tolist(), covariance_units="AUD squared",
        formula="(1-alpha) * sample_covariance + alpha * diag(sample_variances)",
        alpha_fit="Ledoit-Wolf on standardized nonconstant eligible histories only", ddof=1))


def load_covariance(directory):
    """Reload a saved ABN-indexed covariance model for downstream portfolio work."""
    settings = json.loads((directory / "settings.json").read_text())
    factors = pd.read_parquet(directory / "factors.parquet")
    cols = [f"centered_{w:02}" for w in range(len(settings["week_ends"]))]
    return Covariance(factors.merchant_abn.to_numpy(), factors[cols].to_numpy().T,
                      factors.sample_variance.to_numpy(), float(settings["alpha"]))


def portfolio_diagnostics(weekly, selection, model, coverage_end):
    """Compare past estimates with later full weeks for an unchanged selected portfolio."""
    as_of = selection.as_of.iloc[0]
    ids = selection.merchant_abn.to_numpy()
    dates = risk_window(as_of)
    history = weekly.loc[weekly.week_end.isin(dates) & weekly.merchant_abn.isin(ids)]
    past = history.groupby("week_end").gross_fees.sum().reindex(dates)
    require(len(history) == len(ids) * len(dates) and past.notna().all(), "Incomplete selected history")
    observed = float(past.std(ddof=1))
    sample = portfolio_volatility(model, ids, "sample")
    require(np.isclose(observed, sample, rtol=1e-10, atol=1e-7), "Portfolio sample SD does not reconcile")
    rows = [pd.DataFrame(dict(as_of=as_of, phase="history", week_end=dates, gross_fees=past.to_numpy()))]
    result = dict(as_of=as_of, selected_merchants=len(ids), expected_monthly_fees=float(selection.expected_monthly_fees.sum()),
        history_weeks=len(dates), history_weekly_mean=float(past.mean()), history_weekly_sd=observed,
        regularized_weekly_sd=portfolio_volatility(model, ids, "shrunk"), shrinkage_alpha=model.alpha,
        future_weeks=0, future_weekly_mean=np.nan, future_weekly_sd=np.nan,
        future_full_week_fees=np.nan, future_status="insufficient_complete_outcome_period")
    end = as_of + pd.offsets.MonthEnd(3)
    if end <= coverage_end:
        future_dates = pd.date_range(as_of + pd.Timedelta(days=1), end, freq="W-SUN")
        future_dates = future_dates[future_dates - pd.Timedelta(days=6) > as_of]
        future = weekly.loc[weekly.week_end.isin(future_dates) & weekly.merchant_abn.isin(ids)]
        require(len(future) == len(ids) * len(future_dates), "Incomplete future portfolio weeks")
        totals = future.groupby("week_end").gross_fees.sum().reindex(future_dates)
        result.update(future_weeks=len(totals), future_weekly_mean=float(totals.mean()),
            future_weekly_sd=float(totals.std(ddof=1)), future_full_week_fees=float(totals.sum()), future_status="observed")
        rows.append(pd.DataFrame(dict(as_of=as_of, phase="future", week_end=future_dates, gross_fees=totals.to_numpy())))
    return result, pd.concat(rows, ignore_index=True)


def load_audited_inputs():
    """Check existing source evidence and independently reconcile daily and monthly totals."""
    _, monthly, merchants, audit = validated_inputs()
    path = DAILY_OUT / "merchant_day.parquet"
    saved = json.loads((DAILY_OUT / "daily_audit.json").read_text())
    require(saved["archives"] == audit["archives"] and saved["monthly_digest"] == digest(monthly), "Daily provenance changed")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == saved["daily_sha256"], "Daily panel checksum changed")
    daily = pd.read_parquet(path)
    start, end = monthly.month.min(), monthly.month.max() + pd.offsets.MonthEnd(0)
    validate_daily(daily, merchants, start, end)
    reconstructed = daily.assign(month=daily.day.dt.to_period("M").dt.to_timestamp()).groupby(
        ["merchant_abn", "month"])[["sales", "orders"]].sum().sort_index()
    expected = monthly.set_index(["merchant_abn", "month"])[["sales", "orders"]].sort_index()
    pd.testing.assert_frame_equal(reconstructed, expected, check_dtype=False, rtol=1e-12, atol=1e-7)
    require(int(daily.orders.sum()) == saved["known_orders"] and np.isclose(daily.sales.sum(), saved["known_sales"], rtol=1e-12),
            "Known-merchant audit totals changed")
    return daily, monthly, merchants, saved


def write_findings(features, metrics):
    """Explain the measured quantities and report results without implying optimization gains."""
    latest = features.loc[features.as_of.eq(CUTOFFS[-1])]
    lines = ["# Weekly fee-revenue variability", "",
        "Expected return remains the expanding historical mean monthly gross fees. This workflow measures risk "
        "for the revenue-only top 100; it does not select a risk-optimized portfolio.", "",
        "Each merchant's volatility is the sample standard deviation (ddof=1) of its last 26 full Monday-Sunday "
        "weekly gross fees. Portfolio sample SD is calculated after summing merchant fees. Covariance therefore "
        "captures contemporaneous co-movement. The regularized estimate shrinks correlations, preserving individual "
        "variances; a smaller estimate is not an achieved reduction in risk.", "",
        f"Latest cutoff: 2022-09-30; window: 2022-03-28 through 2022-09-25. All {len(latest):,} merchants have features; "
        f"{int(latest.eligible.sum()):,} are eligible. {int(latest.weekly_volatility.eq(0).sum())} have zero measured "
        "volatility; inactivity does not imply a safe or eligible merchant.", "",
        "| Cutoff | Expected monthly fees | Historical weekly SD | Regularized weekly SD | Later weekly SD | Later full weeks |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        future = f"A${row.future_weekly_sd:,.2f}" if pd.notna(row.future_weekly_sd) else "Unavailable"
        lines.append(f"| {row.as_of:%Y-%m-%d} | A${row.expected_monthly_fees:,.2f} | A${row.history_weekly_sd:,.2f} | "
                     f"A${row.regularized_weekly_sd:,.2f} | {future} | {row.future_weeks} |")
    lines += ["", "Later SD uses only whole weeks inside the following three calendar months, excluding boundary "
        "weeks; total fees in these weeks are not a complete three-month revenue total. All selections stay fixed. "
        "The September cutoff has no complete future evaluation period in the audited panel.", "",
        "The 26-week window is a fixed starting choice, not a proven optimum. Regularization is an estimation "
        "choice, not evidence of accurate future risk. Trends and calendar effects remain in the histories. "
        "Weekly SD is not settlement cash-flow risk, credit loss, a confidence interval, or annual/monthly total "
        "uncertainty. Do not scale it by the square root of time without examining serial dependence.", "",
        "Categories and take rates are assumed stable and historically available. Gross fees exclude fraud, "
        "funding and operating costs. Transactions for merchants without metadata remain outside fee-based selection. "
        "The data are synthetic and the short overlapping evaluations are retrospective, with no untouched holdout. "
        "This report makes no statistical-significance or improved-selection claim.", "",
        "Reproduce: `python code/cash_flow_variability.py`. Validate: "
        "`python -m pytest -p no:capture code/tests/test_cash_flow_variability.py -q`. "
        "Use the existing requirements-forecast.txt environment; no GPU or network service is needed.", "",
        "Covariance factors are stored per cutoff with ABN labels and JSON settings. "
        "Use load_covariance, covariance_submatrix and portfolio_volatility with estimator='sample' or 'shrunk'. "
        "Sample SD reproduces the observed historical feature; shrunk SD is separately labelled. "
        "Artifacts are uncompressed Parquet, with CSV views of feature and summary tables."]
    (OUT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    """Audit, freeze past-only measures, then score fixed baseline portfolios chronologically."""
    previous_checks = (json.loads((OUT / "acceptance_checks.json").read_text())
                       if (OUT / "acceptance_checks.json").exists() else None)
    daily, monthly, merchants, audit = load_audited_inputs()
    weekly = weekly_panel(daily, merchants)
    protocol = dict(cutoffs=CUTOFFS.strftime("%Y-%m-%d").tolist(), lookback_weeks=LOOKBACK,
        weeks="Complete Monday-Sunday weeks ending on or before cutoff", ddof=1,
        expected_return="Expanding complete-month mean sales * take_rate / 100",
        alpha="Ledoit-Wolf on standardized eligible nonconstant histories; diagonal-preserving shrinkage",
        selection="Existing eligible revenue-only top 100; ascending numeric ABN breaks ties",
        evaluation="Whole weeks inside the next three calendar months; fixed merchant selection",
        retrospective=True, untouched_holdout=False, portfolio_optimization=False,
        inputs=dict(daily_sha256=audit["daily_sha256"], monthly=digest(monthly), merchants=digest(merchants)),
        archives=audit["archives"], versions={p: importlib.metadata.version(p) for p in
            ["numpy", "pandas", "pyarrow", "scikit-learn", "threadpoolctl"]})
    if (OUT / "protocol.json").exists():
        require(json.loads((OUT / "protocol.json").read_text()) == protocol, "Changed cash-flow protocol or inputs")
    save_json(OUT / "protocol.json", protocol)
    weekly.to_parquet(OUT / "merchant_week.parquet", index=False, compression=None)
    old = pd.read_parquet(SIMPLE_OUT / "predictions.parquet",
        columns=["merchant_abn", "origin", "method", "predicted_fees", "rank", "selected"])
    frames, selections, models, checks = [], [], {}, []
    for as_of in CUTOFFS:
        print(f"Building weekly risk measures at {as_of.date()}", flush=True)
        f, model = features_at_cutoff(weekly, monthly, merchants, as_of)
        selected = baseline_selection(f)
        ids = selected.merchant_abn.to_numpy()
        for estimator in ["sample", "shrunk"]:
            sub = covariance_submatrix(model, ids, estimator)
            np.testing.assert_allclose(np.diag(sub), selected.weekly_variance, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(sub, sub.T, rtol=1e-12, atol=1e-7)
            require(np.linalg.eigvalsh(sub).min() >= -1e-9 * max(1, np.diag(sub).max()), "Non-PSD covariance")
            require(np.isclose(sub.to_numpy().sum(), portfolio_volatility(model, ids, estimator) ** 2,
                               rtol=1e-10, atol=1e-6), "Covariance and factor portfolio variance differ")
        if as_of in ORIGINS:
            saved = old.loc[old.origin.eq(as_of) & old.method.eq("expanding_mean")].set_index("merchant_abn").sort_index()
            current = f.loc[f.eligible].set_index("merchant_abn").sort_index()
            require(current.index.equals(saved.index), "Baseline eligible pool changed")
            np.testing.assert_allclose(current.expected_monthly_fees * 3, saved.predicted_fees, rtol=1e-12, atol=1e-7)
            require(ids.tolist() == saved.loc[saved.selected].sort_values("rank").index.tolist(), "Baseline top 100 changed")
            changed_daily = daily.copy()
            changed_daily.loc[changed_daily.day.gt(as_of), "sales"] *= 1000
            changed_monthly = monthly.copy()
            changed_monthly.loc[changed_monthly.month.gt(as_of), "sales"] *= 1000
            changed_weekly = weekly_panel(changed_daily, merchants)
            f2, model2 = features_at_cutoff(changed_weekly, changed_monthly, merchants, as_of)
            pd.testing.assert_frame_equal(f, f2, check_exact=True)
            require(model.alpha == model2.alpha and np.array_equal(model.centered, model2.centered), "Future covariance leakage")
            pd.testing.assert_frame_equal(selected, baseline_selection(f2), check_exact=True)
            for estimator in ["sample", "shrunk"]:
                require(portfolio_volatility(model, ids, estimator) == portfolio_volatility(model2, ids, estimator),
                        "Future leakage in historical portfolio volatility")
        directory = OUT / "covariance" / as_of.strftime("%Y-%m-%d")
        save_covariance(model, directory, as_of)
        restored = load_covariance(directory)
        require(np.array_equal(restored.ids, model.ids) and np.array_equal(restored.centered, model.centered)
                and np.array_equal(restored.variances, model.variances) and restored.alpha == model.alpha, "Covariance roundtrip changed")
        frames.append(f)
        selections.append(selected)
        models[as_of] = model
        checks.append(dict(as_of=as_of, sample_diagonal_and_portfolio_reconciled=True, covariance_psd=True,
            covariance_roundtrip_exact=True, baseline_reproduced=as_of in ORIGINS,
            future_transaction_perturbation=as_of in ORIGINS))
    features, selection = pd.concat(frames, ignore_index=True), pd.concat(selections, ignore_index=True)
    require(not features.duplicated(["merchant_abn", "as_of"]).any(), "Duplicate merchant feature key")
    write_table(features, OUT, "merchant_features")
    write_table(selection, OUT, "baseline_selections")
    save_json(OUT / "feature_freeze.json", dict(features=digest(features), selections=digest(selection),
        saved_at=pd.Timestamp.now(tz="UTC"), stage="Features and selections saved before future diagnostics"))
    metrics, histories = [], []
    for as_of, selected in selection.groupby("as_of", sort=True):
        result, weeks = portfolio_diagnostics(weekly, selected, models[as_of], daily.day.max())
        result["eligible_merchants"] = int(features.loc[features.as_of.eq(as_of), "eligible"].sum())
        metrics.append(result)
        histories.append(weeks)
    metrics = pd.DataFrame(metrics)
    result_digests = dict(feature_digest=digest(features), selection_digest=digest(selection),
                          diagnostics_digest=digest(metrics))
    if previous_checks is not None:
        require(all(previous_checks[key] == value for key, value in result_digests.items()),
                "Rerun changed results under the same frozen protocol")
    write_table(metrics, OUT, "portfolio_diagnostics")
    write_table(pd.concat(histories, ignore_index=True), OUT, "portfolio_weekly_fees")
    write_table(pd.DataFrame(checks), OUT, "cutoff_checks")
    save_json(OUT / "acceptance_checks.json", dict(source_metadata_and_daily_checksum_verified=True,
        complete_merchant_calendar=True, daily_monthly_reconciliation=True, daily_weekly_sales_and_orders_reconciled=True,
        baseline_reproductions=8, future_transaction_perturbations=8, covariance_roundtrips=9,
        unique_merchant_features=True, exactly_100_eligible_unique_selections_per_cutoff=True,
        reproducible_rerun=previous_checks is not None, cutoff_checks=checks, **result_digests))
    write_findings(features, metrics)
    print(metrics.round(3).to_string(index=False), flush=True)
    return features, metrics


if __name__ == "__main__":
    run()
