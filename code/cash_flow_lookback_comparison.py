"""Matched comparison of 26-week, 52-week and expanding weekly fee-risk estimates."""

import importlib.metadata
import json

import numpy as np
import pandas as pd

import cash_flow_variability as cf
from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from forecast_comparison import digest, ORIGINS

OUT = cf.OUT.parent / "cash_flow_lookback_comparison"
WINDOWS = ["26w", "52w", "all"]
LABELS = {"26w": "26 weeks", "52w": "52 weeks", "all": "All available weeks"}


def history_dates(weekly, as_of, window):
    """Return only contiguous, fully observed past weeks; unavailable windows return None."""
    require(window in WINDOWS, "Unknown lookback")
    as_of = pd.Timestamp(as_of)
    dates = pd.DatetimeIndex(weekly.loc[weekly.week_end.le(as_of), "week_end"].unique()).sort_values()
    require(len(dates) >= 2 and (dates.weekday == 6).all(), "Need complete Sunday-ending history")
    end = as_of - pd.Timedelta(days=(as_of.weekday() + 1) % 7)
    require(dates.equals(pd.date_range(dates[0], end, freq="W-SUN")), "Missing calendar week in history")
    length = len(dates) if window == "all" else int(window[:-1])
    return dates[-length:] if len(dates) >= length else None


def estimate_windows(weekly, features, selected, as_of):
    """Use identical eligible merchants and fixed selections for every available lookback."""
    features = features.sort_values("merchant_abn")
    ids = features.merchant_abn.to_numpy()
    chosen = selected.merchant_abn.to_numpy()
    rows, availability = [], []
    for window in WINDOWS:
        dates = history_dates(weekly, as_of, window)
        availability.append(dict(as_of=as_of, window=window, available=dates is not None,
                                 reason="available" if dates is not None else "fewer_than_52_complete_weeks"))
        if dates is None:
            continue
        history = weekly.loc[weekly.week_end.isin(dates)]
        require(not history.duplicated(["merchant_abn", "week_end"]).any(), "Duplicate merchant-week")
        wide = history.pivot(index="week_end", columns="merchant_abn", values="gross_fees").reindex(index=dates, columns=ids)
        require(wide.notna().all().all(), "Incomplete merchant-week history")
        model = cf.estimate_covariance(wide.to_numpy(), ids, features.eligible.to_numpy())
        observed = float(wide[chosen].sum(axis=1).std(ddof=1))
        sample = cf.portfolio_volatility(model, chosen, "sample")
        require(np.isclose(observed, sample, rtol=1e-10, atol=1e-7), "Direct and covariance SD differ")
        for estimator in ["sample", "shrunk"]:
            rows.append(dict(as_of=as_of, window=window, estimator=estimator, history_weeks=len(dates),
                history_start=dates[0]-pd.Timedelta(days=6), history_end=dates[-1],
                estimated_weekly_sd=cf.portfolio_volatility(model, chosen, estimator), shrinkage_alpha=model.alpha,
                eligible_merchants=int(features.eligible.sum()), selected_merchants=len(chosen),
                selected_ids_digest=digest(selected[["merchant_abn"]].reset_index(drop=True)),
                expected_monthly_fees=float(selected.expected_monthly_fees.sum())))
    return pd.DataFrame(rows), pd.DataFrame(availability)


def future_outcome(weekly, selected, as_of, coverage_end):
    """Measure the fixed portfolio in whole weeks strictly inside the future quarter."""
    end = as_of + pd.offsets.MonthEnd(3)
    row = dict(as_of=as_of, future_start=as_of+pd.Timedelta(days=1), future_end=end,
               future_weeks=0, actual_weekly_sd=np.nan, actual_weekly_mean=np.nan,
               status="insufficient_complete_outcome_period")
    if end > coverage_end:
        return row
    dates = pd.date_range(as_of + pd.Timedelta(days=1), end, freq="W-SUN")
    dates = dates[dates - pd.Timedelta(days=6) > as_of]
    data = weekly.loc[weekly.week_end.isin(dates) & weekly.merchant_abn.isin(selected.merchant_abn)]
    require(not data.duplicated(["merchant_abn", "week_end"]).any()
            and len(data) == len(dates) * len(selected), "Incomplete future merchant weeks")
    totals = data.groupby("week_end").gross_fees.sum().reindex(dates)
    require(totals.notna().all(), "Missing future weeks")
    row.update(future_weeks=len(dates), actual_weekly_sd=float(totals.std(ddof=1)),
               actual_weekly_mean=float(totals.mean()), status="observed")
    return row


def score_estimates(estimates, outcomes):
    """Attach outcomes only after estimates are frozen; unavailable outcomes stay missing."""
    scored = estimates.merge(outcomes, on="as_of", validate="many_to_one")
    scored["signed_error"] = scored.estimated_weekly_sd - scored.actual_weekly_sd
    scored["absolute_error"] = scored.signed_error.abs()
    scored["underestimate_amount"] = (-scored.signed_error).clip(lower=0)
    scored["underestimated"] = scored.signed_error.lt(0).where(scored.status.eq("observed"), pd.NA).astype("boolean")
    return scored


def common_origins(availability, outcomes, windows):
    """Determine matched cutoffs from source availability and outcome coverage, not values."""
    available = availability.loc[availability.window.isin(windows)].pivot(index="as_of", columns="window", values="available")
    observed = set(outcomes.loc[outcomes.status.eq("observed"), "as_of"])
    return pd.DatetimeIndex([o for o in available.index if available.loc[o, windows].all() and o in observed])


def summarize(scored, origins, windows):
    """Equal-origin errors and underestimation, paired within each covariance estimator."""
    data = scored.loc[scored.as_of.isin(origins) & scored.window.isin(windows)].copy()
    require(data.status.eq("observed").all(), "Unscored origin in comparison")
    rows = []
    for estimator, method in data.groupby("estimator"):
        base = method.loc[method.window.eq("26w")].set_index("as_of").absolute_error
        for window in windows:
            group = method.loc[method.window.eq(window)].set_index("as_of").sort_index()
            require(group.index.equals(pd.DatetimeIndex(origins).sort_values()), "Unmatched comparison dates")
            errors = group.absolute_error
            baseline = base.reindex(group.index)
            ties = np.isclose(errors, baseline, rtol=1e-10, atol=1e-7)
            rows.append(dict(estimator=estimator, window=window, origins=len(group),
                mean_absolute_error=float(errors.mean()), mean_signed_error=float(group.signed_error.mean()),
                underestimate_count=int(group.underestimated.sum()),
                mean_underestimate_amount=float(group.underestimate_amount.mean()),
                worst_underestimate_amount=float(group.underestimate_amount.max()),
                wins_vs_26w=int(((errors < baseline) & ~ties).sum()), ties_vs_26w=int(ties.sum()),
                mae_improvement_vs_26w_pct=100*(1-errors.mean()/baseline.mean()) if baseline.mean() > 0 else np.nan))
    return pd.DataFrame(rows)


def report(primary, secondary, scored):
    """Write the observed comparison and limitations without silently replacing the baseline."""
    lines = ["# Cash-flow variability: 26 weeks, 52 weeks or all history?", "",
        "The primary comparison uses the same five cutoffs, February-June 2022, and the same revenue-only top "
        "100 at each cutoff. The target is sample SD of their later weekly gross fees in complete weeks within "
        "the following three calendar months. Lower error means closer to this later observed statistic, "
        "not proven accuracy for an unobserved population risk parameter.", "",
        "## Primary matched comparison", "",
        "| Covariance | History | Origins | Mean absolute SD error (AUD) | Mean signed error (AUD) | Underestimates | Mean underestimation (AUD) | Wins vs 26 weeks |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in primary.itertuples():
        lines.append(f"| {r.estimator} | {LABELS[r.window]} | {r.origins} | {r.mean_absolute_error:,.2f} | "
            f"{r.mean_signed_error:+,.2f} | {r.underestimate_count}/{r.origins} | {r.mean_underestimate_amount:,.2f} | {r.wins_vs_26w} |")
    lines += ["", "Mean underestimation includes zero for origins that overestimate risk. Positive signed error "
        "means overestimation. The target is identical across windows; raw and shrinkage methods are compared separately.", ""]
    for estimator, group in primary.groupby("estimator"):
        winner = group.sort_values("mean_absolute_error", kind="stable").iloc[0]
        lines.append(f"For {estimator} covariance, the lowest observed mean absolute error is **{LABELS[winner.window]}** "
                     f"(A${winner.mean_absolute_error:,.2f}). This is a descriptive result from these five dates.")
    common = scored.loc[scored.as_of.between("2022-02-28", "2022-06-30")]
    if common.signed_error.gt(0).all():
        lines += ["", "Every method overestimated later SD at all five common cutoffs. Longer histories reduced "
            "average error, but did not resolve the substantial overestimation. With shrinkage, 26 weeks was "
            "closer than both longer windows in May and June; conclusions also differ across the non-overlapping "
            "schedules. There is no consistently best lookback across all dates and estimators.", "",
            "Treat 52 weeks as the lowest-MAE shrinkage candidate in this comparison and all history as the "
            "lowest-MAE raw-covariance candidate. Keep the existing 26-week feature as the documented baseline "
            "and report the alternatives as sensitivity results; this evidence does not establish an optimal window."]
    lines += ["", "## Secondary comparison on all eight earlier cutoffs", "",
        "52 weeks are unavailable in November 2021-January 2022. Compare only 26 weeks and all history here; "
        "do not rank these eight-origin averages against the five-origin averages above.", "",
        "| Covariance | History | Mean absolute SD error (AUD) | Underestimates | Mean underestimation (AUD) | Wins vs 26 weeks |",
        "|---|---|---:|---:|---:|---:|"]
    for r in secondary.itertuples():
        lines.append(f"| {r.estimator} | {LABELS[r.window]} | {r.mean_absolute_error:,.2f} | "
                     f"{r.underestimate_count}/{r.origins} | {r.mean_underestimate_amount:,.2f} | {r.wins_vs_26w} |")
    lines += ["", "## Every scored origin", "",
        "| Cutoff | History | Past weeks | Future weeks | Later SD (AUD) | Raw estimate (AUD) | Shrunk estimate (AUD) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for (as_of, window), group in scored.loc[scored.status.eq("observed")].groupby(["as_of", "window"]):
        raw = group.loc[group.estimator.eq("sample")].iloc[0]
        shrunk = group.loc[group.estimator.eq("shrunk")].iloc[0]
        lines.append(f"| {as_of:%Y-%m-%d} | {LABELS[window]} | {raw.history_weeks} | {raw.future_weeks} | "
                     f"{raw.actual_weekly_sd:,.2f} | {raw.estimated_weekly_sd:,.2f} | {shrunk.estimated_weekly_sd:,.2f} |")
    lines += ["", "## Interpretation and limits", "",
        "All-history means complete weeks beginning 1 March 2021, ending before or on each cutoff. It ranges "
        "from 52 to 69 weeks in the primary comparison. At February 2022, 52 weeks and all history are exactly "
        "the same observations; they are not independent alternatives at that date.", "",
        "The three schedules of non-overlapping future quarters are exported separately. They contain only "
        "two, two and one primary origins; the five overlapping results are not five independent macroeconomic "
        "replications. Every future SD itself uses only 11-13 weeks and is noisy. Synthetic data, retained trends "
        "and calendar effects, and previously examined outcomes limit interpretation. There is no fresh holdout, "
        "statistical-significance claim or evidence of improved portfolio selection.", "",
        "September 2022 estimates are exported for reference but have no complete future quarter and are excluded "
        "from all error summaries. Expected return, eligibility and merchant selection stay fixed. This experiment "
        "does not overwrite the existing 26-week feature or optimize a portfolio.", "",
        "Reproduce: `python code/cash_flow_lookback_comparison.py`. Tests: "
        "`python -m pytest -p no:capture code/tests/test_cash_flow_lookback_comparison.py -q`. "
        "The notebook 05_cash_flow_lookback_comparison.ipynb explains the comparison. Outputs are ignored "
        "uncompressed Parquet and CSV under tables/analysis/cash_flow_lookback_comparison."]
    (OUT / "findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    """Freeze the comparison, audit inputs, estimate all histories, then attach later outcomes."""
    previous = json.loads((OUT / "acceptance_checks.json").read_text()) if (OUT / "acceptance_checks.json").exists() else None
    daily, monthly, merchants, audit = cf.load_audited_inputs()
    weekly = cf.weekly_panel(daily, merchants)
    protocol = dict(cutoffs=cf.CUTOFFS.strftime("%Y-%m-%d").tolist(), windows=WINDOWS, ddof=1,
        primary="All-three-available cutoffs with a complete future quarter; determined without outcome values",
        secondary="26w vs all on all eight original evaluation origins",
        target="SD of total fees from the fixed top 100 over full weeks in next three calendar months",
        primary_metric="Equal-origin mean absolute error in weekly portfolio SD (AUD)",
        diagnostics=["signed error", "underestimation frequency and dollar gap", "paired wins", "non-overlapping schedules"],
        estimators=["sample", "shrunk"], shrinkage="Unchanged standardized Ledoit-Wolf diagonal-preserving estimator, refit for each history",
        selection="Existing past-only eligibility and expanding-mean monthly gross-fee top 100; ascending ABN ties",
        changes_to_existing_baseline=False, retrospective=True, untouched_holdout=False,
        inputs=dict(daily=audit["daily_sha256"], monthly=digest(monthly), merchants=digest(merchants)),
        versions={p:importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "scikit-learn"]})
    if (OUT / "protocol.json").exists():
        require(json.loads((OUT / "protocol.json").read_text()) == protocol, "Changed lookback protocol or inputs")
    save_json(OUT / "protocol.json", protocol)
    original_selection = pd.read_parquet(cf.OUT / "baseline_selections.parquet")
    records, availability, selections, checks = [], [], [], []
    for as_of in cf.CUTOFFS:
        print(f"Estimating all lookbacks at {as_of.date()}", flush=True)
        features, _ = cf.features_at_cutoff(weekly, monthly, merchants, as_of)
        selected = cf.baseline_selection(features)
        saved = original_selection.loc[original_selection.as_of.eq(as_of)].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, saved, check_exact=True)
        estimated, available = estimate_windows(weekly, features, selected, as_of)
        require(estimated.selected_ids_digest.nunique() == 1 and estimated.expected_monthly_fees.nunique() == 1,
                "Selection or expected return changed between lookbacks")
        if as_of in ORIGINS:
            perturbed = daily.copy()
            perturbed.loc[perturbed.day.gt(as_of), "sales"] *= 1000
            modified_monthly = monthly.copy()
            modified_monthly.loc[modified_monthly.month.gt(as_of), "sales"] *= 1000
            altered_weekly = cf.weekly_panel(perturbed, merchants)
            altered_features, _ = cf.features_at_cutoff(altered_weekly, modified_monthly, merchants, as_of)
            altered_selection = cf.baseline_selection(altered_features)
            pd.testing.assert_frame_equal(selected, altered_selection, check_exact=True)
            again, availability_again = estimate_windows(altered_weekly, altered_features, altered_selection, as_of)
            pd.testing.assert_frame_equal(estimated, again, check_exact=True)
            pd.testing.assert_frame_equal(available, availability_again, check_exact=True)
        records.append(estimated)
        availability.append(available)
        selections.append(selected)
        checks.append(dict(as_of=as_of, baseline_reproduced=True, identical_selection_across_windows=True,
                           covariance_reconciled=True, future_transaction_perturbation=as_of in ORIGINS))
    estimates = pd.concat(records, ignore_index=True)
    available = pd.concat(availability, ignore_index=True)
    selection = pd.concat(selections, ignore_index=True)
    write_table(estimates, OUT, "unscored_estimates")
    write_table(available, OUT, "availability")
    write_table(selection[["as_of", "merchant_abn", "rank", "expected_monthly_fees"]], OUT, "baseline_selections")
    save_json(OUT / "estimate_freeze.json", dict(estimates=digest(estimates), selection=digest(selection),
        saved_at=pd.Timestamp.now(tz="UTC"), stage="All lookback estimates saved before attaching future outcomes"))
    outcomes = pd.DataFrame([future_outcome(weekly, selected, as_of, daily.day.max())
                            for as_of, selected in selection.groupby("as_of")])
    old = pd.read_parquet(cf.OUT / "portfolio_diagnostics.parquet").set_index("as_of")
    for as_of, group in estimates.loc[estimates.window.eq("26w")].groupby("as_of"):
        np.testing.assert_allclose(group.set_index("estimator").estimated_weekly_sd.loc[["sample", "shrunk"]],
            old.loc[as_of, ["history_weekly_sd", "regularized_weekly_sd"]].to_numpy(dtype=float), rtol=1e-12, atol=1e-7)
    for row in outcomes.loc[outcomes.status.eq("observed")].itertuples():
        require(row.future_weeks == old.loc[row.as_of, "future_weeks"], "Future week count changed")
        require(np.isclose(row.actual_weekly_sd, old.loc[row.as_of, "future_weekly_sd"], rtol=1e-12), "Future SD changed")
        ids = selection.loc[selection.as_of.eq(row.as_of), "merchant_abn"]
        raw = daily.loc[daily.merchant_abn.isin(ids)].copy()
        raw["week_end"] = raw.day.dt.to_period("W-SUN").dt.end_time.dt.normalize()
        raw = raw.loc[(raw.week_end-pd.Timedelta(days=6)).gt(row.as_of) & raw.week_end.le(row.future_end)]
        rates = merchants.set_index("merchant_abn").take_rate / 100
        direct = (raw.sales * raw.merchant_abn.map(rates)).groupby(raw.week_end).sum()
        require(len(direct) == row.future_weeks and np.isclose(direct.std(ddof=1), row.actual_weekly_sd, rtol=1e-12),
                "Independent daily future outcome differs")
    scored = score_estimates(estimates, outcomes)
    primary_dates = common_origins(available, outcomes, WINDOWS)
    secondary_dates = common_origins(available, outcomes, ["26w", "all"])
    require(primary_dates.equals(pd.date_range("2022-02-28", "2022-06-30", freq="ME")), "Primary dates changed")
    require(secondary_dates.equals(ORIGINS), "Secondary dates changed")
    primary = summarize(scored, primary_dates, WINDOWS)
    secondary = summarize(scored, secondary_dates, ["26w", "all"])
    schedules = []
    for schedule in range(3):
        dates = pd.DatetimeIndex([o for o in primary_dates if ((o.year-2021)*12+o.month-11) % 3 == schedule])
        schedules.append(summarize(scored, dates, WINDOWS).assign(schedule=schedule,
            cutoffs=", ".join(dates.strftime("%Y-%m-%d"))))
    schedules = pd.concat(schedules, ignore_index=True)
    signatures = dict(estimates=digest(estimates), scored=digest(scored), primary=digest(primary), secondary=digest(secondary))
    if previous:
        require(previous["digests"] == signatures, "Lookback rerun changed results")
    for frame, name in [(outcomes,"outcomes"), (scored,"fold_metrics"), (primary,"summary_common_five"),
                        (secondary,"summary_eight_26w_vs_all"), (schedules,"nonoverlapping_schedules"),
                        (pd.DataFrame(checks),"cutoff_checks")]:
        write_table(frame, OUT, name)
    save_json(OUT / "acceptance_checks.json", dict(audited_inputs_and_complete_weeks=True,
        baseline_reproductions=9, unchanged_26w_estimates=9, independent_future_outcomes=8,
        future_transaction_perturbations=8, matched_primary_origins=5, matched_secondary_origins=8,
        reproducible_rerun=previous is not None, digests=signatures))
    report(primary, secondary, scored)
    print("PRIMARY: identical five origins\n"+primary.round(3).to_string(index=False), flush=True)
    print("SECONDARY: identical eight origins\n"+secondary.round(3).to_string(index=False), flush=True)
    return scored, primary, secondary


if __name__ == "__main__":
    run()
