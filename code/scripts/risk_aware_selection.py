"""Select exactly 100 using covariance risk and a 99% expected-fee floor."""

import importlib.metadata
import json

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array
from threadpoolctl import threadpool_limits

import cash_flow_variability as cf
import cash_flow_risk_comparison as risk
from cash_flow_lookback_comparison import history_dates
from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from forecast_comparison import ORIGINS, digest

OUT = cf.OUT.parent / "risk_aware_selection"
WINDOWS = ["26w", "52w"]
METHODS = ["revenue_only", "raw_risk", "detrended_risk"]
RETENTION = .99
SOLVER = dict(max_cuts=30, node_limit=200, relative_gap=.001, max_swaps=200)


def binary_objective(factors, diagonal, z):
    """Convex extension ||Bz||^2 + d'z; equals shrunk variance for binary selections."""
    total = factors @ z
    return float(total @ total + diagonal @ z)


def improve_swaps(factors, diagonal, means, initial, floor, max_swaps=200):
    """Deterministic best one-for-one exchanges; all eligible alternatives are examined."""
    z = np.asarray(initial, dtype=float).copy()
    norms = np.sum(factors**2, axis=0)
    value = binary_objective(factors, diagonal, z)
    for iteration in range(max_swaps):
        inside, outside = np.flatnonzero(z), np.flatnonzero(1-z)
        if len(outside) == 0:
            return z, iteration, True
        gradient = 2*factors.T@(factors@z)+diagonal
        cross = factors[:, inside].T@factors[:, outside]
        delta = (gradient[outside][None, :]-gradient[inside][:, None]
                 +norms[inside][:, None]+norms[outside][None, :]-2*cross)
        feasible = means@z-means[inside][:, None]+means[outside][None, :] >= floor-1e-10
        delta[~feasible] = np.inf
        i, j = np.unravel_index(np.argmin(delta), delta.shape)
        if delta[i, j] >= -1e-10*max(1., value):
            return z, iteration, True
        z[inside[i]], z[outside[j]] = 0., 1.
        new_value = binary_objective(factors, diagonal, z)
        require(new_value <= value+1e-9, "Exchange increased objective")
        value = new_value
    return z, max_swaps, False


def optimize_portfolio(model, means, count=100, retention=RETENTION, settings=None):
    """Outer-approximation MILPs give feasible portfolios and valid global lower bounds.

    For binary z, z_i^2=z_i. Thus shrunk variance is ||Bz||^2+d'z, with
    B=sqrt(1-alpha)*centered/sqrt(n-1) and d=alpha*sample_variances.
    Tangent planes lower-bound the convex squared norm. The MILP dual bound
    therefore bounds the true binary quadratic optimum; its incumbent alone
    is NOT a certificate of optimality for the quadratic problem.
    """
    settings = SOLVER if settings is None else settings
    means = np.asarray(means, dtype=float)
    require(len(means) == len(model.ids) and len(means) >= count and count >= 1, "Invalid selection dimensions")
    require(np.isfinite(means).all() and (means >= 0).all() and 0 < retention <= 1, "Invalid fee inputs")
    require(pd.Index(model.ids).is_unique and np.all(np.diff(model.ids) > 0), "ABNs must be unique and sorted")
    baseline = np.zeros(len(means))
    baseline[np.lexsort((model.ids, -means))[:count]] = 1
    revenue = means@baseline
    require(revenue > 0, "Positive baseline expected fees required")
    scaled_means = means/(revenue/count)
    floor = retention*count
    factors = np.sqrt(1-model.alpha)*model.centered/np.sqrt(len(model.centered)-1)
    diagonal = model.alpha*model.variances
    scale = binary_objective(factors, diagonal, baseline)
    if scale <= 1e-20:
        return baseline, dict(variance=scale, lower_bound=0., gap_upper_fraction=0., certified=True,
            cuts=0, status="zero_risk_baseline", swaps=0), pd.DataFrame()
    factors, diagonal = factors/np.sqrt(scale), diagonal/scale
    with threadpool_limits(limits=1):
        best, swaps, _ = improve_swaps(factors, diagonal, scaled_means, baseline, floor, settings["max_swaps"])
        upper = binary_objective(factors, diagonal, best)
        lower = 0.
        planes, rhs, seen, logs = [], [], set(), []
        for point in [baseline, best]:
            key = tuple(np.flatnonzero(point))
            if key not in seen:
                seen.add(key)
                projected = factors@point
                planes.append(np.r_[2*factors.T@projected, -1.])
                rhs.append(float(projected@projected))
        for iteration in range(settings["max_cuts"]):
            matrix = np.vstack([np.r_[np.ones(len(means)), 0.], np.r_[scaled_means, 0.], *planes])
            constraints = LinearConstraint(csc_array(matrix),
                np.r_[count, floor, np.full(len(planes), -np.inf)], np.r_[count, np.inf, rhs])
            result = milp(np.r_[diagonal, 1.], integrality=np.r_[np.ones(len(means)), 0],
                bounds=Bounds(np.zeros(len(means)+1), np.r_[np.ones(len(means)), np.inf]),
                constraints=constraints, options=dict(node_limit=settings["node_limit"], mip_rel_gap=1e-6, presolve=True))
            # SciPy 1.16 can expose HiGHS' solution-budget stop as status 4.
            # It is a resource limit, not infeasibility or an optimality certificate.
            solution_limit = "HiGHS Status 16: Solution limit reached" in result.message
            require(result.status in [0, 1] or solution_limit, f"MILP failed: {result.message}")
            bound = getattr(result, "mip_dual_bound", None)
            if bound is not None and np.isfinite(bound):
                lower = max(lower, float(bound))
            candidate = None
            candidate_value = np.nan
            if result.x is not None:
                candidate = np.rint(result.x[:-1])
                require(np.max(np.abs(candidate-result.x[:-1])) < 1e-5, "Nonintegral MILP incumbent")
                require(candidate.sum() == count and scaled_means@candidate >= floor-1e-8, "Infeasible MILP incumbent")
                candidate_value = binary_objective(factors, diagonal, candidate)
                polished, _, _ = improve_swaps(factors, diagonal, scaled_means, candidate, floor, settings["max_swaps"])
                polished_value = binary_objective(factors, diagonal, polished)
                if polished_value < upper-1e-10 or (abs(polished_value-upper) <= 1e-10
                        and tuple(np.flatnonzero(polished)) < tuple(np.flatnonzero(best))):
                    best, upper = polished, polished_value
            require(lower <= upper+1e-6, "Invalid optimization lower bound")
            gap = max(0., (upper-lower)/max(upper, 1e-15))
            logs.append(dict(iteration=iteration, milp_status=result.status,
                milp_nodes=getattr(result, "mip_node_count", None), lower_bound=lower*scale,
                best_variance=upper*scale, candidate_variance=candidate_value*scale, gap_upper_fraction=gap))
            if gap <= settings["relative_gap"]:
                break
            if candidate is None:
                break
            added = False
            for point in [candidate, best]:
                key = tuple(np.flatnonzero(point))
                if key not in seen:
                    seen.add(key)
                    projected = factors@point
                    planes.append(np.r_[2*factors.T@projected, -1.])
                    rhs.append(float(projected@projected))
                    added = True
            if not added:
                break
    require(best.sum() == count and means@best >= retention*revenue-1e-6, "Final selection violates constraints")
    value = binary_objective(factors, diagonal, best)*scale
    require(value <= scale+1e-6, "Optimized in-sample risk is worse than feasible baseline")
    gap = max(0., (value-lower*scale)/max(value, 1e-15))
    return best, dict(variance=value, lower_bound=lower*scale, gap_upper_fraction=gap,
        certified=gap <= settings["relative_gap"], cuts=len(planes), swaps=swaps,
        status="certified_gap" if gap <= settings["relative_gap"] else "bounded_best_found"), pd.DataFrame(logs)


def models_at(weekly, features, as_of, window):
    """Estimate on all eligible histories, without an arbitrary candidate shortlist."""
    dates = history_dates(weekly, as_of, window)
    require(dates is not None, "Unavailable lookback")
    f = features.loc[features.eligible].sort_values("merchant_abn").reset_index(drop=True)
    x = risk.history_matrix(weekly, dates, f.merchant_abn.to_numpy())
    _, _, residual = risk.decompose(x)
    args = (f.merchant_abn.to_numpy(), np.ones(len(f), dtype=bool))
    return f, dates, x, {
        "raw_risk": cf.estimate_covariance(x, *args),
        "detrended_risk": cf.estimate_covariance(residual*np.sqrt((len(dates)-1)/(len(dates)-2)), *args)}


def select_at(weekly, features, as_of, window):
    """Return frozen estimates, selected merchants and numerical-search evidence."""
    f, dates, x, models = models_at(weekly, features, as_of, window)
    baseline = cf.baseline_selection(f)
    baseline_ids = set(baseline.merchant_abn)
    baseline_expected = float(baseline.expected_monthly_fees.sum())
    dates26 = cf.risk_window(as_of)
    fees26 = risk.history_matrix(weekly, dates26, f.merchant_abn.to_numpy())
    reference = float(fees26[:, f.merchant_abn.isin(baseline_ids)].sum(axis=1).mean())
    estimates, selections, logs = [], [], []
    for method in METHODS:
        if method == "revenue_only":
            z = f.merchant_abn.isin(baseline_ids).to_numpy(dtype=float)
            info = dict(variance=np.nan, lower_bound=np.nan, gap_upper_fraction=np.nan,
                        certified=False, cuts=0, swaps=0, status="revenue_baseline")
        else:
            z, info, log = optimize_portfolio(models[method], f.expected_monthly_fees.to_numpy())
            if len(log):
                logs.append(log.assign(as_of=as_of, window=window, method=method))
        info["solver_status"] = info.pop("status")
        chosen = f.loc[z.astype(bool)].sort_values(["expected_monthly_fees", "merchant_abn"], ascending=[False, True]).copy()
        require(len(chosen) == 100 and chosen.merchant_abn.nunique() == 100 and chosen.eligible.all(), "Invalid top 100")
        chosen["rank_by_expected_fees"] = np.arange(1, 101)
        chosen["as_of"], chosen["window"], chosen["method"] = as_of, window, method
        chosen["in_revenue_baseline"] = chosen.merchant_abn.isin(baseline_ids)
        selections.append(chosen)
        history = x[:, z.astype(bool)].sum(axis=1)
        stats, _, _ = risk.decompose(history[:, None])
        raw_sd = cf.portfolio_volatility(models["raw_risk"], chosen.merchant_abn, "sample")
        residual_sd = cf.portfolio_volatility(models["detrended_risk"], chosen.merchant_abn, "sample")
        np.testing.assert_allclose([raw_sd, residual_sd], stats.loc[0, ["raw_sd", "residual_sd"]].to_numpy(dtype=float), rtol=1e-10, atol=1e-7)
        row = dict(as_of=as_of, window=window, method=method, history_weeks=len(dates),
            history_start=dates[0]-pd.Timedelta(days=6), history_end=dates[-1],
            eligible_merchants=len(f), expected_monthly_fees=float(chosen.expected_monthly_fees.sum()),
            baseline_expected_monthly_fees=baseline_expected,
            expected_fee_retention_pct=100*chosen.expected_monthly_fees.sum()/baseline_expected,
            common_reference_weekly_fees=reference, own_reference_weekly_fees=float(fees26[:, z.astype(bool)].sum(axis=1).mean()),
            selected_merchants=len(chosen), overlap_with_baseline=int(chosen.in_revenue_baseline.sum()),
            estimated_raw_sd=raw_sd, estimated_residual_sd=residual_sd,
            estimated_raw_shrunk_sd=cf.portfolio_volatility(models["raw_risk"], chosen.merchant_abn, "shrunk"),
            estimated_residual_shrunk_sd=cf.portfolio_volatility(models["detrended_risk"], chosen.merchant_abn, "shrunk"),
            raw_alpha=models["raw_risk"].alpha, residual_alpha=models["detrended_risk"].alpha,
            trend_slope_per_week=float(stats.trend_slope_per_week.iloc[0]),
            raw_covariance_digest=digest(pd.DataFrame(models["raw_risk"].centered)),
            residual_covariance_digest=digest(pd.DataFrame(models["detrended_risk"].centered)), **info)
        if method != "revenue_only":
            sd = row["estimated_raw_shrunk_sd" if method == "raw_risk" else "estimated_residual_shrunk_sd"]
            np.testing.assert_allclose(sd**2, info["variance"], rtol=1e-10, atol=1e-6)
        require(row["expected_fee_retention_pct"] >= 100*RETENTION-1e-8, "Expected-fee guardrail failed")
        estimates.append(row)
    return pd.DataFrame(estimates), pd.concat(selections, ignore_index=True), pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def portfolio_outcome(weekly, monthly, daily, merchants, estimate, selection):
    """Full-quarter fees plus common-week downside; verify fees from independent daily sums."""
    as_of = estimate["as_of"]
    key = {k: estimate[k] for k in ["as_of", "window", "method"]}
    dates = risk.future_dates(as_of, daily.day.max())
    if dates is None:
        return dict(**key, status="unavailable", future_weeks=0), pd.DataFrame()
    ids = selection.merchant_abn.to_numpy()
    x = risk.history_matrix(weekly, dates, ids).sum(axis=1)
    reference, own = estimate["common_reference_weekly_fees"], estimate["own_reference_weekly_fees"]
    losses = np.maximum(reference-x, 0)
    end = as_of+pd.offsets.MonthEnd(3)
    months = monthly.loc[monthly.merchant_abn.isin(ids) & monthly.month.gt(as_of) & monthly.month.le(end)]
    require(len(months) == 300, "Incomplete selected-merchant quarter")
    rates = merchants.set_index("merchant_abn").take_rate/100
    actual_fees = float((months.sales*months.merchant_abn.map(rates)).sum())
    days = daily.loc[daily.merchant_abn.isin(ids) & daily.day.gt(as_of) & daily.day.le(end)].copy()
    direct_fees = days.sales*days.merchant_abn.map(rates)
    np.testing.assert_allclose(actual_fees, direct_fees.sum(), rtol=1e-12, atol=1e-7)
    week_ends = days.day.dt.to_period("W-SUN").dt.end_time.dt.normalize()
    direct_weeks = direct_fees.groupby(week_ends).sum().reindex(dates)
    np.testing.assert_allclose(x, direct_weeks.to_numpy(), rtol=1e-12, atol=1e-7)
    stats, _, _ = risk.decompose(x[:, None])
    result = dict(**key, status="observed", future_weeks=len(x), actual_quarter_fees=actual_fees,
        actual_weekly_mean=float(x.mean()), actual_weekly_sd=float(x.std(ddof=1)),
        actual_weekly_cv=float(x.std(ddof=1)/x.mean()), actual_residual_sd=float(stats.residual_sd.iloc[0]),
        downside_common=float(risk.downside(x, reference)), downside_own=float(risk.downside(x, own)),
        mean_shortfall=float(losses.mean()), worst_shortfall=float(losses.max()),
        below_reference_weeks=int((x < reference).sum()), worst_week_fees=float(x.min()))
    return result, pd.DataFrame(dict(**key, week_end=dates, gross_fees=x,
        common_reference=reference, own_reference=own, shortfall=losses))


def compare_outcomes(estimates, outcomes):
    joined = estimates.merge(outcomes, on=["as_of", "window", "method"], validate="one_to_one")
    metrics = ["actual_quarter_fees", "actual_weekly_sd", "actual_weekly_cv", "downside_common", "downside_own"]
    baseline = joined.loc[joined.method.eq("revenue_only"), ["as_of", "window", *metrics]].rename(columns={c:"baseline_"+c for c in metrics})
    paired = joined.merge(baseline, on=["as_of", "window"], validate="many_to_one")
    paired["actual_fee_change_pct"] = 100*(paired.actual_quarter_fees/paired.baseline_actual_quarter_fees-1)
    paired["actual_fee_change_aud"] = paired.actual_quarter_fees-paired.baseline_actual_quarter_fees
    paired["actual_sd_change_pct"] = 100*(paired.actual_weekly_sd/paired.baseline_actual_weekly_sd-1)
    paired["actual_cv_change_pct"] = 100*(paired.actual_weekly_cv/paired.baseline_actual_weekly_cv-1)
    paired["downside_change_aud"] = paired.downside_common-paired.baseline_downside_common
    paired["downside_change_reference_pct"] = 100*paired.downside_change_aud/paired.common_reference_weekly_fees
    paired["downside_own_change_aud"] = paired.downside_own-paired.baseline_downside_own
    # No undefined relative percentage when the baseline downside is zero.
    paired["downside_change_pct"] = 100*paired.downside_change_aud/paired.baseline_downside_common.where(paired.baseline_downside_common > 0)
    paired["actual_fee_guardrail_met"] = paired.actual_fee_change_pct.ge(-1.-1e-8).where(paired.status.eq("observed"), pd.NA).astype("boolean")
    paired["downside_lower"] = paired.downside_change_aud.lt(-1e-7).where(paired.status.eq("observed"), pd.NA).astype("boolean")
    paired["downside_tied"] = paired.downside_change_aud.abs().le(1e-7).where(paired.status.eq("observed"), pd.NA).astype("boolean")
    return paired


def summarize(paired, dates, windows, label):
    rows = []
    for window in windows:
        for method in METHODS[1:]:
            g = paired.loc[paired.as_of.isin(dates) & paired.window.eq(window) & paired.method.eq(method)]
            require(set(g.as_of) == set(dates) and g.status.eq("observed").all(), "Unmatched outcome comparison")
            rows.append(dict(comparison=label, window=window, method=method, origins=len(g),
                mean_expected_fee_retention_pct=g.expected_fee_retention_pct.mean(),
                mean_actual_fee_change_pct=g.actual_fee_change_pct.mean(), mean_actual_fee_change_aud=g.actual_fee_change_aud.mean(),
                worst_actual_fee_change_pct=g.actual_fee_change_pct.min(), fee_guardrail_passes=int(g.actual_fee_guardrail_met.sum()),
                mean_actual_sd_change_pct=g.actual_sd_change_pct.mean(), mean_actual_cv_change_pct=g.actual_cv_change_pct.mean(),
                mean_downside_change_aud=g.downside_change_aud.mean(), mean_downside_change_reference_pct=g.downside_change_reference_pct.mean(),
                mean_own_downside_change_aud=g.downside_own_change_aud.mean(), downside_wins=int(g.downside_lower.sum()),
                downside_ties=int(g.downside_tied.sum()), downside_losses=int((~g.downside_lower & ~g.downside_tied).sum()),
                joint_successes=int((g.downside_lower & g.actual_fee_guardrail_met).sum()),
                mean_overlap=g.overlap_with_baseline.mean(), max_optimization_gap_pct=100*g.gap_upper_fraction.max()))
    return pd.DataFrame(rows)


def report(paired, summary, estimates):
    primary = summary.loc[summary.comparison.eq("primary_26w_eight")]
    lines = ["# Does covariance risk improve merchant selection?", "",
        "The experiment selects exactly 100 eligible merchants and retains at least 99% of the revenue-only "
        "portfolio's expected monthly gross fees. Expected return remains the expanding historical complete-month "
        "mean. The revenue-only selection, raw-fee covariance selection and detrended covariance selection all "
        "use the same candidate pool and dates. Both risk methods use the existing variance-preserving Ledoit-Wolf "
        "shrinkage; 'raw' means fees are not detrended, not that covariance is unregularized.", "",
        "Primary: eight November 2021-June 2022 cutoffs with 26 historical weeks. Sensitivity: 26 vs 52 weeks "
        "on the same five February-June dates. September 2022 selections are unscored snapshots. Settings and "
        "all selections were frozen before attaching outcomes. There is no new untouched holdout.", "",
        "## Primary results", "",
        "All changes below are candidate minus revenue-only baseline. Negative downside and SD changes are "
        "improvements; negative fee changes are revenue sacrificed. Means give each origin equal weight.", "",
        "| Risk objective | Fee change | SD change | Downside change (A$/week RMS) | Downside wins/ties/losses | Realized 99% fee guardrail | Both improve downside and pass guardrail |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in primary.itertuples():
        lines.append(f"| {r.method} | {r.mean_actual_fee_change_pct:+.3f}% | {r.mean_actual_sd_change_pct:+.3f}% | "
            f"{r.mean_downside_change_aud:+,.2f} | {r.downside_wins}/{r.downside_ties}/{r.downside_losses} | "
            f"{r.fee_guardrail_passes}/{r.origins} | {r.joint_successes}/{r.origins} |")
    if primary.downside_wins.eq(0).all() and primary.mean_actual_fee_change_pct.lt(0).all():
        lines += ["", "**Decision: retain the revenue-only top 100.** Neither tested risk-aware policy improved "
            "realized downside at any primary origin, while both gave up revenue. The two zero-downside ties "
            "do not count as improvements. A lower predicted variance did not deliver the intended economic benefit.", "",
            "Using each portfolio's own income reference gives an apparently better average downside result "
            "here, but that reference falls when lower-income merchants are selected. Against the common baseline "
            "income requirement, the conclusion reverses. The common-reference comparison is the predeclared test."]
    lines += ["", "The 99% selection constraint is on EXPECTED fees. Actual next-quarter fees can violate it; "
        "these failures are reported rather than excluded. Lower estimated variance is not the same as lower "
        "realized downside. Results are for this fixed 1% revenue budget, not an optimized business trade-off.", "",
        "## Every historical primary comparison", "",
        "| Cutoff | Method | Expected retention | Actual fee change | Actual SD change | Downside change (A$) | Overlap |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in paired.loc[paired.window.eq("26w") & paired.status.eq("observed") & paired.method.ne("revenue_only")].itertuples():
        lines.append(f"| {r.as_of:%Y-%m-%d} | {r.method} | {r.expected_fee_retention_pct:.4f}% | {r.actual_fee_change_pct:+.3f}% | "
            f"{r.actual_sd_change_pct:+.3f}% | {r.downside_change_aud:+,.2f} | {r.overlap_with_baseline}/100 |")
    lines += ["", "## Fair downside evaluation", "",
        "At each cutoff, fix one common reference: the revenue-only baseline's mean gross weekly fees over "
        "the latest 26 complete weeks. Every policy, including the 52-week sensitivity, is evaluated against "
        "that SAME dollar reference. Downside is sqrt(mean(max(reference - total weekly fees, 0)^2)); sum "
        "merchants before applying the shortfall. A policy cannot look safer simply by lowering its own reference. "
        "A separate own-reference downside sensitivity is exported and must not replace this main test.", "",
        "Full-quarter fee revenue includes every day in the next three complete calendar months. Weekly risk "
        "uses only complete Monday-Sunday weeks entirely inside that quarter (11-13 observations). Risk percentages "
        "are not annualized. When baseline downside is zero, percentage downside change is undefined; report "
        "dollar differences and changes divided by the common reference instead. Volatility/mean is also reported "
        "to check whether an apparent SD reduction merely reflects lower income.", "",
        "## Numerical search", "",
        "Minimize z' Sigma z subject to sum(z)=100, z binary, and expected fees >=99% of baseline. No candidate "
        "shortlist, fraud penalty, customer-concentration rule, trend filter or category constraint is added. "
        "For binary z, shrinkage variance can be written ||Bz||^2 + d'z. A deterministic improving exchange "
        "search initializes the incumbent, then tangent-plane mixed-integer linear problems give a valid global "
        "lower bound. The recorded gap bounds the remaining possible variance improvement; it does not measure "
        "statistical uncertainty. Numerical solver inputs are sorted by numeric ABN. Baseline and display ranks "
        "break expected-fee ties by ABN. Exact equal-objective incumbents prefer the lower sorted ABN tuple.", "",
        f"Search settings: {SOLVER}. Each MILP uses a fixed node budget, not a machine-dependent time budget. "
        "All achieved gaps and solver logs are exported; budget-limited results are called best-found, not exact optima. "
        "See [SciPy's MILP documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.milp.html).", ""]
    solved = estimates.loc[estimates.method.ne("revenue_only")]
    lines.append(f"{int(solved.certified.sum())}/{len(solved)} risk portfolios meet the predeclared 0.1% variance-gap "
        f"tolerance; the largest recorded bound is {100*solved.gap_upper_fraction.max():.4f}%.")
    lines += ["", "## Interpretation", ""]
    for r in primary.itertuples():
        direction = "lower" if r.mean_downside_change_aud < 0 else "higher or unchanged"
        lines.append(f"For {r.method}, average realized downside is {direction}; downside improves in "
            f"{r.downside_wins}/8 tests and improves while retaining 99% of realized fees in {r.joint_successes}/8. "
            f"The worst fee change is {r.worst_actual_fee_change_pct:+.3f}%. These results describe the trade-off; "
            "they do not establish a universally better policy.")
    lines += ["", "Use the same-origin 52-week sensitivity and the three non-overlapping schedules to inspect "
        "stability. The existing revenue-only baseline is preserved; no method is silently promoted after seeing "
        "these already-used outcomes. Detrending can hide smooth decline, while raw variance can penalize growth. "
        "The economic test, not the in-sample risk objective, determines whether either helps here.", "",
        "This conclusion concerns these covariance estimates, this data and the fixed 1% expected-fee budget. "
        "It does not show that diversification can never help, or address fraud and customer-concentration risks.", "",
        "Assume take rates/categories were historically known and stable, and onboarding a merchant would not "
        "change its historical sales or the sales of other merchants. Gross fees exclude credit/fraud losses, "
        "funding and operating expenses, and settlement timing. Synthetic history, few economic periods and "
        "overlapping quarters limit generalization. No statistical-significance or causal claim is made.", "",
        "## Reproduce", "",
        "Run `python code/scripts/risk_aware_selection.py`; tests: "
        "`python -m pytest -p no:capture code/tests/test_risk_aware_selection.py -q`. "
        "The explanatory notebook is code/notebooks/07_risk_aware_selection.ipynb. Use existing pinned "
        "requirements-forecast.txt; no additional optimizer package, GPU or API is required. "
        "Outputs are uncompressed Parquet and CSV in ignored tables/analysis/risk_aware_selection/. "
        "Latest 26-week candidate lists are explicitly labelled by method; they are not validated September recommendations."]
    (OUT/"findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    previous = json.loads((OUT/"acceptance_checks.json").read_text()) if (OUT/"acceptance_checks.json").exists() else None
    daily, monthly, merchants, audit = cf.load_audited_inputs()
    weekly = cf.weekly_panel(daily, merchants)
    protocol = dict(cutoffs=cf.CUTOFFS.strftime("%Y-%m-%d").tolist(), windows=WINDOWS, methods=METHODS,
        retention=RETENTION, cardinality=100, expected_return="Unchanged expanding mean monthly gross fees",
        covariance="Diagonal-preserving Ledoit-Wolf, separately fitted on raw fees or n-2-adjusted OLS residuals",
        primary="26 weeks, eight November-June origins", sensitivity="26 vs 52 weeks, same five February-June origins",
        downside_reference="Revenue-only baseline mean of latest 26 complete weeks; identical dollars for every arm/window",
        downside="RMS shortfall, denominator all future complete weeks", outcome="Full three-month fees and 11-13 whole-week risks",
        realized_fee_guardrail="Report whether actual fees retain 99% of actual revenue-only baseline; never guaranteed by expected constraint",
        search=SOLVER, tuning=False, untouched_holdout=False, retrospective=True,
        inputs=dict(daily=audit["daily_sha256"], monthly=digest(monthly), merchants=digest(merchants)),
        versions={p:importlib.metadata.version(p) for p in ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "threadpoolctl"]})
    if (OUT/"protocol.json").exists():
        require(json.loads((OUT/"protocol.json").read_text()) == protocol, "Changed portfolio protocol or inputs")
    save_json(OUT/"protocol.json", protocol)
    old = pd.read_parquet(cf.OUT/"baseline_selections.parquet")
    estimates, selections, logs, checks, availability = [], [], [], [], []
    for as_of in cf.CUTOFFS:
        print(f"Selecting portfolios at {as_of.date()}", flush=True)
        f, _ = cf.features_at_cutoff(weekly, monthly, merchants, as_of)
        baseline = cf.baseline_selection(f)
        pd.testing.assert_frame_equal(baseline, old.loc[old.as_of.eq(as_of)].reset_index(drop=True), check_exact=True)
        if as_of in ORIGINS:
            changed = daily.copy()
            changed.loc[changed.day.gt(as_of), "sales"] *= 1000
            changed_monthly = monthly.copy()
            changed_monthly.loc[changed_monthly.month.gt(as_of), "sales"] *= 1000
            changed_weekly = cf.weekly_panel(changed, merchants)
            changed_f, _ = cf.features_at_cutoff(changed_weekly, changed_monthly, merchants, as_of)
            pd.testing.assert_frame_equal(f, changed_f, check_exact=True)
        for window in WINDOWS:
            available = history_dates(weekly, as_of, window) is not None
            availability.append(dict(as_of=as_of, window=window, available=available))
            if not available:
                continue
            e, s, log = select_at(weekly, f, as_of, window)
            print(f"  {window}: max variance-gap bound {100*e.gap_upper_fraction.max():.4f}%", flush=True)
            if as_of in ORIGINS:
                e2, s2, log2 = select_at(changed_weekly, changed_f, as_of, window)
                pd.testing.assert_frame_equal(e, e2, check_exact=True)
                pd.testing.assert_frame_equal(s, s2, check_exact=True)
                pd.testing.assert_frame_equal(log, log2, check_exact=True)
            estimates.append(e)
            selections.append(s)
            logs.append(log)
            checks.append(dict(as_of=as_of, window=window, baseline_reproduced=True,
                exactly_100_eligible=True, expected_fee_floor=True, covariance_variance_reconciled=True,
                future_daily_perturbation=as_of in ORIGINS))
    estimates, selections, logs = pd.concat(estimates, ignore_index=True), pd.concat(selections, ignore_index=True), pd.concat(logs, ignore_index=True)
    require(not selections.duplicated(["as_of", "window", "method", "merchant_abn"]).any(), "Duplicate selections")
    require(selections.groupby(["as_of", "window", "method"]).size().eq(100).all(), "Selection count changed")
    require(estimates.groupby("as_of").common_reference_weekly_fees.nunique().eq(1).all(), "Different downside references")
    for name, frame in [("unscored_estimates", estimates), ("selections", selections), ("solver_log", logs),
                        ("availability", pd.DataFrame(availability)), ("cutoff_checks", pd.DataFrame(checks))]:
        write_table(frame, OUT, name)
    save_json(OUT/"selection_freeze.json", dict(estimates=digest(estimates), selections=digest(selections),
        saved_at=pd.Timestamp.now(tz="UTC"), stage="Every selection and reference saved before scoring future outcomes"))
    outcomes, future_series = [], []
    for estimate in estimates.to_dict("records"):
        selected = selections.loc[selections.as_of.eq(estimate["as_of"]) & selections.window.eq(estimate["window"]) & selections.method.eq(estimate["method"])]
        o, series = portfolio_outcome(weekly, monthly, daily, merchants, estimate, selected)
        outcomes.append(o)
        if len(series):
            future_series.append(series)
    outcomes = pd.DataFrame(outcomes)
    paired = compare_outcomes(estimates, outcomes)
    common = pd.date_range("2022-02-28", "2022-06-30", freq="ME")
    summary = pd.concat([summarize(paired, ORIGINS, ["26w"], "primary_26w_eight"),
        summarize(paired, common, WINDOWS, "sensitivity_common_five")], ignore_index=True)
    schedules = pd.concat([summarize(paired, ORIGINS[k::3], ["26w"], f"schedule_{k}").assign(
        cutoffs=", ".join(ORIGINS[k::3].strftime("%Y-%m-%d"))) for k in range(3)], ignore_index=True)
    composition = selections.groupby(["as_of", "window", "method", "category"], as_index=False).agg(
        merchants=("merchant_abn", "size"), expected_monthly_fees=("expected_monthly_fees", "sum"))
    changes = []
    for (as_of, window), group in selections.groupby(["as_of", "window"]):
        b = group.loc[group.method.eq("revenue_only")]
        for method in METHODS[1:]:
            c = group.loc[group.method.eq(method)]
            entrants = c.loc[~c.merchant_abn.isin(b.merchant_abn)].assign(change="entrant")
            removals = b.loc[~b.merchant_abn.isin(c.merchant_abn)].assign(method=method, change="removed")
            changes.extend([entrants, removals])
    signatures = dict(estimates=digest(estimates), selections=digest(selections), outcomes=digest(outcomes),
        summary=digest(summary), solver_log=digest(logs))
    if previous:
        require(previous["digests"] == signatures, "Portfolio rerun changed frozen results")
    for name, frame in [("outcomes", outcomes), ("fold_comparison", paired), ("summary", summary),
        ("nonoverlapping_schedules", schedules), ("future_weekly_fees", pd.concat(future_series, ignore_index=True)),
        ("category_composition", composition), ("changed_merchants", pd.concat(changes, ignore_index=True))]:
        write_table(frame, OUT, name)
    for method in METHODS:
        latest = selections.loc[selections.as_of.eq(cf.CUTOFFS[-1]) & selections.window.eq("26w") & selections.method.eq(method)]
        write_table(latest, OUT, f"latest_{method}_100")
    save_json(OUT/"acceptance_checks.json", dict(source_signatures_checksum_calendar_reconciliation=True,
        unique_exactly_100_eligible=True, expected_fee_floor=True, same_downside_reference_across_all_methods_windows=True,
        numerical_variance_reconciliation=True, baseline_reproductions=len(cf.CUTOFFS),
        future_daily_perturbations=int(sum(c["future_daily_perturbation"] for c in checks)),
        independent_daily_quarter_and_weekly_fee_checks=int(outcomes.status.eq("observed").sum()),
        frozen_before_outcomes=True, september_unscored=True, reproducible_rerun=previous is not None, digests=signatures))
    report(paired, summary, estimates)
    print(summary.round(4).to_string(index=False), flush=True)
    return selections, paired, summary


if __name__ == "__main__":
    run()
