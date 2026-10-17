"""Frozen practical frontiers and explicitly non-deployable hindsight diagnostics.

Run select -> evaluate -> oracle -> report, or all. Existing experiments are read-only.
The oracle epigraph MILPs bound convex objectives globally; incumbents alone do not.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array
from threadpoolctl import threadpool_limits

import cash_flow_variability as cf
import cash_flow_risk_comparison as risk
import risk_aware_selection as rs
from cash_flow_lookback_comparison import history_dates
from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from forecast_comparison import ORIGINS, digest

OUT = cf.OUT.parent / "diversification_frontier"
RETENTIONS = [.995, .99, .98, .95]
KEY = ["as_of", "window", "method", "retention"]
ORACLE_KEY = ["as_of", "retention", "objective", "regime"]
SECOND_BUDGET = dict(max_cuts=100, node_limit=2000, relative_gap=.001, max_swaps=200)
COMMON = pd.date_range("2022-02-28", "2022-06-30", freq="ME")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def previous_inventory():
    """Preserve all earlier generated artifacts, including their timestamps and sizes."""
    return {str(p.relative_to(cf.OUT.parent)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(cf.OUT.parent.rglob("*")) if p.is_file() and OUT not in p.parents}


def inputs():
    daily, monthly, merchants, audit = cf.load_audited_inputs()
    return daily, monthly, merchants, cf.weekly_panel(daily, merchants), audit


def practical_protocol(audit, monthly, merchants):
    return dict(retentions=RETENTIONS, methods=rs.METHODS, windows=rs.WINDOWS,
        dates=cf.CUTOFFS.strftime("%Y-%m-%d").tolist(), count=100, solver=rs.SOLVER,
        second_oracle_budget=SECOND_BUDGET, absolute_gap_baseline_fraction=1e-6,
        material_risk_reduction=.05, expected_return="expanding complete-month mean gross fees",
        common_reference="baseline mean of latest 26 complete weeks, frozen for every arm",
        inputs=dict(daily=audit["daily_sha256"], monthly=digest(monthly), merchants=digest(merchants)),
        helpers={p:file_hash(Path(__file__).with_name(p+".py")) for p in
                 ["risk_aware_selection", "cash_flow_variability", "cash_flow_risk_comparison", "cash_flow_lookback_comparison"]},
        selection_implementation=hashlib.sha256(inspect.getsource(practical_at).encode()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in
                  ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "threadpoolctl"]},
        untouched_holdout=False, retrospective=True)


def practical_at(weekly, features, as_of, window):
    """Past-only selections; same solver calls/order within each 99% optimization as before."""
    f, dates, x, models = rs.models_at(weekly, features, as_of, window)
    baseline = cf.baseline_selection(f)
    base_ids = set(baseline.merchant_abn)
    base_fees = float(baseline.expected_monthly_fees.sum())
    fees26 = risk.history_matrix(weekly, cf.risk_window(as_of), f.merchant_abn.to_numpy())
    reference = float(fees26[:, f.merchant_abn.isin(base_ids)].sum(axis=1).mean())
    estimates, selections, logs = [], [], []
    points = [("revenue_only", 1.)] + [(m, r) for r in RETENTIONS for m in rs.METHODS[1:]]
    for method, retention in points:
        if method == "revenue_only":
            z = f.merchant_abn.isin(base_ids).to_numpy(dtype=float)
            info = dict(variance=np.nan, lower_bound=np.nan, gap_upper_fraction=np.nan,
                        certified=False, cuts=0, swaps=0, status="revenue_baseline")
        else:
            z, info, log = rs.optimize_portfolio(models[method], f.expected_monthly_fees.to_numpy(), retention=retention)
            if len(log):
                logs.append(log.assign(as_of=as_of, window=window, method=method, retention=retention))
        info["solver_status"] = info.pop("status")
        chosen = f.loc[z.astype(bool)].sort_values(["expected_monthly_fees", "merchant_abn"], ascending=[False, True]).copy()
        chosen["rank_by_expected_fees"] = np.arange(1, 101)
        chosen["as_of"], chosen["window"], chosen["method"], chosen["retention"] = as_of, window, method, retention
        chosen["in_revenue_baseline"] = chosen.merchant_abn.isin(base_ids)
        require(len(chosen) == 100 and chosen.merchant_abn.nunique() == 100 and chosen.eligible.all(), "Invalid selection")
        expected = float(chosen.expected_monthly_fees.sum())
        require(expected >= retention*base_fees-1e-6, "Expected fee floor failed")
        selected_history = x[:, z.astype(bool)].sum(axis=1)
        stats, _, _ = risk.decompose(selected_history[:, None])
        row = dict(as_of=as_of, window=window, method=method, retention=retention,
            history_weeks=len(dates), history_start=dates[0]-pd.Timedelta(days=6), history_end=dates[-1],
            eligible_merchants=len(f), expected_monthly_fees=expected, baseline_expected_monthly_fees=base_fees,
            expected_fee_retention_pct=100*expected/base_fees, selected_merchants=100,
            overlap_with_baseline=int(chosen.in_revenue_baseline.sum()), common_reference_weekly_fees=reference,
            own_reference_weekly_fees=float(fees26[:, z.astype(bool)].sum(axis=1).mean()),
            estimated_raw_sd=cf.portfolio_volatility(models["raw_risk"], chosen.merchant_abn, "sample"),
            estimated_residual_sd=cf.portfolio_volatility(models["detrended_risk"], chosen.merchant_abn, "sample"),
            estimated_raw_shrunk_sd=cf.portfolio_volatility(models["raw_risk"], chosen.merchant_abn, "shrunk"),
            estimated_residual_shrunk_sd=cf.portfolio_volatility(models["detrended_risk"], chosen.merchant_abn, "shrunk"),
            raw_alpha=models["raw_risk"].alpha, residual_alpha=models["detrended_risk"].alpha,
            trend_slope_per_week=float(stats.trend_slope_per_week.iloc[0]),
            raw_covariance_digest=digest(pd.DataFrame(models["raw_risk"].centered)),
            residual_covariance_digest=digest(pd.DataFrame(models["detrended_risk"].centered)), **info)
        np.testing.assert_allclose(row["estimated_raw_sd"], selected_history.std(ddof=1), rtol=1e-10)
        if method != "revenue_only":
            sd = row["estimated_raw_shrunk_sd" if method == "raw_risk" else "estimated_residual_shrunk_sd"]
            np.testing.assert_allclose(sd**2, info["variance"], rtol=1e-10)
        row["historical_diversification_ratio"] = float(np.sqrt(models["raw_risk"].variances[z.astype(bool)]).sum()/row["estimated_raw_sd"]) if row["estimated_raw_sd"] else np.nan
        estimates.append(row)
        selections.append(chosen)
    return pd.DataFrame(estimates), pd.concat(selections, ignore_index=True), pd.concat(logs, ignore_index=True)


def verify_old_99(estimates, selections):
    old_e = pd.read_parquet(rs.OUT/"unscored_estimates.parquet")
    old_s = pd.read_parquet(rs.OUT/"selections.parquet")
    for old, new in [(old_e, estimates), (old_s, selections)]:
        actual = new.loc[new.retention.eq(.99) | new.method.eq("revenue_only"), old.columns]
        order = ["as_of", "window", "method"] + (["merchant_abn"] if "merchant_abn" in old else [])
        pd.testing.assert_frame_equal(old.sort_values(order).reset_index(drop=True),
            actual.sort_values(order).reset_index(drop=True), check_exact=True)


def validate_freeze(data=None):
    require((OUT/"selection_freeze.json").exists(), "Run select: no frozen-selection manifest")
    frozen = json.loads((OUT/"selection_freeze.json").read_text())
    protocol = json.loads((OUT/"protocol.json").read_text())
    require(protocol["retentions"] == RETENTIONS and protocol["solver"] == rs.SOLVER, "Frozen settings changed")
    require(protocol["selection_implementation"] == hashlib.sha256(inspect.getsource(practical_at).encode()).hexdigest(), "Frozen selection implementation changed")
    for module, checksum in protocol["helpers"].items():
        require(file_hash(Path(__file__).with_name(module+".py")) == checksum, "Frozen helper implementation changed")
    if data is not None:
        _, monthly, merchants, _, audit = data
        require(protocol == practical_protocol(audit, monthly, merchants), "Frozen inputs/settings changed")
    for name, checksum in frozen["files"].items():
        require(file_hash(OUT/name) == checksum, f"Frozen selection file changed: {name}")
    require(previous_inventory() == frozen["previous_inventory"], "Previous experiment outputs changed")
    return frozen


def select_stage(data=None):
    data = inputs() if data is None else data
    daily, monthly, merchants, weekly, audit = data
    OUT.mkdir(parents=True, exist_ok=True)
    before = previous_inventory()
    previous = json.loads((OUT/"selection_freeze.json").read_text()) if (OUT/"selection_freeze.json").exists() else None
    protocol = practical_protocol(audit, monthly, merchants)
    if previous:
        validate_freeze(data)
    estimates, selections, logs, checks, availability = [], [], [], [], []
    for as_of in cf.CUTOFFS:
        print(f"Selecting {as_of.date()} (past information only)", flush=True)
        f, _ = cf.features_at_cutoff(weekly, monthly, merchants, as_of)
        # Change all later fees and monthly sales. Equality of complete optimizer inputs
        # plus deterministic rerun/toy selection tests establishes practical chronology.
        changed_weekly, changed_monthly = weekly.copy(), monthly.copy()
        changed_weekly.loc[changed_weekly.week_end.gt(as_of), "gross_fees"] *= 1000
        changed_monthly.loc[changed_monthly.month.gt(as_of), "sales"] *= 1000
        f2, _ = cf.features_at_cutoff(changed_weekly, changed_monthly, merchants, as_of)
        pd.testing.assert_frame_equal(f, f2, check_exact=True)
        for window in rs.WINDOWS:
            dates = history_dates(weekly, as_of, window)
            availability.append(dict(as_of=as_of, window=window, available=dates is not None))
            if dates is None:
                continue
            a = rs.models_at(weekly, f, as_of, window)
            b = rs.models_at(changed_weekly, f2, as_of, window)
            np.testing.assert_array_equal(a[2], b[2])
            for m in a[3]:
                require(a[3][m].alpha == b[3][m].alpha, "Future changed shrinkage")
                np.testing.assert_array_equal(a[3][m].centered, b[3][m].centered)
            e, s, log = practical_at(weekly, f, as_of, window)
            estimates.append(e); selections.append(s); logs.append(log)
            checks.append(dict(as_of=as_of, window=window, future_inputs_unchanged=True))
            print(f"  {window}: eight practical points; max variance gap {100*e.gap_upper_fraction.max():.4f}%", flush=True)
    e, s, logs = [pd.concat(v, ignore_index=True) for v in [estimates, selections, logs]]
    verify_old_99(e, s)
    require(e.groupby("as_of").common_reference_weekly_fees.nunique().eq(1).all(), "References differ")
    require(not s.duplicated(KEY+["merchant_abn"]).any() and s.groupby(KEY).size().eq(100).all(), "Invalid cardinality")
    result_digests = dict(estimates=digest(e), selections=digest(s), solver_log=digest(logs))
    if previous:
        require(previous["digests"] == result_digests, "Practical rerun changed results")
    tables = dict(unscored_estimates=e, selections=s, practical_solver_log=logs,
                  availability=pd.DataFrame(availability), chronology_checks=pd.DataFrame(checks))
    for name, frame in tables.items():
        write_table(frame, OUT, name)
    save_json(OUT/"protocol.json", protocol)
    require(before == previous_inventory(), "Previous outputs were modified")
    files = ["protocol.json"]+[name+ext for name in tables for ext in [".parquet", ".csv"]]
    save_json(OUT/"selection_freeze.json", dict(files={n:file_hash(OUT/n) for n in files}, digests=result_digests,
        previous_inventory=before, original_99_exactly_reproduced=True, reproducible_rerun=previous is not None,
        frozen_at=str(pd.Timestamp.now(tz="UTC")), stage="All practical selections frozen before outcome scoring/oracles"))
    return data


def matching(frame, row, keys=KEY):
    mask = np.ones(len(frame), dtype=bool)
    for key in keys:
        mask &= frame[key].eq(row[key])
    return frame.loc[mask]


def verify_practical_rerun():
    """Recompute every practical solve without rewriting the frozen artifact files."""
    data = inputs()
    frozen = validate_freeze(data)
    _, monthly, merchants, weekly, _ = data
    results = [[], [], []]
    for as_of in cf.CUTOFFS:
        f, _ = cf.features_at_cutoff(weekly, monthly, merchants, as_of)
        for window in rs.WINDOWS:
            if history_dates(weekly, as_of, window) is None:
                continue
            print(f"Verifying reproducibility: {as_of.date()} {window}", flush=True)
            for target, frame in zip(results, practical_at(weekly, f, as_of, window)):
                target.append(frame)
    signatures = dict(zip(["estimates", "selections", "solver_log"],
        [digest(pd.concat(frames, ignore_index=True)) for frames in results]))
    require(signatures == frozen["digests"], "Practical rerun differs")
    save_json(OUT/"practical_rerun_check.json", dict(digests=signatures, all_searches_recomputed=True,
        selection_manifest_sha256=file_hash(OUT/"selection_freeze.json")))
    print("Every practical search reproduced exactly; frozen files untouched.", flush=True)


def compare(estimates, outcomes):
    paired = estimates.merge(outcomes, on=KEY, validate="one_to_one")
    metrics = ["actual_quarter_fees", "actual_weekly_sd", "actual_weekly_cv", "downside_common", "downside_own"]
    base = paired.loc[paired.method.eq("revenue_only"), ["as_of", "window", *metrics]]
    paired = paired.merge(base.rename(columns={c:"baseline_"+c for c in metrics}), on=["as_of", "window"], validate="many_to_one")
    paired["actual_fee_retention_pct"] = 100*paired.actual_quarter_fees/paired.baseline_actual_quarter_fees
    paired["actual_fee_change_aud"] = paired.actual_quarter_fees-paired.baseline_actual_quarter_fees
    paired["sd_reduction_pct"] = 100*(1-paired.actual_weekly_sd/paired.baseline_actual_weekly_sd.where(paired.baseline_actual_weekly_sd.gt(0)))
    paired["cv_reduction_pct"] = 100*(1-paired.actual_weekly_cv/paired.baseline_actual_weekly_cv.where(paired.baseline_actual_weekly_cv.gt(0)))
    paired["normalized_downside"] = paired.downside_common/paired.common_reference_weekly_fees
    paired["baseline_normalized_downside"] = paired.baseline_downside_common/paired.common_reference_weekly_fees
    paired["downside_change_aud"] = paired.downside_common-paired.baseline_downside_common
    observed = paired.status.eq("observed")
    paired["actual_fee_guardrail_met"] = paired.actual_fee_retention_pct.ge(100*paired.retention-1e-8).where(observed, pd.NA).astype("boolean")
    paired["sd_improves"] = paired.actual_weekly_sd.lt(paired.baseline_actual_weekly_sd-1e-7).where(observed, pd.NA).astype("boolean")
    paired["downside_improves"] = paired.downside_change_aud.lt(-1e-7).where(observed, pd.NA).astype("boolean")
    return paired


def summarize(paired, dates, windows, label):
    rows = []
    for (window, method, retention), g in paired.loc[paired.as_of.isin(dates) & paired.window.isin(windows) & paired.method.ne("revenue_only")].groupby(["window", "method", "retention"], sort=True):
        require(set(g.as_of) == set(dates) and g.status.eq("observed").all(), "Unmatched summary origins")
        base_d = float(g.baseline_normalized_downside.mean())
        reduction = 100*(1-g.normalized_downside.mean()/base_d) if base_d > 0 else np.nan
        fee_pass = g.actual_fee_retention_pct.mean() >= 100*retention-1e-8
        sd_joint = int((g.sd_improves & g.actual_fee_guardrail_met).sum())
        d_joint = int((g.downside_improves & g.actual_fee_guardrail_met).sum())
        rows.append(dict(comparison=label, window=window, method=method, retention=retention, origins=len(g),
            mean_actual_fee_retention_pct=g.actual_fee_retention_pct.mean(), mean_actual_fee_change_aud=g.actual_fee_change_aud.mean(),
            worst_actual_fee_retention_pct=g.actual_fee_retention_pct.min(), fee_guardrail_passes=int(g.actual_fee_guardrail_met.sum()),
            mean_sd_reduction_pct=g.sd_reduction_pct.mean(), mean_cv_reduction_pct=g.cv_reduction_pct.mean(),
            mean_normalized_downside=g.normalized_downside.mean(), baseline_mean_normalized_downside=base_d,
            normalized_downside_reduction_pct=reduction, mean_downside_change_aud=g.downside_change_aud.mean(),
            sd_joint_successes=sd_joint, downside_joint_successes=d_joint, downside_wins=int(g.downside_improves.sum()),
            downside_ties=int(g.downside_change_aud.abs().le(1e-7).sum()), mean_overlap=g.overlap_with_baseline.mean(),
            investigate_sd=bool(label == "primary_26w_eight" and g.sd_reduction_pct.mean() >= 5 and fee_pass and sd_joint >= 6),
            investigate_downside=bool(label == "primary_26w_eight" and reduction >= 5 and fee_pass and d_joint >= 6)))
    return pd.DataFrame(rows)


def evaluate_stage(data=None):
    data = inputs() if data is None else data
    validate_freeze(data)
    daily, monthly, merchants, weekly, _ = data
    e, s = [pd.read_parquet(OUT/(n+".parquet")) for n in ["unscored_estimates", "selections"]]
    outcomes, series = [], []
    for row in e.to_dict("records"):
        chosen = matching(s, row)
        o, ts = rs.portfolio_outcome(weekly, monthly, daily, merchants, row, chosen)
        o["retention"] = row["retention"]
        if len(ts):
            y = risk.history_matrix(weekly, ts.week_end, chosen.merchant_abn.to_numpy())
            o["diversification_ratio"] = float(y.std(axis=0, ddof=1).sum()/o["actual_weekly_sd"]) if o["actual_weekly_sd"] else np.nan
            o["shortfall_frequency"] = o["below_reference_weeks"]/o["future_weeks"]
            series.append(ts.assign(retention=row["retention"]))
        outcomes.append(o)
    paired = compare(e, pd.DataFrame(outcomes))
    old = pd.read_parquet(rs.OUT/"outcomes.parquet").sort_values(["as_of", "window", "method"]).reset_index(drop=True)
    new = paired.loc[paired.retention.eq(.99) | paired.method.eq("revenue_only"), old.columns].sort_values(["as_of", "window", "method"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    summary = pd.concat([summarize(paired, ORIGINS, ["26w"], "primary_26w_eight"),
        summarize(paired, COMMON, rs.WINDOWS, "sensitivity_common_five")], ignore_index=True)
    schedules = pd.concat([summarize(paired, ORIGINS[k::3], ["26w"], f"schedule_{k}").assign(
        cutoffs=", ".join(ORIGINS[k::3].strftime("%Y-%m-%d"))) for k in range(3)], ignore_index=True)
    composition = s.groupby(KEY+["category"], as_index=False).agg(merchants=("merchant_abn", "size"), expected_monthly_fees=("expected_monthly_fees", "sum"))
    changes = []
    for row in e.loc[e.method.ne("revenue_only")].to_dict("records"):
        c = matching(s, row)
        b = matching(s, dict(row, method="revenue_only", retention=1.))
        changes += [c.loc[~c.merchant_abn.isin(b.merchant_abn)].assign(change="entrant"),
                    b.loc[~b.merchant_abn.isin(c.merchant_abn)].assign(method=row["method"], retention=row["retention"], change="removed")]
    tables = dict(outcomes=pd.DataFrame(outcomes), frontier_metrics=paired, summary=summary,
        nonoverlapping_schedules=schedules, future_weekly_fees=pd.concat(series, ignore_index=True),
        category_composition=composition, changed_merchants=pd.concat(changes, ignore_index=True),
        latest_unscored_candidates=s.loc[s.as_of.eq(cf.CUTOFFS[-1])].reset_index(drop=True))
    save_stage_results("evaluation", tables)
    print("Evaluation complete: prior 99% outcomes reproduce exactly.", flush=True)
    return data


def save_stage_results(stage, tables):
    record = OUT/(stage+"_checks.json")
    signatures = {name:digest(frame) for name, frame in tables.items()}
    previous = json.loads(record.read_text()) if record.exists() else None
    if previous:
        require(previous["digests"] == signatures, f"{stage} rerun changed results")
    for name, frame in tables.items():
        write_table(frame, OUT, name)
    save_json(record, dict(digests=signatures, reproducible_rerun=previous is not None,
        selection_manifest_sha256=file_hash(OUT/"selection_freeze.json")))


def validate_stage(stage):
    path = OUT/(stage+"_checks.json")
    require(path.exists(), f"Run {stage} before using its results")
    checks = json.loads(path.read_text())
    require(checks["selection_manifest_sha256"] == file_hash(OUT/"selection_freeze.json"), "Results belong to a different selection freeze")
    for name, checksum in checks["digests"].items():
        require(digest(pd.read_parquet(OUT/(name+".parquet"))) == checksum, f"Changed {stage} result: {name}")


class OracleObjective:
    """Differentiable convex variance or squared hinge of aggregate future income."""
    def __init__(self, future, kind, reference=0.):
        self.future = np.asarray(future, dtype=float)
        require(self.future.ndim == 2 and len(self.future) >= 2 and np.isfinite(self.future).all(), "Invalid future observations")
        require(kind in ["variability", "downside"] and np.isfinite(reference), "Invalid oracle objective")
        self.kind, self.reference = kind, float(reference)
        self.factors = (self.future-self.future.mean(axis=0))/np.sqrt(len(self.future)-1)

    def value_gradient(self, z):
        if self.kind == "variability":
            total = self.factors@z
            return float(total@total), 2*self.factors.T@total
        loss = np.maximum(self.reference-self.future@z, 0)
        return float(np.mean(loss**2)), -2*self.future.T@loss/len(loss)

    def value(self, z):
        return self.value_gradient(z)[0]


def oracle_swaps(objective, initial, constraints, floors, scale, max_swaps=200):
    """Exact one-for-one objective changes; enforce every income floor simultaneously."""
    z = initial.copy()
    matrix = objective.factors if objective.kind == "variability" else objective.future
    for step in range(max_swaps):
        inside, outside = np.flatnonzero(z), np.flatnonzero(1-z)
        if not len(outside):
            break
        feasible = np.ones((len(inside), len(outside)), dtype=bool)
        for c, floor in zip(constraints, floors):
            feasible &= c@z-c[inside, None]+c[outside] >= floor-1e-9
        total = matrix@z
        values = np.zeros(feasible.shape)
        for w in range(len(matrix)):
            amounts = total[w]-matrix[w, inside, None]+matrix[w, outside]
            values += (amounts**2 if objective.kind == "variability" else np.maximum(objective.reference-amounts, 0)**2/len(matrix))
        values[~feasible] = np.inf
        i, j = np.unravel_index(np.argmin(values), values.shape)
        if values[i, j] >= objective.value(z)-1e-10*scale:
            break
        z[inside[i]], z[outside[j]] = 0., 1.
    return z


def optimize_oracle(objective, means, count=100, retention=.99, actual_fees=None,
                    seeds=(), budgets=None):
    """Both budgets continue one outer approximation; dual bounds are global.

    Actual fees, when supplied, add the income-protected constraint. Objective
    bounds are in AUD squared. Never call this during practical selection.
    """
    means = np.asarray(means, dtype=float)
    n = len(means)
    require(n >= count >= 1 and 0 < retention <= 1 and np.isfinite(means).all() and (means >= 0).all(), "Invalid oracle constraints")
    require(objective.future.shape[1] == n, "Oracle dimension mismatch")
    baseline = np.zeros(n)
    baseline[np.lexsort((np.arange(n), -means))[:count]] = 1
    require(means@baseline > 0, "Zero expected income")
    constraints = [means/(means@baseline/count)]
    floors = [retention*count]
    if actual_fees is not None:
        actual_fees = np.asarray(actual_fees, dtype=float)
        require(actual_fees.shape == means.shape and np.isfinite(actual_fees).all() and (actual_fees >= 0).all(), "Invalid realized income")
        actual_base = float(actual_fees@baseline)
        # Future inactivity is a valid zero outcome; a zero actual floor is vacuous.
        constraints.append(actual_fees/(actual_base/count if actual_base > 0 else 1.))
        floors.append(retention*count if actual_base > 0 else 0.)
    def feasible(z):
        return z.shape == baseline.shape and np.isin(z, [0., 1.]).all() and z.sum() == count and all(c@z >= floor-1e-8 for c, floor in zip(constraints, floors))
    scale = objective.value(baseline)
    if scale <= 1e-20:
        return baseline, dict(baseline_objective=scale, objective_value=scale, lower_bound=0.,
            gap_upper_fraction=0., certified=True, solver_status="zero_baseline_objective", iterations=0, budget_tier=0), pd.DataFrame()
    budgets = [rs.SOLVER, SECOND_BUDGET] if budgets is None else budgets
    points = [baseline]+[np.asarray(s, dtype=float) for s in seeds if feasible(np.asarray(s, dtype=float))]
    best = min(points, key=lambda z:(objective.value(z), tuple(np.flatnonzero(z)))).copy()
    planes, rhs, seen, logs = [], [], set(), []
    lower = 0.
    def add_plane(z):
        key = tuple(np.flatnonzero(z))
        if key in seen:
            return False
        seen.add(key)
        value, grad = objective.value_gradient(z)
        planes.append(np.r_[grad/scale, -1.]); rhs.append(float((grad@z-value)/scale))
        return True
    with threadpool_limits(limits=1):
        best = oracle_swaps(objective, best, constraints, floors, scale, budgets[0]["max_swaps"])
        for z in points+[best]:
            add_plane(z)
        upper = objective.value(best)/scale
        certified = False
        for tier, settings in enumerate(budgets, 1):
            if tier > 1 and n > 100:
                print(f"  continuing oracle at second fixed budget; current relative gap {100*(upper-lower)/max(upper,1e-30):.3f}%", flush=True)
            for iteration in range(settings["max_cuts"]):
                matrix = np.vstack([np.r_[np.ones(n), 0.], *[np.r_[c, 0.] for c in constraints], *planes])
                lc = LinearConstraint(csc_array(matrix), np.r_[count, floors, np.full(len(planes), -np.inf)],
                                      np.r_[count, np.full(len(floors), np.inf), rhs])
                result = milp(np.r_[np.zeros(n), 1.], integrality=np.r_[np.ones(n), 0],
                    bounds=Bounds(np.zeros(n+1), np.r_[np.ones(n), np.inf]), constraints=lc,
                    options=dict(node_limit=settings["node_limit"], mip_rel_gap=1e-6, presolve=True))
                solution_limit = "HiGHS Status 16: Solution limit reached" in result.message
                require(result.status in [0, 1] or solution_limit, f"Oracle MILP failure: {result.message}")
                bound = getattr(result, "mip_dual_bound", None)
                if bound is not None and np.isfinite(bound):
                    lower = max(lower, float(bound))
                candidate = None
                if result.x is not None:
                    candidate = np.rint(result.x[:-1])
                    require(np.max(np.abs(candidate-result.x[:-1])) < 1e-5 and feasible(candidate), "Infeasible oracle incumbent")
                    polished = oracle_swaps(objective, candidate, constraints, floors, scale, settings["max_swaps"])
                    value = objective.value(polished)/scale
                    if value < upper-1e-12 or (abs(value-upper) <= 1e-12 and tuple(np.flatnonzero(polished)) < tuple(np.flatnonzero(best))):
                        best, upper = polished, value
                require(lower <= upper+1e-6, "Oracle lower bound exceeds feasible objective")
                # Downward numerical adjustment only; never strengthen a reported bound.
                lower = min(lower, upper)
                gap = max(0., upper-lower)
                certified = gap <= settings["relative_gap"]*upper or gap <= 1e-6
                logs.append(dict(budget_tier=tier, iteration=iteration, milp_status=result.status,
                    milp_nodes=getattr(result, "mip_node_count", None), lower_bound=lower*scale,
                    objective_value=upper*scale, gap_upper_fraction=gap/max(upper, 1e-30)))
                if certified:
                    break
                added = add_plane(best)
                if candidate is not None:
                    added = add_plane(candidate) or added
                if not added:
                    break  # Next tier increases the node budget on the same relaxation.
            if certified:
                break
    require(feasible(best), "Final oracle infeasible")
    return best, dict(baseline_objective=scale, objective_value=objective.value(best), lower_bound=lower*scale,
        gap_upper_fraction=max(0., upper-lower)/max(upper, 1e-30), certified=certified,
        solver_status="certified_gap" if certified else "bounded_best_found", iterations=len(logs), budget_tier=tier), pd.DataFrame(logs)


def opportunity(info):
    base, upper, lower = [float(info[k]) for k in ["baseline_objective", "objective_value", "lower_bound"]]
    if base <= 1e-20:
        return dict(opportunity="no_baseline_risk", demonstrated_reduction_pct=0., possible_reduction_pct=0., risk_lower_bound=0., risk_upper_bound=0.)
    demonstrated, possible = 100*(1-np.sqrt(max(upper, 0)/base)), 100*(1-np.sqrt(max(lower, 0)/base))
    label = "demonstrated" if demonstrated >= 5-1e-8 else "ruled_out_by_bound" if possible < 5-1e-8 else "unresolved"
    return dict(opportunity=label, demonstrated_reduction_pct=demonstrated, possible_reduction_pct=possible,
                risk_lower_bound=float(np.sqrt(max(lower, 0))), risk_upper_bound=float(np.sqrt(max(upper, 0))))


def oracle_stage(data=None, verify_only=False):
    data = inputs() if data is None else data
    validate_freeze(data)
    validate_stage("evaluation")
    daily, monthly, merchants, weekly, _ = data
    estimates, selections = [pd.read_parquet(OUT/(n+".parquet")) for n in ["unscored_estimates", "selections"]]
    records, chosen_rows, logs, outcomes, series = [], [], [], [], []
    for as_of in ORIGINS:
        features, _ = cf.features_at_cutoff(weekly, monthly, merchants, as_of)
        f = features.loc[features.eligible].sort_values("merchant_abn").reset_index(drop=True)
        ids, means = f.merchant_abn.to_numpy(), f.expected_monthly_fees.to_numpy()
        baseline_ids = set(cf.baseline_selection(f).merchant_abn)
        base = f.merchant_abn.isin(baseline_ids).to_numpy(dtype=float)
        dates = risk.future_dates(as_of, daily.day.max())
        require(dates is not None, "Oracle needs complete future quarter")
        y = risk.history_matrix(weekly, dates, ids)
        quarter = monthly.loc[monthly.month.gt(as_of) & monthly.month.le(as_of+pd.offsets.MonthEnd(3))]
        fees = quarter.groupby("merchant_abn").sales.sum().reindex(ids).to_numpy()*f.take_rate.to_numpy()/100
        require(np.isfinite(fees).all(), "Missing eligible future quarter")
        reference = float(estimates.loc[estimates.as_of.eq(as_of), "common_reference_weekly_fees"].iloc[0])
        prior_solutions = []
        for retention in RETENTIONS:
            practical = selections.loc[selections.as_of.eq(as_of)]
            seeds = [f.merchant_abn.isin(g.merchant_abn).to_numpy(dtype=float)
                     for _, g in practical.groupby(["window", "method", "retention"])] + prior_solutions
            for objective_name in ["variability", "downside"]:
                obj = OracleObjective(y, objective_name, reference)
                for regime in ["matched", "income_protected"]:
                    print(f"HINDSIGHT ONLY {as_of.date()} r={retention:.3f} {objective_name} {regime}", flush=True)
                    z, info, log = optimize_oracle(obj, means, retention=retention,
                        actual_fees=fees if regime == "income_protected" else None, seeds=seeds+prior_solutions)
                    prior_solutions.append(z)
                    key = dict(as_of=as_of, retention=retention, objective=objective_name, regime=regime)
                    chosen = f.loc[z.astype(bool)].copy()
                    for k, value in key.items():
                        chosen[k] = value
                    chosen["hindsight_only"] = True
                    require(len(chosen) == 100 and chosen.merchant_abn.nunique() == 100, "Oracle count failed")
                    require(means@z >= retention*(means@base)-1e-6, "Oracle expected floor failed")
                    if regime == "income_protected":
                        require(fees@z >= retention*(fees@base)-1e-6, "Oracle realized floor failed")
                    e = dict(as_of=as_of, window="hindsight", method=objective_name+"_"+regime,
                        common_reference_weekly_fees=reference,
                        own_reference_weekly_fees=float(risk.history_matrix(weekly, cf.risk_window(as_of), ids[z.astype(bool)]).sum(axis=1).mean()))
                    o, ts = rs.portfolio_outcome(weekly, monthly, daily, merchants, e, chosen)
                    measured = o["actual_weekly_sd"] if objective_name == "variability" else o["downside_common"]
                    np.testing.assert_allclose(measured**2, info["objective_value"], rtol=1e-9, atol=1e-6)
                    row = dict(**key, **info, **opportunity(info), hindsight_only=True, future_weeks=len(dates),
                        expected_fee_retention_pct=100*(means@z)/(means@base), actual_fee_retention_pct=100*(fees@z)/(fees@base),
                        common_reference_weekly_fees=reference, overlap_with_baseline=int(chosen.merchant_abn.isin(baseline_ids).sum()),
                        **{k:v for k,v in o.items() if k not in ["as_of", "window", "method", "future_weeks"]})
                    records.append(row); chosen_rows.append(chosen)
                    print(f"  {row['opportunity']}; achievable reduction {row['demonstrated_reduction_pct']:.2f}% "
                          f"to bound {row['possible_reduction_pct']:.2f}%; certified={info['certified']}", flush=True)
                    if len(log):
                        logs.append(log.assign(**key))
                    outcomes.append(dict(**key, **{k:v for k,v in o.items() if k not in ["as_of", "window", "method"]}))
                    series.append(ts.drop(columns=["window", "method"]).assign(**{k:v for k,v in key.items() if k != "as_of"}))
                    # Separate progress checkpoint; never a practical recommendation.
                    if not verify_only:
                        write_table(pd.DataFrame(records), OUT/"hindsight_only", "progress_bounds")
    table = pd.DataFrame(records)
    nested = check_nested_bounds(table, ORACLE_KEY[:-3]+["objective", "regime"])
    protected = table.loc[table.regime.eq("income_protected")].merge(table.loc[table.regime.eq("matched")],
        on=["as_of", "retention", "objective"], suffixes=("_protected", "_matched"), validate="one_to_one")
    require((protected.lower_bound_matched <= protected.objective_value_protected+1e-5*np.maximum(1, protected.baseline_objective_matched)).all(), "Constraint-regime bounds contradict")
    tables = dict(oracle_bounds=table, oracle_selections=pd.concat(chosen_rows, ignore_index=True),
        oracle_solver_log=pd.concat(logs, ignore_index=True), oracle_outcomes=pd.DataFrame(outcomes),
        oracle_future_weekly_fees=pd.concat(series, ignore_index=True), oracle_nested_bounds=nested)
    # Explicit filenames/flags and separate copies keep hindsight IDs distinct from deployable lists.
    history_manifest = OUT/"hindsight_only"/"manifest.json"
    previous = json.loads(history_manifest.read_text()) if history_manifest.exists() else None
    signatures = {name:digest(frame) for name, frame in tables.items()}
    if verify_only:
        require(previous is not None and previous["digests"] == signatures, "Hindsight verification differs or original run incomplete")
        save_json(OUT/"oracle_rerun_check.json", dict(digests=signatures, all_searches_recomputed=True,
            selection_manifest_sha256=file_hash(OUT/"selection_freeze.json")))
        print("Every hindsight search reproduced exactly; original artifacts untouched.", flush=True)
        return data
    if previous:
        require(previous["digests"] == signatures, "Hindsight selections/logs changed on rerun")
    for name, frame in tables.items():
        write_table(frame, OUT/"hindsight_only", name)
    save_stage_results("oracle", {"oracle_bounds":table, "oracle_nested_bounds":nested})
    save_json(history_manifest, dict(hindsight_only=True, digests=signatures,
        reproducible_rerun=previous is not None,
        selection_manifest_sha256=file_hash(OUT/"selection_freeze.json"),
        files={p.name:file_hash(p) for p in sorted((OUT/"hindsight_only").glob("*.parquet"))}))
    return data


def check_nested_bounds(table, groups):
    rows = []
    value_name = "objective_value" if "objective_value" in table else "variance"
    for keys, g in table.groupby(groups):
        g = g.sort_values("retention", ascending=False)
        for a, b in zip(g.to_dict("records"), g.to_dict("records")[1:]):
            # A looser feasible set's global LB cannot exceed a feasible stricter solution.
            tolerance = 1e-6*max(1., a[value_name], b[value_name])
            valid = b["lower_bound"] <= a[value_name]+tolerance
            require(valid, "Nested feasible-set bounds contradict")
            rows.append(dict(**dict(zip(groups, keys if isinstance(keys, tuple) else (keys,))),
                strict_retention=a["retention"], loose_retention=b["retention"],
                strict_feasible_objective=a[value_name], loose_lower_bound=b["lower_bound"], consistent=valid))
    return pd.DataFrame(rows)


def report_stage():
    validate_freeze()
    validate_stage("evaluation")
    validate_stage("oracle")
    history_manifest = json.loads((OUT/"hindsight_only"/"manifest.json").read_text())
    for name, checksum in history_manifest["files"].items():
        require(file_hash(OUT/"hindsight_only"/name) == checksum, "Changed hindsight artifact")
    e, paired, summary, oracles = [pd.read_parquet(OUT/(n+".parquet")) for n in
                                    ["unscored_estimates", "frontier_metrics", "summary", "oracle_bounds"]]
    practical_nested = check_nested_bounds(e.loc[e.method.ne("revenue_only")], ["as_of", "window", "method"])
    write_table(practical_nested, OUT, "practical_nested_bounds")
    practical_oracle = paired.loc[paired.method.ne("revenue_only") & paired.status.eq("observed")].merge(
        oracles[[*ORACLE_KEY, "opportunity", "demonstrated_reduction_pct", "possible_reduction_pct", "risk_lower_bound", "risk_upper_bound", "certified"]],
        on=["as_of", "retention"], validate="many_to_many")
    practical_oracle["practical_risk"] = np.where(practical_oracle.objective.eq("variability"), practical_oracle.actual_weekly_sd, practical_oracle.downside_common)
    practical_oracle["practical_satisfies_regime"] = practical_oracle.regime.eq("matched") | practical_oracle.actual_fee_guardrail_met.fillna(False)
    practical_oracle["baseline_risk"] = np.where(practical_oracle.objective.eq("variability"),
        practical_oracle.baseline_actual_weekly_sd, practical_oracle.baseline_downside_common)
    protected_labels = oracles.loc[oracles.regime.eq("income_protected"), ["as_of", "retention", "objective", "opportunity"]].rename(columns={"opportunity":"protected_opportunity"})
    practical_oracle = practical_oracle.merge(protected_labels, on=["as_of", "retention", "objective"], validate="many_to_one")
    practical_oracle["interpretation"] = "material_opportunity_not_captured_with_income_floor"
    captured = practical_oracle.practical_risk.le(.95*practical_oracle.baseline_risk) & practical_oracle.actual_fee_guardrail_met.fillna(False)
    practical_oracle.loc[captured, "interpretation"] = "material_reduction_captured"
    larger_cost = practical_oracle.regime.eq("matched") & practical_oracle.opportunity.eq("demonstrated") & practical_oracle.protected_opportunity.eq("ruled_out_by_bound")
    practical_oracle.loc[larger_cost, "interpretation"] = "material_opportunity_requires_larger_actual_revenue_cost"
    for label, meaning in [("ruled_out_by_bound", "little_material_opportunity"),
                           ("unresolved", "numerically_unresolved"), ("no_baseline_risk", "no_baseline_risk")]:
        practical_oracle.loc[practical_oracle.opportunity.eq(label), "interpretation"] = meaning
    comparable = practical_oracle.loc[practical_oracle.practical_satisfies_regime]
    require((comparable.risk_upper_bound <= comparable.practical_risk+1e-5*np.maximum(1, comparable.practical_risk)).all(), "Oracle worse than feasible practical seed")
    write_table(practical_oracle, OUT, "practical_versus_oracle")
    oracle_summary = []
    for (obj, regime, retention), g in oracles.groupby(["objective", "regime", "retention"]):
        if obj == "variability":
            low, high = g.demonstrated_reduction_pct.mean(), g.possible_reduction_pct.mean()
        else:
            baseline_downside = (np.sqrt(g.baseline_objective)/g.common_reference_weekly_fees).mean()
            low = 100*(1-(g.risk_upper_bound/g.common_reference_weekly_fees).mean()/baseline_downside) if baseline_downside else 0.
            high = 100*(1-(g.risk_lower_bound/g.common_reference_weekly_fees).mean()/baseline_downside) if baseline_downside else 0.
        oracle_summary.append(dict(objective=obj, regime=regime, retention=retention,
            mean_risk_reduction_lower_pct=low, mean_risk_reduction_upper_pct=high,
            mean_actual_fee_retention_pct=g.actual_fee_retention_pct.mean(),
            material_opportunity_origins=int(g.opportunity.eq("demonstrated").sum()),
            ruled_out_origins=int(g.opportunity.eq("ruled_out_by_bound").sum()),
            unresolved_origins=int(g.opportunity.eq("unresolved").sum()),
            zero_baseline_origins=int(g.opportunity.eq("no_baseline_risk").sum())))
    oracle_summary = pd.DataFrame(oracle_summary)
    write_table(oracle_summary, OUT, "oracle_summary")
    make_plots(e, paired, summary, oracles)
    primary = summary.loc[summary.comparison.eq("primary_26w_eight")]
    lines = ["# Diversification opportunity and revenue-risk frontiers", "",
        "Exactly 100 eligible merchants; expected return is unchanged historical mean monthly gross fees. "
        "All 120 practical risk selections (plus 15 revenue-only references) were frozen before outcomes and hindsight searches. "
        "The earlier 99% selections, estimates and outcomes reproduce exactly. Previous experiment artifacts are unchanged.", "",
        "## Practical results: eight 26-week tests", "",
        "Positive risk-reduction percentages mean improvement. Downside averages first normalize each origin's RMS "
        "shortfall by its common frozen income reference; zero-downside origins are retained. Negative reduction means worse risk.", "",
        "| Method | Expected floor | Actual fee retention | SD reduction | CV reduction | Downside reduction | SD + fee wins | Downside + fee wins | Screen SD / downside |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in primary.itertuples():
        lines.append(f"| {r.method} | {100*r.retention:g}% | {r.mean_actual_fee_retention_pct:.3f}% | {r.mean_sd_reduction_pct:+.2f}% | {r.mean_cv_reduction_pct:+.2f}% | {r.normalized_downside_reduction_pct:+.2f}% | {r.sd_joint_successes}/8 | {r.downside_joint_successes}/8 | {r.investigate_sd} / {r.investigate_downside} |")
    if not primary[["investigate_sd", "investigate_downside"]].any().any():
        lines += ["", "**No practical method passes the predeclared screen. Retain the revenue-only baseline.** "
            "A lower absolute SD alone is insufficient: examine the revenue sacrificed, relative variability, "
            "common-target downside and the frequency of meeting the realized income floor."]
    lines += ["", f"{int(e.certified.sum())}/{int(e.method.ne('revenue_only').sum())} practical searches met the "
        "0.1% variance-gap tolerance. Where such a search fails to improve later risk, the failure is in transferring "
        "the historical risk estimate to the future, rather than a large unresolved optimization gap."]
    lines += ["", "A screen passes only with at least 5% average risk reduction, average actual fee retention at its own floor, "
        "and lower risk while meeting that floor in at least six of eight tests. This is a research screen, not a business "
        "requirement or statistical-significance test. No retention level was chosen after seeing results; the baseline is not automatically replaced.", "",
        "## Hindsight: what was achievable on the observed future path?", "",
        "These diagnostics are NOT recommendations. They use future fees to minimize either observed sample variance "
        "or mean squared shortfall. Matched means the same historical expected-fee constraint. Income-protected adds "
        "the corresponding actual full-quarter fee floor. Bounds measure numerical uncertainty only.", "",
        "| Objective | Regime | Floor | Demonstrated ≥5% | Ruled out by bound | Unresolved | Zero baseline risk | Certified |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for (obj, regime, retention), g in oracles.groupby(["objective", "regime", "retention"]):
        counts = g.opportunity.value_counts()
        lines.append(f"| {obj} | {regime} | {100*retention:g}% | {counts.get('demonstrated',0)} | {counts.get('ruled_out_by_bound',0)} | {counts.get('unresolved',0)} | {counts.get('no_baseline_risk',0)} | {int(g.certified.sum())}/{len(g)} |")
    ceiling99 = oracle_summary.loc[oracle_summary.regime.eq("income_protected") & oracle_summary.retention.eq(.99)].set_index("objective")
    lines += ["", "At the 99% floor, income-protected hindsight portfolios allow an equal-origin average SD reduction "
        f"of {ceiling99.loc['variability','mean_risk_reduction_lower_pct']:.2f}% to "
        f"{ceiling99.loc['variability','mean_risk_reduction_upper_pct']:.2f}%, but only "
        f"{ceiling99.loc['downside','mean_risk_reduction_lower_pct']:.2f}% to "
        f"{ceiling99.loc['downside','mean_risk_reduction_upper_pct']:.2f}% reduction in normalized average downside. "
        "These intervals are numerical bounds on this observed path, not confidence intervals. "
        "The complete mean bounds for every floor/regime are in oracle_summary.csv.", "",
        "## Interpretation by opportunity", ""]
    for (obj, retention), g in oracles.groupby(["objective", "retention"]):
        protected = g.loc[g.regime.eq("income_protected")]
        matched = g.loc[g.regime.eq("matched")]
        demonstrated = int(protected.opportunity.eq("demonstrated").sum())
        cost_only = int((matched.set_index("as_of").opportunity.eq("demonstrated") & protected.set_index("as_of").opportunity.eq("ruled_out_by_bound")).sum())
        lines.append(f"- {obj}, {100*retention:g}% floor: income-protected material opportunity demonstrated at {demonstrated}/8 origins; "
            f"{cost_only}/8 show a matched opportunity ruled out once actual income is protected. Inspect the per-origin practical-versus-oracle table "
            "to distinguish opportunity missed by estimates from limited available opportunity. Unresolved bounds are not evidence of impossibility.")
    lines += ["", "## Measures and limitations", "",
        "Revenue uses every day of the next three calendar months. Risk uses only the 11–13 complete Monday–Sunday weeks "
        "wholly inside those months. The common downside target is always the revenue-only portfolio's latest-26-week mean; "
        "it is identical across retention levels, methods and lookbacks. Own-reference downside is secondary because lowering "
        "the income requirement can manufacture an apparent improvement. Future inactivity is zero revenue, not exclusion.", "",
        "SD/mean separates scale from relative variability. Diversification ratio is sum of individual SDs divided by "
        "portfolio SD: it compares with hypothetical perfect co-movement, not with the baseline portfolio or an optimization gain. "
        "Raw variability retains growth and calendar effects; detrended covariance removes a fitted linear trend but can also hide smooth decline.", "",
        "The 26-versus-52-week sensitivity uses only the five common February–June dates. Three exported schedules use "
        "non-overlapping quarters within each schedule; schedules are not independent of one another. All points, including "
        "dominated points, remain visible. These are discrete approximate frontiers, not guaranteed smooth efficient curves.", "",
        "September candidates are unscored. There is no untouched holdout. Short synthetic history and overlapping outcomes "
        "limit generalization. A hindsight oracle can exploit chance patterns in only 11-13 weeks; its opportunity does not "
        "prove that the same improvement was predictable. Merchant-level weekly fields in selection lists come from the "
        "original 26-week feature panel; window-specific portfolio risks are in unscored_estimates. "
        "Categories/take rates are assumed stable and historically known; hypothetical onboarding "
        "does not change sales. Gross fees exclude fraud, funding, operating costs and settlement timing.", "",
        "## Reproduction", "",
        "Use existing pinned code/requirements-forecast.txt. Run `python code/scripts/diversification_frontier.py select`, then "
        "`evaluate`, `oracle`, `report`; `all` runs all stages. Repeating all stages recomputes and checks exact table digests. "
        "`python -m pytest -p no:capture code/tests -q` runs the complete suite. The notebook reads frozen outputs "
        "and regenerates the report/plots; it does not silently refit on future data. Outputs are ignored, uncompressed Parquet "
        "and CSV. Hindsight merchant lists live separately in hindsight_only/.", "",
        "Practical budgets are unchanged: 30 iterations, 200 MILP nodes per iteration, 200 exchange steps, 0.1% variance gap. "
        "Uncertified oracles continue with 100 iterations and 2,000 nodes; stop at 0.1% relative objective gap or 1e-6 "
        "of baseline objective in absolute gap. A noncertified incumbent gives only an achievable risk reduction; the global "
        "lower bound limits any further possible reduction. Squared-objective bounds are square-rooted before comparing SD or RMS downside."]
    (OUT/"findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    freeze = validate_freeze()
    rerun_path = OUT/"practical_rerun_check.json"
    rerun = json.loads(rerun_path.read_text()) if rerun_path.exists() else {}
    practical_reproduced = freeze["reproducible_rerun"] or (rerun.get("all_searches_recomputed", False)
        and rerun.get("digests") == freeze["digests"]
        and rerun.get("selection_manifest_sha256") == file_hash(OUT/"selection_freeze.json"))
    oracle_rerun_path = OUT/"oracle_rerun_check.json"
    oracle_rerun = json.loads(oracle_rerun_path.read_text()) if oracle_rerun_path.exists() else {}
    oracle_reproduced = json.loads((OUT/"oracle_checks.json").read_text())["reproducible_rerun"] or (
        oracle_rerun.get("all_searches_recomputed", False) and oracle_rerun.get("digests") == history_manifest["digests"]
        and oracle_rerun.get("selection_manifest_sha256") == file_hash(OUT/"selection_freeze.json"))
    save_json(OUT/"acceptance_checks.json", dict(original_99_exact_reproduction=True, previous_outputs_unchanged=True,
        all_practical_frozen_before_outcomes=True, input_audit_verified=True, exactly_100_eligible_expected_floors=True,
        common_reference_all_grid=True, independent_daily_quarter_weekly_reconciliations=int(paired.status.eq("observed").sum())+len(oracles),
        protected_realized_constraints=True, objective_reconciliations=True, nested_bounds=True,
        september_unscored=bool(paired.loc[paired.as_of.eq(cf.CUTOFFS[-1]), "status"].eq("unavailable").all()),
        practical_reproducible_rerun=practical_reproduced,
        evaluation_reproducible_rerun=json.loads((OUT/"evaluation_checks.json").read_text())["reproducible_rerun"],
        oracle_reproducible_rerun=oracle_reproduced,
        practical_portfolios=len(e), oracle_cases=len(oracles), certified_oracles=int(oracles.certified.sum())))
    print(primary.to_string(index=False), flush=True)


def make_plots(estimates, paired, summary, oracles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = OUT/"plots"
    path.mkdir(exist_ok=True)
    colors = dict(raw_risk="#2776a1", detrended_risk="#cd742d")
    labels = dict(raw_risk="Raw covariance", detrended_risk="Detrended covariance")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    primary = estimates.loc[estimates.window.eq("26w") & estimates.as_of.isin(ORIGINS)].copy()
    b = primary.loc[primary.method.eq("revenue_only")].set_index("as_of")
    for ax, metric, title in zip(axes, ["estimated_raw_shrunk_sd", "estimated_residual_shrunk_sd"], ["Estimated raw variability", "Estimated detrended variability"]):
        primary["relative_risk"] = 100*primary[metric]/primary.as_of.map(b[metric])
        for method, color in colors.items():
            g = primary.loc[primary.method.eq(method)].groupby("retention").agg(x=("expected_fee_retention_pct", "mean"), y=("relative_risk", "mean"))
            ax.plot(np.r_[g.x, 100], np.r_[g.y, 100], "o-", color=color, label=labels[method])
        ax.scatter([100], [100], marker="*", color="black", s=90, label="revenue-only")
        ax.set(title=title, xlabel="Expected fees retained (%)", ylabel="Estimated risk / baseline (%)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Estimated frontier · equal-origin means, eight 26-week tests")
    fig.tight_layout(); fig.savefig(path/"estimated_frontier.png", dpi=160); plt.close(fig)
    p = summary.loc[summary.comparison.eq("primary_26w_eight")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for ax, metric, title in zip(axes, ["mean_sd_reduction_pct", "mean_cv_reduction_pct", "normalized_downside_reduction_pct"], ["Weekly SD", "SD / weekly mean", "Common-reference downside"]):
        for method, color in colors.items():
            g = p.loc[p.method.eq(method)].sort_values("retention")
            ax.plot(np.r_[100-g.mean_actual_fee_retention_pct, 0], np.r_[g[metric], 0], "o-", color=color, label=labels[method])
            if method == "detrended_risk":
                for r in g.itertuples():
                    ax.annotate(f"{100*r.retention:g}%", (100-r.mean_actual_fee_retention_pct, getattr(r, metric)),
                        xytext=(6, 7), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="gray", lw=.8); ax.axhline(5, color="green", ls="--", lw=.8, label="5% risk screen")
        ax.margins(x=.12, y=.15)
        ax.set(title=title, xlabel="Actual fees sacrificed (%)", ylabel="Risk reduction (%)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Realized frontier · positive means safer; shared revenue floors labelled on orange series")
    fig.tight_layout(); fig.savefig(path/"realized_frontier.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, obj, metric in zip(axes, ["variability", "downside"], ["mean_sd_reduction_pct", "normalized_downside_reduction_pct"]):
        for method, color in colors.items():
            g = p.loc[p.method.eq(method)].sort_values("retention")
            ax.plot(100*(1-g.retention), g[metric], "o-", color=color, label=labels[method])
        for regime, style in [("matched", "--"), ("income_protected", ":")]:
            g = oracles.loc[oracles.objective.eq(obj) & oracles.regime.eq(regime)].copy()
            # Downside aggregation matches practical normalized-mean convention, including zeros.
            rows = []
            for retention, h in g.groupby("retention"):
                if obj == "variability":
                    lo, hi = h.demonstrated_reduction_pct.mean(), h.possible_reduction_pct.mean()
                else:
                    denominator = (np.sqrt(h.baseline_objective)/h.common_reference_weekly_fees).mean()
                    lo = 100*(1-(h.risk_upper_bound/h.common_reference_weekly_fees).mean()/denominator)
                    hi = 100*(1-(h.risk_lower_bound/h.common_reference_weekly_fees).mean()/denominator)
                rows.append((100*(1-retention), lo, hi))
            x, lo, hi = np.array(rows).T
            order = np.argsort(x)
            ax.plot(x[order], lo[order], style, color="#525252" if regime == "matched" else "#8158a0",
                    label="Hindsight: "+("expected floor" if regime == "matched" else "both income floors"))
            ax.fill_between(x[order], lo[order], hi[order], alpha=.12, color="gray")
        ax.axhline(5, color="green", ls="--", lw=.7, label="5% risk screen")
        ax.set(title=obj.capitalize(), xlabel="Allowed expected-fee sacrifice (%)", ylabel="Risk reduction (%)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Practical results versus hindsight opportunity · shading = numerical bounds")
    fig.tight_layout(); fig.savefig(path/"practical_vs_hindsight.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=["select", "evaluate", "oracle", "report", "all"])
    args = parser.parse_args()
    if args.stage == "all":
        data = select_stage()
        evaluate_stage(data); oracle_stage(data); report_stage()
    elif args.stage == "report":
        report_stage()
    else:
        {"select":select_stage, "evaluate":evaluate_stage, "oracle":oracle_stage}[args.stage]()


if __name__ == "__main__":
    main()
