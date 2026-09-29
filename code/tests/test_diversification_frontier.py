"""Oracle mathematics, global bounds, chronology and frozen-stage controls."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diversification_frontier as df
import cash_flow_variability as cf
from test_risk_aware_selection import toy


@pytest.mark.parametrize("kind", ["variability", "downside"])
@pytest.mark.parametrize("protected", [False, True])
def test_oracle_matches_exhaustive(kind, protected):
    rng = np.random.default_rng(19)
    y = rng.uniform(0, 18, (8, 7))
    means = np.array([12., 11., 10., 9., 8., 7., 6.])
    actual = np.array([20., 18., 24., 12., 30., 4., 9.])
    obj = df.OracleObjective(y, kind, 30.)
    budgets = [dict(max_cuts=100, node_limit=2000, relative_gap=1e-10, max_swaps=100)]
    z, info, log = df.optimize_oracle(obj, means, count=3, retention=.8,
        actual_fees=actual if protected else None, budgets=budgets)
    feasible = []
    for c in combinations(range(7), 3):
        zz = np.zeros(7); zz[list(c)] = 1
        if means@zz >= .8*33 and (not protected or actual@zz >= .8*62):
            feasible.append(obj.value(zz))
    optimum = min(feasible)
    assert info["objective_value"] == pytest.approx(optimum, rel=1e-8, abs=1e-7)
    assert info["lower_bound"] <= optimum+1e-7
    assert info["certified"]
    z2, info2, log2 = df.optimize_oracle(obj, means, count=3, retention=.8,
        actual_fees=actual if protected else None, budgets=budgets)
    np.testing.assert_array_equal(z, z2)
    assert info == info2
    pd.testing.assert_frame_equal(log, log2, check_exact=True)


@pytest.mark.parametrize("kind", ["variability", "downside"])
def test_gradient_and_global_tangent(kind):
    y = np.array([[1., 5., 8.], [3., 2., 9.], [7., 8., 1.]])
    obj = df.OracleObjective(y, kind, 9.)
    z = np.array([.3, .7, .4])
    value, gradient = obj.value_gradient(z)
    for i in range(3):
        eps = np.zeros(3); eps[i] = 1e-6
        assert gradient[i] == pytest.approx((obj.value(z+eps)-obj.value(z-eps))/2e-6, rel=1e-6, abs=1e-6)
    for x in np.random.default_rng(20).uniform(0, 1, (30, 3)):
        assert obj.value(x) >= value+gradient@(x-z)-1e-10
    if kind == "variability":
        assert value == pytest.approx(np.var(y@z, ddof=1))
    else:
        assert value == pytest.approx(np.mean(np.maximum(9-y@z, 0)**2))


def test_hand_downside_sum_before_shortfall_and_zero_case():
    y = np.array([[4., 12.], [12., 4.]])
    obj = df.OracleObjective(y, "downside", 10.)
    assert obj.value(np.ones(2)) == 0  # Not the sum of separate merchant shortfalls.
    assert obj.value(np.array([1., 0.])) == 18
    z, info, log = df.optimize_oracle(obj, np.array([2., 1.]), count=2)
    assert info["certified"] and info["objective_value"] == 0 and log.empty
    assert df.opportunity(info)["opportunity"] == "no_baseline_risk"


def test_income_constraint_excludes_low_revenue_safe_solution():
    y = np.array([[1., 3., 2.], [5., 1., 2.], [1., 3., 2.]])
    obj = df.OracleObjective(y, "variability")
    means = np.array([10., 9., 8.])
    z, info, _ = df.optimize_oracle(obj, means, count=1, retention=.8)
    assert z[2] == 1 and info["objective_value"] == 0
    z, info, _ = df.optimize_oracle(obj, means, count=1, retention=.8, actual_fees=np.array([10., 9., 1.]))
    assert z[2] == 0


def test_budget_limited_bound_is_not_impossibility_claim():
    y = np.array([[1., 4., 8.], [4., 2., 3.], [8., 9., 1.]])
    z, info, _ = df.optimize_oracle(df.OracleObjective(y, "variability"), np.array([10., 8., 6.]),
        count=2, retention=.99, budgets=[dict(max_cuts=0, node_limit=1, relative_gap=.001, max_swaps=0)])
    assert not info["certified"] and info["lower_bound"] == 0
    assert df.opportunity(info)["opportunity"] == "unresolved"
    assert z.sum() == 2


def test_bound_classification_uses_square_roots():
    info = dict(baseline_objective=100., objective_value=90.25, lower_bound=90.)
    assert df.opportunity(info)["demonstrated_reduction_pct"] == pytest.approx(5)
    assert df.opportunity(info)["opportunity"] == "demonstrated"
    info.update(objective_value=98., lower_bound=96.)
    assert df.opportunity(info)["opportunity"] == "ruled_out_by_bound"
    info.update(lower_bound=0.)
    assert df.opportunity(info)["opportunity"] == "unresolved"


def test_nested_sets_use_bound_not_incumbent_monotonicity():
    t = pd.DataFrame(dict(as_of="a", method="m", retention=[.99, .95],
                          variance=[90., 92.], lower_bound=[89., 88.]))
    assert df.check_nested_bounds(t, ["as_of", "method"]).consistent.all()
    t.loc[1, "lower_bound"] = 91.
    with pytest.raises(ValueError, match="Nested"):
        df.check_nested_bounds(t, ["as_of", "method"])


def test_frontier_chronology_including_selections(toy, monkeypatch):
    daily, monthly, merchants = toy
    weekly = cf.weekly_panel(daily, merchants)
    cutoff = pd.Timestamp("2022-06-30")
    f, _ = cf.features_at_cutoff(weekly, monthly, merchants, cutoff)
    monkeypatch.setattr(df, "RETENTIONS", [.995, .95])
    e, s, log = df.practical_at(weekly, f, cutoff, "26w")
    changed = daily.copy(); changed.loc[changed.day.gt(cutoff), "sales"] *= 100
    cm = monthly.copy(); cm.loc[cm.month.gt(cutoff), "sales"] *= 100
    cw = cf.weekly_panel(changed, merchants)
    ff, _ = cf.features_at_cutoff(cw, cm, merchants, cutoff)
    ee, ss, ll = df.practical_at(cw, ff, cutoff, "26w")
    for a, b in [(f, ff), (e, ee), (s, ss), (log, ll)]:
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert e.common_reference_weekly_fees.nunique() == 1
    assert s.groupby(df.KEY).size().eq(100).all() and s.eligible.all()


def test_grid_specific_guardrails_and_zero_downside():
    e = pd.DataFrame(dict(as_of=pd.Timestamp("2022-06-30"), window="26w",
        method=["revenue_only", "raw_risk", "raw_risk"], retention=[1., .995, .95], common_reference_weekly_fees=100.))
    o = e[df.KEY].assign(status="observed", actual_quarter_fees=[1000., 994., 960.],
        actual_weekly_sd=[10., 9., 8.], actual_weekly_cv=[.1, .09, .08], downside_common=[0., 0., 2.], downside_own=0.)
    p = df.compare(e, o)
    assert p.actual_fee_guardrail_met.tolist() == [True, False, True]
    assert not p.downside_improves.any()
    assert p.normalized_downside.tolist() == [0., 0., .02]


def test_no_manifest_no_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(df, "OUT", tmp_path)
    with pytest.raises(ValueError, match="manifest"):
        df.validate_freeze()


def test_future_perturbation_can_change_oracle():
    y = np.array([[0., 2.], [4., 2.]])
    means = np.ones(2)
    a, _, _ = df.optimize_oracle(df.OracleObjective(y, "variability"), means, count=1)
    b, _, _ = df.optimize_oracle(df.OracleObjective(y[:, ::-1], "variability"), means, count=1)
    assert not np.array_equal(a, b)


def test_zero_realized_income_is_not_ineligible():
    obj = df.OracleObjective(np.zeros((3, 4)), "downside", 10.)
    z, info, _ = df.optimize_oracle(obj, np.ones(4), count=2, actual_fees=np.zeros(4))
    assert z.sum() == 2 and info["objective_value"] == 100


def test_solution_limit_keeps_feasible_incumbent(monkeypatch):
    from scipy.optimize import OptimizeResult
    def limited(*args, **kwargs):
        return OptimizeResult(status=4, message="HiGHS Status 16: Solution limit reached", x=None,
                              mip_dual_bound=None, mip_node_count=None)
    monkeypatch.setattr(df.rs, "milp", limited)
    x = np.array([[2., 4., 8.], [8., 1., 5.], [3., 6., 2.]])
    model = cf.Covariance(np.arange(3), x-x.mean(axis=0), x.var(axis=0, ddof=1), .5)
    z, info, _ = df.rs.optimize_portfolio(model, np.array([3., 2., 1.]), count=2)
    assert z.sum() == 2 and not info["certified"]
    monkeypatch.setattr(df, "milp", limited)
    z, info, _ = df.optimize_oracle(df.OracleObjective(x, "variability"), np.array([3., 2., 1.]), count=2)
    assert z.sum() == 2 and not info["certified"] and info["budget_tier"] == 2
