"""Binary feasibility, certified bounds, diversification, chronology and fair downside scoring."""

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
import cash_flow_variability as cf
import risk_aware_selection as rs


def covariance(x, alpha):
    centered = x-x.mean(axis=0)
    return cf.Covariance(np.arange(1, x.shape[1]+1), centered, x.var(axis=0, ddof=1), alpha)


@pytest.mark.parametrize("alpha", [0., .4, 1.])
def test_optimizer_matches_exhaustive_search_and_bounds(alpha):
    rng = np.random.default_rng(2026)
    model = covariance(rng.normal(size=(12, 7))*np.arange(1, 8), alpha)
    means = np.array([12., 11., 10., 9., 8., 6., 5.])
    settings = dict(max_cuts=100, node_limit=1000, relative_gap=1e-9, max_swaps=100)
    z, info, log = rs.optimize_portfolio(model, means, count=3, retention=.9, settings=settings)
    brute = min(cf.portfolio_volatility(model, model.ids[list(c)], "shrunk")**2
                for c in combinations(range(7), 3) if means[list(c)].sum() >= .9*33)
    assert z.sum() == 3 and set(z) <= {0., 1.} and means@z >= .9*33
    assert info["variance"] == pytest.approx(brute, rel=1e-8)
    assert info["lower_bound"] <= brute+1e-6
    assert info["certified"] and info["gap_upper_fraction"] <= 1e-9
    z2, info2, log2 = rs.optimize_portfolio(model, means, count=3, retention=.9, settings=settings)
    np.testing.assert_array_equal(z, z2)
    assert info == info2
    pd.testing.assert_frame_equal(log, log2, check_exact=True)


def test_binary_low_rank_objective_equals_dense_covariance():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(26, 8))
    model = covariance(x, .35)
    b = np.sqrt(.65)*model.centered/np.sqrt(25)
    d = .35*model.variances
    z = np.array([1., 0., 0., 1., 1., 0., 0., 1.])
    q = cf.covariance_submatrix(model, model.ids, "shrunk").to_numpy()
    assert rs.binary_objective(b, d, z) == pytest.approx(z@q@z)


def test_covariance_selects_offsetting_merchant_not_individual_sd_only():
    shock = np.array([1., -1., 1., -1.])
    x = np.column_stack([100+shock, 100+shock, 100-shock, np.full(4, 100.)])
    model = covariance(x, .2)
    means = np.array([10., 10., 9.95, 1.])
    z, info, _ = rs.optimize_portfolio(model, means, count=2)
    assert z[2] == 1 and z[3] == 0
    assert z[0]+z[1] == 1
    assert info["variance"] < cf.portfolio_volatility(model, [1, 2], "shrunk")**2
    assert means@z >= .99*20


def test_zero_variance_ties_keep_numeric_abn_baseline():
    model = covariance(np.ones((4, 5)), .5)
    z, info, _ = rs.optimize_portfolio(model, np.ones(5), count=3)
    assert np.flatnonzero(z).tolist() == [0, 1, 2]
    assert info["variance"] == 0 and info["certified"]


def test_budget_limited_result_is_feasible_and_not_falsely_certified():
    x = np.array([[1., 2., 3., 4.], [4., 1., 5., 2.], [2., 4., 1., 3.]])
    model = covariance(x, .5)
    z, info, _ = rs.optimize_portfolio(model, np.array([10., 9., 8., 7.]), count=2,
        settings=dict(max_cuts=0, node_limit=1, relative_gap=.001, max_swaps=100))
    assert z.sum() == 2 and np.array([10., 9., 8., 7.])@z >= .99*19
    assert info["status"] == "bounded_best_found" and not info["certified"]


@pytest.fixture
def toy():
    dates, ids = pd.date_range("2021-03-01", "2022-09-30"), np.arange(1, 104)
    daily = pd.MultiIndex.from_product([ids, dates], names=["merchant_abn", "day"]).to_frame(index=False)
    daily["sales"] = (daily.merchant_abn+10)*(10+(daily.day.dt.dayofyear%7))
    daily["orders"] = 1
    daily.loc[daily.merchant_abn.eq(103), ["sales", "orders"]] = 0
    monthly = daily.assign(month=daily.day.dt.to_period("M").dt.to_timestamp(),
        sales_26=np.where(daily.day.dt.day.le(26), daily.sales, 0)).groupby(["merchant_abn", "month"], as_index=False)[["sales", "orders", "sales_26"]].sum()
    merchants = pd.DataFrame(dict(merchant_abn=ids, name="toy", category="toy", take_rate=5., valid_take_rate=True))
    return daily, monthly, merchants


def test_exact_eligibility_and_full_future_perturbation(toy):
    daily, monthly, merchants = toy
    weekly = cf.weekly_panel(daily, merchants)
    f, _ = cf.features_at_cutoff(weekly, monthly, merchants, pd.Timestamp("2022-06-30"))
    e, selected, log = rs.select_at(weekly, f, pd.Timestamp("2022-06-30"), "26w")
    assert selected.groupby("method").size().eq(100).all()
    assert selected.eligible.all() and 103 not in set(selected.merchant_abn)
    assert e.expected_fee_retention_pct.ge(99-1e-8).all()
    assert e.common_reference_weekly_fees.nunique() == 1
    assert "solver_status" in e and "status" not in e  # Outcome status must remain distinct.
    changed = daily.copy()
    changed.loc[changed.day.gt("2022-06-30"), "sales"] *= 1000
    changed_monthly = monthly.copy()
    changed_monthly.loc[changed_monthly.month.gt("2022-06-30"), "sales"] *= 1000
    altered = cf.weekly_panel(changed, merchants)
    f2, _ = cf.features_at_cutoff(altered, changed_monthly, merchants, pd.Timestamp("2022-06-30"))
    e2, selected2, log2 = rs.select_at(altered, f2, pd.Timestamp("2022-06-30"), "26w")
    for a,b in [(e,e2),(selected,selected2),(log,log2)]:
        pd.testing.assert_frame_equal(a,b,check_exact=True)


def test_full_month_fees_and_whole_week_downside_are_distinct(toy):
    daily, monthly, merchants = toy
    weekly = cf.weekly_panel(daily, merchants)
    chosen = pd.DataFrame(dict(merchant_abn=np.arange(1,101)))
    estimate = dict(as_of=pd.Timestamp("2022-06-30"), window="26w", method="raw_risk",
                    common_reference_weekly_fees=50000., own_reference_weekly_fees=40000.)
    row, series = rs.portfolio_outcome(weekly, monthly, daily, merchants, estimate, chosen)
    direct = daily.loc[daily.merchant_abn.le(100)&daily.day.gt("2022-06-30"), "sales"].sum()*.05
    assert row["actual_quarter_fees"] == pytest.approx(direct)
    assert row["future_weeks"] == 12
    assert series.week_end.min() == pd.Timestamp("2022-07-10")
    assert series.week_end.max() == pd.Timestamp("2022-09-25")
    assert row["actual_quarter_fees"] > series.gross_fees.sum()
    assert row["downside_common"] == pytest.approx(np.sqrt(np.mean(np.maximum(50000-series.gross_fees,0)**2)))
    estimate["as_of"] = pd.Timestamp("2022-09-30")
    missing, series = rs.portfolio_outcome(weekly, monthly, daily, merchants, estimate, chosen)
    assert missing["status"] == "unavailable" and series.empty


def test_common_reference_does_not_reward_lower_income_reference():
    x = np.array([80., 100., 120.])
    y = .9*x
    assert rs.risk.downside(y, 100.) > rs.risk.downside(x, 100.)
    assert rs.risk.downside(y, 90.) < rs.risk.downside(x, 100.)


def test_zero_baseline_downside_and_realized_fee_guardrail():
    estimates = pd.DataFrame(dict(as_of=pd.Timestamp("2022-06-30"), window="26w", method=rs.METHODS,
        common_reference_weekly_fees=100., solver_status="certified_gap"))
    outcomes = estimates[["as_of","window","method"]].assign(status="observed", future_weeks=12,
        actual_quarter_fees=[1000., 980., 995.], actual_weekly_sd=[10., 9., 9.5],
        actual_weekly_cv=[.1, .09, .095], downside_common=[0., 2., 0.], downside_own=[0., 1., 0.])
    p=rs.compare_outcomes(estimates,outcomes).set_index("method")
    assert p.downside_change_pct.isna().all()
    assert p.loc["raw_risk","downside_change_aud"] == 2
    assert not p.loc["raw_risk","actual_fee_guardrail_met"]
    assert p.loc["detrended_risk","actual_fee_guardrail_met"]
    assert p.loc["detrended_risk","downside_tied"]


def test_invalid_inputs_rejected():
    model = covariance(np.ones((4, 3)), .5)
    with pytest.raises(ValueError, match="Invalid selection dimensions"):
        rs.optimize_portfolio(model, np.ones(3), count=4)
    with pytest.raises(ValueError, match="Invalid fee inputs"):
        rs.optimize_portfolio(model, np.ones(3), count=2, retention=1.1)
