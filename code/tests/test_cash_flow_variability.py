"""Check weekly fee arithmetic, covariance semantics, chronology and saved interfaces."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
import cash_flow_variability as cf


def monthly_from_daily(daily):
    """Independently aggregate complete toy months, including first-26-day sales."""
    d = daily.assign(month=daily.day.dt.to_period("M").dt.to_timestamp(),
                     sales_26=np.where(daily.day.dt.day.le(26), daily.sales, 0))
    return d.groupby(["merchant_abn", "month"], as_index=False)[["sales", "orders", "sales_26"]].sum()


@pytest.fixture
def toy():
    """A full 110-merchant calendar, with one always-inactive merchant."""
    ids = np.arange(1, 111)
    dates = pd.date_range("2021-03-01", "2021-12-31")
    daily = pd.MultiIndex.from_product([ids, dates], names=["merchant_abn", "day"]).to_frame(index=False)
    daily["sales"] = ((daily.merchant_abn + 1) * (1 + (daily.day.dt.dayofyear // 7) % 5) + daily.day.dt.weekday).astype(float)
    daily.loc[daily.merchant_abn.eq(110), "sales"] = 0
    daily["orders"] = daily.sales.gt(0).astype(int)
    merchants = pd.DataFrame(dict(merchant_abn=ids, name=[str(i) for i in ids], category="toy",
                                 take_rate=(1 + ids % 7).astype(float), valid_take_rate=True))
    return daily, monthly_from_daily(daily), merchants


def test_hand_calculated_fees_and_complete_weeks():
    dates = pd.date_range("2021-03-01", "2021-03-24")
    d = pd.DataFrame(dict(merchant_abn=1, day=dates, sales=[10.] * 7 + [20.] * 7 + [30.] * 7 + [900.] * 3, orders=1))
    m = pd.DataFrame(dict(merchant_abn=[1], take_rate=[10.]))
    weekly = cf.weekly_panel(d, m)
    assert weekly.gross_fees.tolist() == [7., 14., 21.]
    assert weekly.gross_fees.std(ddof=1) == 7
    assert weekly.orders.sum() == 21
    assert weekly.week_start.dt.weekday.eq(0).all()
    assert weekly.week_end.dt.weekday.eq(6).all()
    d.loc[d.day.between("2021-03-08", "2021-03-14"), ["sales", "orders"]] = 0
    assert cf.weekly_panel(d, m).gross_fees.tolist() == [7., 0., 21.]
    # Partial leading weeks also stay out, without discarding later complete weeks.
    assert cf.weekly_panel(d.loc[d.day.ge("2021-03-03")], m).gross_fees.tolist() == [0., 21.]


def test_missing_duplicate_dates_and_invalid_units_are_rejected(toy):
    daily, _, merchants = toy
    with pytest.raises(ValueError, match="Incomplete merchant calendar"):
        cf.weekly_panel(daily.iloc[1:], merchants)
    with pytest.raises(ValueError, match="Duplicate merchant-day"):
        cf.weekly_panel(pd.concat([daily, daily.iloc[:1]]), merchants)
    without_day = daily.loc[daily.day.ne(pd.Timestamp("2021-06-01"))]
    with pytest.raises(ValueError, match="Incomplete daily coverage"):
        cf.weekly_panel(without_day, merchants)
    invalid = merchants.copy()
    invalid.loc[0, "take_rate"] = 500
    with pytest.raises(ValueError, match="Invalid take rate"):
        cf.weekly_panel(daily, invalid)
    with pytest.raises(ValueError, match="Incomplete daily coverage"):
        cf.validate_daily(daily.loc[daily.day.gt(daily.day.min())], merchants, daily.day.min(), daily.day.max())


def test_sunday_and_friday_cutoffs_and_known_window():
    assert cf.risk_window("2022-09-25")[-1] == pd.Timestamp("2022-09-25")
    dates = cf.risk_window("2022-09-30")
    assert len(dates) == 26
    assert dates[-1] == pd.Timestamp("2022-09-25")
    assert dates[0] - pd.Timedelta(days=6) == pd.Timestamp("2022-03-28")


def test_covariance_diversification_and_id_order():
    offsetting = np.array([[1., 3.], [2., 2.], [3., 1.]])
    model = cf.estimate_covariance(offsetting, [11, 22], [True, True])
    assert model.variances.tolist() == [1., 1.]
    np.testing.assert_allclose(cf.covariance_submatrix(model, [22, 11], "sample"), [[1, -1], [-1, 1]])
    assert cf.covariance_submatrix(model, [22, 11], "sample").index.tolist() == [22, 11]
    assert cf.portfolio_volatility(model, [11, 22], "sample") == 0
    regularized = replace(model, alpha=.25)
    assert cf.portfolio_volatility(regularized, [11, 22], "shrunk") == pytest.approx(np.sqrt(.5))
    synchronized = cf.estimate_covariance(np.array([[1, 1], [2, 2], [3, 3]]), [11, 22], [True, True])
    assert cf.portfolio_volatility(synchronized, [11, 22], "sample") == 2
    with pytest.raises(ValueError, match="unique merchants"):
        cf.portfolio_volatility(model, [11, 11], "sample")
    with pytest.raises(ValueError, match="Unknown merchant"):
        cf.covariance_submatrix(model, [33], "sample")
    with pytest.raises(ValueError, match="Estimator"):
        cf.portfolio_volatility(model, [11], "mystery")


def test_shrinkage_matches_standardized_reference_and_preserves_variances():
    rng = np.random.default_rng(2026)
    x = rng.uniform(1, 100, size=(26, 8)) * np.arange(1, 9)
    model = cf.estimate_covariance(x, np.arange(8), np.ones(8, dtype=bool))
    std = x.std(axis=0, ddof=1)
    ref = LedoitWolf(assume_centered=True, store_precision=False).fit((x - x.mean(0)) / std)
    expected = ref.covariance_ * (26 / 25) * np.outer(std, std)
    actual = cf.covariance_submatrix(model, list(range(8)), "shrunk").to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(np.diag(actual), std ** 2)
    assert np.linalg.eigvalsh(actual).min() >= -1e-7
    assert cf.portfolio_volatility(model, [7, 1, 4], "sample") == pytest.approx(x[:, [7, 1, 4]].sum(1).std(ddof=1))


def test_constants_and_ineligible_histories_do_not_fit_shrinkage():
    x = np.column_stack([np.arange(26), np.arange(26) ** 2, np.zeros(26), np.ones(26) * 7])
    model = cf.estimate_covariance(x, [1, 2, 3, 4], [True, True, False, False])
    assert model.variances[2:].tolist() == [0, 0]
    assert cf.portfolio_volatility(model, [3, 4], "shrunk") == 0
    changed = x.copy()
    changed[:, 2:] = np.arange(26)[:, None] * 1000
    assert cf.estimate_covariance(changed, [1, 2, 3, 4], [True, True, False, False]).alpha == model.alpha
    constant = cf.estimate_covariance(np.ones((26, 2)), [1, 2], [True, True])
    assert constant.alpha == 0
    assert cf.portfolio_volatility(constant, [1, 2], "sample") == 0


def test_features_expected_return_and_inactivity(toy):
    daily, monthly, merchants = toy
    weekly = cf.weekly_panel(daily, merchants)
    features, model = cf.features_at_cutoff(weekly, monthly, merchants, "2021-09-30")
    assert len(features) == 110 and features.n_weeks.eq(26).all()
    one = features.set_index("merchant_abn").loc[1]
    past_sales = daily.loc[daily.merchant_abn.eq(1) & daily.day.le("2021-09-30"), "sales"].sum()
    assert one.expected_monthly_fees == pytest.approx(past_sales / 7 * .02)
    weeks = weekly.loc[weekly.merchant_abn.eq(1) & weekly.week_end.isin(cf.risk_window("2021-09-30"))]
    assert one.weekly_volatility == pytest.approx(weeks.gross_fees.std(ddof=1))
    assert one.weekly_mean_fees == pytest.approx(weeks.gross_fees.mean())
    inactive = features.set_index("merchant_abn").loc[110]
    assert inactive.weekly_volatility == 0 and not inactive.eligible and inactive.active_weeks == 0
    selected = cf.baseline_selection(features)
    assert len(selected) == selected.merchant_abn.nunique() == 100
    assert 110 not in set(selected.merchant_abn)
    with pytest.raises(ValueError, match="Incomplete 26-week"):
        cf.features_at_cutoff(weekly.loc[weekly.week_end.ne(pd.Timestamp("2021-09-19"))], monthly, merchants, "2021-09-30")


def test_future_transaction_perturbation_cannot_change_past(toy):
    daily, monthly, merchants = toy
    weekly = cf.weekly_panel(daily, merchants)
    features, model = cf.features_at_cutoff(weekly, monthly, merchants, "2021-09-30")
    modified = daily.copy()
    modified.loc[modified.day.gt("2021-09-30"), "sales"] *= 1000
    f2, m2 = cf.features_at_cutoff(cf.weekly_panel(modified, merchants), monthly_from_daily(modified), merchants, "2021-09-30")
    pd.testing.assert_frame_equal(features, f2, check_exact=True)
    assert model.alpha == m2.alpha
    np.testing.assert_array_equal(model.centered, m2.centered)
    selected = cf.baseline_selection(features)
    pd.testing.assert_frame_equal(selected, cf.baseline_selection(f2))
    for estimator in ["sample", "shrunk"]:
        assert cf.portfolio_volatility(model, selected.merchant_abn, estimator) == cf.portfolio_volatility(m2, selected.merchant_abn, estimator)


def test_deterministic_ties_and_eligibility():
    f = pd.DataFrame(dict(merchant_abn=np.arange(102, 0, -1), expected_monthly_fees=100., eligible=True))
    f.loc[f.merchant_abn.eq(102), ["expected_monthly_fees", "eligible"]] = [999999., False]
    selected = cf.baseline_selection(f)
    assert selected.merchant_abn.tolist() == list(range(1, 101))
    assert selected["rank"].tolist() == list(range(1, 101))


def test_covariance_roundtrip_and_future_week_boundaries(toy, tmp_path):
    daily, monthly, merchants = toy
    weekly = cf.weekly_panel(daily, merchants)
    features, model = cf.features_at_cutoff(weekly, monthly, merchants, "2021-09-30")
    cf.save_covariance(model, tmp_path, pd.Timestamp("2021-09-30"))
    restored = cf.load_covariance(tmp_path)
    np.testing.assert_array_equal(model.ids, restored.ids)
    np.testing.assert_array_equal(model.centered, restored.centered)
    np.testing.assert_array_equal(model.variances, restored.variances)
    assert model.alpha == restored.alpha
    selected = cf.baseline_selection(features)
    metrics, history = cf.portfolio_diagnostics(weekly, selected, restored, daily.day.max())
    future = history.loc[history.phase.eq("future")]
    assert metrics["future_weeks"] == len(future) == 12
    assert (future.week_end - pd.Timedelta(days=6)).gt("2021-09-30").all()
    assert future.week_end.le("2021-12-31").all()
    assert metrics["future_weekly_sd"] == pytest.approx(future.gross_fees.std(ddof=1))
    # Incomplete future quarters are unscored, rather than treated as zero outcomes.
    metrics, history = cf.portfolio_diagnostics(weekly, selected, restored, pd.Timestamp("2021-11-30"))
    assert metrics["future_weeks"] == 0 and np.isnan(metrics["future_weekly_sd"])
    assert history.phase.eq("history").all()
