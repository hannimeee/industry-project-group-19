"""Hand arithmetic, trend blindness, aggregation, residual covariance and frozen references."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
import cash_flow_variability as cf
import cash_flow_risk_comparison as risk


@pytest.fixture
def histories():
    dates = pd.date_range("2021-03-07", periods=82, freq="W-SUN")
    ids = np.arange(1, 111)
    frame = pd.MultiIndex.from_product([dates, ids], names=["week_end", "merchant_abn"]).to_frame(index=False)
    t = frame.week_end.map(dict(zip(dates, range(len(dates)))))
    frame["gross_fees"] = frame.merchant_abn * (10 + .1*t + (t%5))
    frame.loc[frame.merchant_abn.eq(110), "gross_fees"] = 0
    base = pd.DataFrame(dict(merchant_abn=ids, name="toy", category="toy", take_rate=5.,
        eligible=ids != 110, eligibility_reason=np.where(ids == 110, "inactive", "eligible"),
        expected_monthly_fees=ids*100., as_of=pd.Timestamp("2022-06-30")))
    return frame, base, cf.baseline_selection(base)


def test_hand_calculated_trend_variance_and_downside():
    stats, fitted, residual = risk.decompose(np.array([[1.], [2.], [6.], [7.]]))
    r = stats.iloc[0]
    assert r.weekly_mean == 4
    assert r.trend_slope_per_week == pytest.approx(2.2)
    np.testing.assert_allclose(fitted[:, 0], [.7, 2.9, 5.1, 7.3])
    np.testing.assert_allclose(residual[:, 0], [.3, -.9, .9, -.3])
    assert r.raw_sd == pytest.approx(np.sqrt(26/3))
    assert r.trend_variance == pytest.approx(24.2/3)
    assert r.residual_descriptive_sd**2 == pytest.approx(1.8/3)
    assert r.residual_sd**2 == pytest.approx(1.8/2)
    assert r.downside_mean == pytest.approx(np.sqrt(13/4))
    assert r.downside_trend == pytest.approx(np.sqrt(.9/4))


def test_growth_and_decline_have_equal_raw_risk_but_zero_residual_risk():
    rows = risk.toy_scenarios().set_index("scenario")
    up, down = rows.loc["steady_growth"], rows.loc["steady_decline"]
    assert up.trend_slope_per_week == 5 and down.trend_slope_per_week == -5
    for name in ["weekly_mean", "raw_sd", "downside_mean"]:
        assert up[name] == pytest.approx(down[name])
    assert up.raw_sd > 0 and up.downside_mean > 0
    assert up.residual_sd == down.residual_sd == 0
    assert up.downside_trend == down.downside_trend == 0
    assert up.trend_variance_share == down.trend_variance_share == 1
    assert rows.loc["constant", "raw_sd"] == 0
    assert np.isnan(rows.loc["constant", "trend_variance_share"])


def test_df_correction_can_raise_residual_sd_when_trend_is_zero():
    stats, _, _ = risk.decompose(np.array([[11.], [9.], [9.], [11.]]))
    r = stats.iloc[0]
    assert r.trend_slope_per_week == 0
    assert r.residual_descriptive_sd == r.raw_sd
    assert r.residual_sd == pytest.approx(np.sqrt(2))
    assert r.residual_sd > r.raw_sd


def test_downside_divides_by_all_weeks_and_aggregates_before_clipping():
    assert risk.downside([0., 10., 10., 10.], 10.) == 5  # NOT 10 (only shortfall weeks).
    x = np.array([[0., 20.], [20., 0.], [0., 20.], [20., 0.]])
    individual = risk.downside(x, [10., 10.])
    assert (individual > 0).all()
    assert risk.downside(x.sum(axis=1), 20.) == 0


def test_residual_covariance_matches_direct_portfolio_and_survives_roundtrip(histories, tmp_path):
    weekly, base, selected = histories
    f, portfolio, h, model = risk.estimate_at(weekly, base, selected, "2022-06-30", "26w")
    dates = risk.history_dates(weekly, "2022-06-30", "26w")
    sub = cf.covariance_submatrix(model, selected.merchant_abn, "sample")
    assert np.sqrt(sub.to_numpy().sum()) == pytest.approx(portfolio["residual_sd"])
    assert portfolio["raw_sd"] == pytest.approx(h.gross_fees.std(ddof=1))
    assert portfolio["trend_variance"] + portfolio["residual_descriptive_sd"]**2 == pytest.approx(portfolio["raw_sd"]**2)
    assert not f.loc[f.merchant_abn.eq(110), "eligible"].item()
    assert f.loc[f.merchant_abn.eq(110), "residual_sd"].item() == 0
    risk.save_residual_covariance(model, dates, tmp_path)
    restored = cf.load_covariance(tmp_path)
    np.testing.assert_array_equal(restored.centered, model.centered)
    np.testing.assert_array_equal(restored.variances, model.variances)
    assert restored.alpha == model.alpha


def test_offsetting_and_shared_residual_shocks():
    shock = np.array([1., -1., -1., 1.])  # Orthogonal to intercept and linear trend.
    x = np.column_stack([10+np.arange(4)+shock, 20+2*np.arange(4)-shock])
    individual, _, residual = risk.decompose(x)
    assert individual.residual_sd.gt(0).all()
    model = cf.estimate_covariance(residual*np.sqrt(3/2), [1, 2], [True, True])
    assert cf.portfolio_volatility(model, [1, 2], "sample") == 0
    total, _, _ = risk.decompose(x.sum(axis=1)[:, None])
    assert total.residual_sd.item() == 0
    shared, _, _ = risk.decompose(np.column_stack([10+shock, 20+shock]).sum(axis=1)[:, None])
    assert shared.residual_sd.item() == pytest.approx(2*np.sqrt(2))


def test_future_changes_cannot_change_any_historical_measure(histories):
    weekly, base, selected = histories
    changed = weekly.copy()
    changed.loc[changed.week_end.gt("2022-06-30"), "gross_fees"] *= 1000
    for window in risk.WINDOWS:
        f, p, h, model = risk.estimate_at(weekly, base, selected, "2022-06-30", window)
        f2, p2, h2, model2 = risk.estimate_at(changed, base, selected, "2022-06-30", window)
        pd.testing.assert_frame_equal(f, f2, check_exact=True)
        pd.testing.assert_frame_equal(pd.DataFrame([p]), pd.DataFrame([p2]), check_exact=True)
        pd.testing.assert_frame_equal(h, h2, check_exact=True)
        np.testing.assert_array_equal(model.centered, model2.centered)
        assert model.alpha == model2.alpha


def test_future_reference_is_frozen_and_elapsed_calendar_weeks_are_used():
    past = pd.DataFrame(dict(history_end=[pd.Timestamp("2022-06-26")], history_weeks=[4],
                             weekly_mean=[12.5], trend_slope_per_week=[5.]))
    dates = pd.date_range("2022-07-10", periods=3, freq="W-SUN")
    stats, line = risk.realized_measures(np.full((3, 1), 20.), dates, past)
    np.testing.assert_allclose(line[:, 0], [30., 35., 40.])
    assert stats.future_residual_sd.item() == 0  # A retrospective refit is flat.
    assert stats.future_downside_mean.item() == 0  # Historical mean stays 12.5.
    assert stats.future_trend_error_rms.item() == pytest.approx(np.sqrt((100+225+400)/3))
    assert stats.future_downside_trend.item() == stats.future_trend_error_rms.item()


def test_negative_forecast_clipped_only_in_future():
    past = pd.DataFrame(dict(history_end=[pd.Timestamp("2022-06-26")], history_weeks=[4],
                             weekly_mean=[12.5], trend_slope_per_week=[-5.]))
    dates = pd.date_range("2022-07-10", periods=3, freq="W-SUN")
    stats, line = risk.realized_measures(np.zeros((3, 1)), dates, past)
    assert (line == 0).all() and stats.forecast_clipped_weeks.item() == 3
    assert stats.future_downside_trend.item() == 0
    assert stats.future_downside_mean.item() == 12.5  # Complete decline remains visible.


def test_future_boundaries_no_partial_quarter_and_missing_weeks(histories):
    weekly, base, selected = histories
    f, p, _, _ = risk.estimate_at(weekly, base, selected, "2022-06-30", "26w")
    outcome, merchants, h = risk.evaluate_future(weekly, f, p, selected, pd.Timestamp("2022-09-30"))
    assert outcome["future_weeks"] == 12 and merchants.future_weeks.eq(12).all()
    assert h.week_end.min() == pd.Timestamp("2022-07-10")
    assert h.week_end.max() == pd.Timestamp("2022-09-25")
    missing, m, h = risk.evaluate_future(weekly, f, p, selected, pd.Timestamp("2022-08-31"))
    assert missing["status"] == "unavailable" and len(m) == len(h) == 0
    dates = risk.history_dates(weekly, "2022-06-30", "26w")
    with pytest.raises(ValueError, match="Incomplete merchant-week"):
        risk.history_matrix(weekly.drop(weekly.loc[weekly.week_end.eq(dates[0])].index[:1]), dates, base.merchant_abn)


def test_summary_uses_matched_dates_and_known_error():
    dates = pd.date_range("2022-02-28", periods=2, freq="ME")
    scored = pd.DataFrame(dict(as_of=dates, window="26w", measure="raw_sd", target="future_raw_sd", status="observed",
        estimate=[80., 140.], actual=[100., 120.], signed_error=[-20., 20.], absolute_error=20.))
    r = risk.summarize_errors(scored, dates, ["26w"], "toy").iloc[0]
    assert r.mean_absolute_error == 20 and r.mean_signed_error == 0 and r.underestimate_count == 1
    with pytest.raises(ValueError, match="Unmatched summary"):
        risk.summarize_errors(scored.iloc[:1], dates, ["26w"], "toy")


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="three finite"):
        risk.decompose(np.ones((2, 1)))
    with pytest.raises(ValueError, match="three finite"):
        risk.decompose(np.array([[1.], [2.], [np.nan]]))
    with pytest.raises(ValueError, match="Nonfinite downside"):
        risk.downside([1., np.nan], 5.)
