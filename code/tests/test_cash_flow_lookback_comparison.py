"""Matched-history comparisons, unavailable windows, future boundaries and error accounting."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cash_flow_variability as cf
import cash_flow_lookback_comparison as cmp


@pytest.fixture
def weekly_toy():
    """Complete histories with changing fee levels and a fixed eligible universe."""
    dates = pd.date_range("2021-03-07", periods=82, freq="W-SUN")
    ids = np.arange(1, 103)
    weekly = pd.MultiIndex.from_product([dates, ids], names=["week_end", "merchant_abn"]).to_frame(index=False)
    week_number = weekly.week_end.map(dict(zip(dates, range(82))))
    weekly["gross_fees"] = (weekly.merchant_abn + 1) * (10 + (week_number % 5) + week_number / 10)
    features = pd.DataFrame(dict(merchant_abn=ids, eligible=True, expected_monthly_fees=ids*100., as_of=pd.Timestamp("2022-06-30")))
    return weekly, features, cf.baseline_selection(features)


def test_window_boundaries_and_unavailability(weekly_toy):
    weekly, _, _ = weekly_toy
    assert cmp.history_dates(weekly, "2022-01-31", "52w") is None
    at_52 = cmp.history_dates(weekly, "2022-02-28", "52w")
    assert len(at_52) == 52
    assert at_52.equals(cmp.history_dates(weekly, "2022-02-28", "all"))
    assert cmp.history_dates(weekly, "2022-06-30", "26w")[0] == pd.Timestamp("2022-01-02")
    assert len(cmp.history_dates(weekly, "2022-06-30", "all")) == 69
    assert cmp.history_dates(weekly, "2022-06-26", "all")[-1] == pd.Timestamp("2022-06-26")
    assert cmp.history_dates(weekly, "2022-06-30", "52w")[-1] == pd.Timestamp("2022-06-26")


def test_history_gaps_are_not_shorter_windows(weekly_toy):
    weekly, features, selected = weekly_toy
    broken = weekly.loc[weekly.week_end.ne(pd.Timestamp("2022-03-13"))]
    with pytest.raises(ValueError, match="Missing calendar week"):
        cmp.history_dates(broken, "2022-06-30", "all")
    broken = weekly.drop(weekly.loc[weekly.week_end.eq("2022-03-13")].index[:1])
    with pytest.raises(ValueError, match="Incomplete merchant-week"):
        cmp.estimate_windows(broken, features, selected, pd.Timestamp("2022-06-30"))


def test_raw_estimates_equal_direct_summed_fee_sd(weekly_toy):
    weekly, features, selected = weekly_toy
    estimates, availability = cmp.estimate_windows(weekly, features, selected, pd.Timestamp("2022-06-30"))
    assert availability.available.all() and len(estimates) == 6
    assert estimates.selected_ids_digest.nunique() == 1
    assert estimates.expected_monthly_fees.nunique() == 1
    for row in estimates.loc[estimates.estimator.eq("sample")].itertuples():
        dates = cmp.history_dates(weekly, row.as_of, row.window)
        summed = weekly.loc[weekly.week_end.isin(dates) & weekly.merchant_abn.isin(selected.merchant_abn)].groupby("week_end").gross_fees.sum()
        assert row.estimated_weekly_sd == pytest.approx(summed.std(ddof=1))
        assert row.history_weeks == len(dates)


def test_future_perturbation_does_not_change_any_window(weekly_toy):
    weekly, features, selected = weekly_toy
    original, availability = cmp.estimate_windows(weekly, features, selected, pd.Timestamp("2022-06-30"))
    changed = weekly.copy()
    changed.loc[changed.week_end.gt("2022-06-30"), "gross_fees"] *= 1000
    modified, availability2 = cmp.estimate_windows(changed, features, selected, pd.Timestamp("2022-06-30"))
    pd.testing.assert_frame_equal(original, modified, check_exact=True)
    pd.testing.assert_frame_equal(availability, availability2, check_exact=True)


def test_52_and_all_identical_at_first_available_origin(weekly_toy):
    weekly, features, selected = weekly_toy
    estimates, _ = cmp.estimate_windows(weekly, features, selected, pd.Timestamp("2022-02-28"))
    cols = ["history_weeks", "history_start", "history_end", "estimated_weekly_sd", "shrinkage_alpha"]
    a = estimates.loc[estimates.window.eq("52w")].reset_index(drop=True)[cols]
    b = estimates.loc[estimates.window.eq("all")].reset_index(drop=True)[cols]
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_future_periods_exclude_crossing_weeks_and_stay_missing(weekly_toy):
    weekly, _, selected = weekly_toy
    row = cmp.future_outcome(weekly, selected, pd.Timestamp("2022-06-30"), pd.Timestamp("2022-09-30"))
    assert row["future_weeks"] == 12 and row["status"] == "observed"
    direct = weekly.loc[weekly.week_end.between("2022-07-10", "2022-09-25") & weekly.merchant_abn.isin(selected.merchant_abn)].groupby("week_end").gross_fees.sum()
    assert row["actual_weekly_sd"] == pytest.approx(direct.std(ddof=1))
    missing = cmp.future_outcome(weekly, selected, pd.Timestamp("2022-09-30"), pd.Timestamp("2022-09-30"))
    assert missing["future_weeks"] == 0 and np.isnan(missing["actual_weekly_sd"])
    estimates = pd.DataFrame(dict(as_of=[pd.Timestamp("2022-09-30")], estimated_weekly_sd=[100.]))
    scored = cmp.score_estimates(estimates, pd.DataFrame([missing]))
    assert pd.isna(scored.underestimated.iloc[0]) and np.isnan(scored.absolute_error.iloc[0])


def test_hand_calculated_errors_and_underestimation():
    origins = pd.date_range("2022-02-28", periods=2, freq="ME")
    estimates = pd.DataFrame([
        dict(as_of=origins[0], window="26w", estimator="sample", estimated_weekly_sd=80.),
        dict(as_of=origins[1], window="26w", estimator="sample", estimated_weekly_sd=140.),
        dict(as_of=origins[0], window="all", estimator="sample", estimated_weekly_sd=110.),
        dict(as_of=origins[1], window="all", estimator="sample", estimated_weekly_sd=130.)])
    outcomes = pd.DataFrame(dict(as_of=origins, actual_weekly_sd=[100., 120.], status="observed"))
    summary = cmp.summarize(cmp.score_estimates(estimates, outcomes), origins, ["26w", "all"]).set_index("window")
    assert summary.loc["26w", "mean_absolute_error"] == 20
    assert summary.loc["26w", "mean_signed_error"] == 0
    assert summary.loc["26w", "underestimate_count"] == 1
    assert summary.loc["26w", "mean_underestimate_amount"] == 10
    assert summary.loc["all", "mean_absolute_error"] == 10
    assert summary.loc["all", "underestimate_count"] == 0
    assert summary.loc["all", "wins_vs_26w"] == 2
    assert summary.loc["all", "mae_improvement_vs_26w_pct"] == 50


def test_common_origins_do_not_depend_on_outcome_values():
    origins = pd.date_range("2022-01-31", periods=3, freq="ME")
    availability = pd.DataFrame([dict(as_of=o, window=w, available=not(o==origins[0] and w=="52w"))
                                 for o in origins for w in cmp.WINDOWS])
    outcomes = pd.DataFrame(dict(as_of=origins, status=["observed", "observed", "insufficient_complete_outcome_period"],
                                actual_weekly_sd=[999., 1., np.nan]))
    assert cmp.common_origins(availability, outcomes, cmp.WINDOWS).tolist() == [origins[1]]
    assert cmp.common_origins(availability, outcomes, ["26w", "all"]).tolist() == list(origins[:2])
    outcomes["actual_weekly_sd"] = [1., 999., np.nan]
    assert cmp.common_origins(availability, outcomes, cmp.WINDOWS).tolist() == [origins[1]]
