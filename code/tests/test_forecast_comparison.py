"""Forecast horizon, chronological leakage and economic comparison checks."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
import forecast_comparison as exp


@pytest.fixture
def toy():
    ids = np.arange(1000, 1160)
    merchants = pd.DataFrame(dict(merchant_abn=ids, name="merchant", category="category",
                                 take_rate=3.0, valid_take_rate=True))
    panel = pd.concat([pd.DataFrame(dict(merchant_abn=ids, month=month,
                          sales=(ids - 990) * (j + 1.0), sales_26=(ids - 990) * (j + 1.0) * 0.8))
                       for j, month in enumerate(pd.date_range("2021-03-01", "2022-09-01", freq="MS"))],
                      ignore_index=True)
    return panel, merchants


def test_lags_and_three_month_target_by_hand(toy):
    panel, merchants = toy
    features = exp.build_features(panel, merchants)
    row = features.loc[features.origin.eq("2021-08-31") & features.merchant_abn.eq(1000)].iloc[0]
    assert row[exp.LAGS].tolist() == [60, 50, 40, 30, 20, 10]
    assert row.recent_monthly_sales == 50
    assert row.own_growth == 1.5
    train = exp.mdl.training_at(panel, features, pd.Timestamp("2021-11-30"), "3m")
    assert train.origin.nunique() == 1
    assert train.loc[train.merchant_abn.eq(1000), "target_sales"].item() == 70 + 80 + 90


def test_future_perturbation_cannot_change_training_or_forecast(toy):
    panel, merchants = toy
    origin = pd.Timestamp("2022-03-31")
    changed = panel.copy()
    changed.loc[changed.month.gt(origin), ["sales", "sales_26"]] *= 1000
    before = exp.build_features(panel, merchants, through=origin)
    after = exp.build_features(changed, merchants, through=origin)
    pd.testing.assert_frame_equal(before, after)
    a = exp.mdl.training_at(panel, before, origin, "3m")
    b = exp.mdl.training_at(changed, after, origin, "3m")
    pd.testing.assert_frame_equal(a, b)
    assert a.origin.max() == pd.Timestamp("2021-12-31")
    test = before.loc[before.origin.eq(origin) & before.eligible].reset_index(drop=True)
    prediction, _ = exp.pooled_prediction(a, test)
    repeated, _ = exp.pooled_prediction(b, test)
    assert np.array_equal(prediction, repeated)
    holt, _ = exp.damped_predictions(panel, test.iloc[:3], origin, 1)
    again, _ = exp.damped_predictions(changed, test.iloc[:3], origin, 1)
    assert np.array_equal(holt, again)


def test_future_zero_outcome_retained(toy):
    panel, merchants = toy
    panel.loc[panel.merchant_abn.eq(1000) & panel.month.ge("2022-04-01"), "sales"] = 0
    features = exp.build_features(panel, merchants)
    train = exp.mdl.training_at(panel, features, pd.Timestamp("2022-06-30"), "3m")
    assert train.loc[train.merchant_abn.eq(1000) & train.origin.eq("2022-03-31"), "target_sales"].item() == 0


def test_rank_ties_fee_conversion_and_change_attribution(toy):
    panel, merchants = toy
    features = exp.build_features(panel, merchants)
    candidates = features.loc[features.origin.eq("2022-06-30")].reset_index(drop=True)
    repeat = np.ones(160) * 1000
    alternative = repeat.copy()
    alternative[100] = 2000
    predictions = exp.make_predictions(candidates, dict(repeat=repeat, minimal=repeat, pooled_lags=alternative))
    base = predictions.loc[predictions.method.eq("repeat")]
    assert base.predicted_fees.eq(30).all()
    assert base.loc[base.selected, "merchant_abn"].tolist() == list(range(1000, 1100))
    scored = exp.attach_outcomes(predictions, panel)
    metrics, changes = exp.summarize(scored)
    new = metrics.loc[metrics.method.eq("pooled_lags")].iloc[0]
    assert new.overlap_with_repeat == 99
    # ABN 1100 replaces ABN 1099; next sales differ by 17+18+19, at 3%.
    assert new.delta_vs_repeat == pytest.approx(54 * 0.03)
    assert changes.fee_contribution.sum() == pytest.approx(54 * 0.03)


def test_damped_constant_history():
    result = exp.damped_one(np.ones(12) * 100)
    assert result["predicted_sales"] == 300
    assert result["status"] == "constant_history"


def test_incomplete_and_duplicate_history_rejected(toy):
    panel, merchants = toy
    with pytest.raises(ValueError, match="Incomplete"):
        exp.build_features(panel.iloc[1:], merchants)
    with pytest.raises(ValueError, match="Duplicate"):
        exp.build_features(pd.concat([panel, panel.iloc[:1]]), merchants)


def test_incomplete_future_target_rejected(toy):
    panel, merchants = toy
    features = exp.build_features(panel, merchants)
    candidates = features.loc[features.origin.eq("2022-06-30")].reset_index(drop=True)
    predictions = exp.make_predictions(candidates, dict(repeat=np.ones(160)))
    with pytest.raises(ValueError, match="Incomplete target"):
        exp.attach_outcomes(predictions, panel.loc[~panel.month.eq("2022-09-01")])
