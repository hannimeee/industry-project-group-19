"""Independent arithmetic and historical-availability checks for simple rules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from naive_return_comparison import naive_forecasts


def history():
    return pd.DataFrame(dict(merchant_abn=1, month=pd.date_range("2021-03-01", "2022-09-01", freq="MS"),
                             sales=np.arange(1.,20.)))


def test_averages_blend_and_ewma_against_independent_arithmetic():
    result = naive_forecasts(history(), [1], pd.Timestamp("2021-08-31"))
    assert result["six_month_mean"].item() == 3*3.5
    assert result["blend_50_50"].item() == (15+10.5)/2
    reference = 3*pd.Series([1.,2.,3.,4.,5.,6.]).ewm(halflife=3, adjust=True).mean().iloc[-1]
    assert result["ewma_3m"].item() == pytest.approx(reference)


def test_seasonal_availability_and_exact_month_alignment():
    panel = history()
    assert "seasonal_naive_annual" not in naive_forecasts(panel, [1], pd.Timestamp("2022-01-31"))
    feb = naive_forecasts(panel, [1], pd.Timestamp("2022-02-28"))
    assert feb["seasonal_naive_annual"].item() == 1+2+3  # March-May 2021
    june = naive_forecasts(panel, [1], pd.Timestamp("2022-06-30"))
    assert june["seasonal_naive_annual"].item() == 5+6+7  # July-September 2021


def test_future_data_does_not_affect_rules_or_availability():
    panel = history()
    origin = pd.Timestamp("2022-02-28")
    first = naive_forecasts(panel, [1], origin)
    panel.loc[panel.month.gt(origin), "sales"] *= 99999
    second = naive_forecasts(panel, [1], origin)
    assert first.keys() == second.keys()
    for method in first:
        assert np.array_equal(first[method], second[method])


def test_zero_sales_are_retained_and_missing_month_is_rejected():
    panel = history()
    panel.loc[panel.month.eq("2021-08-01"), "sales"] = 0
    result = naive_forecasts(panel, [1], pd.Timestamp("2021-08-31"))
    assert result["six_month_mean"].item() == 3*(1+2+3+4+5+0)/6
    with pytest.raises(ValueError, match="Incomplete"):
        naive_forecasts(panel.loc[panel.month.ne("2021-06-01")], [1], pd.Timestamp("2021-08-31"))
