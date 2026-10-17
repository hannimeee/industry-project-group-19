"""Forecast semantics and chronology without requiring downloaded neural weights."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
from timesfm_comparison import past_contexts, quantile_totals, seasonal_naive


def test_quantile_integral_is_explicit_and_differs_from_median():
    q = np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 100.],
                   [20, 20, 20, 20, 20, 20, 20, 20, 20.]]])
    mean, median = quantile_totals(q)
    # Day 1: triangle from .8-.9 plus constant upper .1 tail = 5+10.
    assert mean.tolist() == pytest.approx([15 + 20])
    assert median.tolist() == [20]


def test_weekly_repetition_and_exact_calendar_horizon():
    assert seasonal_naive(np.array([[100, 1, 2, 3, 4, 5, 6, 7.]]), 9).item() == 28 + 1 + 2
    origin = pd.Timestamp("2022-06-30")
    assert (origin + pd.offsets.MonthEnd(3) - origin).days == 92
    origin = pd.Timestamp("2022-01-31")
    assert (origin + pd.offsets.MonthEnd(3) - origin).days == 89


def test_future_perturbation_cannot_enter_context():
    dates = pd.date_range("2021-03-01", "2021-04-30")
    wide = pd.DataFrame(np.ones((2, len(dates))), index=[11, 22], columns=dates)
    origin = pd.Timestamp("2021-03-31")
    original = past_contexts(wide, [22, 11], origin)
    wide.loc[:, wide.columns > origin] *= 999999
    assert np.array_equal(past_contexts(wide, [22, 11], origin), original)
    assert original.shape == (2, 31)


def test_invalid_quantiles_and_missing_context_are_rejected():
    with pytest.raises(ValueError, match="Nonmonotone"):
        quantile_totals(np.array([[[9, 8, 7, 6, 5, 4, 3, 2, 1]]]))
    with pytest.raises(ValueError, match="Invalid daily history"):
        past_contexts(pd.DataFrame([[1.]], index=[1], columns=[pd.Timestamp("2021-03-01")]),
                      [1], pd.Timestamp("2021-03-02"))
