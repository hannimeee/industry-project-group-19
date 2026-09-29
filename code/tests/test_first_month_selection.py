"""Independent small examples for fixed selection and hindsight accounting."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_month_selection import FIRST, freeze_selection, score_month


def example():
    merchants = pd.DataFrame(dict(merchant_abn=[4, 3, 2, 1], name="m", category="c",
                                 take_rate=[5., 10., 5., 10.], valid_take_rate=True))
    march = pd.DataFrame(dict(merchant_abn=[4, 3, 2, 1], month=FIRST, sales=[100., 50., 60., 0.]))
    future = pd.DataFrame(dict(merchant_abn=[4, 3, 2, 1], month=pd.Timestamp("2021-04-01"),
                               sales=[0., 10., 100., 200.]))
    return merchants, march, future


def test_hand_calculated_ties_fees_zero_activity_and_hindsight():
    merchants, march, future = example()
    frozen = freeze_selection(march, merchants, top_n=2)
    assert frozen.loc[frozen.selected, "merchant_abn"].tolist() == [3, 4]
    assert frozen.march_fees.tolist() == [5., 5., 3., 0.]
    scored, metrics = score_month(frozen, future, top_n=2)
    assert len(scored) == 4
    assert scored.loc[scored.hindsight_selected, "merchant_abn"].tolist() == [1, 2]
    assert metrics["selected_actual_fees"] == 1.
    assert metrics["hindsight_best_fees"] == 25.
    assert metrics["capture_pct"] == 4.
    assert metrics["overlap"] == 0
    assert metrics["fee_wape_pct"] == pytest.approx(100 * 31 / 26)


def test_future_changes_and_partial_month_do_not_change_selection():
    merchants, march, future = example()
    frozen = freeze_selection(pd.concat([march, future]), merchants, top_n=2)
    future.sales *= 1000
    changed = freeze_selection(pd.concat([march, future]), merchants, top_n=2)
    pd.testing.assert_frame_equal(frozen, changed, check_exact=True)
    full, _ = score_month(frozen, future, top_n=2)
    partial, _ = score_month(frozen, future, forecast_scale=26 / 31, top_n=2)
    pd.testing.assert_series_equal(full.selected, partial.selected)
    assert partial.predicted_fees.sum() == pytest.approx(full.predicted_fees.sum() * 26 / 31)


def test_missing_outcome_is_not_silently_imputed_as_zero():
    merchants, march, future = example()
    frozen = freeze_selection(march, merchants, top_n=2)
    with pytest.raises(ValueError, match="Incomplete"):
        score_month(frozen, future.iloc[1:], top_n=2)
