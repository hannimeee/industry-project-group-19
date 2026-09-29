"""Tests for economic definitions, historical availability, and selection integrity."""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import abs_momentum_comparison as comparison


@pytest.fixture
def synthetic_data():
    """Provide a small complete panel with both mapping cohorts and changing industry signals."""
    ids = np.arange(1, 141)
    series_ids = list(comparison.SERIES)
    merchants = pd.DataFrame({"merchant_abn": ids, "category": "test retail", "take_rate": 4.0,
                              "valid_take_rate": True, "abs_matched": ids <= 70,
                              "series_id": [series_ids[i % 7] if i < 70 else "" for i in range(140)],
                              "match_quality": ["approximate" if i < 70 else "unmapped" for i in range(140)]})
    rows = []
    for t, month in enumerate(pd.date_range(comparison.START, comparison.END, freq="MS")):
        for merchant in ids:
            rows.append((merchant, month, (100 + merchant * 8) * (1 + .01 * t + .02 * np.sin(merchant + t)), 1))
    panel = pd.DataFrame(rows, columns=["merchant_abn", "month", "sales", "orders"])
    abs_rows = []
    for t, origin in enumerate(comparison.FEATURE_ORIGINS):
        for j, series_id in enumerate(series_ids):
            abs_rows.append(dict(origin=origin, series_id=series_id, abs_momentum=.03 * np.sin(t + j),
                                 abs_reference_month=origin.to_period("M").to_timestamp() - pd.DateOffset(months=2),
                                 abs_published_at=(origin - pd.Timedelta(days=2)).tz_localize("Australia/Melbourne"),
                                 abs_source_url="https://www.abs.gov.au/test.xls", abs_page_url="https://www.abs.gov.au/test"))
    return panel, merchants, pd.DataFrame(abs_rows)


def make_features(panel, merchants, abs_features):
    """Construct each historical row independently using the production feature builder."""
    return pd.concat([comparison.features_at(panel, merchants, origin, abs_features)
                      for origin in comparison.FEATURE_ORIGINS], ignore_index=True)


def test_hand_calculated_growth_targets_and_zero_sales(synthetic_data):
    """Verify period boundaries, an observed zero month, and percentage take-rate units."""
    panel, merchants, abs_features = synthetic_data
    # March-August: 1,2,3 then 4,5,6; September-November: 0,8,9.
    mask = panel.merchant_abn.eq(1) & panel.month.le("2021-11-01")
    panel.loc[mask, "sales"] = [1, 2, 3, 4, 5, 6, 0, 8, 9]
    features = comparison.features_at(panel, merchants, "2021-08-31", abs_features)
    row = features.set_index("merchant_abn").loc[1]
    assert row.recent_monthly_sales == 5
    assert row.own_growth == 1.5
    assert row.log_own_growth == pytest.approx(np.log(2.5))
    target = comparison.targets_at(panel, pd.Index([1], name="merchant_abn"), "2021-08-31")
    assert target.loc[1] == 17
    assert target.loc[1] * row.take_rate / 100 == pytest.approx(.68)


def test_eligibility_is_past_only_and_denominator_zero_is_excluded(synthetic_data):
    """Do not discard future inactivity or assign infinite growth to absent history."""
    panel, merchants, abs_features = synthetic_data
    panel.loc[panel.merchant_abn.eq(1) & panel.month.lt("2021-06-01"), "sales"] = 0
    panel.loc[panel.merchant_abn.eq(2) & panel.month.between("2021-06-01", "2021-08-01"), "sales"] = 0
    panel.loc[panel.merchant_abn.eq(3) & panel.month.gt("2021-08-01"), "sales"] = 0
    merchants.loc[merchants.merchant_abn.eq(4), "valid_take_rate"] = False
    features = comparison.features_at(panel, merchants, "2021-08-31", abs_features).set_index("merchant_abn")
    assert features.loc[1, "eligibility_reason"] == "no_sales_in_previous_three_months"
    assert features.loc[2, "eligibility_reason"] == "no_sales_in_recent_three_months"
    assert features.loc[3, "eligible"]
    assert features.loc[4, "eligibility_reason"] == "invalid_take_rate"
    assert comparison.targets_at(panel, pd.Index([3], name="merchant_abn"), "2021-08-31").loc[3] == 0


def test_missing_month_and_duplicate_month_fail(synthetic_data):
    """Distinguish missing source coverage from a genuine observed zero."""
    panel, merchants, abs_features = synthetic_data
    with pytest.raises(ValueError, match="Incomplete merchant history"):
        comparison.features_at(panel.drop(index=0), merchants, "2021-08-31", abs_features)
    with pytest.raises(ValueError, match="Duplicate merchant-month"):
        comparison.features_at(pd.concat([panel, panel.iloc[:1]]), merchants, "2021-08-31", abs_features)
    with pytest.raises(ValueError, match="Three complete outcome"):
        comparison.targets_at(panel.loc[panel.month.ne("2022-09-01")], merchants.merchant_abn, comparison.FINAL_ORIGIN)


def test_release_parser_uses_detailed_date_and_supports_single_issue():
    """Verify the detailed date and the documented November single-release exception."""
    html = '''Released 29/06/2022 <p>Release date and time 05/07/2022 11:30am AEST</p>
    <p>An additional information release on 5 July 2022 included the detailed results.</p>
    <a href="/may-2022/8501012.xlsx">Table 12</a>'''
    release = comparison.parse_release_page(html, "2022-05-01", comparison.BASE_URL + "may-2022")
    assert release["published_at"] == pd.Timestamp("2022-07-05 11:30", tz="Australia/Melbourne")
    with pytest.raises(ValueError, match="Conflicting detailed-release"):
        comparison.parse_release_page(html.replace("5 July 2022 included", "4 July 2022 included"), "2022-05-01", "https://www.abs.gov.au/test")
    holiday = '''Release date and time 11/01/2022 11:30am AEDT
    All tables have been made available on 11 January 2022.
    <a href="/nov-2021/8501012.xlsx">Table 12</a>'''
    assert comparison.parse_release_page(holiday, "2021-11-01", comparison.BASE_URL)["published_at"].day == 11


def test_late_release_and_later_revisions_cannot_enter_previous_origin():
    """A reference month alone cannot establish historical data availability."""
    releases = pd.DataFrame({"reference_month": pd.to_datetime(["2022-04-01", "2022-05-01"]),
                             "published_at": pd.DatetimeIndex(["2022-06-02 11:30", "2022-07-05 11:30"], tz="Australia/Melbourne")})
    assert comparison.choose_release(releases, "2022-06-30").reference_month == pd.Timestamp("2022-04-01")
    revised = pd.concat([releases, pd.DataFrame({"reference_month": [pd.Timestamp("2022-04-01")],
                                               "published_at": [pd.Timestamp("2022-07-10", tz="Australia/Melbourne")]})])
    assert comparison.choose_release(revised, "2022-06-30").equals(comparison.choose_release(releases, "2022-06-30"))
    with pytest.raises(ValueError, match="No historical ABS release"):
        comparison.choose_release(releases, "2022-05-31")


def test_hand_calculated_abs_momentum_and_missing_observation():
    """Verify three-versus-three momentum without dropping a missing ABS month."""
    months = pd.date_range("2021-11-01", periods=6, freq="MS")
    values = pd.DataFrame([(s, m, v) for s in comparison.SERIES for m, v in zip(months, [10, 20, 30, 20, 30, 40])],
                          columns=["series_id", "month", "turnover"])
    release = pd.Series(dict(reference_month=pd.Timestamp("2022-04-01"),
                             published_at=pd.Timestamp("2022-06-02", tz="Australia/Melbourne"),
                             source_url="test.xls", page_url="test"))
    feature = comparison.momentum_from_vintage(values, release, comparison.FINAL_ORIGIN)
    np.testing.assert_allclose(feature.abs_momentum, .5)
    with pytest.raises(ValueError, match="Six consecutive"):
        comparison.momentum_from_vintage(values.iloc[1:], release, comparison.FINAL_ORIGIN)


def test_workbook_parser_validates_geography_adjustment_and_vintage():
    """Reject a wrong geography, unadjusted data or a replacement future workbook."""
    def workbook(geography="Total (State)", adjustment="Seasonally Adjusted", end=pd.Timestamp("2022-04-01")):
        """Construct a real Excel payload with deliberate metadata variations."""
        labels = ["", "Unit", "Series Type", "Data Type", "Frequency", "Collection Month", "Series Start", "Series End", "No. Obs", "Series ID", pd.Timestamp("2022-04-01")]
        data = [labels]
        for series_id, description in comparison.SERIES.items():
            data.append([f"Turnover ; {geography} ; {description} ;", "$ Millions", adjustment, "FLOW", "Month", 1,
                         pd.Timestamp("2022-04-01"), end, 1, series_id, 100.0])
        buffer = io.BytesIO()
        pd.DataFrame(data).T.to_excel(buffer, sheet_name="Data1", index=False, header=False)
        return buffer.getvalue()
    assert len(comparison.parse_workbook(workbook(), "2022-04-01", ".xlsx")) == 7
    with pytest.raises(ValueError, match="Unexpected series description"):
        comparison.parse_workbook(workbook(geography="South Australia"), "2022-04-01", ".xlsx")
    with pytest.raises(ValueError, match="Wrong adjustment"):
        comparison.parse_workbook(workbook(adjustment="Original"), "2022-04-01", ".xlsx")
    with pytest.raises(ValueError, match="end date"):
        comparison.parse_workbook(workbook(end=pd.Timestamp("2022-05-01")), "2022-04-01", ".xlsx")


def test_future_sales_and_future_abs_perturbations_leave_predictions_unchanged(synthetic_data):
    """Rebuild and refit after altering unavailable information to detect time leakage."""
    panel, merchants, abs_features = synthetic_data
    features = make_features(panel, merchants, abs_features)
    original, details = comparison.predict_at(panel, features, comparison.FINAL_ORIGIN)
    changed_panel = panel.copy()
    changed_panel.loc[changed_panel.month.gt(comparison.FINAL_ORIGIN), "sales"] *= 10000
    later_abs = abs_features.loc[abs_features.origin.eq(comparison.FINAL_ORIGIN)].copy()
    later_abs["origin"] = pd.Timestamp("2022-07-31")
    later_abs["abs_published_at"] = pd.Timestamp("2022-07-05", tz="Australia/Melbourne")
    later_abs["abs_momentum"] = 999
    changed_abs = pd.concat([abs_features, later_abs])
    rebuilt = make_features(changed_panel, merchants, changed_abs)
    pd.testing.assert_frame_equal(features, rebuilt)
    again, new_details = comparison.predict_at(changed_panel, rebuilt, comparison.FINAL_ORIGIN)
    pd.testing.assert_frame_equal(original, again)
    assert details == new_details
    assert details["matched_baseline"]["training_origins"] == comparison.TRAIN_ORIGINS.strftime("%Y-%m-%d").tolist()
    assert details["matched_baseline"]["training_rows"] == details["matched_enhanced"]["training_rows"]


def test_identical_unmatched_predictions_and_independent_fee_scoring(synthetic_data):
    """Check the shared fallback and reconcile realized fees directly to source months."""
    panel, merchants, abs_features = synthetic_data
    predictions, _ = comparison.predict_at(panel, make_features(panel, merchants, abs_features), comparison.FINAL_ORIGIN)
    unmatched = predictions.loc[~predictions.abs_matched].pivot(index="merchant_abn", columns="method", values="predicted_fees")
    np.testing.assert_array_equal(unmatched.baseline, unmatched.enhanced)
    for _, group in predictions.groupby("method"):
        assert group.loc[group.selected, "merchant_abn"].nunique() == 100
        assert sorted(group["rank"]) == list(range(1, 141))
    scored, metrics, fold, changes, _ = comparison.score_at(predictions, panel, comparison.FINAL_ORIGIN)
    assert np.isfinite(metrics[["mae", "wape_pct", "bias_pct"]]).all().all()
    assert fold["delta_fees"] == pytest.approx(changes.fee_contribution.sum())
    selected = scored.loc[scored.method.eq("enhanced") & scored.selected]
    direct = panel.loc[panel.month.between("2022-07-01", "2022-09-01") & panel.merchant_abn.isin(selected.merchant_abn), "sales"].sum() * .04
    assert fold["enhanced_fees"] == pytest.approx(direct)


def test_ties_use_ascending_abn_and_do_not_select_over_100(synthetic_data, monkeypatch):
    """Make selection deterministic even when all forecast scores are equal."""
    panel, merchants, abs_features = synthetic_data
    def constant_model(training, candidates, columns):
        """Force equal scores to test the ranking contract independently of regression."""
        return np.full(len(candidates), 100.0), dict(training_rows=len(training), scaler_mean=[0.] * len(columns))
    monkeypatch.setattr(comparison, "fit_component", constant_model)
    prediction, _ = comparison.predict_at(panel, make_features(panel, merchants, abs_features), comparison.FINAL_ORIGIN)
    for method in ["baseline", "enhanced"]:
        selected = prediction.loc[prediction.method.eq(method) & prediction.selected]
        assert selected.merchant_abn.tolist() == list(range(1, 101))
        assert selected.predicted_fees.eq(4).all()


def test_unmatured_training_targets_do_not_enter_december_fit(synthetic_data):
    """The first fold can train only on the August and September origins."""
    panel, merchants, abs_features = synthetic_data
    features = make_features(panel, merchants, abs_features)
    _, details = comparison.predict_at(panel, features, pd.Timestamp("2021-12-31"))
    assert details["matched_baseline"]["training_origins"] == ["2021-08-31", "2021-09-30"]


def test_future_abs_publication_is_rejected_in_feature_join(synthetic_data):
    """Ensure a malformed feature join cannot bypass the publication cutoff."""
    panel, merchants, abs_features = synthetic_data
    abs_features.loc[abs_features.origin.eq(comparison.FINAL_ORIGIN), "abs_published_at"] = pd.Timestamp("2022-07-05", tz="Australia/Melbourne")
    with pytest.raises(ValueError, match="Future ABS publication"):
        comparison.features_at(panel, merchants, comparison.FINAL_ORIGIN, abs_features)


def test_category_normalization_preserves_words():
    """Match existing ETL punctuation rules without silently repairing category words."""
    assert comparison.normalize_category("  ANTIQUE SHOPS- sales , repairs,AND restoration services ") == "antique shops - sales, repairs, and restoration services"
    assert comparison.normalize_category("equipment rent al") == "equipment rent al"


def test_protocol_rejects_changed_settings(tmp_path):
    """Prevent silent replacement of an evaluated protocol with tuned parameters."""
    mapping = pd.read_csv(comparison.ROOT / "code/abs_category_mapping.csv", keep_default_na=False)
    comparison.freeze_protocol(tmp_path, mapping)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol["model"]["alpha"] = 999
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Frozen protocol differs"):
        comparison.freeze_protocol(tmp_path, mapping)


def test_duplicate_order_ids_across_archives_are_rejected(tmp_path):
    """Detect globally repeated identifiers even when each source partition is unique."""
    frame = pd.DataFrame(dict(order_id=["same-id"], merchant_abn=[1], dollar_value=[10.0]))
    for part, day in [(2, "2021-03-01"), (3, "2021-03-02"), (4, "2021-03-03")]:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        with zipfile.ZipFile(tmp_path / f"project-2-bnpl-tables-part{part}.zip", "w") as archive:
            archive.writestr(f"tables/snapshot{part}/order_datetime={day}/part.parquet", buffer.getvalue())
    with pytest.raises(ValueError, match="Duplicate order ID"):
        comparison.aggregate_transactions(tmp_path, tmp_path / "output")


@pytest.mark.parametrize("earlier,final,expected", [
    ([.1, .1, .1, -.1], 1., True),
    ([.1, .1, .1, .1], .999, False),
    ([.1, .1, 0., 0.], 5., False),
    ([0., 0., -.1, 0.], 0., False),
])
def test_retention_threshold_requires_both_conditions(earlier, final, expected):
    """Test the inclusive 1% threshold and strictly positive count of earlier gains."""
    folds = pd.DataFrame(dict(origin=list(comparison.DEV_ORIGINS) + [comparison.FINAL_ORIGIN],
                              uplift_pct=earlier + [final], delta_fees=[0.] * 5))
    assert comparison.retention_decision(folds)["retain_abs_momentum"] == expected
