"""Independent checks for time boundaries, economic quantities and paired model controls."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
import external_data_sources as src
import external_data_models as mdl
import external_data_comparison as exp


@pytest.fixture
def toy():
    """Construct a small panel with different trends, missing source matches and future zero sales."""
    ids = np.arange(1000, 1160)
    merchants = pd.DataFrame(dict(merchant_abn=ids, name=['m'] * 160,
                                 category=np.where(ids % 2 == 0, 'a', 'b'), take_rate=np.where(ids % 3 == 0, 3.0, 1.0),
                                 valid_take_rate=True, cpi_proxy=False, cabee_proxy=False, trends_proxy=False))
    rows = []
    for j, month in enumerate(pd.date_range('2021-03-01', '2022-09-01', freq='MS')):
        sales = (ids - 990) * 10 * (1 + 0.03 * j + 0.08 * np.sin(j + ids))
        if j >= 13:
            sales[0] = 0
        rows.append(pd.DataFrame(dict(merchant_abn=ids, month=month, sales=sales, sales_26=sales * 0.8,
                                      orders=np.ceil(sales / 50).astype(int), orders_26=np.ceil(sales / 70).astype(int))))
    panel = pd.concat(rows, ignore_index=True)
    external = pd.DataFrame([dict(category=c, origin=o, C=j * (0.01 if c == 'a' else -0.02),
                                  G=np.nan, E=0.08 if j < 4 else (0.1 if c == 'a' else 0.06))
                             for j, o in enumerate(src.ORIGINS) for c in ['a', 'b']])
    features = mdl.build_features(panel, merchants, external)
    return panel, merchants, external, features


def test_growth_target_and_rate_by_hand(toy):
    panel, merchants, external, _ = toy
    panel = panel.copy()
    mask = panel.merchant_abn.eq(1001)
    panel.loc[mask, 'sales'] = np.arange(1, 20) * 10.0
    panel.loc[mask, 'sales_26'] = np.arange(1, 20) * 8.0
    o = pd.Timestamp('2021-08-31')
    row = mdl.features_at(panel, merchants, external, o).set_index('merchant_abn').loc[1001]
    assert row.recent_monthly_sales == 50
    assert row.own_growth == 1.5
    assert row.log_own_growth == pytest.approx(np.log(2.5))
    target = mdl.target_at(panel, pd.Series([1001]), o, '3m')[0]
    assert target == 70 + 80 + 90
    assert target * 3 / 100 == pytest.approx(7.2)
    assert mdl.target_at(panel, pd.Series([1001]), o, '26d')[0] == 56
    assert row.reference_26 == 40


def test_inactivity_is_zero_not_exclusion(toy):
    panel, _, _, features = toy
    train = mdl.training_at(panel, features, pd.Timestamp('2022-06-30'), '3m')
    r = train.loc[train.merchant_abn.eq(1000) & train.origin.eq(pd.Timestamp('2022-03-31'))].iloc[0]
    assert r.target_sales == 0
    assert r.target_shortfall == pytest.approx(0.5 * r.reference_sales)


def test_incomplete_history_and_duplicate_keys_fail(toy):
    panel, merchants, external, _ = toy
    with pytest.raises(ValueError, match='Incomplete'):
        mdl.features_at(panel.iloc[1:], merchants, external, pd.Timestamp('2021-08-31'))
    with pytest.raises(ValueError, match='Duplicate'):
        mdl.features_at(pd.concat([panel, panel.iloc[:1]]), merchants, external, pd.Timestamp('2021-08-31'))


def test_training_maturity_and_26_day_boundary(toy):
    panel, _, _, features = toy
    train = mdl.training_at(panel, features, pd.Timestamp('2022-06-30'), '3m')
    assert train.origin.max() == pd.Timestamp('2022-03-31')
    assert train.origin.min() == pd.Timestamp('2021-08-31')
    short = mdl.training_at(panel, features, pd.Timestamp('2022-09-30'), '26d')
    assert short.origin.max() == pd.Timestamp('2022-08-31')
    assert mdl.horizon_end(pd.Timestamp('2022-09-30'), '26d') == pd.Timestamp('2022-10-26')


def test_abs_publication_gate_and_future_revisions():
    releases = pd.DataFrame(dict(vintage=['old', 'new'], published_at=pd.to_datetime(['2022-04-27 11:30', '2022-07-27 11:30']).tz_localize('Australia/Sydney')))
    assert src.choose_asof(releases, pd.Timestamp('2022-06-30')).vintage == 'old'
    changed = releases.copy()
    changed.loc[1, 'vintage'] = 'arbitrary_future_revision'
    pd.testing.assert_series_equal(src.choose_asof(releases, pd.Timestamp('2022-06-30')), src.choose_asof(changed, pd.Timestamp('2022-06-30')))
    with pytest.raises(ValueError, match='No verified'):
        src.choose_asof(releases, pd.Timestamp('2021-01-31'))


def test_future_feature_publication_rejected(toy):
    panel, merchants, external, _ = toy
    ext = external.copy()
    ext['C_published_at'] = pd.Timestamp('2023-01-01', tz='Australia/Sydney')
    with pytest.raises(ValueError, match='Future ABS'):
        mdl.features_at(panel, merchants, ext, pd.Timestamp('2021-08-31'))


def test_cpi_calculation_and_mapping_conflict():
    data = [[None, 'Index Numbers ; Footwear ; Australia ;'], ['Unit', 'Index Numbers'],
            ['Series Type', 'Original'], ['Data Type', 'INDEX'], ['Series End', pd.Timestamp('2022-03-01')],
            ['Series ID', 'A2326386V'], [pd.Timestamp('2021-03-01'), 100], [pd.Timestamp('2022-03-01'), 110]]
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(data).to_excel(writer, sheet_name='Data1', header=False, index=False)
    values = src.cpi_values(output.getvalue(), '.xlsx', pd.Timestamp('2022-03-01')).set_index('series_id')
    assert values.loc['A2326386V', 'value'] == pytest.approx(np.log(1.1))
    assert values.loc['A2329311F', 'status'] == 'mapping_conflict'


def test_cabee_national_total_not_state_average():
    rows = [[None] * 32 for _ in range(8)]
    rows[0][0] = 'Released at 11.30 am (Canberra time) 16 February 2021'
    rows[1][0] = 'Industry Class by Annualised Employment Size Ranges'
    rows[5][0:3] = ['NSW', 4252, 'Footwear Retailing']
    rows[5][7], rows[5][17] = 20, 9
    rows[6][0:3] = [None, 4252, 'Total Footwear Retailing']
    rows[6][7], rows[6][17] = 100, 10
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='June 2020 a ', header=False, index=False)
    values, _ = src.cabee_values(output.getvalue(), '.xlsx', 2020, '2021-02-16')
    assert values.set_index('industry_code').loc['4252', 'value'] == 0.1
    with pytest.raises(ValueError, match='date'):
        src.cabee_values(output.getvalue(), '.xlsx', 2020, '2020-02-16')


def test_trends_calendar_ratio_and_scale_invariance():
    start, end = src.trends_window(pd.Timestamp('2021-08-31'))
    assert start == pd.Timestamp('2021-01-31') and end == pd.Timestamp('2021-07-31')
    data = pd.DataFrame({'Day': pd.date_range(start, end), 'shoes: (Australia)': np.r_[np.ones(91)*20, np.ones(91)*40]})
    estimate, weekly = src.parse_trends_csv(data.to_csv(index=False).encode(), start, end, 'shoes')
    assert estimate == pytest.approx(np.log(2)) and len(weekly) == 26
    data.iloc[:, 1] *= 2
    assert src.parse_trends_csv(data.to_csv(index=False).encode(), start, end, 'shoes')[0] == pytest.approx(estimate)
    with pytest.raises(ValueError, match='Incomplete'):
        src.parse_trends_csv(data.iloc[:-1].to_csv(index=False).encode(), start, end, 'shoes')
    with pytest.raises(ValueError, match='query or geography'):
        src.parse_trends_csv(data.to_csv(index=False).encode(), start, end, 'furniture')
    data.iloc[:91, 1] = 0
    with pytest.raises(ValueError, match='Insufficient'):
        src.parse_trends_csv(data.to_csv(index=False).encode(), start, end, 'shoes')


@pytest.mark.parametrize('family', ['tweedie', 'boosting'])
def test_missing_source_fallback_and_finite_models(toy, family):
    panel, _, _, features = toy
    features = features.copy()
    features['G'] = np.sin(features.origin.dt.month) * np.where(features.category.eq('a'), 0.2, -0.1)
    features.loc[features.category.eq('b'), 'C'] = np.nan
    features.loc[features.merchant_abn.mod(3).eq(0), 'G'] = np.nan
    train, test, design = exp.input_at(panel, features, pd.Timestamp('2022-06-30'), '3m')
    predictions, records = mdl.subset_predictions(design, train, test, list(mdl.ARMS), {'return': family, 'risk': family})
    missing = test.C.isna().to_numpy()
    for target in ['return', 'risk']:
        np.testing.assert_array_equal(predictions[target,'C'][missing], predictions[target,'B'][missing])
        np.testing.assert_array_equal(predictions[target,'CE'][missing], predictions[target,'E'][missing])
        missing_g = test.G.isna().to_numpy()
        np.testing.assert_array_equal(predictions[target,'CGE'][missing_g], predictions[target,'CE'][missing_g])
        np.testing.assert_array_equal(predictions[target,'CG'][missing_g], predictions[target,'C'][missing_g])
        np.testing.assert_array_equal(predictions[target,'CGE'][missing], predictions[target,'GE'][missing])
        assert np.isfinite(predictions[target, 'CE']).all()
    assert all(r['training_rows'] == len(train) for r in records)
    assert all(r['scaler_mean'] == records[0]['scaler_mean'] for r in records)
    assert all(r['iterations'] > 0 for r in records if not r.get('constant_zero_target'))


def test_all_zero_risk_is_valid(toy):
    panel, _, _, features = toy
    train, test, design = exp.input_at(panel, features, pd.Timestamp('2022-06-30'), '3m')
    train.target_shortfall = 0
    pred, info = mdl.fit_model(design, train, test, 'B', 'tweedie', 'risk')
    assert (pred == 0).all() and info['constant_zero_target']
    assert np.isfinite(mdl.deviance(np.zeros(len(pred)), pred))


def test_future_perturbation_keeps_earlier_fit_forecasts_and_selection(toy):
    panel, merchants, external, features = toy
    origin = pd.Timestamp('2022-03-31')
    changed = panel.copy()
    changed.loc[changed.month.gt(origin), ['sales', 'sales_26']] *= 10000
    ext = external.copy()
    ext.loc[ext.origin.gt(origin), ['C', 'E']] += 10000
    changed_features = mdl.build_features(changed, merchants, ext)
    before = exp.input_at(panel, features, origin, '3m')
    after = exp.input_at(changed, changed_features, origin, '3m')
    pd.testing.assert_frame_equal(before[0], after[0], check_exact=True)
    pd.testing.assert_frame_equal(before[1], after[1], check_exact=True)
    for target in ['return', 'risk']:
        a, _ = mdl.fit_model(before[2], before[0], before[1], 'CE', 'tweedie', target)
        b, _ = mdl.fit_model(after[2], after[0], after[1], 'CE', 'tweedie', target)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(mdl.ranks(before[1].merchant_abn, a), mdl.ranks(after[1].merchant_abn, b))


def test_model_family_ignores_immature_validation():
    history = [dict(origin=pd.Timestamp(o), target='return', family=f, deviance=d)
               for o in ['2021-11-30','2021-12-31','2022-06-30'] for f,d in [('tweedie', 3), ('boosting', 2)]]
    origin = pd.Timestamp('2022-03-31')
    assert mdl.choose_family(history, origin, 'return', '3m')[0] == 'boosting'
    history[-1]['deviance'] = 1e20
    assert mdl.choose_family(history, origin, 'return', '3m')[0] == 'boosting'
    assert mdl.choose_family(history, pd.Timestamp('2022-02-28'), 'return', '3m')[0] == 'tweedie'


def test_ranking_and_independent_fee_audit(toy):
    panel, _, _, features = toy
    origin = pd.Timestamp('2022-06-30')
    _, candidates, _ = exp.input_at(panel, features, origin, '3m')
    preds = {('return','B'): np.ones(len(candidates)) * 100, ('risk','B'): np.zeros(len(candidates))}
    frames = mdl.policy_frames(candidates, preds, ['B'])
    scored, folds, _, changes = mdl.score_policies(frames, panel, origin, '3m')
    assert frames.groupby(['arm','policy']).selected.sum().eq(100).all()
    selected = scored.loc[scored.arm.eq('B') & scored.policy.eq('return') & scored.selected]
    months = panel.loc[panel.month.ge(pd.Timestamp('2022-07-01'))].merge(selected[['merchant_abn','take_rate']], on='merchant_abn')
    assert folds.loc[folds.arm.eq('B') & folds.policy.eq('return'), 'fees'].iloc[0] == pytest.approx((months.sales * months.take_rate / 100).sum())
    ids = np.arange(120)[::-1]
    ranks = mdl.ranks(ids, np.ones(120))
    assert set(ids[ranks <= 100]) == set(range(100))
    assert changes.loc[changes.arm.eq('B'), 'fee_contribution'].sum() == 0
    repeat_delta = folds.loc[folds.arm.eq('repeat'), 'delta_fees'].iloc[0]
    assert changes.loc[changes.arm.eq('repeat'), 'fee_contribution'].sum() == pytest.approx(repeat_delta)


def test_placebos_preserve_whole_paths_and_missingness(toy):
    _, _, external, _ = toy
    a, assignments = exp.shuffled_external(external, 7)
    b, _ = exp.shuffled_external(external, 7)
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(a[mdl.SOURCES].isna(), external[mdl.SOURCES].isna())
    for item in assignments:
        source = item['source']
        np.testing.assert_array_equal(a.loc[a.category.eq(item['category']), source],
                                      external.loc[external.category.eq(item['donor_category']), source])


def test_nonoverlapping_windows_count():
    assert exp.nonoverlap_count(mdl.DEV_ORIGINS) == 3
    assert exp.nonoverlap_count(pd.date_range('2022-03-31','2022-06-30',freq='ME')) == 2


def test_resume_rejects_changed_prepared_inputs(toy, tmp_path):
    panel, _, _, features = toy
    exp.check_backtest_inputs(panel, features, tmp_path)
    exp.check_backtest_inputs(panel.copy(), features.copy(), tmp_path)
    changed = features.copy()
    changed.loc[0, 'C'] = 99.0
    with pytest.raises(ValueError, match='Prepared inputs changed'):
        exp.check_backtest_inputs(panel, changed, tmp_path)


def test_protocol_and_october_guard(tmp_path):
    mapping = pd.read_csv(src.MAPPING, keep_default_na=False)
    exp.freeze_protocol(mapping, tmp_path)
    exp.freeze_protocol(mapping, tmp_path)
    other = mapping.copy()
    other.loc[0, 'trends_query'] = 'changed'
    with pytest.raises(ValueError, match='Frozen protocol'):
        exp.freeze_protocol(other, tmp_path)
    with pytest.raises(ValueError, match='October access requires'):
        exp.october_panel(tmp_path, pd.DataFrame(), tmp_path)


def test_decision_requires_every_gate():
    origins = pd.date_range('2021-11-30','2022-06-30',freq='ME')
    folds = pd.DataFrame([dict(origin=o, arm=a, policy='return', risk_lambda=0.0, fees=f, utility=f)
                          for o in origins for a,f in [('B',100), ('C',102)]])
    base = dict(arm='C', policy='return', informative_origins=8, nonoverlapping_windows=3,
                origins=','.join(origins.strftime('%Y-%m-%d')), mean_uplift_pct=2., mean_repeat_uplift_pct=1.5,
                mean_utility_gain_pct=2., positive_fraction=1., score=2., placebo_95_pct=0.5,
                fragile=False, mean_shortfall_deviance_improvement_pct=np.nan)
    _, winner = exp.decide(pd.DataFrame([base]), folds)
    assert winner['return'] == 'C'
    for key, value in [('mean_repeat_uplift_pct',0.5), ('informative_origins',3), ('positive_fraction',0.5), ('fragile',True), ('placebo_95_pct',2.0)]:
        _, choice = exp.decide(pd.DataFrame([base | {key:value}]), folds)
        assert choice['return'] == 'B'
