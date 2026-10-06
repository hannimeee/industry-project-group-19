"""Focused tests for the transaction-level fraud adjustment and scope mapping."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fraud_adjusted_portfolios as run


def test_percentage_conversion_and_extremes():
    np.testing.assert_allclose(run.discounted_fees([100,100,100],[5,5,5],[0,20,100]),[5,4,0])


@pytest.mark.parametrize('score',[np.nan,np.inf,-1,101])
def test_missing_or_invalid_scores_never_become_zero_risk(score):
    with pytest.raises(ValueError):run.discounted_fees([100],[5],[score])


def test_discount_before_aggregation_not_unweighted_average():
    result=run.discounted_fees([100,900],[10,10],[100,0]).sum()
    assert result==90
    assert result != 1000*.1*(1-np.mean([100,0])/100)


def test_same_customer_across_all_merchants_has_one_day_score():
    orders=pd.DataFrame({'user_id':[1,1,2],'merchant_abn':[11,999,11],'dollar_value':[20.,80.,30.]})
    f=run.daily_features(orders).set_index('user_id')
    assert f.loc[1,'same_day_spending']==100
    assert f.loc[1,'same_day_order_count']==2
    assert f.loc[1,'same_day_average_order_value']==50


def test_observed_scores_preserved_missing_reconstructed_and_flagged():
    def predict(model,frame):
        return pd.DataFrame({'predicted_score':[10.,20.],'low_support':[True,True],
                             'prediction_clipped':[True,False]},index=frame.index)
    result=run.complete_scores(pd.DataFrame({'user_id':[1,2]}),pd.Series({1:100.}),None,{'predict_score_model':predict})
    assert result.consumer_fraud_probability.tolist()==[100.,20.]
    assert result.consumer_score_source.tolist()==['observed','retrospective_model']
    assert result.low_support.tolist()==[False,True]
    assert not result.prediction_clipped.any()


def test_uniform_discount_preserves_relative_risk_but_halves_expected_fees():
    from portfolio_optimizer.data import build_snapshot
    f=pd.DataFrame({'merchant_abn':[1,2],'name':['a','b'],'eligible':[True,True],'expected_monthly_fees':[10.,20.]})
    w=pd.DataFrame({'merchant_abn':[1,1,2,2],'week_end':['2022-09-18','2022-09-25']*2,'gross_fees':[1.,3.,4.,6.]})
    c=pd.DataFrame({'merchant_abn':[1,2,2],'consumer_id':[8,8,9],'gross_fees':[4.,5.,5.]})
    meta=dict(as_of='2022-09-30',weeks=2,window_start='2022-09-12',window_end='2022-09-25',
              expected_return_start='2021-03-01',expected_return_end='2022-09-30',currency='AUD',complete_coverage=True)
    old=build_snapshot(f,w,c,meta,count=2)
    f.expected_monthly_fees*=.5;w.gross_fees*=.5;c.gross_fees*=.5
    new=build_snapshot(f,w,c,meta,count=2)
    assert new.model.base_fees==old.model.base_fees*.5
    assert new.model.base_cv==pytest.approx(old.model.base_cv)
    assert new.model.base_hhi==pytest.approx(old.model.base_hhi)


def test_category_mapping_covers_original_pool_and_matches_existing_outputs():
    f=pd.read_parquet(run.ROOT/'handoff/input_snapshot/merchants.parquet')
    cats=[c for values in run.SEGMENTS.values() for c in values]
    assert len(cats)==len(set(cats))==25
    assert set(cats)==set(f.category)
    for scope,categories in run.SEGMENTS.items():
        old=pd.read_parquet(run.ROOT/'handoff/final'/(run.scope_filename(scope)+'.parquet'))
        assert old.merchant_category.isin(categories).all()


def test_reuses_the_team_model_with_raw_squares_before_scaling():
    namespace,_=run.notebook_model()
    values=np.array([[4.,2.,2.],[9.,3.,3.]])
    np.testing.assert_array_equal(namespace['_consumer_design'](values,'retrospective'),[[4,2,16,4],[9,3,81,9]])
