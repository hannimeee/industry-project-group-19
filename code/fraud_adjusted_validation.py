"""Independently reconcile exported adjusted portfolios and write the findings."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json
import ast
import io
import sys
import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf_shrinkage
from threadpoolctl import threadpool_limits
from fraud_adjusted_portfolios import ROOT,OUT,DELIVERY,SEGMENTS,scope_filename,notebook_model
sys.path.insert(0,str(ROOT/'handoff'))
from portfolio_optimizer.data import file_hash,save_json


def validate_score_reconstruction():
    """Reproduce the notebook's saved score benchmarks without running its ETL."""
    cells=json.loads((ROOT/'code/01_etl.ipynb').read_text(encoding='utf-8'))['cells']
    cell=next(c for c in cells if 'def evaluate_score_models(' in ''.join(c.get('source',[])))
    tree=ast.parse(''.join(cell['source']))
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))
               or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCORE_AVAILABILITY_DELAY_DAYS' for t in n.targets))]
    namespace,_=notebook_model()
    exec(compile(tree,'team_score_evaluation','exec'),namespace)
    frame=pd.read_parquet(OUT/'consumer_training.parquet').sort_values(['order_datetime','user_id']).reset_index(drop=True)
    temporal,_,_=namespace['evaluate_score_models'](frame,'consumer')
    random,_=namespace['evaluate_random_consumer_split'](frame)
    actual=pd.concat([temporal,random],ignore_index=True)
    texts=[''.join(o.get('data',{}).get('text/plain',[])) for o in cell.get('outputs',[])]
    reference=next(t for t in texts if 'random_70_30_diagnostic' in t and 'linear_regression' in t)
    expected=pd.read_csv(io.StringIO(reference),sep=r'\s+')
    keys=['entity','split','model']
    pd.testing.assert_frame_equal(actual[keys],expected[keys])
    np.testing.assert_array_equal(actual[['train_rows','test_rows','test_entities']],expected[['train_rows','test_rows','test_entities']])
    np.testing.assert_allclose(actual[['mae','rmse','r2']],expected[['mae','rmse','r2']],atol=5.1e-7,rtol=0)
    actual.to_csv(OUT/'consumer_validation.csv',index=False)
    actual.to_csv(DELIVERY/'provenance/consumer_validation.csv',index=False)


def validate():
    """Recompute monthly fees, weekly CV and customer HHI without optimizer methods."""
    validate_score_reconstruction()
    inp=OUT/'input_snapshot';dest=DELIVERY/'final'
    f=pd.read_parquet(inp/'merchants.parquet').set_index('merchant_abn')
    weekly=pd.read_parquet(inp/'weekly_fees.parquet').pivot(index='week_end',columns='merchant_abn',values='gross_fees')
    customers=pd.read_parquet(inp/'customer_fees.parquet')
    original_customers=pd.read_parquet(ROOT/'handoff/input_snapshot/customer_fees.parquet')
    matched=customers.merge(original_customers,on=['consumer_id','merchant_abn'],how='outer',
                            suffixes=('_adjusted','_original'),indicator=True,validate='one_to_one')
    assert matched._merge.eq('both').all()
    assert matched.gross_fees_adjusted.ge(0).all()
    assert (matched.gross_fees_adjusted<=matched.gross_fees_original+1e-7).all()
    del original_customers,matched
    totals=customers.groupby('merchant_abn').gross_fees.sum()
    np.testing.assert_allclose(totals.reindex(f.index,fill_value=0),weekly.sum().reindex(f.index),rtol=1e-10,atol=1e-7)
    daily=pd.read_parquet(OUT/'merchant_day.parquet')
    np.testing.assert_allclose(daily.groupby('merchant_abn').adjusted_fees.sum().reindex(f.index)/19,f.expected_monthly_fees,rtol=1e-10,atol=1e-7)
    del daily
    compared=[];allkeys=[]
    for scope in ['global',*SEGMENTS]:
        eligible=f.loc[f.eligible & (True if scope=='global' else f.category.isin(SEGMENTS[scope]))]
        x=weekly.reindex(columns=eligible.index).to_numpy()
        centered=x-x.mean(axis=0);centered[:,np.ptp(x,axis=0)==0]=0
        variance=np.var(centered,axis=0,ddof=1);nz=variance>0
        alpha=float(ledoit_wolf_shrinkage(centered[:,nz]/np.sqrt(variance[nz]),assume_centered=True,block_size=256)) if nz.sum()>1 else 0.
        current=pd.read_parquet(dest/(scope_filename(scope)+'.parquet'))
        csv=pd.read_csv(dest/(scope_filename(scope)+'.csv'),dtype={'merchant_abn':str})
        pd.testing.assert_frame_equal(current,csv,check_dtype=False,rtol=1e-12,atol=1e-12)
        old=pd.read_parquet(ROOT/'handoff/final'/(scope_filename(scope)+'.parquet'))
        baseline=list(eligible.reset_index().sort_values(['expected_monthly_fees','merchant_abn'],ascending=[False,True]).head(100).merchant_abn)
        def manual(ids):
            ids=list(ids);mu=eligible.loc[ids,'expected_monthly_fees']
            w=weekly[ids];series=w.sum(axis=1)
            cv=np.sqrt((1-alpha)*series.var(ddof=1)+alpha*w.var(ddof=1).sum())/series.mean()
            c=customers.loc[customers.merchant_abn.isin(ids)].copy()
            c['exposure']=c.gross_fees*c.merchant_abn.map(mu/totals.reindex(ids))
            customer=c.groupby('consumer_id').exposure.sum()
            hhi=float(((customer/mu.sum())**2).sum())
            return float(mu.sum()),float(cv),hhi
        base=manual(baseline)
        for retention,g in current.groupby('retention'):
            row=g.iloc[0];ids=g.merchant_abn.astype('int64').tolist()
            assert len(ids)==len(set(ids))==100 and set(ids)<=set(eligible.index)
            assert g.fee_basis.eq('consumer_fraud_adjusted').all()
            assert g.portfolio_id.nunique()==1
            allkeys.append(row.portfolio_id)
            a=manual(ids)
            np.testing.assert_allclose(a,[row.portfolio_expected_monthly_adjusted_fees,row.portfolio_variability_cv,row.portfolio_customer_hhi],rtol=1e-10,atol=1e-9)
            assert a[0]>=retention*base[0]-1e-6
            score=50*(a[1]/base[1]+a[2]/base[2])
            np.testing.assert_allclose(score,row.portfolio_risk_score,rtol=1e-10)
            ordered=g.sort_values(['merchant_expected_monthly_adjusted_fees','merchant_abn'],ascending=[False,True])
            assert ordered.rank_by_expected_fees.tolist()==list(range(1,101))
            assert row.portfolio_score_lower_bound<=score+1e-7
            gap=(score-row.portfolio_score_lower_bound)/score
            np.testing.assert_allclose(gap,row.portfolio_optimality_gap,atol=1e-10)
            if row.portfolio_certified:assert gap<=.001+1e-9 or score-row.portfolio_score_lower_bound<=.0001+1e-9
            old_ids=old.loc[old.retention.eq(retention)].merchant_abn.astype('int64').tolist()
            b=manual(old_ids)
            compared.append(dict(scope=scope,retention=retention,old_selection_adjusted_fees=b[0],new_selection_adjusted_fees=a[0],
                old_selection_adjusted_cv=b[1],new_selection_adjusted_cv=a[1],old_selection_adjusted_hhi=b[2],new_selection_adjusted_hhi=a[2],
                old_selection_adjusted_score=50*(b[1]/base[1]+b[2]/base[2]),new_selection_adjusted_score=score,
                old_selection_meets_new_floor=bool(b[0]>=retention*base[0]-1e-6),certified=bool(row.portfolio_certified)))
        for marker in (OUT/'runs'/scope).glob('retention_*/complete.json'):
            info=json.loads(marker.read_text())
            for name,digest in info['files'].items():assert file_hash(marker.parent/name)==digest
        print(f'Independently validated {scope}.',flush=True)
    assert len(allkeys)==len(set(allkeys))==30
    pd.DataFrame(compared).to_csv(DELIVERY/'old_and_new_on_adjusted_basis.csv',index=False)
    audit=json.loads((OUT/'preparation_complete.json').read_text())
    summary=pd.read_csv(DELIVERY/'comparison.csv')
    checks=dict(portfolios=30,rows=3000,unique_portfolio_identifiers=True,exactly_100_eligible=True,all_fee_floors_satisfied=True,
        monthly_return_from_daily_reconciled=True,weekly_customer_totals_reconciled=True,independent_cv_hhi_score_recomputation=True,
        original_customer_pair_keys_preserved=True,adjusted_customer_fees_never_exceed_original=True,
        csv_parquet_equivalent=True,solver_checkpoint_hashes_verified=True,deterministic_ranks=True,
        consumer_validation_matches_team_notebook=True,certified=int(summary.certified.sum()),
        noncertified=int((~summary.certified).sum()),no_new_holdout=True)
    save_json(DELIVERY/'acceptance_checks.json',checks)
    dictionary=pd.read_csv(ROOT/'handoff/final/data_dictionary.csv')
    dictionary.loc[dictionary.column.eq('portfolio_id'),'description']='Unique fee basis / scope / snapshot / size / weight / revenue-floor combination'
    dictionary.loc[dictionary.column.eq('merchant_category'),'description']='Original merchant category; category-group scopes restrict the candidate pool using the documented lookup'
    dictionary.loc[dictionary.column.eq('retention'),'description']='Minimum adjusted expected fees divided by the same scope adjusted-revenue-only baseline; .995 means 99.5%'
    for column in ['merchant_expected_monthly_gross_fees','portfolio_expected_monthly_gross_fees']:
        dictionary.loc[dictionary.column.eq(column),'description']='Legacy compatibility alias: fraud-adjusted monthly fees in this dataset, not unadjusted gross fees.'
    for column in ['portfolio_variability_cv','portfolio_customer_hhi','portfolio_risk_score','portfolio_expected_fee_retention']:
        mask=dictionary.column.eq(column)
        dictionary.loc[mask,'description']='Computed on the fraud-adjusted fee basis. '+dictionary.loc[mask,'description']
    extra=[dict(column='fee_basis',unit='text',aggregation='first per portfolio',description='consumer_fraud_adjusted; distinguishes these amounts from the preserved original outputs'),
        dict(column='scope',unit='text',aggregation='group key',description='global, tech, home, fashion, arts or auto'),
        dict(column='merchant_expected_monthly_adjusted_fees',unit='AUD / month',aggregation='sum within ONE portfolio',description='Mean monthly transaction fees after the consumer-fraud discount'),
        dict(column='portfolio_expected_monthly_adjusted_fees',unit='AUD / month',aggregation='FIRST ONLY; never SUM',description='Total expected monthly fraud-adjusted fees, repeated on merchant rows')]
    pd.concat([dictionary,pd.DataFrame(extra)],ignore_index=True).to_csv(dest/'data_dictionary.csv',index=False)
    manifest=json.loads((dest/'manifest.json').read_text())
    manifest['files']['data_dictionary.csv']=file_hash(dest/'data_dictionary.csv')
    save_json(dest/'manifest.json',manifest)
    overall=summary.loc[summary.scope.eq('global')].sort_values('retention',ascending=False)
    table=['| Revenue floor | Adjusted monthly fees (AUD) | Weekly CV | Customer HHI | Risk score | Replaced vs original | Gap | Certified |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in overall.itertuples():table.append(f'| {r.retention:.1%} | {r.expected_monthly_adjusted_fees:,.2f} | {r.adjusted_cv:.3%} | {r.adjusted_hhi:.8f} | {r.risk_score:.3f} | {r.replaced} | {r.optimality_gap:.3%} | {r.certified} |')
    text=f'''# Fraud-adjusted portfolio findings

Recalculated all 30 final portfolios: the global merchant pool and five category groups, each at 100%, 99.5%, 99%, 98% and 95% expected-revenue retention. The date, 100-merchant size and 50/50 risk weights are unchanged. Both expected fees and risk inputs use transaction-level consumer-fraud-adjusted fees.

Across all metadata merchants, historical mean monthly fees decrease from A${audit['gross_monthly_fees']:,.2f} to A${audit['adjusted_monthly_fees']:,.2f}, a {1-audit['adjusted_monthly_fees']/audit['gross_monthly_fees']:.2%} discount. This is the requested fee proxy, not measured fraud losses or net profit.

## Global results

'''+ '\n'.join(table)+f'''

## Validation and limitations

- All {audit['rows']:,} transactions across 579 complete-history days were processed. {audit['known_orders']:,} have merchant metadata; {audit['unknown_merchant_orders']:,} remain outside fee-based selection. No take rate was invented.
- {audit['observed_orders']:,} orders retain supplied scores; {audit['imputed_orders']:,} use the exact existing consumer OLS specification. There are no missing completed scores. The reconstructed temporal and random-split benchmark metrics match the teammate notebook to printed precision.
- {audit['low_support_orders']:,} orders ({audit['low_support_orders']/audit['rows']:.2%}) receive the model's low-support flag, meaning at least one activity predictor lies outside the observed labelled range. This is not a count of confirmed fraud or a prediction-error estimate. Good validation on observed scores does not establish calibration on these orders.
- Only {audit['clipped_orders']} order(s) use a clipped missing-score prediction. Supplied scores are never clipped or overwritten.
- {checks['certified']} of 30 portfolios meet the 0.1% optimization-gap criterion; {checks['noncertified']} are best-found feasible portfolios with explicit bounds. No claim of exact global optimality is made for them.
- Old and new selections are also recomputed on the SAME adjusted fee/CV/HHI basis in `old_and_new_on_adjusted_basis.csv`. A former gross-fee portfolio may not satisfy the new adjusted-revenue floor.
- Covariance is re-estimated and customer exposures are rebuilt after discounting. Retention is relative to each scope's adjusted-revenue-only baseline, not its old gross-fee baseline. Lower historical score does not establish future downside protection.
- No new forecasting model, merchant-fraud multiplier, net-loss assumption or untouched backtest was introduced. The full original result files remain preserved under `handoff/final/`.

Use `final/` in this folder for the new files. They include explicit adjusted-fee aliases and `fee_basis`; legacy gross-fee field names remain compatibility aliases only in this directory. The old offline dashboard bundle has not been silently relabelled or overwritten.
'''
    (DELIVERY/'FINDINGS.md').write_text(text,encoding='utf-8')
    print(json.dumps(checks),flush=True)

if __name__=='__main__':
    if os.name=='nt':
        from portfolio_optimizer.resource_limits import apply_worker_limits
        apply_worker_limits()
    with threadpool_limits(limits=1):validate()
