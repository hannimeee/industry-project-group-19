"""Rebuild all final portfolios using the team's consumer-fraud fee discount.

Stages are serial, bounded in memory, restartable, and preserve the gross-fee study.
The notebook's exact consumer model is reused when its ignored export is absent.
"""
from pathlib import Path
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable]='1'
import argparse
import ast
import gc
import hashlib
import io
import json
import re
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'handoff'))
from portfolio_optimizer.data import file_hash,save_json,write_snapshot,load_snapshot,build_snapshot
from portfolio_optimizer.cli import optimize_snapshot,verify_records
from portfolio_optimizer.core import require

OUT=ROOT/'tables/analysis/fraud_adjusted_portfolios'
DELIVERY=ROOT/'handoff/fraud_adjusted'
SOURCE=ROOT.parent/'project 2 materials'
START=pd.Timestamp('2021-03-01'); END=pd.Timestamp('2022-09-30')
RISK_START=pd.Timestamp('2022-03-28'); RISK_END=pd.Timestamp('2022-09-25')
RETENTIONS=[1.,.995,.99,.98,.95]
# Exact labels, resolving the explicit positions in 04_categories.ipynb's saved table.
SEGMENTS={
 'tech':['digital goods: books, movies, music','computer programming, data processing, and integrated systems design services','computers, computer peripheral equipment, and software','cable, satellite, and other pay television and radio services','telecom'],
 'home':['furniture, home furnishings and equipment shops, and manufacturers, except appliances','florists supplies, nursery stock, and flowers','stationery, office supplies and printing and writing paper','lawn and garden supply outlets, including nurseries','equipment, tool, furniture, and appliance rent al and leasing'],
 'fashion':['shoe shops','watch, clock, and jewelry repair shops','health and beauty spas','opticians, optical goods, and eyeglasses','jewelry, watch, clock, and silverware shops'],
 'arts':['artist supply and craft shops','gift, card, novelty, and souvenir shops','music shops - musical instruments, pianos, and sheet music','books, periodicals, and newspapers','hobby, toy and game shops','antique shops - sales, repairs, and restoration services','art dealers and galleries'],
 'auto':['tent and awning shops','bicycle shops - sales and service','motor vehicle supplies and new parts']}


def dump(frame,path):
    """Write a compact analytical table without Parquet compression."""
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(path,index=False,compression=None)


def discounted_fees(value,rate,score):
    """Convert both supplied percentages before discounting each transaction."""
    value,rate,score=[np.asarray(x,dtype=float) for x in [value,rate,score]]
    require(np.isfinite(value).all() and (value>=0).all(),'Invalid transaction values')
    require(np.isfinite(rate).all() and ((rate>0)&(rate<=100)).all(),'Invalid percentage take rates')
    require(np.isfinite(score).all() and ((score>=0)&(score<=100)).all(),'Incomplete or invalid consumer scores')
    return value*rate/100*(1-score/100)


def notebook_model():
    """Load only the inspected definitions cell, never execute notebook ETL cells."""
    path=ROOT/'code/01_etl.ipynb'
    cells=json.loads(path.read_text(encoding='utf-8'))['cells']
    matches=[''.join(c.get('source',[])) for c in cells if 'def fit_score_model(' in ''.join(c.get('source',[]))]
    require(len(matches)==1,'Expected one team model definitions cell')
    source=matches[0]
    tree=ast.parse(source)
    require(all(isinstance(n,(ast.Expr,ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef)) for n in tree.body),'Unexpected executable model cell')
    namespace={}
    exec(compile(tree,str(path)+'::score_model','exec'),namespace)
    return namespace,hashlib.sha256(source.encode()).hexdigest()


def source_days(source=SOURCE,start=None,end=None):
    """Read one complete date at a time, combining all its Parquet fragments."""
    for part in [2,3,4]:
        with zipfile.ZipFile(source/f'project-2-bnpl-tables-part{part}.zip') as archive:
            members={}
            for name in archive.namelist():
                if not name.endswith('.parquet'):continue
                match=re.search(r'order_datetime=(\d{4}-\d{2}-\d{2})/',name)
                require(match is not None,'Undated transaction partition')
                date=pd.Timestamp(match[1])
                if (start is None or date>=start) and (end is None or date<=end):members.setdefault(date,[]).append(name)
            for date,names in sorted(members.items()):
                frames=[]
                for name in sorted(names):
                    table=pq.ParquetFile(io.BytesIO(archive.read(name))).read(columns=['order_id','user_id','merchant_abn','dollar_value'])
                    frames.append(table.to_pandas())
                frame=pd.concat(frames,ignore_index=True)
                require(frame.order_id.is_unique and frame.notna().all().all(),'Invalid daily order keys/values')
                require(np.isfinite(frame.dollar_value).all() and frame.dollar_value.ge(0).all(),'Invalid source sales')
                yield date,frame


def daily_features(orders):
    """Use ALL merchants' orders for a consumer's complete same-day activity."""
    f=orders.groupby('user_id',as_index=False).agg(same_day_spending=('dollar_value','sum'),same_day_order_count=('dollar_value','size'))
    f['same_day_average_order_value']=f.same_day_spending/f.same_day_order_count
    f['same_day_activity_available']=True
    return f


def complete_scores(features,observed,model,namespace):
    """Keep supplied entity/day scores and fill only missing values with team OLS."""
    pred=namespace['predict_score_model'](model,features)
    f=features[['user_id']].copy()
    f['observed_score']=f.user_id.map(observed)
    f['consumer_fraud_probability']=f.observed_score.fillna(pred.predicted_score)
    f['consumer_score_source']=np.where(f.observed_score.notna(),'observed','retrospective_model')
    f['low_support']=pred.low_support.to_numpy() & f.observed_score.isna()
    f['prediction_clipped']=pred.prediction_clipped.to_numpy() & f.observed_score.isna()
    require(f.consumer_fraud_probability.between(0,100).all(),'Missing completed scores')
    return f


def fingerprint(source):
    """Verify original audit signatures and add strong archive hashes."""
    audit=json.loads((ROOT/'tables/analysis/timesfm_comparison/daily_audit.json').read_text())
    result={}
    for item in audit['archives']:
        p=source/item['name'];stat=p.stat()
        require(stat.st_size==item['size'] and stat.st_mtime_ns==item['mtime_ns'],'Source archive differs from audited transaction-ID scan')
        result[item['name']]=file_hash(p)
    p=ROOT/'tables/analysis/timesfm_comparison/merchant_day.parquet'
    require(file_hash(p)==audit['daily_sha256'],'Original daily audit checksum changed')
    result['audited_daily_panel']=audit['daily_sha256']
    result['etl_notebook']=file_hash(ROOT/'code/01_etl.ipynb')
    result['original_snapshot']=file_hash(ROOT/'handoff/input_snapshot/snapshot.json')
    return result


def prepare(source=SOURCE):
    """Reconstruct imputation and audit adjusted daily, weekly and customer fees."""
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(sources=fingerprint(source),formula='sales * take_rate / 100 * (1 - consumer_score / 100)',
                  history=[str(START.date()),str(END.date())],risk_window=[str(RISK_START.date()),str(RISK_END.date())],
                  code_sha256=file_hash(Path(__file__)))
    if (OUT/'preparation_complete.json').exists():
        saved=json.loads((OUT/'preparation_complete.json').read_text())
        require(saved['protocol']==protocol,'Prepared inputs changed; use a new output directory')
        for name,digest in saved['files'].items():require(file_hash(OUT/name)==digest,'Prepared artifact changed')
        print('Reused audited fraud-adjusted preparation.',flush=True);return
    with zipfile.ZipFile(source/'project-2-bnpl-tables-part1.zip') as z:
        labels=pd.read_csv(io.BytesIO(z.read('tables/consumer_fraud_probability.csv'))).drop_duplicates()
        users=pd.read_parquet(io.BytesIO(z.read('tables/consumer_user_details.parquet')))
        known=pd.read_parquet(io.BytesIO(z.read('tables/tbl_merchants.parquet'))).reset_index()
        known['take_rate']=pd.to_numeric(known.tags.str.extract(r'(?i)take rate:\s*([0-9]+(?:\.[0-9]+)?)',expand=False))
        require(known.merchant_abn.is_unique and known.take_rate.between(0,100,inclusive='right').all(),'Invalid original merchant rates')
    labels['order_datetime']=pd.to_datetime(labels.order_datetime)
    require(not labels.duplicated(['user_id','order_datetime']).any(),'Conflicting fraud labels')
    require(labels.fraud_probability.between(0,100).all(),'Invalid supplied score')
    require(users.user_id.is_unique and users.consumer_id.is_unique and users.notna().all().all(),'Ambiguous customer identity')
    user_map=users.set_index('user_id').consumer_id
    namespace,model_hash=notebook_model()
    training=[]
    label_dates=set(labels.order_datetime)
    for date,orders in source_days(source,labels.order_datetime.min(),labels.order_datetime.max()):
        if date not in label_dates:continue
        today=labels.loc[labels.order_datetime.eq(date)].rename(columns={'fraud_probability':'target_score'})
        f=today.merge(daily_features(orders),on='user_id',how='left',validate='one_to_one')
        for c in ['same_day_spending','same_day_order_count','same_day_average_order_value']:f[c]=f[c].fillna(0.)
        f['same_day_activity_available']=f.same_day_activity_available.fillna(False).astype(bool)
        training.append(f)
    training=pd.concat(training,ignore_index=True).sort_values(['order_datetime','user_id']).reset_index(drop=True)
    require(len(training)==len(labels),'Label dates missing from archives')
    model=namespace['fit_score_model'](training,'consumer')
    require(model['kind']=='ols','Team OLS unavailable')
    dump(training,OUT/'consumer_training.parquet')
    rawcoef=model['model'].coef_/model['preprocessor'].scale_
    save_json(OUT/'consumer_model.json',dict(kind='team_notebook_ols',definition_cell_sha256=model_hash,
        labelled_rows=len(training),fitted_rows=model['n_fit_rows'],label_start=str(labels.order_datetime.min().date()),
        label_end=str(labels.order_datetime.max().date()),features=model['design_columns'],coefficients=rawcoef.tolist(),
        intercept=float(model['model'].intercept_-rawcoef@model['preprocessor'].mean_),
        note='Exact existing specification refitted because ignored completed predictions are absent; no new model selection.'))
    rates=known.set_index('merchant_abn').take_rate
    dates=[];daily=[];partitions=[];score_audits=[]
    for date,orders in source_days(source,START,END):
        require(date not in dates,'Overlapping source dates');dates.append(date)
        features=daily_features(orders)
        observed=labels.loc[labels.order_datetime.eq(date)].set_index('user_id').fraud_probability
        scores=complete_scores(features,observed,model,namespace)
        scores['order_datetime']=date
        dump(scores,OUT/'consumer_scores'/f'{date.date()}.parquet')
        orders=orders.merge(scores.drop(columns='order_datetime'),on='user_id',validate='many_to_one')
        orders['consumer_id']=orders.user_id.map(user_map)
        require(orders.consumer_id.notna().all(),'Unmapped customer')
        orders['take_rate']=orders.merchant_abn.map(rates)
        unknown=orders.take_rate.isna()
        score_audits.append(dict(day=str(date.date()),orders=len(orders),observed_orders=int(orders.consumer_score_source.eq('observed').sum()),
            imputed_orders=int(orders.consumer_score_source.ne('observed').sum()),low_support_orders=int(orders.low_support.sum()),
            clipped_orders=int(orders.prediction_clipped.sum()),unknown_merchant_orders=int(unknown.sum())))
        partitions.append(dict(day=date,orders=len(orders),sales=orders.dollar_value.sum()))
        orders=orders.loc[~unknown].copy()
        orders['unadjusted_fees']=orders.dollar_value*orders.take_rate/100
        orders['adjusted_fees']=discounted_fees(orders.dollar_value,orders.take_rate,orders.consumer_fraud_probability)
        require((orders.adjusted_fees<=orders.unadjusted_fees+1e-10).all(),'Adjustment increased fees')
        grouped=orders.groupby('merchant_abn',as_index=False).agg(sales=('dollar_value','sum'),orders=('order_id','size'),
                      unadjusted_fees=('unadjusted_fees','sum'),adjusted_fees=('adjusted_fees','sum'))
        grouped['day']=date;daily.append(grouped)
        if RISK_START<=date<=RISK_END:
            pairs=orders.groupby(['consumer_id','merchant_abn'],as_index=False).agg(gross_fees=('adjusted_fees','sum'))
            dump(pairs,OUT/'customer_parts'/f'{date.date()}.parquet')
        if len(dates)%30==0:print(f'Adjusted {len(dates)}/579 complete-history days.',flush=True)
    require(pd.DatetimeIndex(dates).sort_values().equals(pd.date_range(START,END)),'Incomplete source calendar')
    daily=pd.concat(daily,ignore_index=True)
    original=pd.read_parquet(ROOT/'tables/analysis/timesfm_comparison/merchant_day.parquet')
    new=daily.set_index(['merchant_abn','day'])[['sales','orders']].reindex(original.set_index(['merchant_abn','day']).index,fill_value=0)
    np.testing.assert_allclose(new,original[['sales','orders']],rtol=1e-11,atol=1e-7)
    grid=pd.MultiIndex.from_product([known.merchant_abn,pd.date_range(START,END)],names=['merchant_abn','day'])
    daily=daily.set_index(['merchant_abn','day']).reindex(grid,fill_value=0).reset_index()
    dump(daily,OUT/'merchant_day.parquet');dump(pd.DataFrame(partitions),OUT/'source_partitions.parquet')
    pd.DataFrame(score_audits).to_csv(OUT/'score_coverage.csv',index=False)
    merchants=pd.read_parquet(ROOT/'handoff/input_snapshot/merchants.parquet')
    gross=daily.groupby('merchant_abn').unadjusted_fees.sum()/19
    adjusted=daily.groupby('merchant_abn').adjusted_fees.sum()/19
    np.testing.assert_allclose(merchants.expected_monthly_fees,merchants.merchant_abn.map(gross),rtol=1e-10,atol=1e-7)
    merchants['unadjusted_expected_monthly_fees']=merchants.expected_monthly_fees
    merchants['expected_monthly_fees']=merchants.merchant_abn.map(adjusted)
    window=daily.loc[daily.day.between(RISK_START,RISK_END)].copy()
    window['week_end']=window.day.dt.to_period('W-SUN').dt.end_time.dt.normalize()
    weekly=window.groupby(['merchant_abn','week_end'],as_index=False).agg(gross_fees=('adjusted_fees','sum'))
    zero=merchants.expected_monthly_fees.le(0)|merchants.merchant_abn.map(weekly.groupby('merchant_abn').gross_fees.sum()).le(0)
    merchants.loc[merchants.eligible & zero,'eligibility_reason']='zero fraud-adjusted fees'
    merchants['eligible']=merchants.eligible & ~zero
    del daily,window,original,new;gc.collect()
    customer=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'customer_parts').glob('*.parquet'))],ignore_index=True)
    customer=customer.groupby(['consumer_id','merchant_abn'],as_index=False).gross_fees.sum()
    customer['consumer_id']=customer.consumer_id.astype('int64')
    meta=json.loads((ROOT/'handoff/input_snapshot/snapshot.json').read_text())
    for key in ['files','sha256','source_selection_freeze_sha256']:meta.pop(key,None)
    meta.update(fee_basis='consumer_fraud_adjusted',fraud_score_units='percentage_0_100',
        expected_return_definition='Mean of 19 complete monthly sums of transaction sales * take_rate / 100 * (1 - consumer_fraud_probability / 100)',
        imputation_scope='Retrospective all-label team OLS; observed scores preserved; not point-in-time backtest predictions',
        source_model_sha256=model_hash,eligibility_definition='Original sales eligibility, additionally positive adjusted mean and risk-window fees')
    write_snapshot(OUT/'input_snapshot',merchants,weekly,customer,meta)
    coverage=pd.DataFrame(score_audits)
    audit=dict(protocol=protocol,rows=int(coverage.orders.sum()),known_orders=int(coverage.orders.sum()-coverage.unknown_merchant_orders.sum()),
        observed_orders=int(coverage.observed_orders.sum()),imputed_orders=int(coverage.imputed_orders.sum()),
        low_support_orders=int(coverage.low_support_orders.sum()),clipped_orders=int(coverage.clipped_orders.sum()),
        unknown_merchant_orders=int(coverage.unknown_merchant_orders.sum()),eligible=int(merchants.eligible.sum()),
        gross_monthly_fees=float(gross.sum()),adjusted_monthly_fees=float(adjusted.sum()),
        all_579_days_reconciled_to_original=True,missing_scores=0,customer_weekly_reconciled=True,
        label_scope='Supplied consumer/date scores from 2021-02-28 to 2022-02-27; later orders are model extrapolation',
        files={str(p.relative_to(OUT)).replace('\\','/'):file_hash(p) for p in [OUT/'merchant_day.parquet',OUT/'consumer_training.parquet',OUT/'consumer_model.json',OUT/'input_snapshot/snapshot.json']})
    save_json(OUT/'preparation_complete.json',audit)
    print(f'Preparation complete: {audit["rows"]:,} transactions; {audit["eligible"]} eligible.',flush=True)


def scope_filename(scope):
    return ('portfolio_dashboard' if scope=='global' else scope+'_portfolio_dashboard')


def run_scope(scope):
    """Rebuild each scope's baseline, covariance and exposure from adjusted fees."""
    require((OUT/'preparation_complete.json').exists(),'Run prepare first')
    inp=OUT/'input_snapshot'
    base=load_snapshot(inp)
    f=pd.read_parquet(inp/'merchants.parquet')
    if scope!='global':f['eligible']=f.eligible & f.category.isin(SEGMENTS[scope])
    categories=[c for cats in SEGMENTS.values() for c in cats]
    require(len(categories)==len(set(categories))==25 and set(categories)==set(f.category),'Segment mapping mismatch')
    old=pd.read_parquet(ROOT/'handoff/final'/(scope_filename(scope)+'.parquet'))
    if scope!='global':require(old.merchant_category.isin(SEGMENTS[scope]).all(),'Mapping differs from original category portfolios')
    if scope=='global':snap=base
    else:
        del base;gc.collect()
        snap=build_snapshot(f,pd.read_parquet(inp/'weekly_fees.parquet'),pd.read_parquet(inp/'customer_fees.parquet'),json.loads((inp/'snapshot.json').read_text()))
    snap.fingerprints={p.name:file_hash(p) for p in inp.iterdir() if p.is_file()}
    snap.fingerprints['segments']=hashlib.sha256(json.dumps(SEGMENTS,sort_keys=True).encode()).hexdigest()
    # Existing lists are rescored; only merchant identities are used as warm starts.
    seed=OUT/(scope+'_seeds.parquet')
    old[['merchant_abn','as_of','retention']].to_parquet(seed,index=False,compression=None)
    t=time.monotonic()
    metrics,selections=optimize_snapshot(snap,OUT/'runs'/scope,RETENTIONS,.5,seed_portfolios=seed,
        iterations=30,nodes=200,time_limit=30,max_seconds=300,patience=8,gap=.001,bound_iterations=20,
        resume=(OUT/'runs'/scope/'protocol.json').exists())
    verify_records(snap,metrics,selections)
    save_json(OUT/'runs'/scope/'elapsed.json',dict(seconds=time.monotonic()-t))
    print(f'Finished {scope}: {len(metrics)} portfolios.',flush=True)


def report():
    """Export all thirty selections with explicit adjusted-fee labels and comparisons."""
    dest=DELIVERY/'final';dest.mkdir(parents=True,exist_ok=True)
    rows=[];files={}
    f=pd.read_parquet(OUT/'input_snapshot/merchants.parquet').set_index('merchant_abn')
    for scope in ['global',*SEGMENTS]:
        path=OUT/'runs'/scope/'portfolio_dashboard.parquet'
        require(path.exists(),f'Incomplete scope {scope}')
        current=pd.read_parquet(path)
        old=pd.read_parquet(ROOT/'handoff/final'/(scope_filename(scope)+'.parquet'))
        for r,g in current.groupby('retention'):
            before=old.loc[old.retention.eq(r)]
            before_ids=set(before.merchant_abn.astype('int64'));after_ids=set(g.merchant_abn.astype('int64'))
            row=g.iloc[0]
            rows.append(dict(scope=scope,retention=r,expected_monthly_adjusted_fees=row.portfolio_expected_monthly_gross_fees,
                old_selection_rescored_adjusted_fees=float(f.loc[list(before_ids),'expected_monthly_fees'].sum()),
                expected_monthly_unadjusted_fees=float(f.loc[list(after_ids),'unadjusted_expected_monthly_fees'].sum()),
                old_expected_monthly_unadjusted_fees=float(before.iloc[0].portfolio_expected_monthly_gross_fees),
                adjusted_cv=row.portfolio_variability_cv,adjusted_hhi=row.portfolio_customer_hhi,
                risk_score=row.portfolio_risk_score,achieved_adjusted_retention=row.portfolio_expected_fee_retention,
                overlap=len(before_ids&after_ids),replaced=100-len(before_ids&after_ids),
                certified=bool(row.portfolio_certified),optimality_gap=row.portfolio_optimality_gap,status=row.portfolio_status))
        current['fee_basis']='consumer_fraud_adjusted'
        current['scope']=scope
        current['portfolio_id']='fraud_adjusted_'+scope+'_'+current.portfolio_id
        current['merchant_expected_monthly_adjusted_fees']=current.merchant_expected_monthly_gross_fees
        current['portfolio_expected_monthly_adjusted_fees']=current.portfolio_expected_monthly_gross_fees
        for suffix in ['parquet','csv']:
            p=dest/(scope_filename(scope)+'.'+suffix)
            if suffix=='parquet':dump(current,p)
            else:current.to_csv(p,index=False)
            files[p.name]=file_hash(p)
    summary=pd.DataFrame(rows);summary.to_csv(DELIVERY/'comparison.csv',index=False)
    dump(summary,DELIVERY/'comparison.parquet')
    save_json(dest/'manifest.json',dict(rows=3000,portfolios=30,files=files,fee_basis='consumer_fraud_adjusted',
        formula='gross fee * (1 - consumer_fraud_probability / 100)',score='50% baseline-relative adjusted weekly CV + 50% baseline-relative adjusted customer HHI',
        baseline='Recomputed adjusted-revenue-only top 100 separately for each scope',
        original_outputs_preserved=True,legacy_gross_columns='Compatibility aliases for adjusted fees in THIS directory; use explicit adjusted columns',
        no_new_holdout=True,certified=int(summary.certified.sum())))
    print(summary.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','select','report','all'])
    parser.add_argument('--scope',choices=['global',*SEGMENTS])
    args=parser.parse_args()
    if os.name=='nt':
        from portfolio_optimizer.resource_limits import apply_worker_limits
        apply_worker_limits()
    with threadpool_limits(limits=1):
        if args.stage in ['prepare','all']:prepare()
        if args.stage in ['select','all']:
            for scope in ([args.scope] if args.scope else ['global',*SEGMENTS]):run_scope(scope)
        if args.stage in ['report','all']:report()

if __name__=='__main__':main()
