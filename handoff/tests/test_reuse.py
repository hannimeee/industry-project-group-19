"""Input-contract, dashboard and actual out-of-repository CLI reuse checks."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from portfolio_optimizer import core
from portfolio_optimizer.data import (build_snapshot,write_snapshot,load_snapshot,week_ends,
    merchant_features_from_monthly,read_table)
from portfolio_optimizer.cli import record,optimize_snapshot,verify_records
from portfolio_optimizer.export import merge_dashboard,write_dashboard


def frames():
    """Different dates, IDs, size and history from the real study."""
    dates=week_ends('2024-04-30',8)
    ids=np.array([901,902,903,904,905,906])
    merchants=pd.DataFrame(dict(merchant_abn=ids,name=['A','B','C','D','E','Inactive'],
        expected_monthly_fees=[100.,90.,80.,70.,60.,0.],eligible=[True]*5+[False]))
    x=np.array([25.,22.,19.,18.,17.,0.])[None,:]+np.sin(np.arange(8)[:,None]+np.arange(6)[None,:])*np.array([2,3,1,2,1,0])[None,:]
    weekly=pd.DataFrame([dict(merchant_abn=int(ids[i]),week_end=w,gross_fees=x[j,i])
        for j,w in enumerate(dates) for i in range(6)])
    fractions=np.array([[.8,.7,0,.1,.4],[.2,.3,.2,0,.3],[0,0,.8,.4,.1],[0,0,0,.5,.2]])
    customer=pd.DataFrame([dict(consumer_id=100+j,merchant_abn=int(ids[i]),gross_fees=fractions[j,i]*x[:,i].sum())
        for j in range(4) for i in range(5) if fractions[j,i]>0])
    meta=dict(as_of='2024-04-30',weeks=8,window_start=str((dates[0]-pd.Timedelta(days=6)).date()),
        window_end=str(dates[-1].date()),expected_return_start='2023-11-01',expected_return_end='2024-04-30',
        complete_coverage=True,currency='AUD')
    return merchants,weekly,customer,meta


def snapshot():
    return build_snapshot(*frames(),count=2)


def test_input_reconciliation_covariance_and_overlap():
    snap=snapshot(); o=snap.model
    assert len(snap.merchants)==5 and snap.metadata['as_of']=='2024-04-30'
    z=o.baseline
    x=snap.weekly[:,z.astype(bool)]
    variance=(1-snap.alpha)*x.sum(axis=1).var(ddof=1)+snap.alpha*x.var(axis=0,ddof=1).sum()
    assert o.components(z)[0]*o.base_cv == pytest.approx(np.sqrt(variance)/x.sum(axis=1).mean())
    f,w,c,meta=frames()
    by_merchant=c.groupby('merchant_abn').gross_fees.sum()
    c['allocated']=c.gross_fees/c.merchant_abn.map(by_merchant)*c.merchant_abn.map(f.set_index('merchant_abn').expected_monthly_fees)
    values=c[c.merchant_abn.isin([901,902])].groupby('consumer_id').allocated.sum()
    assert o.base_hhi==pytest.approx(((values/values.sum())**2).sum())


@pytest.mark.parametrize('problem',['missing_week','duplicate_week','duplicate_customer','wrong_total','wrong_window','unknown','negative','float_id','missing_id','coverage','fractional_weeks','partial_month'])
def test_bad_input_rejected(problem):
    f,w,c,m=frames()
    if problem=='missing_week': w=w.iloc[1:]
    if problem=='duplicate_week': w=pd.concat([w,w.iloc[:1]])
    if problem=='duplicate_customer': c=pd.concat([c,c.iloc[:1]])
    if problem=='wrong_total': c.loc[0,'gross_fees']+=1
    if problem=='wrong_window': m['window_start']='2024-01-01'
    if problem=='unknown': c.loc[0,'merchant_abn']=999
    if problem=='negative': w.loc[0,'gross_fees']=-1
    if problem=='float_id': f['merchant_abn']=f.merchant_abn.astype(float)
    if problem=='missing_id': c.loc[0,'consumer_id']=np.nan
    if problem=='coverage': m['complete_coverage']=False
    if problem=='fractional_weeks': m['weeks']=8.5
    if problem=='partial_month': m['expected_return_end']='2024-04-15'
    with pytest.raises((ValueError,AssertionError)):
        build_snapshot(f,w,c,m,count=2)


def test_monthly_mean_take_rate_and_eligibility():
    f=pd.DataFrame(dict(merchant_abn=[1,2],name=['a','b'],take_rate=[2.5,5.]))
    months=pd.date_range('2023-11-01',periods=6,freq='MS')
    panel=pd.DataFrame([dict(merchant_abn=i,month=t,sales=100. if i==1 or j>=3 else 0.)
        for i in [1,2] for j,t in enumerate(months)])
    output=merchant_features_from_monthly(f,panel,'2023-11-01','2024-04-30')
    np.testing.assert_allclose(output.expected_monthly_fees,[2.5,2.5])
    assert output.eligible.tolist()==[True,False]
    later=pd.concat([panel,pd.DataFrame([dict(merchant_abn=1,month=pd.Timestamp('2024-05-01'),sales=1e12)])],ignore_index=True)
    pd.testing.assert_frame_equal(output,merchant_features_from_monthly(f,later,'2023-11-01','2024-04-30'))
    with pytest.raises(ValueError,match='coverage'):
        merchant_features_from_monthly(f,panel.iloc[1:],'2023-11-01','2024-04-30')


def test_roundtrip_and_corrupt_checksum(tmp_path):
    write_snapshot(tmp_path/'inputs',*frames(),count=2)
    restored=load_snapshot(tmp_path/'inputs',2)
    np.testing.assert_allclose(restored.model.components(np.array([1,0,1,0,0.])),snapshot().model.components(np.array([1,0,1,0,0.])))
    path=tmp_path/'inputs/merchants.parquet'
    f=pd.read_parquet(path); f.loc[0,'expected_monthly_fees']+=1
    f.to_parquet(path,index=False,compression=None)
    with pytest.raises(ValueError,match='checksum'):
        load_snapshot(tmp_path/'inputs',2)


def test_dashboard_single_table_no_individual_score(tmp_path):
    snap=snapshot()
    info=dict(score=100.,lower_bound=100.,gap=0.,certified=True,status='unique',iterations=0)
    m,s=record(snap,snap.model.baseline,1.,.5,info)
    out=write_dashboard(m,s,tmp_path)
    assert len(out)==2 and out.merchant_abn.dtype==object
    assert 'merchant_risk_score' not in out
    assert out.drop_duplicates('portfolio_id').portfolio_expected_monthly_gross_fees.sum()==pytest.approx(190)
    assert out.merchant_expected_monthly_gross_fees.sum()==pytest.approx(190)
    assert out.portfolio_expected_monthly_gross_fees.sum()==pytest.approx(380)  # explains the aggregation trap
    csv=pd.read_csv(tmp_path/'portfolio_dashboard.csv',dtype={'merchant_abn':str})
    pd.testing.assert_frame_equal(csv,out,check_dtype=False)
    with pytest.raises(ValueError,match='Duplicate portfolio'):
        merge_dashboard(pd.concat([m,m]),s)
    wrong=s.copy(); wrong['rank_by_expected_fees']=[2,1]
    with pytest.raises(ValueError,match='ranks'):
        merge_dashboard(m,wrong)


def test_budget_stop_and_resume_integrity(tmp_path):
    write_snapshot(tmp_path/'inputs',*frames(),count=2)
    snap=load_snapshot(tmp_path/'inputs',2)
    opts=dict(iterations=2,nodes=20,bound_iterations=0,max_seconds=1e-9,time_limit=.1)
    m,s=optimize_snapshot(snap,tmp_path/'out',[1.,.973],.4,**opts)
    assert s.groupby('retention').size().eq(2).all()
    assert m.loc[m.retention.eq(.973),'status'].item()=='bounded_best_found_time_budget'
    verify_records(snap,m,s)
    m2,s2=optimize_snapshot(snap,tmp_path/'out',[1.,.973],.4,resume=True,**opts)
    pd.testing.assert_frame_equal(m,m2);pd.testing.assert_frame_equal(s,s2)
    with pytest.raises(ValueError,match='changed'):
        optimize_snapshot(snap,tmp_path/'out',[1.,.973],.3,resume=True,**opts)


def test_arbitrary_percent_weight_and_complete_pool():
    snap=snapshot(); o=snap.model
    z,info,_=core.optimize(o,.3,.873,budgets=[dict(cuts=40,nodes=1000)],relative_gap=1e-9)
    from itertools import combinations
    values=[]
    for ids in combinations(range(o.n),o.count):
        trial=np.zeros(o.n);trial[list(ids)]=1
        if core.feasible(o,trial,.873): values.append(o.value(trial,.3))
    assert o.value(z,.3)==pytest.approx(min(values),rel=1e-9)
    assert info['lower_bound']/100 <= min(values)+1e-8


def test_component_bounds_and_zero_optimum():
    snap=snapshot();o=snap.model
    from itertools import combinations
    components=[]
    for ids in combinations(range(o.n),o.count):
        z=np.zeros(o.n);z[list(ids)]=1
        if core.feasible(o,z,.85): components.append(o.components(z))
    for j in [0,1]:
        assert core.component_relaxation_bound(o,.85,j,15)<=np.min(components,axis=0)[j]+1e-7
    o=core.JointRisk([3.,2.,1.],[3.,2.,1.],np.array([[1.,1.,-1.]]),[0.,0.,0.],sparse.diags([3.,2.,1.]),count=2)
    z,info,_=core.optimize(o,1.,.8,budgets=[dict(cuts=10,nodes=100)])
    assert info['score']==0 and info['gap']==0 and info['certified']


def test_sunday_and_partial_weeks():
    assert week_ends('2024-04-28',3)[-1]==pd.Timestamp('2024-04-28')
    assert week_ends('2024-04-30',3)[-1]==pd.Timestamp('2024-04-28')


def test_portable_cli_outside_repository(tmp_path):
    home=Path(__file__).resolve().parents[1]
    portable=tmp_path/'standalone';portable.mkdir()
    shutil.copyfile(home/'optimize.py',portable/'optimize.py')
    shutil.copytree(home/'portfolio_optimizer',portable/'portfolio_optimizer',ignore=shutil.ignore_patterns('__pycache__'))
    f,w,c,meta=frames()
    for name,frame in [('merchants',f),('weekly',w),('customers',c)]:frame.to_csv(portable/(name+'.csv'),index=False)
    (portable/'meta.json').write_text(json.dumps(meta))
    def run(*args):
        result=subprocess.run([sys.executable,'optimize.py',*args],cwd=portable,capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    run('prepare','--merchants','merchants.csv','--weekly-fees','weekly.csv','--customer-fees','customers.csv',
        '--metadata','meta.json','--count','2','--output','inputs')
    run('optimize','--input','inputs','--output','results','--retentions','100','97.3','--count','2',
        '--weight','0.3','--iterations','8','--nodes','100','--max-seconds','5','--time-limit','1','--bound-iterations','3')
    run('export','--input','inputs','--metrics','results/metrics.parquet','--selections','results/selections.parquet','--output','dashboard')
    output=pd.read_parquet(portable/'dashboard/portfolio_dashboard.parquet')
    assert len(output)==4 and set(output.retention)=={1.,.973}
    assert output.as_of.eq('2024-04-30').all() and output.variability_weight.eq(.3).all()
    assert not (portable/'code').exists() and not (portable/'tables').exists()
