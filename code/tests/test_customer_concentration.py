"""Concentration mathematics, exact normalization and chronological controls."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0,str((Path(__file__).resolve().parents[1] / "scripts")))
import customer_concentration as cc
from test_risk_aware_selection import toy as panel_toy


def toy():
    p=np.array([[.8,.7,0,.1,.4],[.2,.3,.2,0,.3],[0,0,.8,.4,.1],[0,0,0,.5,.2]])
    mu=np.array([100.,90.,80.,70.,60.])
    return p,mu


def test_fee_shares_and_zero():
    # 2% of $100/$300 -> $2/$6: HHI .25^2+.75^2=.625.
    m=cc.shares_metrics(np.array([100.,300.])*.02)
    assert m['fees']==8
    assert m['hhi']==.625
    assert m['effective_customers']==1.6
    assert m['top_1_share']==.75
    assert m['top_10_share']==1
    assert np.isnan(cc.shares_metrics([0.,0.])['hhi'])
    with pytest.raises(ValueError): cc.shares_metrics([-1,2])


def test_shared_disjoint_and_order():
    shared=cc.Exposure(np.arange(2),np.array([11,22]),sparse.csc_matrix([[50,50],[50,50]]))
    disjoint=cc.Exposure(np.arange(4),np.array([11,22]),sparse.csc_matrix([[50,0],[50,0],[0,50],[0,50]]))
    s,d=cc.concentration(shared),cc.concentration(disjoint)
    assert s['hhi']==.5 and d['hhi']==.25
    assert s['individual_hhi']==d['individual_hhi']==.25
    assert s['shared_hhi']==.25 and d['shared_hhi']==0
    assert s['shared_customer_fee_share']==1 and d['shared_customer_fee_share']==0
    o=cc.overlap_matrix(shared,[22,11])
    assert o.index.tolist()==[22,11]
    np.testing.assert_allclose(o,.5)


@pytest.mark.parametrize('kind',cc.METHODS)
@pytest.mark.parametrize('protected',[False,True])
@pytest.mark.parametrize('money_scale',[1.,1e6])
def test_optimizer_exhaustive_and_reproducible(kind,protected,money_scale):
    p,mu=toy()
    mu*=money_scale
    actual=np.array([80.,110.,95.,70.,100.])*money_scale if protected else mu
    a=sparse.csc_matrix(p*actual)
    budget=[dict(max_cuts=100,node_limit=2000,max_swaps=100,relative_gap=1e-9)]
    floor=.85*actual[:2].sum() if protected else None
    z,info,log=cc.optimize(a,mu,kind,count=2,retention=.85,actual_floor=floor,budgets=budget)
    obj=cc.FractionalHHI(a,kind,2)
    feasible=[]
    for ids in combinations(range(len(mu)),2):
        x=np.zeros(len(mu));x[list(ids)]=1
        if mu@x>=.85*190*money_scale and (floor is None or actual@x>=floor): feasible.append(obj.value(x))
    assert mu@z>=.85*190*money_scale-1e-6
    if floor is not None: assert actual@z>=floor-1e-6
    assert info['objective_value']==pytest.approx(min(feasible),rel=1e-8)
    assert info['lower_bound']<=min(feasible)+1e-9
    assert info['certified']
    zz,ii,ll=cc.optimize(a,mu,kind,count=2,retention=.85,actual_floor=floor,budgets=budget)
    np.testing.assert_array_equal(z,zz)
    assert info==ii
    pd.testing.assert_frame_equal(log,ll,check_exact=True)


@pytest.mark.parametrize('kind',cc.METHODS)
def test_gradient_tangent_normalization_and_scale(kind):
    p,mu=toy();a=sparse.csc_matrix(p*mu)
    obj=cc.FractionalHHI(a,kind,2)
    v=np.array([.4,.7,.2,.1,.9])
    val,grad=obj.value_gradient(v)
    for j in range(5):
        e=np.zeros(5);e[j]=1e-6
        assert grad[j]==pytest.approx((obj.value_gradient(v+e)[0]-obj.value_gradient(v-e)[0])/2e-6,rel=1e-6)
    for w in np.random.default_rng(31).uniform(0,2,(20,5)):
        assert obj.value_gradient(w)[0]>=val+grad@(w-v)-1e-12
    z=np.array([1.,0,1,0,0])
    total=a@z;expected=total@total/(mu@z)**2 if kind=='shared' else np.sum((p*mu*z)**2)/(mu@z)**2
    assert obj.value(z)==pytest.approx(expected)
    assert cc.FractionalHHI(a*1000,kind,2).value(z)==pytest.approx(expected)


def test_expected_allocation_and_ties():
    p,mu=toy()
    e=cc.Exposure(np.arange(4),np.arange(5),sparse.csc_matrix(p*np.arange(1,6)))
    a=cc.expected_exposure(e,mu)
    np.testing.assert_allclose(a.fees.toarray(),p*mu)
    e2=cc.Exposure(np.array([20,10]),np.array([1]),sparse.csc_matrix([[10.],[10.]]))
    assert cc.shock_customers(e2,1).tolist()==[10]


def test_bounded_budget_not_false_certificate():
    p,mu=toy()
    z,i,_=cc.optimize(sparse.csc_matrix(p*mu),mu,'shared',count=2,retention=.85,
                      budgets=[dict(max_cuts=1,node_limit=0,max_swaps=1,relative_gap=1e-12)])
    assert z.sum()==2 and mu@z>=.85*190
    assert i['lower_bound']<=i['objective_value']+1e-12
    if not i['certified']: assert i['solver_status']=='bounded_best_found'


@pytest.mark.parametrize('kind',cc.METHODS)
def test_perspective_tangents_bound_every_binary_portfolio(kind):
    p,mu=toy()
    # Add exclusive-customer rows to exercise the full-HHI perspective component.
    a=sparse.vstack([sparse.csc_matrix(p*mu),sparse.diags(mu*.2)],format='csc')
    obj=cc.FractionalHHI(a,kind,2)
    z=np.array([1.,1,0,0,0]);v=z/(obj.u@z)
    f,gz,gv=obj.binary_tangent(z)
    for c in combinations(range(5),2):
        zz=np.zeros(5);zz[list(c)]=1;vv=zz/(obj.u@zz)
        assert obj.value(zz)>=f+gz@(zz-z)+gv@(vv-v)-1e-12


def test_random_shocks_identity_invariance_and_moments():
    ids=np.arange(1,2001,dtype=np.int64)
    u=cc.splitmix_uniform(ids,10)
    np.testing.assert_array_equal(u[::-1],cc.splitmix_uniform(ids[::-1],10))
    np.testing.assert_array_equal(u,cc.splitmix_uniform(np.r_[ids,9000],10)[:-1])
    fees=np.linspace(1,2,len(ids))
    draws=cc.random_losses(ids,np.column_stack([fees,fees*2]),draws=1000,probabilities=[.1])[0]
    np.testing.assert_array_equal(draws[:,1],2*draws[:,0])
    theoretical=.1*.9*np.sum(fees**2)
    assert abs(draws[:,0].mean()-.1*fees.sum())<6*np.sqrt(theoretical/1000)
    assert draws[:,0].var(ddof=1)==pytest.approx(theoretical,rel=.2)
    assert (draws>=0).all() and (draws<=np.array([fees.sum(),2*fees.sum()])).all()


def test_dynamics_no_activity_is_not_lost_future_fees():
    rows=cc.customer_dynamics(np.array([10.,20,30,0]),np.array([15.,0,0,40.]),np.array([1,2,4]),np.arange(1,5))
    groups={r['customer_group']:r for r in rows}
    assert groups['returning']['future_fees']==15
    assert groups['new_to_portfolio']['future_fees']==40
    assert groups['former_active_elsewhere']['historical_fees']==20
    assert groups['former_network_inactive']['historical_fees']==30
    assert groups['former_network_inactive']['future_fees']==0


def test_break_even_and_unavailable():
    assert cc.break_even(10,20)==(.5,'within_0_1')
    assert cc.break_even(30,20)==(1.5,'above_100_percent')
    assert cc.break_even(-10,20)==(-.5,'already_ahead')
    assert cc.break_even(1,0)[1]=='no_positive_loss_avoidance'


def test_future_perturbation_and_missing_coverage(tmp_path,monkeypatch):
    monkeypatch.setattr(cc,'OUT',tmp_path)
    directory=tmp_path/'customer_day';directory.mkdir()
    dates=pd.to_datetime(['2022-01-02','2022-01-09','2022-01-23'])
    frame=pd.DataFrame({'day':dates.repeat(3),'merchant_abn':np.tile([10,20,30],3),
        'consumer_id':np.tile([1,2,1],3),'gross_fees':np.tile([10.,9.,8.],3)})
    path=directory/'2022-01.parquet';frame.to_parquet(path,index=False,compression=None)
    def calculate():
        h=cc.read_exposure(pd.Timestamp('2022-01-01'),pd.Timestamp('2022-01-16'),np.array([10,20,30]),np.array([1,2]))
        a=cc.expected_exposure(h,np.array([10.,9.,8.]))
        z,info,_=cc.optimize(a.fees,np.array([10.,9.,8.]),'shared',count=2,retention=.9)
        return h.fees.toarray(),z,info,cc.shock_customers(a,1)
    first=calculate()
    frame.loc[frame.day.gt('2022-01-16'),'gross_fees']*=1e6
    frame.to_parquet(path,index=False,compression=None)
    second=calculate()
    for a,b in zip(first,second):
        if isinstance(a,dict): assert a==b
        else: np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='Missing source coverage'):
        cc.read_exposure(pd.Timestamp('2021-12-01'),pd.Timestamp('2022-01-16'),np.array([10,20,30]),np.array([1,2]))
    with pytest.raises(ValueError,match='frozen selections'): cc.validate_freeze()


def test_expected_inactive_and_duplicate_subset_rejected():
    e=cc.Exposure(np.array([1,2]),np.array([10,20]),sparse.csc_matrix([[1,0],[0,0]]))
    with pytest.raises(ValueError): cc.expected_exposure(e,[1,2])
    with pytest.raises(ValueError): e.subset([10,10])
    with pytest.raises(ValueError): e.subset([30])


def test_common_shock_screen_uses_own_revenue_floor():
    dates=pd.date_range('2021-11-30',periods=8,freq='ME')
    f=pd.DataFrame(dict(as_of=dates,window='26w',method='shared',retention=.95,k=100,severity=1.,
        remaining_fee_gain=20.,gain_baseline_pct=.2,actual_fee_retention_pct=96.,loss_avoided=30.,revenue_sacrificed=10.))
    r=cc.stress_summary(f,dates,['26w'],'primary_26w_eight').iloc[0]
    assert r.scenario_protection and r.joint_successes==8
    f['actual_fee_retention_pct']=94.9
    assert not cc.stress_summary(f,dates,['26w'],'primary_26w_eight').iloc[0].scenario_protection


def test_downside_fragility_uses_aggregate_ratio_influence():
    p=pd.DataFrame(dict(as_of=cc.ORIGINS,window='26w',method='shared',retention=.99,
        baseline_normalized_downside=[1.,.1]+[.01]*6,
        normalized_downside=[.955,.08]+[.0095]*6,
        actual_fee_retention_pct=100.,downside_improves=True,actual_fee_guardrail_met=True))
    stress=pd.DataFrame(columns=['window','as_of','method','retention','k','severity'])
    result=cc.extended_fragility(p,stress).iloc[0]
    # The first origin has the largest absolute difference, but removing the
    # second drops aggregate improvement below 5%, reversing the conclusion.
    assert result.removed_origin==cc.ORIGINS[1]
    assert result.remaining_downside_reduction_pct==pytest.approx(100*.048/1.06)
    assert result.conclusion_reverses


@pytest.mark.parametrize('kind',cc.METHODS)
def test_revenue_secant_global_bound_matches_enumeration(kind):
    rng=np.random.default_rng(71)
    a=rng.uniform(.1,2,(6,101))
    mu=a.sum(axis=0)
    baseline=np.argsort(-mu)[:2];floor=.95*mu[baseline].sum()
    obj=cc.FractionalHHI(sparse.csc_matrix(a),kind,2)
    optimum=min(obj.value(np.isin(np.arange(101),ids).astype(float)) for ids in combinations(range(101),2) if mu[list(ids)].sum()>=floor)
    z,info,log=cc.optimize(sparse.csc_matrix(a),mu,kind,count=2,retention=.95,
        budgets=[dict(max_cuts=30,node_limit=200,max_swaps=100,relative_gap=.001)])
    assert 'continuous_presolve' in log.bound_method.to_list()
    assert info['lower_bound']<=optimum+1e-10
    assert info['objective_value']>=optimum-1e-10
    if info['certified']: assert info['objective_value']<=optimum/(1-.001)+1e-10


@pytest.mark.parametrize('kind',cc.METHODS)
def test_continuous_presolve_and_supports_are_global_bounds(kind):
    p,mu=toy();obj=cc.FractionalHHI(sparse.csc_matrix(p*mu),kind,2)
    z=np.array([1.,1,0,0,0]);scale=obj.value(z)
    constraints=[mu/190,obj.u];floors=[.85,.85]
    bound,planes=cc.continuous_presolve(obj,z,constraints,floors,np.zeros(5),np.ones(5),scale,steps=30)
    values=[]
    for ids in combinations(range(5),2):
        zz=np.zeros(5);zz[list(ids)]=1
        if mu@zz>=.85*190: values.append(obj.value(zz))
        numerator=obj.value_gradient(zz)[0]
        for gradient,right in planes: assert numerator/scale>=gradient@zz-right-1e-10
    assert bound<=min(values)+1e-10


def test_full_evaluation_pipeline_with_frozen_toy_portfolios(tmp_path,monkeypatch,panel_toy):
    daily,monthly,merchants=panel_toy
    weekly=cc.cf.weekly_panel(daily,merchants)
    monkeypatch.setattr(cc,'OUT',tmp_path/'study');cc.OUT.mkdir()
    monkeypatch.setattr(cc.cf,'OUT',tmp_path/'cashflow');cc.cf.OUT.mkdir()
    monkeypatch.setattr(cc,'load_inputs',lambda:(daily,monthly,merchants,weekly,{}))
    monkeypatch.setattr(cc,'validate_preparation',lambda *args: {})
    monkeypatch.setattr(cc,'validate_freeze',lambda: {})
    customers=np.arange(1,151)
    cc.table(pd.DataFrame(dict(consumer_id=customers,user_id=customers)),'identities',csv=False,directory=cc.OUT)
    facts=daily.loc[daily.sales.gt(0)].copy()
    facts['consumer_id']=(facts.merchant_abn+facts.day.dt.day)%150+1
    facts['gross_fees']=facts.sales*.05
    for month,g in facts.groupby(facts.day.dt.to_period('M')):
        cc.table(g[['day','merchant_abn','consumer_id','gross_fees']].reset_index(drop=True),str(month),cc.OUT/'customer_day',csv=False)
    activity=facts.assign(month=facts.day.dt.to_period('M').dt.to_timestamp())[['consumer_id','month']].drop_duplicates()
    cc.table(activity,'network_activity',csv=False,directory=cc.OUT)
    all_f,all_e,all_s,all_shocks=[],[],[],[]
    for date in cc.cf.CUTOFFS:
        f,_=cc.cf.features_at_cutoff(weekly,monthly,merchants,date);all_f.append(f)
        b=cc.cf.baseline_selection(f)
        ref=cc.risk.history_matrix(weekly,cc.cf.risk_window(date),b.merchant_abn.to_numpy()).sum(axis=1).mean()
        for window in ['26w','52w']:
            dates=cc.history_dates(weekly,date,window)
            if dates is None: continue
            h=cc.read_exposure(dates[0]-pd.Timedelta(days=6),dates[-1],f.merchant_abn.to_numpy(),customers)
            cc.save_exposure(h,cc.group_directory(date,window))
            if window=='26w':
                eligible=f.loc[f.eligible].sort_values('merchant_abn')
                a=cc.expected_exposure(h.subset(eligible.merchant_abn.to_numpy()),eligible.expected_monthly_fees.to_numpy())
                all_shocks.extend(pd.DataFrame(dict(as_of=date,k=k,consumer_id=cc.shock_customers(a,k))) for k in cc.SHOCK_K)
            for method,r in [('revenue_only',1.)]+[(m,r) for r in cc.RETENTIONS for m in cc.METHODS]:
                key=dict(as_of=date,window=window,method=method,retention=r)
                all_e.append(dict(**key,common_reference_weekly_fees=ref,own_reference_weekly_fees=ref,
                    overlap_with_baseline=100,expected_monthly_fees=b.expected_monthly_fees.sum()))
                all_s.append(b.assign(**key))
    cc.table(pd.concat(all_f,ignore_index=True),'merchant_features',cc.cf.OUT,csv=False)
    for name,f in [('unscored_estimates',pd.DataFrame(all_e)),('selections',pd.concat(all_s,ignore_index=True)),('common_shocks',pd.concat(all_shocks,ignore_index=True))]:
        cc.table(f,name,cc.OUT,csv=False)
    cc.evaluate_stage()
    result=pd.read_parquet(cc.OUT/'frontier_metrics.parquet')
    assert len(result)==135 and result.status.eq('observed').sum()==117
    assert result.loc[result.status.eq('observed'),'hhi_reduction_pct'].abs().max()<1e-10
    assert result.loc[result.as_of.eq('2022-09-30'),'status'].eq('unavailable').all()
    stress=pd.read_parquet(cc.OUT/'common_stress.parquet')
    assert stress.remaining_fee_gain.abs().max()<1e-7
    assert (stress.loss<=stress.fees+1e-7).all()
    own=pd.read_parquet(cc.OUT/'own_customer_stress.parquet')
    assert set(own.k)==set(cc.TOP_K)
    assert len(own)==117*len(cc.TOP_K)
    # Repeat the entire outcome pipeline, not merely a reused manifest.
    cc.evaluate_stage()
