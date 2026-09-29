"""Customer concentration, frozen portfolio comparisons and conditional spending shocks.

Stages: prepare -> select -> evaluate -> oracle -> report. Previous studies are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from threadpoolctl import threadpool_limits

import cash_flow_variability as cf
import cash_flow_risk_comparison as risk
import diversification_frontier as frontier
import risk_aware_selection as rs
from cash_flow_lookback_comparison import history_dates
from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from forecast_comparison import ORIGINS, digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tables/analysis/customer_concentration'
KEY = ['as_of', 'window', 'method', 'retention']
RETENTIONS = [.995, .99, .98, .95]
METHODS = ['individual', 'shared']
BUDGETS = [dict(max_cuts=30, node_limit=200, max_swaps=200, relative_gap=.001),
           dict(max_cuts=100, node_limit=2000, max_swaps=200, relative_gap=.001)]
TOP_K = [1, 10, 100, 1000]
SHOCK_K = [10, 100, 1000]
SEVERITIES = [.25, .5, 1.]
PROBABILITIES = [.01, .05, .1]
SEED, DRAWS = 2026, 500


def fingerprint(path):
    return frontier.file_hash(path)


def previous_inventory():
    return {str(p.relative_to(OUT.parent)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(OUT.parent.rglob('*')) if p.is_file() and OUT not in p.parents}


def table(frame, name, directory=None, csv=True):
    directory=OUT if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / (name+'.parquet'), index=False, compression=None)
    if csv:
        frame.to_csv(directory / (name+'.csv'), index=False)


def load_inputs():
    daily, monthly, merchants, audit = cf.load_audited_inputs()
    return daily, monthly, merchants, cf.weekly_panel(daily, merchants), audit


def protocol(audit):
    return dict(version=1, count=100, dates=cf.CUTOFFS.strftime('%Y-%m-%d').tolist(),
        windows=['26w', '52w'], retentions=RETENTIONS, methods=METHODS, budgets=BUDGETS,
        expected_return='expanding complete-month mean gross fees', fee_rate_divisor=100,
        seed=SEED, draws=DRAWS, top_k=TOP_K, shock_k=SHOCK_K, severities=SEVERITIES,
        probabilities=PROBABILITIES, concentration_screen_pct=5, downside_screen_pct=5,
        stress_screen_baseline_fee_pct=.1, primary_joint_successes=6,
        reference='revenue-only baseline latest 26 complete weeks mean fees',
        common_shocks='top customers across all eligible merchants, estimated fees, 26 weeks',
        shock_rng='SplitMix64(seed, replicate, consumer_id); stable under identity extension',
        absolute_gap_baseline_fraction=1e-6,
        solver_objective_bound='known feasible HHI; excludes no improving portfolio',
        solver_revenue_constraints='AUD units; final absolute feasibility tolerance AUD 1e-6',
        independent_date_workers=6,
        retrospective=True, untouched_holdout=False,
        data=audit, versions={p:importlib.metadata.version(p) for p in
                             ['numpy','pandas','scipy','pyarrow','scikit-learn','threadpoolctl']},
        assumptions=['stable known identity mappings, take rates and categories',
            'selection does not change sales', 'customer shocks remove spending without replacement',
            'gross fees exclude principal losses, defaults, fraud, funding and operating costs'])


def prepare_stage():
    daily, monthly, merchants, weekly, audit = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT/'preparation.json'
    if marker.exists():
        validate_preparation(audit)
        print('Verified existing customer preparation.', flush=True)
        return
    old = previous_inventory()
    source = ROOT.parent/'project 2 materials'
    with zipfile.ZipFile(source/'project-2-bnpl-tables-part1.zip') as archive:
        users = pd.read_parquet(io.BytesIO(archive.read('tables/consumer_user_details.parquet')))
    require(users.notna().all().all() and users.user_id.is_unique and users.consumer_id.is_unique,
            'Ambiguous customer identity map')
    mapping = users.set_index('user_id').consumer_id
    rates = merchants.set_index('merchant_abn').take_rate/100
    table(users.sort_values('consumer_id').reset_index(drop=True), 'identities', csv=False)
    parts, known_daily, networks, month_pieces, part_audit = [], [], [], {}, []
    total, unknown = 0, 0
    for part in [2,3,4]:
        with zipfile.ZipFile(source/f'project-2-bnpl-tables-part{part}.zip') as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith('.parquet')):
                match = re.search(r'order_datetime=(\d{4}-\d{2}-\d{2})/', name)
                require(match is not None, 'Missing transaction partition date')
                day = pd.Timestamp(match[1])
                if not pd.Timestamp('2021-03-01') <= day <= pd.Timestamp('2022-09-30'):
                    continue
                reader = pq.ParquetFile(io.BytesIO(archive.read(name)))
                n, sales = 0, 0.
                for batch in reader.iter_batches(batch_size=100_000, columns=['user_id','merchant_abn','dollar_value']):
                    raw = batch.to_pandas()
                    require(raw.notna().all().all() and np.isfinite(raw.dollar_value).all()
                            and raw.dollar_value.gt(0).all(), 'Invalid source transaction')
                    raw['consumer_id'] = raw.user_id.map(mapping)
                    require(raw.consumer_id.notna().all(), 'Unmapped transaction customer')
                    n += len(raw); sales += float(raw.dollar_value.sum())
                    networks.append(pd.DataFrame({'consumer_id':raw.consumer_id.unique(), 'month':day.to_period('M').start_time}))
                    known = raw.loc[raw.merchant_abn.isin(rates.index)]
                    unknown += len(raw)-len(known)
                    g = known.groupby(['consumer_id','merchant_abn'], as_index=False).dollar_value.agg(sales='sum', orders='size')
                    g['day'] = day
                    g['gross_fees'] = g.sales*g.merchant_abn.map(rates)
                    month_pieces.setdefault(day.strftime('%Y-%m'), []).append(g)
                    known_daily.append(g.groupby('merchant_abn')[['sales','orders']].sum().reset_index().assign(day=day))
                part_audit.append(dict(source_partition=name, day=day, orders=n, sales=sales))
                total += n
        print(f'Customer source archive {part}/4 aggregated.', flush=True)
    source_actual = pd.DataFrame(part_audit).sort_values('source_partition').reset_index(drop=True)
    source_expected = pd.read_parquet(ROOT/'tables/analysis/external_data_comparison/source_partition_audit.parquet')
    source_expected = source_expected.loc[source_expected.day.ge('2021-03-01'), source_actual.columns].sort_values('source_partition').reset_index(drop=True)
    pd.testing.assert_frame_equal(source_actual, source_expected, check_dtype=False, rtol=1e-12, atol=1e-7)
    require(set(source_actual.day) == set(pd.date_range('2021-03-01','2022-09-30')), 'Incomplete source calendar')
    reconstructed = pd.concat(known_daily).groupby(['merchant_abn','day'])[['sales','orders']].sum()
    expected = daily.set_index(['merchant_abn','day'])[['sales','orders']]
    reconstructed = reconstructed.reindex(expected.index, fill_value=0)
    pd.testing.assert_frame_equal(reconstructed, expected, check_dtype=False, rtol=1e-12, atol=1e-7)
    for month, pieces in sorted(month_pieces.items()):
        f = pd.concat(pieces, ignore_index=True).groupby(['day','merchant_abn','consumer_id'], as_index=False)[['sales','orders','gross_fees']].sum()
        require(not f.duplicated(['day','merchant_abn','consumer_id']).any(), 'Duplicate customer-day keys')
        table(f, month, OUT/'customer_day', csv=False)
        parts.append(dict(month=month, rows=len(f), orders=int(f.orders.sum()), sales=float(f.sales.sum()),
                          sha256=fingerprint(OUT/'customer_day'/f'{month}.parquet')))
    activity = pd.concat(networks, ignore_index=True).drop_duplicates().sort_values(['month','consumer_id']).reset_index(drop=True)
    table(activity, 'network_activity', csv=False)
    table(source_actual, 'source_partition_audit', csv=False)
    table(pd.DataFrame(parts), 'customer_partition_audit')
    save_json(OUT/'protocol.json', protocol(audit))
    save_json(marker, dict(audit=audit, identities=fingerprint(OUT/'identities.parquet'),
        network=fingerprint(OUT/'network_activity.parquet'), partitions=parts,
        total_orders=total, known_orders=int(daily.orders.sum()), unknown_merchant_orders=unknown,
        source_reconciled=True, daily_monthly_reconciled=True, previous=old))
    require(previous_inventory() == old, 'Prior artifacts changed')
    print(f'Prepared {total:,} source transactions; {unknown:,} unknown-merchant orders excluded from fees.', flush=True)


def validate_preparation(audit=None):
    path = OUT/'preparation.json'
    require(path.exists(), 'Run prepare first')
    saved = json.loads(path.read_text())
    if audit is not None:
        require(saved['audit'] == audit, 'Source audit changed')
    for name, key in [('identities','identities'),('network_activity','network')]:
        require(fingerprint(OUT/(name+'.parquet')) == saved[key], 'Customer input checksum changed')
    for p in saved['partitions']:
        require(fingerprint(OUT/'customer_day'/(p['month']+'.parquet')) == p['sha256'], 'Customer partition changed')
    require(previous_inventory() == saved['previous'], 'Previous artifacts changed')
    return saved


@dataclass
class Exposure:
    customers: np.ndarray
    merchants: np.ndarray
    fees: sparse.csc_matrix

    def subset(self, ids):
        p = pd.Index(self.merchants).get_indexer(ids)
        require((p >= 0).all() and pd.Index(ids).is_unique, 'Unknown or repeated merchant')
        return Exposure(self.customers, np.asarray(ids), self.fees[:,p])


def read_exposure(start, end, ids, customers):
    """Sparse aggregation; only requested historical/quarter partitions enter the calculation."""
    ids, customers = np.asarray(ids), np.asarray(customers)
    rows, cols, vals = [], [], []
    for month in pd.period_range(pd.Timestamp(start), pd.Timestamp(end), freq='M'):
        path = OUT/'customer_day'/(str(month)+'.parquet')
        require(path.exists(), 'Missing source coverage')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=['day','merchant_abn','consumer_id','gross_fees']):
            f = batch.to_pandas()
            f = f.loc[f.day.between(start,end) & f.merchant_abn.isin(ids)]
            rows.append(pd.Index(customers).get_indexer(f.consumer_id))
            cols.append(pd.Index(ids).get_indexer(f.merchant_abn))
            vals.append(f.gross_fees.to_numpy())
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    require((r >= 0).all() and (c >= 0).all(), 'Customer exposure indexing failed')
    matrix = sparse.coo_matrix((v,(r,c)), shape=(len(customers),len(ids))).tocsc()
    matrix.sum_duplicates(); matrix.sort_indices()
    require(np.isfinite(matrix.data).all() and (matrix.data >= 0).all(), 'Invalid customer fees')
    return Exposure(customers, ids, matrix)


def expected_exposure(history, means):
    totals = np.asarray(history.fees.sum(axis=0)).ravel()
    means = np.asarray(means, dtype=float)
    require((totals > 0).all() and np.isfinite(means).all() and (means > 0).all(), 'Eligible merchant lacks exposure history')
    return Exposure(history.customers, history.merchants, (history.fees @ sparse.diags(means/totals)).tocsc())


def shares_metrics(fees):
    fees = np.asarray(fees, dtype=float).ravel()
    require(np.isfinite(fees).all() and (fees >= 0).all(), 'Invalid fee vector')
    total = float(fees.sum())
    if total <= 0:
        return dict(fees=0., hhi=np.nan, effective_customers=np.nan, active_customers=0,
                    **{f'top_{k}_share':np.nan for k in TOP_K})
    share = fees/total
    hhi = float(share@share)
    order = np.sort(share)[::-1]
    return dict(fees=total, hhi=hhi, effective_customers=1/hhi, active_customers=int((fees>0).sum()),
                **{f'top_{k}_share':float(order[:k].sum()) for k in TOP_K})


def concentration(exposure, ids=None):
    x = exposure.fees if ids is None else exposure.subset(ids).fees
    total = np.asarray(x.sum(axis=1)).ravel()
    result = shares_metrics(total)
    if result['fees'] > 0:
        individual = float(x.multiply(x).sum())/result['fees']**2
        degree = np.asarray((x>0).sum(axis=1)).ravel()
        result.update(individual_hhi=individual, shared_hhi=max(0.,result['hhi']-individual),
                      shared_customer_fee_share=float(total[degree>1].sum()/result['fees']))
    else:
        result.update(individual_hhi=np.nan, shared_hhi=np.nan, shared_customer_fee_share=np.nan)
    return result


def merchant_metrics(exposure):
    rows = []
    for i, abn in enumerate(exposure.merchants):
        rows.append(dict(merchant_abn=int(abn), **shares_metrics(exposure.fees.data[exposure.fees.indptr[i]:exposure.fees.indptr[i+1]])))
    return pd.DataFrame(rows)


def overlap_matrix(exposure, ids):
    x = exposure.subset(ids).fees
    totals = np.asarray(x.sum(axis=0)).ravel()
    require((totals>0).all(), 'Overlap undefined for inactive merchants')
    p = x @ sparse.diags(1/totals)
    return pd.DataFrame((p.T@p).toarray(), index=ids, columns=ids)


def shock_customers(exposure, k):
    totals = np.asarray(exposure.fees.sum(axis=1)).ravel()
    ix = np.flatnonzero(totals>0)
    ix = ix[np.lexsort((exposure.customers[ix], -totals[ix]))][:k]
    return exposure.customers[ix]


class FractionalHHI:
    """Convex quadratic in normalized exposures; fractional value only at binary selections."""
    def __init__(self, exposure, kind, count):
        require(kind in METHODS, 'Unknown concentration objective')
        self.kind = kind
        self.den = np.asarray(exposure.sum(axis=0)).ravel()
        self.upper_revenue = float(np.sort(self.den)[-count:].sum())
        require(self.upper_revenue > 0, 'No positive fee exposure')
        self.u = self.den/self.upper_revenue
        self.a = (exposure/self.upper_revenue).tocsc()
        self.diagonal = np.asarray(self.a.multiply(self.a).sum(axis=0)).ravel()
        # Exclusive-customer rows form a diagonal PSD component. Its perspective
        # is a stronger convex extension with identical values at binary z.
        exclusive=np.asarray((self.a>0).sum(axis=1)).ravel()==1
        only=self.a.multiply(exclusive[:,None])
        self.perspective_diagonal = self.diagonal if kind=='individual' else np.asarray(only.multiply(only).sum(axis=0)).ravel()
        self.cache = {}

    def value_gradient(self, v):
        if self.kind == 'individual':
            return float(self.diagonal@(v*v)), 2*self.diagonal*v
        total = self.a@v
        return float(total@total), 2*np.asarray(self.a.T@total).ravel()

    def value(self, z):
        denominator = float(self.u@z)
        return self.value_gradient(z/denominator)[0] if denominator > 0 else np.inf

    def binary_tangent(self,z):
        t=1/(self.u@z);v=t*z
        value,grad=self.value_gradient(v)
        d=self.perspective_diagonal
        return value,-d*t*t,grad-2*d*v+2*d*t

    def gram_row(self, i):
        if i not in self.cache:
            if self.kind == 'individual':
                row = np.zeros(len(self.u)); row[i] = self.diagonal[i]
            else:
                row = (self.a[:,i].T@self.a).toarray().ravel()
            self.cache[i] = row
        return self.cache[i]


def improve_exchanges(obj, initial, constraints, floors, max_steps):
    z = initial.copy()
    n,grad=obj.value_gradient(z)
    current_den=float(obj.u@z)
    for step in range(max_steps):
        inside, outside = np.flatnonzero(z), np.flatnonzero(1-z)
        if len(outside) == 0:
            break
        cross = np.vstack([obj.gram_row(i)[outside] for i in inside])
        numerator = n+grad[outside][None,:]-grad[inside][:,None]+obj.diagonal[inside,None]+obj.diagonal[outside][None,:]-2*cross
        den = current_den-obj.u[inside,None]+obj.u[outside][None,:]
        feasible = den>0
        for c, floor in zip(constraints,floors):
            feasible &= c@z-c[inside,None]+c[outside][None,:] >= floor-1e-13
        value = np.full(numerator.shape,np.inf)
        np.divide(np.maximum(numerator,0),den**2,out=value,where=feasible)
        i,j = np.unravel_index(np.argmin(value),value.shape)
        current = n/current_den**2
        if value[i,j] >= current-1e-12*current:
            break
        remove,add=inside[i],outside[j]
        z[remove],z[add] = 0.,1.
        grad+=2*(obj.gram_row(add)-obj.gram_row(remove))
        n=float(numerator[i,j]);current_den=float(den[i,j])
    np.testing.assert_allclose(obj.value(z),n/current_den**2,rtol=1e-9,atol=1e-14)
    return z


def continuous_presolve(obj,z,constraints,floors,zlo,zhi,scale,steps=100):
    """Frank-Wolfe LP dual bounds for the convex numerator/secant relaxation.

    No fractional portfolio is deployed. Every support plane globally bounds
    the binary numerator; the LP dual bound is also a genuine global bound.
    """
    x=z.copy();eta=obj.value(z);low=floors[-1];d=obj.perspective_diagonal
    matrix=sparse.diags(np.sqrt(obj.diagonal),format='csc') if obj.kind=='individual' else obj.a
    y=matrix@x;best_bound=0.;planes=[]
    vertices=[z.copy()];weights=[1.]
    for iteration in range(steps):
        value=float(y@y-d@(x*x)+d@x)
        gradient=2*np.asarray(matrix.T@y).ravel()-2*d*x+d
        g=(gradient-eta*(low+1)*obj.u)/scale
        result=linprog(g,A_ub=-np.vstack(constraints),b_ub=-np.asarray(floors),
            A_eq=np.ones((1,len(x))),b_eq=[z.sum()],bounds=list(zip(zlo,zhi)),method='highs',
            options=dict(dual_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9))
        require(result.success,'Continuous concentration relaxation failed')
        diff=(value-eta*((low+1)*(obj.u@x)-low))/scale
        dual=diff+float(result.fun-g@x)
        best_bound=max(best_bound,eta+min(0.,dual*scale)/low**2-1e-8*scale)
        if iteration%20==0 or iteration==steps-1:
            planes.append((gradient.copy()/scale,float((gradient@x-value)/scale)))
        if (eta-best_bound)/eta <= .0009: break
        # Pairwise away steps avoid the slow zig-zagging of plain Frank-Wolfe.
        scores=np.array([g@v if w>1e-14 else -np.inf for v,w in zip(vertices,weights)])
        away=int(np.argmax(scores))
        direction=result.x-vertices[away];ad=matrix@direction
        curvature=float(ad@ad-d@(direction*direction))/scale
        slope=float(g@direction)
        if -slope<=1e-7: break
        maximum=weights[away]
        gamma=float(np.clip(-slope/(2*curvature),0,maximum)) if curvature>1e-20 else maximum
        x+=gamma*direction;y+=gamma*ad
        weights[away]-=gamma
        match=next((j for j,v in enumerate(vertices) if np.array_equal(v,result.x)),None)
        if match is None: vertices.append(result.x.copy());weights.append(gamma)
        else: weights[match]+=gamma
    return max(0.,best_bound),planes


def optimize(exposure, means, kind, count=100, retention=.99, actual_floor=None, seeds=(), budgets=None):
    """Exact Charnes-Cooper normalization plus binary McCormick links and convex OA."""
    means = np.asarray(means,float)
    n = len(means)
    require(n>=count>0 and exposure.shape[1]==n and 0<retention<=1, 'Invalid optimization dimensions')
    require(np.isfinite(means).all() and (means>=0).all(), 'Invalid expected fees')
    baseline = np.zeros(n); baseline[np.lexsort((np.arange(n),-means))[:count]]=1
    base_mu = float(means@baseline)
    require(base_mu>0, 'No baseline expected fees')
    obj = FractionalHHI(exposure,kind,count)
    den_floor = retention*base_mu if actual_floor is None else actual_floor
    require(den_floor>0 and den_floor <= obj.upper_revenue+1e-7, 'Invalid denominator floor')
    lo, hi = 1., obj.upper_revenue/den_floor
    constraints = [means/base_mu, obj.u]
    floors = [retention,den_floor/obj.upper_revenue]
    # HiGHS feasibility tolerances are absolute. Express MILP revenue rows in
    # dollars so its default tolerance cannot silently permit a dollar shortfall
    # when a normalized floor near one represents millions of dollars.
    revenue_scales = [base_mu,obj.upper_revenue]
    milp_floors = np.asarray(floors)*revenue_scales
    # Exact cardinality/revenue propagation: fix only choices proved mandatory
    # or impossible by the best remaining count-1/count merchants, not a shortlist.
    zlo,zhi=np.zeros(n),np.ones(n)
    if n>count:
        for c,floor in zip(constraints,floors):
            order=np.lexsort((np.arange(n),-c));top=order[:count]
            maximum=float(c[top].sum());next_value=c[order[count]]
            forced=top[maximum-c[top]+next_value < floor-1e-10]
            zlo[forced]=1
            outsiders=order[count:]
            impossible=outsiders[maximum-c[top[-1]]+c[outsiders] < floor-1e-10]
            zhi[impossible]=0
    require((zlo<=zhi).all(),'Contradictory feasible-set bounds')
    def feasible(z):
        return (z.shape==(n,) and np.isin(z,[0.,1.]).all() and z.sum()==count
                and means@z>=retention*base_mu-1e-6 and obj.den@z>=den_floor-1e-6)
    require(feasible(baseline), 'Baseline must be feasible')
    seeds = [baseline]+[np.asarray(s,float) for s in seeds if feasible(np.asarray(s,float))]
    best = min(seeds,key=lambda z:(obj.value(z),tuple(np.flatnonzero(z)))).copy()
    scale = obj.value(baseline)
    require(np.isfinite(scale) and scale>0, 'Undefined baseline concentration')
    budgets = BUDGETS if budgets is None else budgets
    # Variables: z[n], v[n], t, theta. v=t*z, u'v=1; t in [1,U/L].
    eye = sparse.eye(n,format='csc'); zero = sparse.csc_matrix((n,n)); ones = sparse.csc_matrix(np.ones((n,1)))
    blank = sparse.csc_matrix((n,1))
    links = sparse.vstack([
        sparse.hstack([-lo*eye,eye,blank,blank]),
        sparse.hstack([-hi*eye,eye,blank,blank]),
        sparse.hstack([-hi*eye,eye,-ones,blank]),
        sparse.hstack([-lo*eye,eye,-ones,blank])],format='csc')
    lower = np.r_[np.zeros(n),np.full(n,-np.inf),np.full(n,-hi),np.full(n,-np.inf)]
    upper = np.r_[np.full(n,np.inf),np.zeros(n),np.full(n,np.inf),np.full(n,-lo)]
    fixed_rows = np.vstack([np.r_[np.ones(n),np.zeros(n+2)],
        *[np.r_[c*s,np.zeros(n+2)] for c,s in zip(constraints,revenue_scales)],np.r_[np.zeros(n),obj.u,0.,0.]])
    fixed = sparse.vstack([sparse.csc_matrix(fixed_rows),links],format='csc')
    fl = np.r_[count,milp_floors,1.,lower]; fu = np.r_[count,np.full(len(floors),np.inf),1.,upper]
    planes, rhs, quick_planes, quick_rhs, seen, logs = [],[],[],[],set(),[]
    def add(z):
        key=tuple(np.flatnonzero(z))
        if key in seen: return False
        seen.add(key)
        v=z/(obj.u@z); value,gz,gv=obj.binary_tangent(z)
        planes.append(np.r_[gz/scale,gv/scale,0.,-1.])
        rhs.append(float((gz@z+gv@v-value)/scale))
        value,g=obj.value_gradient(z)
        d=obj.perspective_diagonal
        g=g-2*d*z+d  # convex extension of the same binary numerator
        quick_planes.append(np.r_[g/scale,-1.])
        quick_rhs.append(float((g@z-value)/scale))
        return True
    lb=0.; certified=False; force_exact=False
    with threadpool_limits(limits=1):
        best=improve_exchanges(obj,best,constraints,floors,budgets[0]['max_swaps'])
        for z in seeds+[best]: add(z)
        ub=obj.value(best)/scale
        if n>100:
            bound,warm_planes=continuous_presolve(obj,best,constraints,floors,zlo,zhi,scale)
            lb=max(lb,bound/scale)
            for gradient,right in warm_planes:
                quick_planes.append(np.r_[gradient,-1.]);quick_rhs.append(right)
            require(lb<=ub+1e-6,'Invalid continuous lower bound')
            lb=min(lb,ub)
            logs.append(dict(tier=0,iteration=0,bound_method='continuous_presolve',status=0,nodes=0,
                lower_bound=lb*scale,objective_value=ub*scale,gap=(ub-lb)/max(ub,1e-30)))
            print(f'  {kind} continuous bound gap {100*(ub-lb)/ub:.5f}%',flush=True)
            if ub-lb <= budgets[0]['relative_gap']*ub or ub-lb<=1e-6:
                return best,dict(objective_value=obj.value(best),lower_bound=lb*scale,baseline_objective=scale,
                    gap=(ub-lb)/ub,certified=True,iterations=0,budget_tier=0,solver_status='certified_gap'),pd.DataFrame(logs)
        for tier,budget in enumerate(budgets,1):
            for iteration in range(budget['max_cuts']):
                started=time.perf_counter()
                # Certified presolve on the revenue interval. R^2 <= (L+U)R-LU.
                # The secant overestimates the denominator, so it gives a valid
                # lower bound on HHI. At these narrow fee floors it is very tight.
                # Reserve the final iteration of each budget for the exact linked
                # formulation when this cheaper relaxation has not certified it.
                quick = n>100 and iteration<budget['max_cuts']-1 and not force_exact
                if quick:
                    low=den_floor/obj.upper_revenue;eta=obj.value(best)
                    matrix=sparse.csc_matrix(np.vstack([np.r_[np.ones(n),0.],
                        *[np.r_[c*s,0.] for c,s in zip(constraints,revenue_scales)],*quick_planes]))
                    # Any improvement has N <= eta*R^2 <= eta*secant(R).
                    # Supply this known incumbent bound rather than making each
                    # new MILP rediscover a usable objective upper bound.
                    objective=np.r_[-eta*(low+1)*obj.u/scale,1.]
                    matrix=sparse.vstack([matrix,sparse.csc_matrix(objective[None,:])],format='csc')
                    result=milp(objective,integrality=np.r_[np.ones(n),0.],
                        bounds=Bounds(np.r_[zlo,0.],np.r_[zhi,np.inf]),
                        constraints=LinearConstraint(matrix,np.r_[count,milp_floors,np.full(len(quick_planes)+1,-np.inf)],
                            np.r_[count,np.full(len(floors),np.inf),quick_rhs,-eta*low/scale+1e-8]),
                        options=dict(node_limit=budget['node_limit'],mip_rel_gap=min(1e-4,budget['relative_gap']/10),presolve=True))
                else:
                    matrix=sparse.vstack([fixed,sparse.csc_matrix(np.vstack(planes))],format='csc')
                    result=milp(np.r_[np.zeros(2*n+1),1.],integrality=np.r_[np.ones(n),np.zeros(n+2)],
                        bounds=Bounds(np.r_[zlo,np.zeros(n),lo,0.],np.r_[zhi,np.full(n,hi),hi,ub+1e-8]),
                        constraints=LinearConstraint(matrix,np.r_[fl,np.full(len(planes),-np.inf)],np.r_[fu,rhs]),
                        options=dict(node_limit=budget['node_limit'],mip_rel_gap=min(1e-4,budget['relative_gap']/10),presolve=True))
                require(result.status in [0,1] or ('HiGHS Status 16:' in result.message and 'Solution limit reached' in result.message),
                        'Concentration MILP failed: '+result.message)
                milp_seconds=time.perf_counter()-started
                bound=getattr(result,'mip_dual_bound',None)
                if bound is not None and np.isfinite(bound):
                    if quick:
                        q_bound=float(bound)*scale+eta*low
                        certified_bound=eta+min(0.,q_bound)/low**2
                        lb=max(lb,certified_bound/scale)
                    else: lb=max(lb,float(bound))
                candidate=None
                if result.x is not None:
                    candidate=np.rint(result.x[:n])
                    require(np.max(np.abs(candidate-result.x[:n]))<1e-5 and feasible(candidate),
                        f'Invalid concentration incumbent: count={candidate.sum()}, '
                        f'integrality_error={np.max(np.abs(candidate-result.x[:n]))}, '
                        f'floor_margins={[float(c@candidate-f) for c,f in zip(constraints,floors)]}, '
                        f'status={result.status}')
                    polished=improve_exchanges(obj,candidate,constraints,floors,budget['max_swaps'])
                    score=obj.value(polished)/scale
                    if score<ub-1e-12 or (abs(score-ub)<=1e-12 and tuple(np.flatnonzero(polished))<tuple(np.flatnonzero(best))):
                        best,ub=polished,score
                require(lb<=ub+1e-6,'Invalid concentration global bound')
                lb=min(lb,ub)
                certified=ub-lb <= budget['relative_gap']*ub or ub-lb<=1e-6
                logs.append(dict(tier=tier,iteration=iteration,bound_method='revenue_secant' if quick else 'exact_normalization',status=result.status,nodes=getattr(result,'mip_node_count',None),
                    lower_bound=lb*scale,objective_value=ub*scale,gap=(ub-lb)/max(ub,1e-30)))
                if n>100 and (iteration%5==0 or certified):
                    print(f'  {kind} tier {tier} iteration {iteration}: gap {100*(ub-lb)/max(ub,1e-30):.5f}%; MILP {milp_seconds:.1f}s, polish {time.perf_counter()-started-milp_seconds:.1f}s',flush=True)
                if certified: break
                added=add(best)
                if candidate is not None: added=add(candidate) or added
                if not added:
                    if quick: force_exact=True
                    else: break
            if certified: break
    require(feasible(best),'Final concentration selection infeasible')
    return best,dict(objective_value=obj.value(best),lower_bound=lb*scale,baseline_objective=scale,
        gap=(ub-lb)/max(ub,1e-30),certified=certified,iterations=len(logs),budget_tier=tier,
        solver_status='certified_gap' if certified else 'bounded_best_found'),pd.DataFrame(logs)


def save_exposure(e, directory):
    coo=e.fees.tocoo()
    table(pd.DataFrame(dict(consumer_id=e.customers[coo.row],merchant_abn=e.merchants[coo.col],gross_fees=coo.data)),
          'exposure',directory,csv=False)


def load_exposure(directory, ids, customers):
    f=pd.read_parquet(directory/'exposure.parquet')
    r=pd.Index(customers).get_indexer(f.consumer_id); c=pd.Index(ids).get_indexer(f.merchant_abn)
    require((r>=0).all() and (c>=0).all(),'Cached exposure identities changed')
    return Exposure(customers,ids,sparse.coo_matrix((f.gross_fees,(r,c)),shape=(len(customers),len(ids))).tocsc())


def group_directory(as_of,window):
    return OUT/'histories'/(pd.Timestamp(as_of).strftime('%Y-%m-%d')+'_'+window)


def frame_match(frame,key,columns=KEY):
    mask=np.ones(len(frame),bool)
    for k in columns: mask &= frame[k].eq(key[k]).to_numpy()
    return frame.loc[mask].copy()


def save_checked(tables,stage,directory=None):
    directory=OUT if directory is None else directory
    hashes={name:digest(frame) for name,frame in tables.items()}
    marker=directory/(stage+'_checks.json')
    if marker.exists():
        require(json.loads(marker.read_text())['digests']==hashes, stage+' rerun changed results')
    for name,frame in tables.items(): table(frame,name,directory)
    save_json(marker,dict(digests=hashes,reproducible=True))


def select_date_worker(args):
    date,verify=args
    try:
        select_stage(verify_only=verify,only_date=date)
    except Exception as exc:
        print(f'Selection failed at {date}: {exc}',flush=True)
        raise


def select_stage(verify_only=False,only_date=None):
    daily,monthly,merchants,weekly,audit=load_inputs()
    saved=validate_preparation(audit)
    if (OUT/'selection_freeze.json').exists() and not verify_only:
        validate_freeze()
        print('Verified frozen selections; use verify-select for independent optimization rerun.',flush=True)
        return
    assembling=only_date is None
    if assembling:
        # Dates are independent. Six local processes keep memory bounded and
        # share no mutable selections; each owns distinct checkpoint directories.
        with ProcessPoolExecutor(max_workers=6) as pool:
            list(pool.map(select_date_worker,[(str(d.date()),verify_only) for d in cf.CUTOFFS]))
    customers=pd.read_parquet(OUT/'identities.parquet').consumer_id.to_numpy()
    existing=pd.read_parquet(cf.OUT/'merchant_features.parquet')
    all_e,all_s,all_l,all_f,shocks,availability=[],[],[],[],[],[]
    for as_of in (cf.CUTOFFS if assembling else pd.DatetimeIndex([only_date])):
        fresh,_=cf.features_at_cutoff(weekly,monthly,merchants,as_of)
        old=existing.loc[existing.as_of.eq(as_of)].sort_values('merchant_abn').reset_index(drop=True)
        pd.testing.assert_frame_equal(fresh.sort_values('merchant_abn').reset_index(drop=True),old,check_exact=True)
        f=fresh.loc[fresh.eligible].sort_values('merchant_abn').reset_index(drop=True)
        ids=f.merchant_abn.to_numpy(); means=f.expected_monthly_fees.to_numpy()
        base=cf.baseline_selection(f); base_ids=base.merchant_abn.to_numpy()
        baseline=np.isin(ids,base_ids).astype(float)
        reference=float(risk.history_matrix(weekly,cf.risk_window(as_of),base_ids).sum(axis=1).mean())
        for window in ['26w','52w']:
            dates=history_dates(weekly,as_of,window)
            availability.append(dict(as_of=as_of,window=window,available=dates is not None))
            if dates is None: continue
            directory=group_directory(as_of,window)
            checkpoint=directory/'selection_checks.json'
            if checkpoint.exists() and (not verify_only or assembling):
                for collection,name in [(all_e,'estimates'),(all_s,'selections'),(all_l,'solver_log'),(all_f,'merchant_features')]:
                    collection.append(pd.read_parquet(directory/(name+'.parquet')))
                if window=='26w': shocks.append(pd.read_parquet(directory/'common_shocks.parquet'))
                print(f'Reused frozen group {as_of.date()} {window}.',flush=True)
                continue
            history=read_exposure(dates[0]-pd.Timedelta(days=6),dates[-1],fresh.merchant_abn.to_numpy(),customers)
            # Independent historical reconciliation against the audited weekly panel.
            expected=weekly.loc[weekly.week_end.isin(dates)].groupby('merchant_abn').gross_fees.sum().reindex(history.merchants)
            np.testing.assert_allclose(np.asarray(history.fees.sum(axis=0)).ravel(),expected,rtol=1e-12,atol=1e-7)
            h=history.subset(ids); a=expected_exposure(h,means)
            if not verify_only: save_exposure(history,directory)
            features=fresh.merge(merchant_metrics(history),on='merchant_abn',validate='one_to_one',suffixes=('','_customer'))
            features['window']=window;features['customer_history_start']=dates[0]-pd.Timedelta(days=6);features['customer_history_end']=dates[-1]
            features['customer_history_weeks']=len(dates)
            if window=='26w':
                shock=pd.concat([pd.DataFrame(dict(consumer_id=shock_customers(a,k),k=k,as_of=as_of)) for k in SHOCK_K],ignore_index=True)
                if verify_only:
                    pd.testing.assert_frame_equal(shock,pd.read_parquet(directory/'common_shocks.parquet'),check_exact=True)
                else: table(shock,'common_shocks',directory)
                shocks.append(shock)
            estimates,selections,logs=[],[],[]
            seeds=[baseline]
            for method,r in [('revenue_only',1.)]+[(m,r) for r in RETENTIONS for m in METHODS]:
                if method=='revenue_only':
                    z=baseline.copy();info=dict(objective_value=np.nan,lower_bound=np.nan,baseline_objective=np.nan,
                        gap=np.nan,certified=False,iterations=0,budget_tier=0,solver_status='revenue_baseline')
                else:
                    z,info,log=optimize(a.fees,means,method,retention=r,seeds=seeds)
                    logs.append(log.assign(as_of=as_of,window=window,method=method,retention=r))
                    seeds.append(z)
                selected=f.loc[z.astype(bool)].sort_values(['expected_monthly_fees','merchant_abn'],ascending=[False,True]).copy()
                selected['as_of']=as_of;selected['window']=window;selected['method']=method;selected['retention']=r
                selected['rank_by_expected_fees']=np.arange(1,101)
                selected['in_revenue_baseline']=selected.merchant_abn.isin(base_ids)
                require(len(selected)==100 and selected.merchant_abn.is_unique and selected.eligible.all(),'Invalid portfolio')
                mu=float(means@z); require(mu>=r*means@baseline-1e-6,'Expected fee floor violated')
                es=concentration(a,selected.merchant_abn.to_numpy())
                hs=concentration(h,selected.merchant_abn.to_numpy())
                own=float(risk.history_matrix(weekly,cf.risk_window(as_of),selected.merchant_abn.to_numpy()).sum(axis=1).mean())
                estimates.append(dict(as_of=as_of,window=window,method=method,retention=r,expected_monthly_fees=mu,
                    baseline_expected_monthly_fees=float(means@baseline),expected_fee_retention_pct=100*mu/(means@baseline),
                    common_reference_weekly_fees=reference,own_reference_weekly_fees=own,
                    history_start=dates[0]-pd.Timedelta(days=6),history_end=dates[-1],history_weeks=len(dates),
                    overlap_with_baseline=int(selected.in_revenue_baseline.sum()),
                    **{'estimated_'+k:v for k,v in es.items()},**{'historical_'+k:v for k,v in hs.items()},**info))
                selections.append(selected)
                print(f'{as_of.date()} {window} {method} {r:g}: HHI {es["hhi"]:.8g}; gap {info["gap"]:.4g}',flush=True)
            e,s,l=pd.DataFrame(estimates),pd.concat(selections,ignore_index=True),pd.concat(logs,ignore_index=True)
            tables=dict(estimates=e,selections=s,solver_log=l,merchant_features=features)
            if verify_only:
                for name,x in tables.items(): pd.testing.assert_frame_equal(x,pd.read_parquet(directory/(name+'.parquet')),check_exact=True)
            else: save_checked(tables,'selection',directory)
            all_e.append(e);all_s.append(s);all_l.append(l);all_f.append(features)
    if not assembling: return
    combined=dict(unscored_estimates=pd.concat(all_e,ignore_index=True),selections=pd.concat(all_s,ignore_index=True),
        solver_log=pd.concat(all_l,ignore_index=True),merchant_features=pd.concat(all_f,ignore_index=True),
        common_shocks=pd.concat(shocks,ignore_index=True),availability=pd.DataFrame(availability))
    if verify_only:
        for name,x in combined.items(): pd.testing.assert_frame_equal(x,pd.read_parquet(OUT/(name+'.parquet')),check_exact=True)
        save_json(OUT/'selection_rerun.json',dict(matched=True,portfolios=len(combined['unscored_estimates'])))
        return
    save_checked(combined,'selection')
    paths=[OUT/(name+'.parquet') for name in combined]+[OUT/'protocol.json']+list((OUT/'histories').rglob('exposure.parquet'))
    save_json(OUT/'selection_freeze.json',dict(files={str(p.relative_to(OUT)):fingerprint(p) for p in paths},
        prepared_identities=saved['identities'],frozen_before_evaluation=not (OUT/'evaluation_checks.json').exists(),
        portfolios=len(combined['unscored_estimates'])))


def validate_freeze():
    marker=OUT/'selection_freeze.json'
    require(marker.exists(),'Evaluation requires frozen selections')
    record=json.loads(marker.read_text())
    require(record['frozen_before_evaluation'],'Selections were not frozen before evaluation')
    for name,h in record['files'].items(): require(fingerprint(OUT/name)==h,'Frozen artifact changed: '+name)
    return record


def customer_dynamics(past,future,network_active,customers):
    """Compare the supplied history window; 'new' does not mean first-ever acquisition."""
    before=past>0;after=future>0
    other=np.isin(customers,network_active)
    categories={'returning':before&after,'new_to_portfolio':~before&after,
                'former_active_elsewhere':before&~after&other,'former_network_inactive':before&~after&~other}
    rows=[]
    for label,mask in categories.items():
        rows.append(dict(customer_group=label,customers=int(mask.sum()),historical_fees=float(past[mask].sum()),
                         future_fees=float(future[mask].sum())))
    return rows


def splitmix_uniform(customers,replicate,seed=SEED):
    """Stateless common random numbers, independent of array order or other customers."""
    x=np.asarray(customers,dtype=np.uint64).copy()
    with np.errstate(over='ignore'):
        x ^= np.uint64(seed)+np.uint64(replicate+1)*np.uint64(0x9E3779B97F4A7C15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        x=x^(x>>np.uint64(31))
    return (x>>np.uint64(11)).astype(float)*(1./9007199254740992.)


def random_losses(customers,fees,draws=DRAWS,probabilities=PROBABILITIES):
    """Customers by portfolios; process each draw once for all portfolios."""
    fees=np.asarray(fees,float)
    if fees.ndim==1: fees=fees[:,None]
    losses=np.empty((len(probabilities),draws,fees.shape[1]))
    with threadpool_limits(limits=1):
        for rep in range(draws):
            u=splitmix_uniform(customers,rep)
            for j,q in enumerate(probabilities): losses[j,rep]=fees[u<q].sum(axis=0)
    return losses


def break_even(revenue_cost,avoided_full_loss):
    if avoided_full_loss>0:
        severity=revenue_cost/avoided_full_loss
        return severity, 'already_ahead' if severity<=0 else 'within_0_1' if severity<=1 else 'above_100_percent'
    return np.nan,'no_positive_loss_avoidance'


def paired_outcomes(estimates,outcomes):
    p=frontier.compare(estimates,outcomes)
    base=p.loc[p.method.eq('revenue_only'),['as_of','window','actual_hhi']].rename(columns={'actual_hhi':'baseline_actual_hhi'})
    p=p.merge(base,on=['as_of','window'],validate='many_to_one')
    p['hhi_reduction_pct']=100*(1-p.actual_hhi/p.baseline_actual_hhi)
    p['hhi_improves']=p.actual_hhi.lt(p.baseline_actual_hhi-1e-15)&p.status.eq('observed')
    return p


def summaries(paired,dates,windows,label):
    result=frontier.summarize(paired,dates,windows,label)
    for idx,row in result.iterrows():
        g=frame_match(paired,row,['window','method','retention'])
        g=g.loc[g.as_of.isin(dates)]
        joint=int((g.hhi_improves&g.actual_fee_guardrail_met.fillna(False)).sum())
        h=float(g.hhi_reduction_pct.mean())
        result.loc[idx,'mean_hhi_reduction_pct']=h
        result.loc[idx,'hhi_joint_successes']=joint
        result.loc[idx,'concentration_benefit']=bool(label=='primary_26w_eight' and h>=5 and
            row.mean_actual_fee_retention_pct>=100*row.retention-1e-8 and joint>=6)
    return result


def evaluate_stage():
    daily,monthly,merchants,weekly,audit=load_inputs()
    validate_preparation(audit);validate_freeze()
    customers=pd.read_parquet(OUT/'identities.parquet').consumer_id.to_numpy()
    network=pd.read_parquet(OUT/'network_activity.parquet')
    estimates,selections,shocks=[pd.read_parquet(OUT/(name+'.parquet')) for name in ['unscored_estimates','selections','common_shocks']]
    outcomes,weekly_rows,dynamics,stress,own_worst,random_rows,random_summaries,stability=[],[],[],[],[],[],[],[]
    for as_of in cf.CUTOFFS:
        ee=estimates.loc[estimates.as_of.eq(as_of)]
        if as_of==cf.CUTOFFS[-1]:
            outcomes.extend([dict(**{k:r[k] for k in KEY},status='unavailable',future_weeks=0) for r in ee.to_dict('records')])
            continue
        f=selections.loc[selections.as_of.eq(as_of)]
        meta=pd.read_parquet(cf.OUT/'merchant_features.parquet')
        eligible=meta.loc[meta.as_of.eq(as_of)&meta.eligible].sort_values('merchant_abn')
        ids=eligible.merchant_abn.to_numpy()
        future=read_exposure(as_of+pd.Timedelta(days=1),as_of+pd.offsets.MonthEnd(3),ids,customers)
        future_merchants=merchant_metrics(future).rename(columns={'hhi':'future_hhi'})
        period=network.loc[network.month.gt(as_of)&network.month.le(as_of+pd.offsets.MonthEnd(3))]
        active=period.consumer_id.unique()
        vectors=[];record_keys=[]
        for window,eg in ee.groupby('window',sort=True):
            all_ids=meta.loc[meta.as_of.eq(as_of)].merchant_abn.to_numpy()
            history=load_exposure(group_directory(as_of,window),all_ids,customers).subset(ids)
            past_m=merchant_metrics(history)[['merchant_abn','hhi']].merge(future_merchants[['merchant_abn','future_hhi']],on='merchant_abn')
            st=past_m.merge(eligible[['merchant_abn','expected_monthly_fees','category']],on='merchant_abn')
            st['as_of']=as_of;st['window']=window;stability.append(st)
            for row in eg.to_dict('records'):
                key={k:row[k] for k in KEY};chosen=frame_match(selections,key)
                chosen_ids=chosen.merchant_abn.to_numpy()
                future_fees=np.asarray(future.subset(chosen_ids).fees.sum(axis=1)).ravel()
                past_fees=np.asarray(history.subset(chosen_ids).fees.sum(axis=1)).ravel()
                o,ts=rs.portfolio_outcome(weekly,monthly,daily,merchants,row,chosen)
                o['retention']=row['retention']
                actual=concentration(future,chosen_ids)
                np.testing.assert_allclose(actual['fees'],o['actual_quarter_fees'],rtol=1e-12,atol=1e-7)
                o.update({'actual_'+k:v for k,v in actual.items() if k!='fees'})
                # Former top customers are selected using history only.
                top=np.lexsort((customers,-past_fees))[:100]
                o['historical_top100_still_active']=int((future_fees[top]>0).sum())
                o['historical_top100_future_fee_share']=float(future_fees[top].sum()/actual['fees']) if actual['fees']>0 else np.nan
                outcomes.append(o);weekly_rows.append(ts.assign(retention=row['retention']))
                dynamics.extend([dict(**key,**d) for d in customer_dynamics(past_fees,future_fees,active,customers)])
                for k in SHOCK_K:
                    shock_ids=shocks.loc[shocks.as_of.eq(as_of)&shocks.k.eq(k),'consumer_id'].to_numpy()
                    full_loss=float(future_fees[np.isin(customers,shock_ids)].sum())
                    for severity in SEVERITIES:
                        loss=severity*full_loss
                        stress.append(dict(**key,k=k,severity=severity,fees=actual['fees'],full_loss=full_loss,
                            loss=loss,loss_share=loss/actual['fees'] if actual['fees'] else np.nan,remaining_fees=actual['fees']-loss))
                for k in TOP_K:
                    top=np.lexsort((customers,-past_fees))[:k]
                    hindsight=np.sort(future_fees)[-k:]
                    own_worst.append(dict(**key,k=k,historical_own_top_future_loss=float(future_fees[top].sum()),
                        hindsight_own_top_loss=float(hindsight.sum()),label='different_customer_shocks_not_common_event'))
                vectors.append(future_fees);record_keys.append(key)
        fees=np.column_stack(vectors)
        losses=random_losses(customers,fees)
        for col,key in enumerate(record_keys):
            total=float(fees[:,col].sum());hhi=shares_metrics(fees[:,col])['hhi']
            for j,q in enumerate(PROBABILITIES):
                x=losses[j,:,col];theory_var=q*(1-q)*hhi*total**2 if total>0 else 0.
                # Exact central fourth moment for independent weighted Bernoulli sums.
                weights=fees[:,col];s2=float(weights@weights);s4=float(np.sum(weights**4))
                fourth=3*theory_var**2+q*(1-q)*(1-6*q*(1-q))*s4
                var_se=np.sqrt(max(0.,(fourth-(DRAWS-3)/(DRAWS-1)*theory_var**2)/DRAWS))
                mean_se=np.sqrt(theory_var/DRAWS)
                mean_ok=abs(x.mean()-q*total)<=6*mean_se+1e-7
                var_ok=abs(x.var(ddof=1)-theory_var)<=6*var_se+1e-7
                require(mean_ok and var_ok,'Random-loss moments outside Monte Carlo tolerance')
                random_summaries.append(dict(**key,probability=q,mean_loss=float(x.mean()),sd_loss=float(x.std(ddof=1)),
                    p95_loss=float(np.quantile(x,.95)),worst5_mean_loss=float(np.sort(x)[-25:].mean()),
                    mean_remaining_fees=total-float(x.mean()),mean_loss_share=float(x.mean()/total) if total else np.nan,
                    sd_loss_share=float(x.std(ddof=1)/total) if total else np.nan,p95_loss_share=float(np.quantile(x,.95)/total) if total else np.nan,
                    worst5_mean_loss_share=float(np.sort(x)[-25:].mean()/total) if total else np.nan,theoretical_mean=q*total,
                    theoretical_sd=np.sqrt(theory_var),mean_mc_se=mean_se,variance_mc_se=var_se,moments_verified=True))
                random_rows.extend([dict(**key,probability=q,replicate=rep,loss=float(v),remaining_fees=total-float(v)) for rep,v in enumerate(x)])
        print(f'Evaluated observed outcomes and fixed shocks at {as_of.date()}.',flush=True)
    paired=paired_outcomes(estimates,pd.DataFrame(outcomes))
    summary=pd.concat([summaries(paired,ORIGINS,['26w'],'primary_26w_eight'),
        summaries(paired,frontier.COMMON,['26w','52w'],'sensitivity_common_five')],ignore_index=True)
    schedules=pd.concat([summaries(paired,ORIGINS[j::3],['26w'],f'schedule_{j}') for j in range(3)],ignore_index=True)
    s=pd.DataFrame(stress)
    base=s.loc[s.method.eq('revenue_only'),['as_of','window','k','severity','fees','loss','full_loss','remaining_fees']]
    s=s.merge(base.rename(columns={c:'baseline_'+c for c in ['fees','loss','full_loss','remaining_fees']}),on=['as_of','window','k','severity'],validate='many_to_one')
    s['loss_avoided']=s.baseline_loss-s.loss
    s['revenue_sacrificed']=s.baseline_fees-s.fees
    s['remaining_fee_gain']=s.remaining_fees-s.baseline_remaining_fees
    s['gain_baseline_pct']=100*s.remaining_fee_gain/s.baseline_fees
    s['actual_fee_retention_pct']=100*s.fees/s.baseline_fees
    be=[break_even(r.revenue_sacrificed,r.baseline_full_loss-r.full_loss) for r in s.itertuples()]
    s['break_even_severity']=[a for a,b in be];s['break_even_status']=[b for a,b in be]
    ss=stress_summary(s,ORIGINS,['26w'],'primary_26w_eight')
    sensitivity_stress=stress_summary(s,frontier.COMMON,['26w','52w'],'sensitivity_common_five')
    fragile=[]
    for row in summary.loc[summary.comparison.eq('primary_26w_eight')].to_dict('records'):
        g=frame_match(paired,row,['window','method','retention']).loc[lambda x:x.as_of.isin(ORIGINS)]
        best=g.loc[g.hhi_reduction_pct.idxmax(),'as_of']
        left=summaries(paired,ORIGINS[ORIGINS!=best],['26w'],'leave_best_out')
        v=frame_match(left,row,['window','method','retention']).iloc[0]
        passes=v.mean_hhi_reduction_pct>=5 and v.mean_actual_fee_retention_pct>=100*row['retention']-1e-8 and v.hhi_joint_successes>=np.ceil(.75*7)
        fragile.append(dict(method=row['method'],retention=row['retention'],window='26w',removed_origin=best,
            remaining_mean_hhi_reduction_pct=v.mean_hhi_reduction_pct,concentration_conclusion_reverses=bool(row['concentration_benefit'] and not passes)))
    changes=[]
    for key in estimates.loc[estimates.method.ne('revenue_only'),KEY].to_dict('records'):
        chosen=frame_match(selections,key);base=frame_match(selections,dict(key,method='revenue_only',retention=1.))
        changes.extend([chosen.loc[~chosen.merchant_abn.isin(base.merchant_abn)].assign(change='entrant'),
            base.loc[~base.merchant_abn.isin(chosen.merchant_abn)].assign(method=key['method'],retention=key['retention'],change='removed')])
    save_checked(dict(outcomes=pd.DataFrame(outcomes),frontier_metrics=paired,summary=summary,nonoverlapping_schedules=schedules,
        future_weekly_fees=pd.concat(weekly_rows,ignore_index=True),customer_dynamics=pd.DataFrame(dynamics),
        merchant_stability=pd.concat(stability,ignore_index=True),common_stress=s,stress_summary=pd.concat([ss,sensitivity_stress],ignore_index=True),
        own_customer_stress=pd.DataFrame(own_worst),random_stress_draws=pd.DataFrame(random_rows),random_stress_summary=pd.DataFrame(random_summaries),
        fragility=pd.DataFrame(fragile),changed_merchants=pd.concat(changes,ignore_index=True),
        category_composition=selections.groupby(KEY+['category'],as_index=False).agg(merchants=('merchant_abn','size'),expected_fees=('expected_monthly_fees','sum')),
        latest_unscored_candidates=selections.loc[selections.as_of.eq(cf.CUTOFFS[-1])].reset_index(drop=True)), 'evaluation')


def stress_summary(frame,dates,windows,label):
    rows=[]
    for keys,g in frame.loc[frame.as_of.isin(dates)&frame.window.isin(windows)&frame.method.ne('revenue_only')].groupby(['window','method','retention','k','severity']):
        window,method,r,k,severity=keys
        require(set(g.as_of)==set(dates),'Unmatched stress origins')
        joint=int(((g.remaining_fee_gain>1e-7)&g.actual_fee_retention_pct.ge(100*r-1e-8)).sum())
        rows.append(dict(comparison=label,window=window,method=method,retention=r,k=k,severity=severity,
            origins=len(g),mean_gain_baseline_pct=g.gain_baseline_pct.mean(),mean_remaining_fee_gain=g.remaining_fee_gain.mean(),
            mean_loss_avoided=g.loss_avoided.mean(),mean_revenue_sacrificed=g.revenue_sacrificed.mean(),
            mean_actual_fee_retention_pct=g.actual_fee_retention_pct.mean(),joint_successes=joint,
            scenario_protection=bool(label=='primary_26w_eight' and g.gain_baseline_pct.mean()>=.1 and
                g.actual_fee_retention_pct.mean()>=100*r-1e-8 and joint>=6)))
    return pd.DataFrame(rows)


def oracle_date_worker(args):
    date,verify=args
    try:
        oracle_stage(verify_only=verify,only_date=date)
    except Exception as exc:
        print(f'Hindsight search failed at {date}: {exc}',flush=True)
        raise


def oracle_stage(verify_only=False,only_date=None):
    validate_preparation();validate_freeze()
    require((OUT/'evaluation_checks.json').exists(),'Score frozen practical portfolios before hindsight')
    customers=pd.read_parquet(OUT/'identities.parquet').consumer_id.to_numpy()
    features=pd.read_parquet(cf.OUT/'merchant_features.parquet')
    selections=pd.read_parquet(OUT/'selections.parquet')
    metrics=pd.read_parquet(OUT/'frontier_metrics.parquet')
    directory=OUT/'hindsight_only';directory.mkdir(parents=True,exist_ok=True)
    assembling=only_date is None
    if assembling:
        with ProcessPoolExecutor(max_workers=6) as pool:
            list(pool.map(oracle_date_worker,[(str(d.date()),verify_only) for d in ORIGINS]))
    bounds,chosen_rows,logs=[],[],[]
    for as_of in (ORIGINS if assembling else pd.DatetimeIndex([only_date])):
        f=features.loc[features.as_of.eq(as_of)&features.eligible].sort_values('merchant_abn').reset_index(drop=True)
        ids=f.merchant_abn.to_numpy();means=f.expected_monthly_fees.to_numpy()
        base=cf.baseline_selection(f);zbase=np.isin(ids,base.merchant_abn).astype(float)
        future=read_exposure(as_of+pd.Timedelta(days=1),as_of+pd.offsets.MonthEnd(3),ids,customers)
        actual=np.asarray(future.fees.sum(axis=0)).ravel();base_actual=float(actual@zbase)
        base_hhi=concentration(future,base.merchant_abn.to_numpy())['hhi']
        seeds=[zbase]
        for _,g in selections.loc[selections.as_of.eq(as_of)].groupby(['window','method','retention']):
            seeds.append(np.isin(ids,g.merchant_abn).astype(float))
        for r in RETENTIONS:
            case=directory/(as_of.strftime('%Y-%m-%d')+'_'+str(r));case.mkdir(parents=True,exist_ok=True)
            if (case/'oracle_checks.json').exists() and (not verify_only or assembling):
                b,s,l=[pd.read_parquet(case/(n+'.parquet')) for n in ['bounds','selections','solver_log']]
                bounds.append(b);chosen_rows.append(s);logs.append(l)
                seeds.append(np.isin(ids,s.merchant_abn).astype(float));continue
            if base_actual<=0:
                b=pd.DataFrame([dict(as_of=as_of,retention=r,opportunity='undefined_zero_baseline_fees',certified=False,
                                    objective_value=np.nan,lower_bound=np.nan,hindsight_only=True)])
                s=f.iloc[:0].copy();l=pd.DataFrame()
            else:
                feasible_seeds=[z for z in seeds if means@z>=r*means@zbase-1e-7 and actual@z>=r*base_actual-1e-7]
                z,info,log=optimize(future.fees,means,'shared',retention=r,actual_floor=r*base_actual,seeds=feasible_seeds)
                realized=concentration(future,ids[z.astype(bool)])
                require(actual@z>=r*base_actual-1e-6,'Oracle actual income floor failed')
                np.testing.assert_allclose(realized['hhi'],info['objective_value'],rtol=1e-12)
                achieved=100*(1-info['objective_value']/base_hhi);possible=100*(1-info['lower_bound']/base_hhi)
                label='demonstrated' if achieved>=5-1e-8 else 'ruled_out_by_bound' if possible<5-1e-8 else 'unresolved'
                for seed in feasible_seeds:
                    require(info['objective_value']<=concentration(future,ids[seed.astype(bool)])['hhi']+1e-12,'Oracle worse than feasible practical seed')
                b=pd.DataFrame([dict(as_of=as_of,retention=r,hindsight_only=True,opportunity=label,
                    achieved_reduction_pct=achieved,possible_reduction_pct=possible,baseline_actual_hhi=base_hhi,
                    actual_fee_retention_pct=100*(actual@z)/base_actual,expected_fee_retention_pct=100*(means@z)/(means@zbase),**info)])
                s=f.loc[z.astype(bool)].assign(as_of=as_of,retention=r,hindsight_only=True).reset_index(drop=True)
                l=log.assign(as_of=as_of,retention=r,hindsight_only=True)
                seeds.append(z)
                print(f'Hindsight {as_of.date()} {r:g}: {achieved:.3f}% to {possible:.3f}% HHI reduction; {label}',flush=True)
            t=dict(bounds=b,selections=s,solver_log=l)
            if verify_only:
                for name,x in t.items(): pd.testing.assert_frame_equal(x,pd.read_parquet(case/(name+'.parquet')),check_exact=True)
            else: save_checked(t,'oracle',case)
            bounds.append(b);chosen_rows.append(s);logs.append(l)
    if not assembling: return
    tables=dict(oracle_bounds=pd.concat(bounds,ignore_index=True),oracle_selections=pd.concat(chosen_rows,ignore_index=True),
                oracle_solver_log=pd.concat(logs,ignore_index=True))
    if verify_only:
        for name,x in tables.items(): pd.testing.assert_frame_equal(x,pd.read_parquet(directory/(name+'.parquet')),check_exact=True)
        save_json(OUT/'oracle_rerun.json',dict(matched=True,cases=len(tables['oracle_bounds'])))
    else: save_checked(tables,'oracle',directory)


def extended_fragility(paired,stress):
    rows=[]
    for (method,r),g in paired.loc[paired.window.eq('26w')&paired.as_of.isin(ORIGINS)&paired.method.ne('revenue_only')].groupby(['method','retention']):
        def reduction(x):
            base=x.baseline_normalized_downside.mean()
            return 100*(1-x.normalized_downside.mean()/base) if base>0 else 0.
        # This screen is a ratio of means. The greatest individual difference
        # need not be the origin that contributes most to the aggregate ratio.
        idx=min(g.index,key=lambda i:(reduction(g.drop(index=i)),g.loc[i,'as_of']))
        rest=g.drop(index=idx)
        def passes(x):
            return reduction(x)>=5 and x.actual_fee_retention_pct.mean()>=100*r-1e-8 and int((x.downside_improves&x.actual_fee_guardrail_met).sum())>=np.ceil(.75*len(x))
        rows.append(dict(measure='downside',method=method,retention=r,k=np.nan,severity=np.nan,
            removed_origin=g.loc[idx,'as_of'],remaining_downside_reduction_pct=reduction(rest),
            conclusion_reverses=bool(passes(g) and not passes(rest))))
    for (method,r,k,severity),g in stress.loc[stress.window.eq('26w')&stress.as_of.isin(ORIGINS)&stress.method.ne('revenue_only')].groupby(['method','retention','k','severity']):
        idx=g.gain_baseline_pct.idxmax();rest=g.drop(index=idx)
        def passes(x):
            joint=((x.remaining_fee_gain>1e-7)&x.actual_fee_retention_pct.ge(100*r-1e-8)).sum()
            return x.gain_baseline_pct.mean()>=.1 and x.actual_fee_retention_pct.mean()>=100*r-1e-8 and joint>=np.ceil(.75*len(x))
        rows.append(dict(measure='common_shock',method=method,retention=r,k=k,severity=severity,
            removed_origin=g.loc[idx,'as_of'],conclusion_reverses=bool(passes(g) and not passes(rest))))
    return pd.DataFrame(rows)


def make_plots(estimates,paired,summary,stress,oracle):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotdir=OUT/'plots';plotdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    hist=estimates.loc[estimates.as_of.isin(ORIGINS)&estimates.window.eq('26w')].copy()
    base=hist.loc[hist.method.eq('revenue_only'),['as_of','estimated_hhi']].rename(columns={'estimated_hhi':'baseline_hhi'})
    hist=hist.merge(base,on='as_of');hist['hhi_reduction_pct']=100*(1-hist.estimated_hhi/hist.baseline_hhi)
    primary=paired.loc[paired.as_of.isin(ORIGINS)&paired.window.eq('26w')]
    for method,label in [('individual','Individual concentration'),('shared','Including shared customers')]:
        h=hist.loc[hist.method.eq(method)].groupby('retention')[['expected_fee_retention_pct','hhi_reduction_pct']].mean()
        g=primary.loc[primary.method.eq(method)].groupby('retention')[['actual_fee_retention_pct','hhi_reduction_pct']].mean()
        ax[0].plot(100-h.expected_fee_retention_pct,h.hhi_reduction_pct,'o-',label=label)
        ax[1].plot(100-g.actual_fee_retention_pct,g.hhi_reduction_pct,'o-',label=label)
    for a,title in zip(ax,['Estimated concentration frontier','Realized concentration frontier']):
        a.scatter([0],[0],marker='*',s=100,c='black',label='Revenue-only')
        a.axhline(5,color='grey',ls='--',lw=1);a.set(title=title,xlabel='Fees sacrificed (%)',ylabel='HHI reduction (%)');a.grid(alpha=.2)
    ax[0].legend(fontsize=9);fig.suptitle('Equal-origin means: eight 26-week tests');fig.savefig(plotdir/'concentration_frontiers.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for a,k in zip(axes,SHOCK_K):
        for method in METHODS:
            for r in [.99,.95]:
                g=stress.loc[stress.window.eq('26w')&stress.method.eq(method)&stress.retention.eq(r)&stress.k.eq(k)].groupby('severity').gain_baseline_pct.mean()
                a.plot(100*g.index,g.values,'o-',label=f'{method}, {100*r:g}% floor')
        a.axhline(0,c='black',lw=.8);a.axhline(.1,c='grey',ls='--',lw=.8)
        a.set(title=f'Common largest {k:,} customers',xlabel='Spending removed (%)',ylabel='Remaining-fee gain / baseline fees (%)');a.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Hypothetical shocks: loss avoided must cover the revenue sacrificed')
    fig.savefig(plotdir/'customer_stress_tradeoff.png',dpi=160);plt.close(fig)
    customers=pd.read_parquet(OUT/'identities.parquet').consumer_id.to_numpy()
    features=pd.read_parquet(cf.OUT/'merchant_features.parquet');as_of=cf.CUTOFFS[-1]
    f=features.loc[features.as_of.eq(as_of)].sort_values('merchant_abn')
    h=load_exposure(group_directory(as_of,'26w'),f.merchant_abn.to_numpy(),customers)
    eligible=f.loc[f.eligible];a=expected_exposure(h.subset(eligible.merchant_abn.to_numpy()),eligible.expected_monthly_fees.to_numpy())
    sels=pd.read_parquet(OUT/'selections.parquet')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    overlap_exports=[]
    for method,r in [('revenue_only',1.),('individual',.99),('shared',.99)]:
        ids=frame_match(sels,dict(as_of=as_of,window='26w',method=method,retention=r)).merchant_abn.to_numpy()
        fees=np.asarray(a.subset(ids).fees.sum(axis=1)).ravel();positive=np.sort(fees[fees>0])[::-1]
        fraction=np.arange(1,len(positive)+1)/len(positive)
        axes[0].plot(100*fraction,100*np.cumsum(positive)/positive.sum(),label=method)
        shared=overlap_matrix(a,ids)
        frame=shared.rename_axis('merchant_abn').reset_index();frame.columns=frame.columns.map(str)
        table(frame,'overlap_'+method,OUT,csv=False)
        if method=='shared':
            corr=shared.to_numpy()/np.sqrt(np.outer(np.diag(shared),np.diag(shared)))
            np.fill_diagonal(corr,np.nan)
            im=axes[1].imshow(corr,cmap='viridis',vmin=0)
    axes[0].set(xlabel='Customers, largest fee contributors first (%)',ylabel='Cumulative expected fees (%)',title='September customer concentration')
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].set(title='Shared-customer exposure cosine similarity\n99% floor; diagonal omitted',xlabel='Selected merchant',ylabel='Selected merchant')
    fig.colorbar(im,ax=axes[1],shrink=.8);fig.savefig(plotdir/'customer_exposure.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7),layout='constrained')
    for method in METHODS:
        g=primary.loc[primary.method.eq(method)].groupby('retention').hhi_reduction_pct.mean()
        ax.plot(100*g.index,g.values,'o-',label=method)
    if len(oracle):
        g=oracle.groupby('retention')[['achieved_reduction_pct','possible_reduction_pct']].mean()
        ax.plot(100*g.index,g.achieved_reduction_pct,'s--',label='Hindsight feasible improvement')
        ax.fill_between(100*g.index,g.achieved_reduction_pct,g.possible_reduction_pct,alpha=.2,label='Numerical opportunity bounds')
    ax.set(xlabel='Expected fee floor (%)',ylabel='Future HHI reduction (%)',title='Practical selections versus income-protected hindsight')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.savefig(plotdir/'concentration_opportunity.png',dpi=160);plt.close(fig)


def report_stage():
    saved=validate_preparation();freeze=validate_freeze()
    e,p,summary,stress,ss,random=[pd.read_parquet(OUT/(n+'.parquet')) for n in
        ['unscored_estimates','frontier_metrics','summary','common_stress','stress_summary','random_stress_summary']]
    oracle=pd.read_parquet(OUT/'hindsight_only/oracle_bounds.parquet')
    # Customer-map size is not the active customer population. Measure how much
    # of the eligible merchants' observed population the baseline already reaches.
    customers=pd.read_parquet(OUT/'identities.parquet').consumer_id.to_numpy()
    metadata=pd.read_parquet(cf.OUT/'merchant_features.parquet')
    population=[]
    for (date,window),g in e.groupby(['as_of','window']):
        f=metadata.loc[metadata.as_of.eq(date)].sort_values('merchant_abn')
        history=load_exposure(group_directory(date,window),f.merchant_abn.to_numpy(),customers)
        eligible=history.subset(f.loc[f.eligible,'merchant_abn'].to_numpy())
        active=int(np.count_nonzero(np.asarray(eligible.fees.sum(axis=1)).ravel()))
        base=g.loc[g.method.eq('revenue_only')].iloc[0]
        reached=int(base.estimated_active_customers)
        require(reached<=active,'Baseline customer count exceeds eligible population')
        population.append(dict(as_of=date,window=window,identity_map_customers=len(customers),
            eligible_active_customers=active,baseline_active_customers=reached,
            baseline_customer_coverage_pct=100*reached/active if active else np.nan,
            eligible_customers_outside_baseline=active-reached))
    population=pd.DataFrame(population);table(population,'historical_customer_population')
    extra=extended_fragility(p,stress);table(extra,'downside_stress_fragility')
    nested=[]
    for (date,window,method),g in e.loc[e.method.ne('revenue_only')].groupby(['as_of','window','method']):
        g=g.sort_values('retention')
        for i,left in enumerate(g.to_dict('records')):
            for right in g.iloc[i+1:].to_dict('records'):
                require(left['lower_bound']<=right['objective_value']+1e-10,'Inconsistent nested HHI bounds')
        nested.append(dict(as_of=date,window=window,method=method,consistent=True))
    for date,g in oracle.groupby('as_of'):
        g=g.sort_values('retention')
        for i,left in enumerate(g.to_dict('records')):
            for right in g.iloc[i+1:].to_dict('records'):
                require(left['lower_bound']<=right['objective_value']+1e-10,'Inconsistent oracle nested bounds')
    table(pd.DataFrame(nested),'nested_bounds')
    # Paired ablation: isolates shared-customer modelling, using the same cutoff and floor.
    a=p.loc[p.method.eq('shared')].merge(p.loc[p.method.eq('individual')],on=['as_of','window','retention'],suffixes=('_shared','_individual'))
    a['incremental_hhi_reduction_pct']=100*(1-a.actual_hhi_shared/a.actual_hhi_individual)
    a['incremental_fees']=a.actual_quarter_fees_shared-a.actual_quarter_fees_individual
    table(a[['as_of','window','retention','incremental_hhi_reduction_pct','incremental_fees']],'shared_customer_ablation')
    stability=pd.read_parquet(OUT/'merchant_stability.parquet')
    stability_rows=[]
    for (date,window),g in stability.groupby(['as_of','window']):
        valid=g[['hhi','future_hhi']].dropna()
        stability_rows.append(dict(as_of=date,window=window,merchants=len(valid),
            spearman_hhi_persistence=float(valid.hhi.corr(valid.future_hhi,method='spearman')),
            spearman_hhi_fee_level=float(g.hhi.corr(g.expected_monthly_fees,method='spearman'))))
    stability_summary=pd.DataFrame(stability_rows);table(stability_summary,'concentration_stability_summary')
    checks=dict(source_and_daily_reconciled=saved['source_reconciled'] and saved['daily_monthly_reconciled'],
        previous_outputs_unchanged=previous_inventory()==saved['previous'],frozen_before_outcomes=freeze['frozen_before_evaluation'],
        practical_portfolios=len(e),optimized_portfolios=int(e.method.ne('revenue_only').sum()),
        certified_practical=int(e.certified.sum()),oracle_cases=len(oracle),certified_oracles=int(oracle.certified.sum()),
        random_moments_verified=bool(random.moments_verified.all()),
        september_unscored=bool(p.loc[p.as_of.eq(cf.CUTOFFS[-1]),'status'].eq('unavailable').all()),
        common_references_identical=bool(e.groupby('as_of').common_reference_weekly_fees.nunique().eq(1).all()),
        exactly_100=bool(pd.read_parquet(OUT/'selections.parquet').groupby(KEY).merchant_abn.nunique().eq(100).all()),
        expected_floors=bool(e.expected_fee_retention_pct.ge(100*e.retention-1e-8).all()),
        oracle_actual_floors=bool(oracle.actual_fee_retention_pct.ge(100*oracle.retention-1e-8).all()),
        selection_rerun=(OUT/'selection_rerun.json').exists(),oracle_rerun=(OUT/'oracle_rerun.json').exists())
    save_json(OUT/'acceptance_checks.json',checks)
    primary=summary.loc[summary.comparison.eq('primary_26w_eight')]
    lines=['# Customer concentration and revenue protection','',
        'Exactly 100 eligible merchants; expected return remains historical mean monthly gross fees. '
        'Customer concentration is measured across the entire portfolio, including customers shared between merchants. '
        'All practical selections and common shock identities were frozen before future evaluation.','',
        '## Primary results: eight 26-week tests','',
        '| Method | Fee floor | Actual fees retained | Future HHI reduction | HHI + fee wins | Downside reduction | Concentration / downside screen |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in primary.itertuples():
        lines.append(f'| {row.method} | {100*row.retention:g}% | {row.mean_actual_fee_retention_pct:.3f}% | '
            f'{row.mean_hhi_reduction_pct:+.2f}% | {int(row.hhi_joint_successes)}/8 | '
            f'{row.normalized_downside_reduction_pct:+.2f}% | {row.concentration_benefit} / {row.investigate_downside} |')
    passed=primary.loc[primary.concentration_benefit.astype(bool)]
    scenario=ss.loc[ss.comparison.eq('primary_26w_eight')]
    lines+=['',f'{len(passed)} of {len(primary)} practical points pass the concentration screen. '
        f'{int(primary.investigate_downside.sum())} pass observed downside protection. '
        f'{int(scenario.scenario_protection.sum())} of {len(scenario)} predeclared common-shock points pass scenario-specific revenue protection.',
        'These are distinct findings; lower HHI alone is not evidence of greater realized profit or better protection against every shock.','',
        '## Baseline customer exposure and incremental overlap value','']
    latest=e.loc[e.as_of.eq(cf.CUTOFFS[-1])&e.window.eq('26w')&e.method.eq('revenue_only')].iloc[0]
    latest_population=population.loc[population.as_of.eq(cf.CUTOFFS[-1])&population.window.eq('26w')].iloc[0]
    lines += [f'The September revenue-only baseline has {int(latest.estimated_active_customers):,} historically active customers '
        f'and an estimated effective customer count of {latest.estimated_effective_customers:,.0f}. Its largest 100 customers '
        f'account for {100*latest.estimated_top_100_share:.3f}% of estimated fees; the largest 1,000 account for '
        f'{100*latest.estimated_top_1000_share:.3f}%. The shared-customer term contributes '
        f'{100*latest.estimated_shared_hhi/latest.estimated_hhi:.2f}% of total estimated HHI. These are exposure descriptions, not estimated loss probabilities.',
        '',f'The identity map contains {len(customers):,} customers, but only '
        f'{int(latest_population.eligible_active_customers):,} were active at eligible merchants in this history. '
        f'The baseline already reaches {latest_population.baseline_customer_coverage_pct:.2f}% of that population; '
        f'{int(latest_population.eligible_customers_outside_baseline):,} eligible-pool customers are outside it. '
        'Customer breadth and balanced customer fee shares are different: a portfolio can serve the same customers '
        'while changing its HHI by redistributing their fee shares. A large shared-customer contribution to HHI '
        'can coexist with low absolute concentration. The population table reports this comparison at every origin and history length.',
        '',f'Across primary origins, merchant concentration has mean historical-to-future Spearman rank correlation '
        f'{stability_summary.loc[stability_summary.window.eq("26w"),"spearman_hhi_persistence"].mean():.3f}. '
        'The historical and future windows have different lengths, so the comparison describes rank persistence, '
        'not a calibrated forecast of the HHI level. All portfolio comparisons use identical future periods.',
        '', '| Fee floor | Full versus individual: future HHI reduction | Full versus individual: actual fee difference |',
        '|---|---:|---:|']
    for r,g in a.loc[a.window.eq('26w')&a.as_of.isin(ORIGINS)].groupby('retention'):
        lines.append(f'| {100*r:g}% | {g.incremental_hhi_reduction_pct.mean():+.3f}% | A${g.incremental_fees.mean():+,.2f} |')
    lines+=['',
        '## Numerical opportunity and robustness','',
        f'{checks["certified_practical"]}/{checks["optimized_portfolios"]} practical searches and '
        f'{checks["certified_oracles"]}/{checks["oracle_cases"]} hindsight searches meet the fixed numerical gap criterion. '
        'Uncertified searches retain explicit feasible solutions and global lower bounds. Bounds describe optimization uncertainty, not statistical confidence.',
        '', '| Fee floor | Hindsight feasible HHI reduction | Upper bound on improvement | Demonstrated / ruled out / unresolved |',
        '|---|---:|---:|---|']
    for r,g in oracle.groupby('retention'):
        lines.append(f'| {100*r:g}% | {g.achieved_reduction_pct.mean():.2f}% | {g.possible_reduction_pct.mean():.2f}% | '
            f'{int(g.opportunity.eq("demonstrated").sum())} / {int(g.opportunity.eq("ruled_out_by_bound").sum())} / {int(g.opportunity.eq("unresolved").sum())} |')
    sensitivity=summary.loc[summary.comparison.eq('sensitivity_common_five')]
    lines+=['','The 26/52-week comparison uses only the same five February–June origins. All values and the three non-overlapping schedules are exported; '
        'the eight overlapping quarters are not eight independent economic replications.','',
        '| History | Method | Fee floor | Actual fees retained | Future HHI reduction |', '|---|---|---:|---:|---:|']
    for r in sensitivity.itertuples(): lines.append(f'| {r.window} | {r.method} | {100*r.retention:g}% | {r.mean_actual_fee_retention_pct:.3f}% | {r.mean_hhi_reduction_pct:+.2f}% |')
    lines+=['','## What the stress results mean','',
        'Common shocks remove 25%, 50% or 100% of the spending of the same historically identified largest 10, 100 or 1,000 customers. '
        'An improvement in remaining fees equals loss avoided minus unshocked revenue sacrificed. The common_stress table includes dollar changes '
        'and break-even severity; a break-even above 100% means even completely losing those customers would not cover the fee sacrifice.',
        '', 'Each portfolio’s own largest-customer removals are separate exposure diagnostics because those shocks affect different customers. '
        'The hindsight own-customer ranking is labelled explicitly and is not available to the selection algorithm.',
        '', 'Random customer-loss probabilities of 1%, 5% and 10% are scenarios, not calibrated churn estimates. '
        'Under independent equal-probability losses the expected lost fraction remains q, while its variance is q(1-q)HHI. '
        'Simulation moments pass a six-Monte-Carlo-standard-error check using analytical weighted-Bernoulli moments. '
        'The 500 draws do not create 500 historical observations.',
        '', 'Customer inactivity is identified only within this dataset. Activity elsewhere distinguishes movement outside the selected portfolio '
        'from no observed network activity. Neither establishes permanent churn or default, and prior exposure is not a measured financial loss.',
        '', 'The new_to_portfolio group means no fees in the specified 26- or 52-week historical exposure window, followed by fees in the future quarter. '
        'It can include customers returning after a gap and does not establish first-ever acquisition. Complete-week histories also omit the '
        'incomplete week immediately before a month-end cutoff; the exported history dates define this comparison precisely.',
        '', '## Interpretation and limits','',
        'The individual-concentration control ignores customer overlap; the shared method includes it. The ablation table reports their paired '
        'future HHI and fee differences. HHI is dimensionless: merely scaling every customer’s fees down cannot improve it. '
        'Effective customer count is the reciprocal of HHI, not a count of independent real-world customers.',
        '', 'Concentration, downside and each stress claim are checked again after removing their most favourable origin. '
        'These are sensitivity diagnostics, not significance tests. No retention level is selected after inspecting results and the revenue-only '
        'baseline is not automatically replaced.',
        '', 'Categories, take rates and identity mappings are assumed stable and historically known. Selection is assumed not to change sales. '
        'Customer-loss scenarios remove spending without replacement. Gross fees exclude principal losses, fraud, funding costs, operating costs and '
        'settlement timing. September is an unscored snapshot. The short synthetic history and already examined outcomes support retrospective '
        'project evidence only; no untouched holdout or causal claim remains.',
        '', '## Reproduction','',
        'Run `python code/customer_concentration.py prepare`, then `select`, `evaluate`, `oracle`, and `report`; `all` runs the sequence. '
        'Use `verify-select` and `verify-oracle` for independent deterministic optimization reruns. Run '
        '`python -m pytest -p no:capture code/tests/test_customer_concentration.py -q` and the full `code/tests` suite. '
        'The notebook reads frozen artifacts and regenerates the report/plots without reselecting portfolios.']
    (OUT/'findings.md').write_text('\n'.join(lines),encoding='utf-8')
    make_plots(e,p,summary,stress,oracle)
    print('Concentration findings and plots written.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',nargs='?',default='all',choices=['prepare','select','evaluate','oracle','report','all','verify-select','verify-oracle'])
    stage=parser.parse_args().stage
    if stage=='verify-select': select_stage(verify_only=True)
    elif stage=='verify-oracle': oracle_stage(verify_only=True)
    else:
        for name,fn in [('prepare',prepare_stage),('select',select_stage),('evaluate',evaluate_stage),('oracle',oracle_stage),('report',report_stage)]:
            if stage in ['all',name]: fn()


if __name__=='__main__': main()
