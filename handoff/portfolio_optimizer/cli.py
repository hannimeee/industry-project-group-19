"""Prepare snapshots, optimize adjustable portfolios, and export dashboard data."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import core
from .data import (load_snapshot, read_table, save_json, file_hash, write_snapshot,
                   merchant_features_from_monthly, week_ends)
from .export import write_dashboard

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def record(snapshot, z, retention, weight, info):
    """Create interoperable portfolio metrics and selected merchant rows."""
    o, f, meta = snapshot.model, snapshot.merchants, snapshot.metadata
    c = o.components(z)
    chosen = f.loc[z.astype(bool)].sort_values(['expected_monthly_fees','merchant_abn'],ascending=[False,True]).copy()
    chosen['as_of'] = meta['as_of']
    chosen['window'] = str(meta['weeks'])+'w'
    chosen['window_start'], chosen['window_end'] = meta['window_start'], meta['window_end']
    chosen['retention'] = retention
    chosen['variability_weight'], chosen['concentration_weight'] = weight, 1-weight
    chosen['rank_by_expected_fees'] = np.arange(1,o.count+1)
    metrics = dict(as_of=meta['as_of'],window=str(meta['weeks'])+'w',retention=retention,
        currency=meta['currency'],
        variability_weight=weight,concentration_weight=1-weight,merchants=o.count,
        expected_monthly_fees=float(o.means@z),baseline_expected_monthly_fees=o.base_fees,
        expected_fee_retention_pct=100*float(o.u@z),estimated_cv=float(c[0]*o.base_cv),
        estimated_hhi=float(c[1]*o.base_hhi),baseline_cv=o.base_cv,baseline_hhi=o.base_hhi,
        shrinkage_alpha=snapshot.alpha,**info)
    return pd.DataFrame([metrics]), chosen


def verify_records(snapshot, metrics, selections):
    """Recompute every saved selection against current inputs; never trust old scores."""
    o, f, meta = snapshot.model, snapshot.merchants, snapshot.metadata
    ids = pd.to_numeric(selections.merchant_abn,errors='raise').astype('int64')
    core.require(ids.isin(f.merchant_abn).all(), 'Selected merchant is not eligible in this snapshot')
    core.require(pd.to_datetime(metrics.as_of).eq(pd.Timestamp(meta['as_of'])).all(), 'Wrong snapshot date in results')
    for r in metrics.itertuples():
        group = selections.loc[selections.retention.eq(r.retention)
            & selections.variability_weight.eq(r.variability_weight)
            & selections.concentration_weight.eq(r.concentration_weight)].copy()
        core.require(len(group)==o.count and group.merchant_abn.nunique()==o.count and r.merchants==o.count,
                     'Saved selection has wrong portfolio size')
        core.require(pd.to_datetime(group.as_of).eq(pd.Timestamp(meta['as_of'])).all(), 'Wrong selection date')
        core.require(str(r.window)==str(meta['weeks'])+'w', 'Wrong risk window')
        core.require(pd.to_datetime(group.window_start).eq(pd.Timestamp(meta['window_start'])).all()
                     and pd.to_datetime(group.window_end).eq(pd.Timestamp(meta['window_end'])).all(), 'Wrong selection window')
        z = f.merchant_abn.isin(pd.to_numeric(group.merchant_abn)).to_numpy(float)
        core.require(core.feasible(o,z,r.retention),'Infeasible saved portfolio')
        comp = o.components(z)
        np.testing.assert_allclose([r.expected_monthly_fees,r.estimated_cv,r.estimated_hhi,r.score,r.expected_fee_retention_pct],
            [o.means@z,comp[0]*o.base_cv,comp[1]*o.base_hhi,100*o.value(z,r.variability_weight),100*o.u@z],rtol=1e-10,atol=1e-8)
        names=group.merchant_abn.astype('int64').map(f.set_index('merchant_abn')['name'])
        core.require(np.array_equal(group['name'].astype(str),names.astype(str)),'Saved merchant name does not match snapshot')
        if hasattr(r,'currency'):
            core.require(r.currency==meta['currency'],'Saved currency does not match snapshot')
        expected = group.merchant_abn.astype('int64').map(f.set_index('merchant_abn').expected_monthly_fees)
        np.testing.assert_allclose(group.expected_monthly_fees,expected,rtol=1e-11,atol=1e-8)


def optimize_snapshot(snapshot, output, retentions, weight=.5, *, iterations=30, nodes=200,
                      time_limit=30., max_seconds=300., patience=8, gap=.001,
                      bound_iterations=20, seed_portfolios=None, resume=False):
    """Select serially over arbitrary percentage floors with fingerprinted checkpoints."""
    core.require(retentions and all(np.isfinite(r) and 0<r<=1 for r in retentions), 'Retention fractions must be in (0,1]')
    core.require(0<=weight<=1 and np.isfinite(weight), 'weight must be in [0,1]')
    core.require(iterations>=1 and nodes>=0 and bound_iterations>=0 and patience>=1
                 and np.isfinite([time_limit,max_seconds,gap]).all() and min(time_limit,max_seconds)>0 and gap>=0, 'Invalid solver settings')
    retentions = sorted(set(retentions),reverse=True)
    output = Path(output)
    settings = dict(retentions=retentions,weight=weight,count=snapshot.model.count,
        iterations=iterations,nodes=nodes,time_limit=time_limit,max_seconds=max_seconds,
        patience=patience,gap=gap,bound_iterations=bound_iterations)
    versions = {p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','pyarrow','scikit-learn']}
    protocol = dict(schema_version=1,inputs=snapshot.fingerprints,metadata=snapshot.metadata,
        settings=settings,versions=versions,
        source={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        seed_sha256=file_hash(seed_portfolios) if seed_portfolios else None)
    signature = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()):
        core.require(resume and (output/'protocol.json').exists(),'Use a new output directory, or --resume with identical inputs/settings')
        core.require(json.loads((output/'protocol.json').read_text()) == protocol,'Input, code or settings changed; use a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'protocol.json',protocol)
    o, f = snapshot.model, snapshot.merchants
    seeds = []
    if seed_portfolios:
        supplied = read_table(seed_portfolios)
        core.require(pd.to_datetime(supplied.as_of).le(pd.Timestamp(snapshot.metadata['as_of'])).all(),'Future-dated seed portfolio')
        for _, group in supplied.groupby(['as_of','retention']):
            z = f.merchant_abn.isin(pd.to_numeric(group.merchant_abn)).to_numpy(float)
            if z.sum()==o.count and len(group)==o.count: seeds.append(z)
    all_m, all_s = [], []
    for r in retentions:
        directory = output/('retention_'+repr(r))
        directory.mkdir(exist_ok=True)
        marker = directory/'complete.json'
        if marker.exists():
            check = json.loads(marker.read_text())
            core.require(check['signature']==signature,'Checkpoint signature changed')
            for name,digest in check['files'].items():
                core.require(file_hash(directory/name)==digest,'Completed checkpoint modified')
            metrics, selected = read_table(directory/'metrics.parquet'), read_table(directory/'selections.parquet')
            verify_records(snapshot,metrics,selected)
            z = f.merchant_abn.isin(selected.merchant_abn).to_numpy(float)
            print(f'Reused verified {100*r:g}% portfolio.',flush=True)
        else:
            warm, inherited_bound = seeds.copy(), 0.
            checkpoint = directory/'checkpoint.json'
            if checkpoint.exists():
                state = json.loads(checkpoint.read_text())
                core.require(state['signature']==signature,'Interrupted checkpoint changed')
                z = f.merchant_abn.isin(state['merchant_abns']).to_numpy(float)
                core.require(core.feasible(o,z,r),'Invalid interrupted selection')
                np.testing.assert_allclose(100*o.value(z,weight),state['progress']['score'],rtol=1e-10)
                warm.append(z); inherited_bound=state['progress']['lower_bound']/100
            ordered=np.sort(o.means)[::-1]
            unique = o.n==o.count or (r==1 and ordered[o.count-1]-ordered[o.count]>1e-6)
            if unique:
                z=o.baseline.copy()
                info=dict(score=100.,lower_bound=100.,gap=0.,certified=True,iterations=0,status='unique_maximum_revenue_portfolio')
                log=pd.DataFrame(columns=['score','lower_bound','gap'])
            else:
                print(f'Optimizing {100*r:g}%: {o.n} eligible, choose {o.count}.',flush=True)
                lower=[core.component_relaxation_bound(o,r,j,bound_iterations)
                       if (weight if j==0 else 1-weight)>0 else 0. for j in [0,1]]
                def progress(z, row):
                    save_json(checkpoint,dict(signature=signature,
                        merchant_abns=f.loc[z.astype(bool),'merchant_abn'].astype(int).tolist(),progress=row))
                    with (directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(row,default=lambda v:v.item())+'\n')
                z,info,log=core.optimize(o,weight,r,seeds=warm,lower_components=lower,
                    initial_lower_bound=inherited_bound,budgets=[dict(cuts=iterations,nodes=nodes)],
                    relative_gap=gap,time_limit=time_limit,max_seconds=max_seconds,patience=patience,progress=progress)
            metrics,selected=record(snapshot,z,r,weight,info)
            verify_records(snapshot,metrics,selected)
            for name,frame in [('metrics',metrics),('selections',selected),('solver_log',log)]:
                frame.to_parquet(directory/(name+'.parquet'),index=False,compression=None)
            save_json(marker,dict(signature=signature,files={name:file_hash(directory/name)
                for name in ['metrics.parquet','selections.parquet','solver_log.parquet']}))
            print(f'{100*r:g}%: score {info["score"]:.5f}, gap {100*info["gap"]:.3f}%, {info["status"]}',flush=True)
        seeds.append(z); all_m.append(metrics); all_s.append(selected)
    metrics,selected=pd.concat(all_m,ignore_index=True),pd.concat(all_s,ignore_index=True)
    metrics.to_parquet(output/'metrics.parquet',index=False,compression=None)
    selected.to_parquet(output/'selections.parquet',index=False,compression=None)
    write_dashboard(metrics,selected,output)
    return metrics,selected


def main(argv=None):
    """Portable command line; numerical work stays sequential and single-threaded."""
    parser=argparse.ArgumentParser(description='Merchant portfolios: expected gross fees versus 50/50 CV and customer HHI.')
    sub=parser.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export',help='Verify and merge existing portfolios, without rerunning optimization')
    e.add_argument('--input',type=Path,default=PACKAGE_ROOT/'input_snapshot')
    e.add_argument('--metrics',type=Path,default=PACKAGE_ROOT/'reference/metrics.parquet')
    e.add_argument('--selections',type=Path,default=PACKAGE_ROOT/'reference/selections.parquet')
    e.add_argument('--output',type=Path,default=PACKAGE_ROOT/'final')
    p=sub.add_parser('prepare',help='Validate your same-schema input tables and create a portable snapshot')
    for name in ['merchants','weekly-fees','customer-fees','metadata','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--monthly-sales',type=Path,help='Optional complete merchant-month sales panel; derive expected fees and eligibility from it')
    p.add_argument('--count',type=int,default=100)
    o=sub.add_parser('optimize',help='Choose portfolios from all eligible merchants at configurable revenue floors')
    o.add_argument('--input',type=Path,default=PACKAGE_ROOT/'input_snapshot')
    o.add_argument('--output',type=Path,required=True)
    o.add_argument('--retentions',nargs='+',type=float,default=[100,99.5,99,98,95],help='PERCENTAGES, e.g. 100 99.5 97.5 (not fractions)')
    o.add_argument('--count',type=int,default=100)
    o.add_argument('--weight',type=float,default=.5,help='CV weight in [0,1]; HHI weight is 1-weight')
    for name,typ,default in [('iterations',int,30),('nodes',int,200),('time-limit',float,30),('max-seconds',float,300),('patience',int,8),('gap',float,.001),('bound-iterations',int,20)]:
        o.add_argument('--'+name,type=typ,default=default)
    o.add_argument('--seed-portfolios',type=Path)
    o.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        if os.name=='nt':
            from .resource_limits import apply_worker_limits
            apply_worker_limits()
        with threadpool_limits(limits=1):
            if args.command=='prepare':
                meta=json.loads(args.metadata.read_text(encoding='utf-8'))
                f=read_table(args.merchants)
                if args.monthly_sales:
                    core.require(meta['expected_return_end']==meta['as_of'],'Monthly derivation requires expected_return_end == as_of (a complete month end)')
                    f=merchant_features_from_monthly(f,read_table(args.monthly_sales),meta['expected_return_start'],meta['as_of'])
                weekly=read_table(args.weekly_fees)
                weekly=weekly.loc[pd.to_datetime(weekly.week_end).isin(week_ends(meta['as_of'],meta['weeks']))]
                write_snapshot(args.output,f,weekly,read_table(args.customer_fees),meta,args.count)
            elif args.command=='export':
                metrics,selected=read_table(args.metrics),read_table(args.selections)
                core.require(metrics.merchants.nunique()==1,'Export one portfolio size at a time')
                snapshot=load_snapshot(args.input,int(metrics.merchants.iloc[0]))
                verify_records(snapshot,metrics,selected)
                metrics=metrics.assign(currency=snapshot.metadata['currency'])
                write_dashboard(metrics,selected,args.output)
            else:
                core.require(all(0<r<=100 for r in args.retentions),'--retentions uses percentages in (0,100]')
                snapshot=load_snapshot(args.input,args.count)
                optimize_snapshot(snapshot,args.output,[r/100 for r in args.retentions],args.weight,
                    iterations=args.iterations,nodes=args.nodes,time_limit=args.time_limit,max_seconds=args.max_seconds,
                    patience=args.patience,gap=args.gap,bound_iterations=args.bound_iterations,
                    seed_portfolios=args.seed_portfolios,resume=args.resume)
        print('Saved to '+str(args.output.resolve()),flush=True)
    except (ValueError,AssertionError,KeyError) as exc:
        parser.exit(2,'Input/validation error: '+str(exc)+'\n')
