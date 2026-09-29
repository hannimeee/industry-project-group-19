"""Repository-only adapter: package the frozen study for independent reuse.

The portable handoff itself never imports this file or any old experiment.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'handoff'))

import pandas as pd
from threadpoolctl import threadpool_limits
from portfolio_optimizer.data import file_hash,load_snapshot,write_snapshot,save_json
from portfolio_optimizer.cli import verify_records
from portfolio_optimizer.export import write_dashboard


def main():
    """Preserve source outputs and export the five verified, previously selected portfolios."""
    source=ROOT/'tables/analysis/joint_risk_portfolios_50_50'
    destination=ROOT/'handoff'
    freeze=json.loads((source/'selection_freeze.json').read_text())
    for name,digest in freeze['files'].items():
        assert file_hash(source/name)==digest, 'Frozen result changed: '+name
    protocol=json.loads((source/'protocol.json').read_text())
    for name,digest in protocol['inputs'].items():
        if isinstance(digest,str):
            assert file_hash(ROOT/name.replace('\\','/'))==digest, 'Source input changed: '+name
    features=pd.read_parquet(ROOT/'tables/analysis/cash_flow_variability/merchant_features.parquet')
    features=features.loc[features.as_of.eq(pd.Timestamp('2022-09-30'))].copy()
    weekly=pd.read_parquet(ROOT/'tables/analysis/cash_flow_variability/merchant_week.parquet')
    weekly=weekly.loc[weekly.week_end.between('2022-04-03','2022-09-25'),['merchant_abn','week_end','gross_fees']]
    customers=pd.read_parquet(ROOT/'tables/analysis/customer_concentration/histories/2022-09-30_26w/exposure.parquet')
    merchants=features[['merchant_abn','name','category','expected_monthly_fees','eligible','eligibility_reason']]
    meta=dict(as_of='2022-09-30',weeks=26,window_start='2022-03-28',window_end='2022-09-25',
        expected_return_start='2021-03-01',expected_return_end='2022-09-30',currency='AUD',
        complete_coverage=True,
        expected_return_definition='Mean monthly gross fees over the 19 complete months; sales * take_rate / 100',
        eligibility_definition='Valid unique metadata and take rate; positive sales in both latest and preceding three complete months',
        source_selection_freeze_sha256=file_hash(source/'selection_freeze.json'))
    snapshot_dir=destination/'input_snapshot'
    if (snapshot_dir/'snapshot.json').exists():
        snapshot=load_snapshot(snapshot_dir)
        assert all(snapshot.metadata.get(k)==v for k,v in meta.items()), 'Existing input bundle belongs to another source'
    else:
        write_snapshot(snapshot_dir,merchants,weekly,customers,meta)
        snapshot=load_snapshot(snapshot_dir)
    metrics=pd.read_parquet(source/'metrics.parquet')
    metrics['currency']=meta['currency']
    selections=pd.read_parquet(source/'selections.parquet')
    verify_records(snapshot,metrics,selections)
    reference=destination/'reference'
    reference.mkdir(exist_ok=True)
    for name in ['metrics.parquet','selections.parquet','execution_stop.json','execution_validation.json']:
        shutil.copyfile(source/name,reference/name)
    merged=write_dashboard(metrics,selections,destination/'final')
    assert len(merged)==500 and merged.portfolio_id.nunique()==5
    save_json(destination/'reference/provenance.json',dict(
        source='tables/analysis/joint_risk_portfolios_50_50',
        source_selection_freeze_sha256=file_hash(source/'selection_freeze.json'),
        score='100 * (0.5 * portfolio_CV / baseline_CV + 0.5 * portfolio_HHI / baseline_HHI)',
        normalization='Baseline-relative index, not a bounded 0-1 loss probability',
        independently_recomputed_against_portable_inputs=True,
        final_95_percent_search_stopped_early=True,
        note='Historical snapshot only. 99%, 98%, 95% remain uncertified at 0.1% tolerance. The 95% search stopped after 30+35 completed iterations; the maximum second-tier budget was not exhausted.',
        files={p.name:file_hash(p) for p in reference.iterdir() if p.is_file() and p.name!='provenance.json'}))
    print(f'Exported {len(merged)} rows / {merged.portfolio_id.nunique()} portfolios to {destination / "final"}',flush=True)


if __name__=='__main__':
    if sys.platform=='win32':
        from portfolio_optimizer.resource_limits import apply_worker_limits
        apply_worker_limits()
    with threadpool_limits(limits=1):
        main()
