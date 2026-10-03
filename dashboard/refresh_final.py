"""Build the offline dashboard bundle from the frozen final handoff; never optimise."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / 'MAST30034 Project 2 Repository/handoff/final'

SCOPES = [('global','Global · all eligible merchants','portfolio_dashboard.csv'),
 ('tech','Technology, Digital & Communications','tech_portfolio_dashboard.csv'),
 ('home','Home, Garden & General Supplies','home_portfolio_dashboard.csv'),
 ('fashion','Fashion, Jewellery & Personal Care','fashion_portfolio_dashboard.csv'),
 ('arts','Arts, Crafts, Gifts & Entertainment','arts_portfolio_dashboard.csv'),
 ('auto','Automotive, Outdoor & Recreational Equipment','auto_portfolio_dashboard.csv')]

def load_scope(source, filename):
    rows = list(csv.DictReader((source / filename).open(newline='', encoding='utf-8-sig')))
    if len(rows) != 500:
        raise ValueError(f'{filename}: expected 500 rows')
    groups = defaultdict(list)
    keys = set()
    for row in rows:
        key = row['portfolio_id'], row['merchant_abn']
        if key in keys or not row['merchant_abn'].isdigit():
            raise ValueError('Duplicate key or invalid merchant ABN')
        keys.add(key)
        groups[row['portfolio_id']].append(row)
    if len(groups) != 5:
        raise ValueError('Source portfolio count does not match manifest')
    def number(row, key):
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite {key}')
        return value
    shared = ['as_of', 'currency', 'variability_weight', 'concentration_weight', 'window_start', 'window_end', 'portfolio_size']
    if any(len({r[k] for r in rows}) != 1 for k in shared):
        raise ValueError('Multiple snapshots, sizes or weights require a separate dashboard build')
    portfolios = []
    for ident, group in groups.items():
        first = group[0]
        repeated = [k for k in first if not k.startswith('merchant_') and k != 'rank_by_expected_fees']
        if any(any(row[k] != first[k] for k in repeated) for row in group):
            raise ValueError(f'Inconsistent repeated metrics in {ident}')
        size = int(first['portfolio_size'])
        if len(group) != size or sorted(int(r['rank_by_expected_fees']) for r in group) != list(range(1, size + 1)):
            raise ValueError(f'Invalid count/rank in {ident}')
        merchants = [{'rank': int(r['rank_by_expected_fees']), 'abn': r['merchant_abn'], 'name': r['merchant_name'], 'category': r['merchant_category'], 'expected_monthly_fees': number(r, 'merchant_expected_monthly_gross_fees')} for r in sorted(group, key=lambda r: int(r['rank_by_expected_fees']))]
        total = number(first, 'portfolio_expected_monthly_gross_fees')
        if not math.isclose(sum(m['expected_monthly_fees'] for m in merchants), total, rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError('Merchant fee total does not reconcile')
        if any(m['expected_monthly_fees'] < 0 for m in merchants):
            raise ValueError('Negative merchant fee')
        if first['portfolio_certified'] not in ('True', 'False'):
            raise ValueError('Invalid certification flag')
        p = {'id': ident, 'retention': number(first, 'retention'), 'merchants': merchants,
             'metrics': {'expected_monthly_fees': total, 'cv': number(first, 'portfolio_variability_cv'), 'hhi': number(first, 'portfolio_customer_hhi'), 'risk_score': number(first, 'portfolio_risk_score'), 'achieved_retention': number(first, 'portfolio_expected_fee_retention')},
             'validation': {'certified': first['portfolio_certified'] == 'True', 'status': first['portfolio_status'], 'lower_bound': number(first, 'portfolio_score_lower_bound'), 'gap': number(first, 'portfolio_optimality_gap')}}
        if not 0 < p['retention'] <= 1 or not 0 <= p['metrics']['hhi'] <= 1 or p['metrics']['cv'] < 0:
            raise ValueError('Invalid risk metric or retention')
        portfolios.append(p)
    if len({p['retention'] for p in portfolios}) != len(portfolios):
        raise ValueError('Retention must identify one portfolio within this snapshot')
    baseline = next(p for p in portfolios if p['retention'] == 1)
    first = rows[0]
    weight = number(first, 'variability_weight')
    hhi_weight = number(first, 'concentration_weight')
    if not math.isclose(weight + hhi_weight, 1) or min(weight, hhi_weight) < 0:
        raise ValueError('Invalid weights')
    for p in portfolios:
        m = p['metrics']
        score = 100 * (weight * m['cv'] / baseline['metrics']['cv'] + hhi_weight * m['hhi'] / baseline['metrics']['hhi'])
        if not math.isclose(score, m['risk_score'], rel_tol=1e-9):
            raise ValueError('Combined risk score does not reconcile')
        achieved = m['expected_monthly_fees'] / baseline['metrics']['expected_monthly_fees']
        if not math.isclose(achieved, m['achieved_retention'], rel_tol=1e-9) or achieved + 1e-9 < p['retention']:
            raise ValueError('Retention does not reconcile')
        if not 0 <= p['validation']['lower_bound'] <= m['risk_score'] + 1e-8:
            raise ValueError('Invalid solver lower bound')
        if not math.isclose((m['risk_score']-p['validation']['lower_bound'])/m['risk_score'],p['validation']['gap'],abs_tol=1e-9):
            raise ValueError('Solver gap does not reconcile')
    return {'currency': first['currency'], 'as_of': first['as_of'], 'window_start': first['window_start'], 'window_end': first['window_end'], 'variability_weight': weight, 'concentration_weight': hhi_weight, 'portfolios': sorted(portfolios,key=lambda p:p['retention'])}

def build(source, output, merchants_path=None):
    import pyarrow.parquet as pq
    source=Path(source)
    manifest=json.loads((source/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source checksum mismatch: {name}')
    if merchants_path is None:
        merchants_path=source.parent.parent/'tables/curated/merchants'
    merchants_path=Path(merchants_path)
    incomes={}
    attributes={}
    fields=['avg_consumer_district_income','avg_consumer_fraud_prob','total_transactions','avg_value_per_transaction','expected_revenue']
    for row in pq.read_table(merchants_path,columns=['merchant_abn']+fields).to_pylist():
        abn=str(row['merchant_abn']);value=row['avg_consumer_district_income']
        if not abn.isdigit() or abn in incomes: raise ValueError('Invalid or duplicate curated merchant ABN')
        if value is not None and (not math.isfinite(value) or value<0): raise ValueError('Invalid income value')
        incomes[abn]=value
        attributes[abn]={k:row[k] for k in fields}
        for k,v in attributes[abn].items():
            if v is not None and (not math.isfinite(v) or v<0):raise ValueError(f'Invalid merchant attribute: {k}')
        if row['avg_consumer_fraud_prob'] is not None and row['avg_consumer_fraud_prob']>100:raise ValueError('Fraud score must be in percentage points')
    result=None;portfolios=[];hashes={};missing=set()
    for scope,name,filename in SCOPES:
        parsed=load_scope(source,filename)
        if result is None:result={k:v for k,v in parsed.items() if k!='portfolios'}
        elif any(parsed[k]!=result[k] for k in result):raise ValueError('Incompatible snapshot or weight metadata')
        for p in parsed['portfolios']:
            p['source_id']=p['id'];p['id']=scope+'::'+p['id'];p['scope']=scope
            for m in p['merchants']:
                if m['abn'] not in incomes:missing.add(m['abn'])
                m.update(attributes.get(m['abn'],{k:None for k in fields}))
            portfolios.append(p)
        hashes[filename]=hashlib.sha256((source/filename).read_bytes()).hexdigest()
    if missing:raise ValueError(f'{len(missing)} selected ABNs absent from curated merchants')
    category_segments={}
    for p in portfolios:
        if p['scope']=='global':continue
        for m in p['merchants']:
            previous=category_segments.setdefault(m['category'],p['scope'])
            if previous!=p['scope']:raise ValueError('Category assigned to multiple segments')
    for p in portfolios:
        for m in p['merchants']:
            if m['category'] not in category_segments:raise ValueError('Selected category has no segment mapping')
            m['segment']=category_segments[m['category']]
    correlations={}
    for kind in ['revenue','customers']:
        name='correlation_'+kind+'.csv';rows=list(csv.DictReader((source/name).open()))
        categories=list(rows[0])[1:]
        if [r['segment'] for r in rows]!=categories or set(categories)!={s[1] for s in SCOPES[1:]}:raise ValueError('Matrix labels do not match segments')
        values=[[float(r[c]) for c in categories] for r in rows]
        for i,row in enumerate(values):
            for j,v in enumerate(row):
                if not math.isfinite(v) or not (-1 if kind=='revenue' else 0)<=v<=1 or not math.isclose(v,values[j][i],abs_tol=1e-8):raise ValueError('Invalid similarity matrix')
        diagnostics_name='correlation_'+kind+'_pair_diagnostics.csv'
        diagnostics=list(csv.DictReader((source/diagnostics_name).open()))
        for d in diagnostics:
            for k in ['candidate_pairs','valid_pairs','excluded_pairs']:d[k]=int(d[k])
            if d['valid_pairs']+d['excluded_pairs']!=d['candidate_pairs']:raise ValueError('Invalid pair diagnostics')
        correlations[kind]={'categories':categories,'values':values,'diagnostics':diagnostics}
        for f in [name,diagnostics_name]:hashes[f]=hashlib.sha256((source/f).read_bytes()).hexdigest()
    files=sorted(merchants_path.rglob('*.parquet')) if merchants_path.is_dir() else [merchants_path]
    result.update(results_date="2022-10-03",demo=False,scopes=[{'id':s,'name':n} for s,n,_ in SCOPES],portfolios=portfolios,correlations=correlations,source={'files':hashes,'curated_merchants_rows':len(incomes),'income_source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'null_income_rows':sum(m['avg_consumer_district_income'] is None for p in portfolios for m in p['merchants'])})
    output=Path(output);temporary=output.with_suffix('.tmp')
    temporary.write_text('window.PORTFOLIO_DATA = '+json.dumps(result,ensure_ascii=True,allow_nan=False,separators=(',',':'))+';\n')
    temporary.replace(output)
    return sum(len(p['merchants']) for p in portfolios),len(portfolios)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT/'dist/data.js')
    parser.add_argument('--merchants',type=Path,default=None,help='Curated merchants Parquet file or directory')
    args = parser.parse_args()
    try:
        rows, portfolios = build(args.source,args.output,args.merchants)
        print(f'Validated and exported {rows} merchant rows across {portfolios} portfolios to {args.output}')
    except (OSError, ValueError, KeyError, StopIteration, ImportError) as exc:
        parser.exit(1, f'Import failed; existing data unchanged: {exc}\n')
