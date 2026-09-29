"""One dashboard table: merchant rows with clearly named repeated portfolio metrics."""
from pathlib import Path
import numpy as np
import pandas as pd

from .core import require
from .data import identifiers, file_hash, save_json

# name: (unit, dashboard aggregation, meaning)
SCHEMA = {
    'portfolio_id':('identifier','group key','Unique snapshot / size / weight / revenue-floor combination'),
    'as_of':('date','first per portfolio','Selection cutoff; not the export date'),
    'currency':('currency code','first per portfolio','Currency shared by all fee amounts in this portfolio'),
    'retention':('fraction','first per portfolio','Minimum expected fees / revenue-only baseline expected fees; .995 means 99.5%'),
    'merchant_abn':('identifier','dimension','Merchant identifier as text; numeric ABN breaks revenue ties'),
    'merchant_name':('text','dimension','Merchant name'),
    'merchant_category':('text','dimension','Category for display; not an extra optimization constraint'),
    'merchant_expected_monthly_gross_fees':('currency / month','sum within ONE portfolio','Historical mean monthly transaction value times take rate / 100'),
    'rank_by_expected_fees':('integer','sort ascending','Revenue ordering within the selected portfolio; not an individual risk rank'),
    'portfolio_expected_monthly_gross_fees':('currency / month','FIRST ONLY; never SUM','Total expected monthly gross fees; repeated on every selected merchant row'),
    'portfolio_variability_cv':('ratio','FIRST ONLY; never SUM','Shrunk weekly fee SD / historical weekly mean; not inherently bounded by 1'),
    'portfolio_customer_hhi':('0 to 1','FIRST ONLY; never SUM','HHI of expected fees allocated to customers; shared customers combined'),
    'portfolio_risk_score':('baseline = 100','FIRST ONLY; never SUM','100 * [weight * CV/base_CV + (1-weight) * HHI/base_HHI]; not a probability'),
    'portfolio_expected_fee_retention':('fraction','FIRST ONLY; never SUM','Achieved expected fee retention; may exceed the minimum retention'),
    'portfolio_size':('integer','FIRST ONLY; never SUM','Exactly this number of merchants is selected'),
    'variability_weight':('0 to 1','first per portfolio','Weight on baseline-normalized CV'),
    'concentration_weight':('0 to 1','first per portfolio','Weight on baseline-normalized HHI'),
    'window_start':('date','first per portfolio','First Monday in the risk history'),
    'window_end':('date','first per portfolio','Last Sunday in the risk history'),
    'portfolio_score_lower_bound':('score points','FIRST ONLY; never SUM','Genuine lower bound on the globally best feasible score'),
    'portfolio_optimality_gap':('fraction','FIRST ONLY; never SUM','(best score - lower bound) / best score; optimization uncertainty'),
    'portfolio_certified':('boolean','first per portfolio','Whether requested optimization tolerance was reached; not an exact-optimum or future-performance claim'),
    'portfolio_status':('text','first per portfolio','Termination/verification status, including uncertified searches'),
}


def portfolio_id(as_of, window, count, weight, retention):
    """Create readable keys without rounding different requested retention levels."""
    return f'{str(pd.Timestamp(as_of).date())}_{window}_n{int(count)}_w{repr(float(weight))}_r{repr(float(retention))}'


def merge_dashboard(metrics, selections):
    """Join many merchants to one portfolio, checking totals and deterministic ranks."""
    m, s = metrics.copy(), identifiers(selections,['merchant_abn'])
    keys = ['as_of','window','retention','variability_weight','concentration_weight']
    for frame in [m,s]:
        frame['as_of'] = pd.to_datetime(frame.as_of)
        if 'window' not in frame and 'n_weeks' in frame:
            frame['window'] = frame.n_weeks.astype(str)+'w'
    require(not m.duplicated(keys).any(), 'Duplicate portfolio metric key')
    require(not s.duplicated(keys+['merchant_abn']).any(), 'Duplicate selected merchant')
    require(np.allclose(m.variability_weight+m.concentration_weight,1), 'Weights must sum to one')
    require(m.variability_weight.between(0,1).all() and m.concentration_weight.between(0,1).all(),'Weights must be in [0,1]')
    require('currency' in m and m.currency.notna().all() and m.currency.astype(str).str.strip().ne('').all(),'Declare result currency')
    require(m.retention.gt(0).all() and m.retention.le(1).all(), 'retention is a fraction in (0,1]')
    m['portfolio_id'] = [portfolio_id(r.as_of,r.window,r.merchants,r.variability_weight,r.retention) for r in m.itertuples()]
    require(m.portfolio_id.is_unique, 'Non-unique portfolio identifiers')
    mapping = dict(expected_monthly_fees='portfolio_expected_monthly_gross_fees',
        estimated_cv='portfolio_variability_cv',estimated_hhi='portfolio_customer_hhi',
        score='portfolio_risk_score',merchants='portfolio_size',lower_bound='portfolio_score_lower_bound',
        gap='portfolio_optimality_gap',certified='portfolio_certified',status='portfolio_status')
    m['portfolio_expected_fee_retention'] = m.expected_fee_retention_pct/100
    s = s.rename(columns={'name':'merchant_name','category':'merchant_category',
                          'expected_monthly_fees':'merchant_expected_monthly_gross_fees'})
    cols = keys+['portfolio_id','portfolio_expected_fee_retention','currency']+list(mapping)
    joined = s.merge(m[cols].rename(columns=mapping),on=keys,how='left',validate='many_to_one',indicator=True)
    require(joined._merge.eq('both').all(), 'Selected portfolio has no metrics')
    require(set(joined.portfolio_id) == set(m.portfolio_id), 'A portfolio metric has no selected merchants')
    for _, group in joined.groupby('portfolio_id',sort=False):
        row = group.iloc[0]
        require(len(group) == row.portfolio_size, 'Portfolio count mismatch')
        np.testing.assert_allclose(group.merchant_expected_monthly_gross_fees.sum(),row.portfolio_expected_monthly_gross_fees,rtol=1e-10,atol=1e-7)
        require(row.portfolio_expected_fee_retention >= row.retention-1e-10, 'Expected-fee floor violated')
        ordered = group.sort_values(['merchant_expected_monthly_gross_fees','merchant_abn'],ascending=[False,True])
        require(np.array_equal(ordered.rank_by_expected_fees,np.arange(1,len(group)+1)), 'Incorrect revenue ranks or ties')
        require(group[['window_start','window_end']].nunique().eq(1).all(), 'Mixed risk windows')
    for col in ['portfolio_variability_cv','portfolio_customer_hhi','portfolio_risk_score','portfolio_score_lower_bound','portfolio_optimality_gap']:
        require(np.isfinite(joined[col]).all() and joined[col].ge(-1e-12).all(), 'Invalid dashboard metric: '+col)
    require(joined.portfolio_customer_hhi.le(1+1e-12).all(), 'HHI outside [0,1]')
    require((joined.portfolio_score_lower_bound <= joined.portfolio_risk_score+1e-7).all(), 'Invalid global lower bound')
    expected_gap=(joined.portfolio_risk_score-joined.portfolio_score_lower_bound)/np.maximum(joined.portfolio_risk_score,1e-13)
    np.testing.assert_allclose(expected_gap,joined.portfolio_optimality_gap,rtol=1e-7,atol=1e-10)
    joined = joined.sort_values(['as_of','retention','rank_by_expected_fees'],ascending=[True,False,True])
    joined['merchant_abn'] = joined.merchant_abn.astype(str)
    for col in ['as_of','window_start','window_end']:
        joined[col] = pd.to_datetime(joined[col]).dt.strftime('%Y-%m-%d')
    return joined[[c for c in SCHEMA if c in joined]].reset_index(drop=True)


def write_dashboard(metrics, selections, directory):
    """Write equivalent CSV/Parquet plus a field dictionary and content checksums."""
    merged = merge_dashboard(metrics,selections)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    merged.to_parquet(directory/'portfolio_dashboard.parquet',index=False,compression=None)
    merged.to_csv(directory/'portfolio_dashboard.csv',index=False)
    pd.DataFrame([dict(column=c,unit=SCHEMA[c][0],aggregation=SCHEMA[c][1],description=SCHEMA[c][2])
                  for c in merged]).to_csv(directory/'data_dictionary.csv',index=False)
    save_json(directory/'manifest.json',dict(schema_version=1,rows=len(merged),
        portfolios=int(merged.portfolio_id.nunique()),grain='one selected merchant per portfolio',
        primary_key=['portfolio_id','merchant_abn'],
        portfolio_aggregation='Filter portfolio_id, then FIRST or drop_duplicates; never sum repeated portfolio_* columns',
        merchant_risk_score_included=False,
        files={name:file_hash(directory/name) for name in ['portfolio_dashboard.parquet','portfolio_dashboard.csv','data_dictionary.csv']}))
    return merged
