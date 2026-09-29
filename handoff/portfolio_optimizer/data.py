"""Explicit, portable input contracts; no original project imports or paths."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.covariance import ledoit_wolf_shrinkage
from threadpoolctl import threadpool_limits

from .core import JointRisk, require


def file_hash(path):
    """Hash a file incrementally, including large exposure tables."""
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    """Replace one JSON checkpoint atomically, without deleting directories."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_table(path):
    """Read Parquet or CSV without interpreting identifiers as floating point."""
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, dtype={'merchant_abn':'string', 'consumer_id':'string'})
    raise ValueError('Use .parquet or .csv: '+str(path))


def identifiers(frame, columns):
    """Canonicalize integer identifiers and reject missing or lossy float IDs."""
    frame = frame.copy()
    for col in columns:
        require(col in frame, 'Missing column: '+col)
        s = frame[col]
        require(not s.isna().any() and not pd.api.types.is_float_dtype(s.dtype),
                'Missing or floating-point identifier: '+col)
        if not pd.api.types.is_integer_dtype(s.dtype):
            require(s.astype(str).str.fullmatch(r'[0-9]+').all(), 'Non-integer identifier: '+col)
        frame[col] = pd.to_numeric(s, errors='raise').astype('int64')
        require(frame[col].ge(0).all(), 'Negative identifier: '+col)
    return frame


def numeric(frame, columns):
    """Require finite, nonnegative amounts in consistent currency units."""
    for col in columns:
        require(col in frame, 'Missing column: '+col)
        frame[col] = pd.to_numeric(frame[col], errors='raise').astype(float)
        require(np.isfinite(frame[col]).all() and frame[col].ge(0).all(), 'Invalid amount: '+col)


def week_ends(as_of, weeks):
    """Latest complete Monday-Sunday weeks, including a Sunday decision date."""
    date = pd.Timestamp(as_of).normalize()
    require(date.tz is None and weeks >= 2, 'Use a local calendar date and at least two weeks')
    last = date-pd.Timedelta(days=(date.weekday()+1) % 7)
    return pd.date_range(end=last, periods=weeks, freq='W-SUN')


def merchant_features_from_monthly(merchants, monthly, history_start, as_of):
    """Derive mean monthly fees and the original two-positive-quarter eligibility."""
    merchants = identifiers(merchants, ['merchant_abn'])
    monthly = identifiers(monthly, ['merchant_abn'])
    require(merchants.merchant_abn.is_unique, 'Duplicate merchant metadata')
    require(not monthly.duplicated(['merchant_abn','month']).any(), 'Duplicate merchant-month')
    start, end = pd.Timestamp(history_start), pd.Timestamp(as_of)
    require(start.is_month_start and end.is_month_end, 'History starts on month start; as_of is month end')
    months = pd.date_range(start, end, freq='MS')
    require(len(months) >= 6, 'At least six complete months required')
    monthly['month'] = pd.to_datetime(monthly.month)
    require(monthly.month.dt.is_month_start.all(), 'month must be a month-start date')
    monthly = monthly.loc[monthly.month.isin(months)].copy()
    require(monthly.merchant_abn.isin(merchants.merchant_abn).all(), 'Unknown merchant in monthly panel')
    numeric(monthly, ['sales'])
    numeric(merchants, ['take_rate'])
    require(merchants.take_rate.between(0,100,inclusive='right').all(), 'take_rate must be percentage points in (0,100]')
    wide = monthly.pivot(index='month', columns='merchant_abn', values='sales').reindex(index=months, columns=merchants.merchant_abn)
    require(wide.notna().all().all(), 'Incomplete monthly coverage; supply explicit genuine zero months')
    merchants['expected_monthly_fees'] = wide.mean().to_numpy()*merchants.take_rate.to_numpy()/100
    merchants['eligible'] = ((wide.iloc[-3:].sum() > 0) & (wide.iloc[-6:-3].sum() > 0)).to_numpy()
    return merchants


@dataclass
class Snapshot:
    """Validated numeric model and its merchant/coverage provenance."""
    model: JointRisk
    merchants: pd.DataFrame
    weekly: np.ndarray
    metadata: dict
    alpha: float
    fingerprints: dict


def build_snapshot(merchants, weekly, customer, metadata, count=100):
    """Validate aligned histories and reconstruct exactly the study's CV and HHI."""
    meta = dict(metadata)
    require(meta.get('complete_coverage') is True, 'Complete source coverage must be explicitly declared')
    require(isinstance(meta['weeks'],int) and not isinstance(meta['weeks'],bool), 'weeks must be an integer')
    dates = week_ends(meta['as_of'], int(meta['weeks']))
    require(str((dates[0]-pd.Timedelta(days=6)).date()) == meta['window_start']
            and str(dates[-1].date()) == meta['window_end'], 'Customer exposure window must match the complete-week window')
    require(pd.Timestamp(meta['expected_return_end']) <= pd.Timestamp(meta['as_of'])
            and pd.Timestamp(meta['expected_return_start']) <= pd.Timestamp(meta['expected_return_end']), 'Invalid expected-return coverage')
    start,end=pd.Timestamp(meta['expected_return_start']),pd.Timestamp(meta['expected_return_end'])
    require(start.is_month_start and end.is_month_end and len(pd.date_range(start,end,freq='MS'))>=6,
            'Expected-return history requires at least six complete calendar months')
    require(bool(meta.get('currency')), 'Declare a common currency')
    f = identifiers(merchants, ['merchant_abn']).sort_values('merchant_abn').reset_index(drop=True)
    require(f.merchant_abn.is_unique and 'eligible' in f and 'name' in f, 'Unique merchant metadata, name and eligible are required')
    require(f.name.notna().all() and f.name.astype(str).str.strip().ne('').all(), 'Missing merchant name')
    mapping = {'true':True, 'false':False}
    if not pd.api.types.is_bool_dtype(f.eligible):
        f['eligible'] = f.eligible.astype(str).str.lower().map(mapping)
    require(f.eligible.notna().all(), 'eligible must be boolean true/false')
    numeric(f, ['expected_monthly_fees'])
    w = identifiers(weekly, ['merchant_abn'])
    c = identifiers(customer, ['merchant_abn','consumer_id'])
    require('week_end' in w, 'Missing week_end')
    w['week_end'] = pd.to_datetime(w.week_end)
    require(not w.duplicated(['merchant_abn','week_end']).any(), 'Duplicate merchant-week')
    require(not c.duplicated(['consumer_id','merchant_abn']).any(), 'Duplicate customer-merchant; aggregate first')
    require(w.week_end.isin(dates).all(), 'Weekly input must contain only the declared window')
    require(w.merchant_abn.isin(f.merchant_abn).all() and c.merchant_abn.isin(f.merchant_abn).all(), 'Unknown merchant in risk input')
    numeric(w, ['gross_fees']); numeric(c, ['gross_fees'])
    wide = w.pivot(index='week_end', columns='merchant_abn', values='gross_fees').reindex(index=dates, columns=f.merchant_abn)
    require(wide.notna().all().all(), 'Missing merchant-week; missing coverage is not zero')
    totals = c.groupby('merchant_abn').gross_fees.sum().reindex(f.merchant_abn, fill_value=0.)
    require(np.allclose(totals,wide.sum(),rtol=1e-10,atol=1e-7), 'Customer fees do not reconcile with weekly fees')
    f = f.loc[f.eligible].reset_index(drop=True)
    require(len(f) >= count > 0, 'Not enough eligible merchants for portfolio size')
    x = wide.reindex(columns=f.merchant_abn).to_numpy(float)
    require((f.expected_monthly_fees > 0).all() and (x.mean(axis=0) > 0).all(), 'Eligible merchants need positive expected and historical fees')
    centered = x-x.mean(axis=0)
    centered[:,np.ptp(x,axis=0)==0] = 0
    variances = np.sum(centered**2,axis=0)/(len(x)-1)
    nonconstant = variances > 0
    alpha = 0.
    if nonconstant.sum() > 1:
        with threadpool_limits(limits=1):
            alpha = float(ledoit_wolf_shrinkage(centered[:,nonconstant]/np.sqrt(variances[nonconstant]), assume_centered=True, block_size=256))
    c = c.loc[c.merchant_abn.isin(f.merchant_abn) & c.gross_fees.gt(0)]
    _, rows = np.unique(c.consumer_id.to_numpy(), return_inverse=True)
    cols = pd.Index(f.merchant_abn).get_indexer(c.merchant_abn)
    amounts = c.gross_fees.to_numpy()*f.expected_monthly_fees.to_numpy()[cols]/x.sum(axis=0)[cols]
    exposure = sparse.csc_matrix((amounts,(rows,cols)),shape=(rows.max()+1,len(f)))
    model = JointRisk(f.expected_monthly_fees.to_numpy(),x.mean(axis=0),
        np.sqrt(1-alpha)*centered/np.sqrt(len(x)-1),alpha*variances,exposure,count=count)
    return Snapshot(model,f,x,meta,alpha,{})


def load_snapshot(directory, count=100):
    """Load a portable bundle and verify its own checksums when provided."""
    directory = Path(directory)
    meta = json.loads((directory/'snapshot.json').read_text(encoding='utf-8'))
    paths = {name:directory/meta['files'][name] for name in ['merchants','weekly_fees','customer_fees']}
    require(all(p.resolve().is_relative_to(directory.resolve()) for p in paths.values()), 'Snapshot paths must stay inside its directory')
    hashes = {name:file_hash(p) for name,p in paths.items()}
    if 'sha256' in meta:
        require(hashes == meta['sha256'], 'Snapshot checksum mismatch; prepare a new bundle after changing data')
    result = build_snapshot(*(read_table(paths[n]) for n in paths),meta,count)
    result.fingerprints = {**hashes,'snapshot.json':file_hash(directory/'snapshot.json')}
    return result


def write_snapshot(directory, merchants, weekly, customer, metadata, count=100):
    """Validate first, then write a new self-contained uncompressed input bundle."""
    directory = Path(directory)
    require(not directory.exists() or not any(directory.iterdir()), 'Snapshot output must be empty')
    result = build_snapshot(merchants,weekly,customer,metadata,count)
    directory.mkdir(parents=True,exist_ok=True)
    frames = dict(merchants=merchants,weekly_fees=weekly,customer_fees=customer)
    for name, frame in frames.items():
        frame.to_parquet(directory/(name+'.parquet'),index=False,compression=None)
    meta = {**metadata,'schema_version':1,'files':{name:name+'.parquet' for name in frames},
            'sha256':{name:file_hash(directory/(name+'.parquet')) for name in frames}}
    save_json(directory/'snapshot.json',meta)
    return result
