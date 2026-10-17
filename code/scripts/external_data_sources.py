"""Audited merchant panels and historically dated external data for experiment 03."""

from __future__ import annotations

import io
import json
import re
import sqlite3
import zipfile
from pathlib import Path
from urllib.parse import urlencode, urljoin

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup

from abs_momentum_comparison import ROOT, START, END, cached_get, load_merchants, require, write_table

ORIGINS = pd.date_range("2021-08-31", "2022-09-30", freq="ME")
OUT = ROOT / "tables/analysis/external_data_comparison"
CACHE = ROOT / "tables/landing/external_data_comparison"
MAPPING = ROOT / "tables/reference/external_category_mapping.csv"
CPI_BASE = "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/"
CABEE_BASE = "https://www.abs.gov.au/statistics/economy/business-indicators/counts-australian-businesses-including-entries-and-exits/"
CPI_SERIES = {
    "A2326386V": "Footwear", "A3604403R": "Furniture and furnishings",
    "A2329311F": "Audio, visual and computing equipment",
    "A3604443J": "Audio, visual and computing media and services",
    "A2330346C": "Equipment for sports, camping and open-air recreation",
    "A3604408A": "Newspapers, books and stationery",
    "A2329266F": "Hairdressing and personal grooming services",
    "A3604393W": "Newspapers, magazines and stationery",
    "A2328726R": "Spare parts and accessories for motor vehicles",
    "A2329086W": "Therapeutic appliances and equipment",
    "A2330391R": "Games, toys and hobbies",
    "A2328546F": "Telecommunication equipment and services",
}
CABEE_CODES = {
    "4272": "Stationery Goods Retailing", "7000": "Computer System Design and Related Services",
    "4252": "Footwear Retailing", "4211": "Furniture Retailing", "4279": "Other Store-Based Retailing n.e.c.",
    "4222": "Computer and Computer Peripheral Retailing", "5622": "Cable and Other Subscription Broadcasting",
    "9499": "Other Repair and Maintenance n.e.c.", "4241": "Sport and Camping Equipment Retailing",
    "4244": "Newspaper and Book Retailing", "9511": "Hairdressing and Beauty Services",
    "4232": "Garden Supplies Retailing", "3921": "Motor Vehicle Parts Retailing",
    "8532": "Optometry and Optical Dispensing", "4243": "Toy and Game Retailing",
    "6639": "Other Goods and Equipment Rental and Hiring n.e.c.", "4273": "Antique and Used Goods Retailing",
    "5801": "Wired Telecommunications Network Operation", "5802": "Other Telecommunications Network Operation",
    "5809": "Other Telecommunications Services", "4253": "Watch and Jewellery Retailing",
}


def save_json(path, value):
    """Write readable experiment metadata without nonstandard NaN JSON tokens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def merchants_and_mapping(source_dir):
    """Reuse original metadata parsing while replacing only the enrichment lookup."""
    old, _ = load_merchants(source_dir, ROOT / "tables/reference/abs_category_mapping.csv")
    mapping = pd.read_csv(MAPPING, keep_default_na=False, dtype=str)
    require(len(mapping) == 25 and mapping.category.is_unique, "Expected 25 unique external mappings")
    require(set(mapping.category) == set(old.category), "External mapping differs from merchant categories")
    for name in ["cpi_proxy", "cabee_proxy", "trends_proxy"]:
        mapping[name] = mapping[name].eq("1")
    return old[["merchant_abn", "name", "category", "take_rate", "valid_take_rate"]].merge(
        mapping, on="category", validate="many_to_one"), mapping


def archive_signature(source_dir):
    """Check source size and modification time before reusing this experiment's own audit."""
    return [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(source_dir.glob("project-2-bnpl-tables-part*.zip"))]


def aggregate_development(source_dir, merchants, out=OUT):
    """Audit pre-October orders; never read October partition payloads in development."""
    stamp = archive_signature(source_dir)
    require(len(stamp) == 4, "Expected four source ZIP archives")
    marker = out / "merchant_audit.json"
    if marker.exists():
        audit = json.loads(marker.read_text(encoding="utf-8"))
        require(audit["archives"] == stamp, "Source archives changed; use a new output directory and rebuild")
        old = pd.read_parquet(out / "merchants.parquet")
        pd.testing.assert_frame_equal(old.reset_index(drop=True), merchants.reset_index(drop=True))
        return pd.read_parquet(out / "merchant_month.parquet")
    out.mkdir(parents=True, exist_ok=True)
    db = out / "development_order_ids.sqlite"
    require(not db.exists(), "Incomplete transaction audit exists; inspect it before restarting")
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA cache_size=-65536")
    con.execute("CREATE TABLE orders (id TEXT PRIMARY KEY) WITHOUT ROWID")
    pieces, daily, dates = [], [], set()
    try:
        for part in [2, 3, 4]:
            print(f"Auditing pre-October transactions, archive {part}/4", flush=True)
            with zipfile.ZipFile(source_dir / f"project-2-bnpl-tables-part{part}.zip") as z:
                for name in sorted(n for n in z.namelist() if n.endswith(".parquet")):
                    match = re.search(r"order_datetime=(\d{4}-\d{2}-\d{2})/", name)
                    require(match is not None, f"Missing partition date: {name}")
                    day = pd.Timestamp(match[1])
                    if day > END:
                        continue  # October remains unopened until confirmation.
                    dates.add(day)
                    df = pq.read_table(io.BytesIO(z.read(name)), columns=["order_id", "merchant_abn", "dollar_value"]).to_pandas()
                    require(df.notna().all().all(), "Null transaction key or amount")
                    require(df.order_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all(), "Invalid order ID")
                    require(np.isfinite(df.dollar_value).all() and df.dollar_value.gt(0).all(), "Invalid transaction value")
                    try:
                        con.executemany("INSERT INTO orders VALUES (?)", ((v,) for v in df.order_id))
                        con.commit()
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("Duplicate transaction identifier") from exc
                    daily.append(dict(day=day, source_partition=name, orders=len(df), sales=float(df.dollar_value.sum())))
                    if day < START:
                        continue
                    g = df.groupby("merchant_abn").dollar_value.agg(sales="sum", orders="size").reset_index()
                    g["month"] = day.to_period("M").to_timestamp()
                    g["sales_26"] = g.sales if day.day <= 26 else 0.0
                    g["orders_26"] = g.orders if day.day <= 26 else 0
                    pieces.append(g)
            print(f"Checked {sum(x['orders'] for x in daily):,} orders through {max(dates).date()}", flush=True)
        count = con.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    finally:
        con.close()
    require(set(pd.date_range(START, END)).issubset(dates), "Incomplete daily source coverage")
    daily = pd.DataFrame(daily)
    raw = pd.concat(pieces).groupby(["merchant_abn", "month"], as_index=False)[["sales", "orders", "sales_26", "orders_26"]].sum()
    covered = daily.loc[daily.day.between(START, END)]
    require(count == int(daily.orders.sum()), "ID reconciliation failed")
    require(int(raw.orders.sum()) == int(covered.orders.sum()), "Aggregation order reconciliation failed")
    require(np.isclose(raw.sales.sum(), covered.sales.sum(), rtol=1e-12), "Aggregation sales reconciliation failed")
    unknown = raw.loc[~raw.merchant_abn.isin(merchants.merchant_abn)]
    idx = pd.MultiIndex.from_product([merchants.merchant_abn, pd.date_range(START, END, freq="MS")], names=["merchant_abn", "month"])
    panel = raw.set_index(["merchant_abn", "month"]).reindex(idx, fill_value=0).reset_index()
    require(int(panel.orders.sum() + unknown.orders.sum()) == int(raw.orders.sum()), "Unknown merchant reconciliation failed")
    require(not panel.duplicated(["merchant_abn", "month"]).any(), "Duplicate merchant-month key")
    require(len(panel) == len(merchants) * 19, "Incorrect panel size")
    write_table(panel, out, "merchant_month")
    write_table(merchants, out, "merchants")
    write_table(unknown, out, "unknown_merchants_monthly")
    write_table(daily, out, "source_partition_audit")
    save_json(marker, dict(archives=stamp, unique_pre_october_orders=count, complete_month_orders=int(raw.orders.sum()),
                          complete_month_sales=float(raw.sales.sum()), unknown_orders=int(unknown.orders.sum()),
                          unknown_sales=float(unknown.sales.sum()), source_dates=len(dates), october_opened=False))
    return panel


def abs_release(html, page_url, filename):
    """Extract an ABS timestamp and an explicitly linked workbook; never infer its URL."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    m = re.search(r"Release date and time\s*(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2})(am|pm)\s+(AEST|AEDT)", text, re.I)
    require(m is not None, f"Unverified ABS publication: {page_url}")
    time = pd.to_datetime(m[1] + " " + m[2] + m[3], format="%d/%m/%Y %I:%M%p").tz_localize("Australia/Sydney")
    require(time.tzname() == m[4].upper(), "ABS publication timezone mismatch")
    links = {urljoin(page_url, a['href']) for a in soup.find_all('a', href=True)
             if re.search(r"/" + filename + r"\.xlsx?(?:\?|$)", a['href'])}
    require(len(links) == 1, f"Missing or ambiguous ABS workbook: {filename}")
    return dict(published_at=time, source_url=links.pop(), page_url=page_url, evidence=m[0])


def cpi_values(payload, extension, reference):
    """Validate CPI national index IDs and calculate within-vintage annual log changes."""
    book = pd.read_excel(io.BytesIO(payload), sheet_name="Data1", header=None,
                         engine="xlrd" if extension == ".xls" else "openpyxl")
    labels = {str(v).strip().lower(): i for i, v in enumerate(book.iloc[:, 0]) if isinstance(v, str)}
    require(all(x in labels for x in ["series id", "series type", "unit", "series end"]), "Missing CPI metadata")
    rows = []
    for sid, description in CPI_SERIES.items():
        cols = book.columns[book.iloc[labels['series id']].eq(sid)]
        record = dict(series_id=sid, description=description, reference_month=reference, value=np.nan, status="mapping_conflict")
        if len(cols) == 1:
            col = cols[0]
            header = [s.strip() for s in str(book.iat[0, col]).split(';') if s.strip()]
            if header == ["Index Numbers", description, "Australia"]:
                require(str(book.iat[labels['unit'], col]) == "Index Numbers", "Wrong CPI unit")
                require(str(book.iat[labels['data type'], col]) == "INDEX", "Wrong CPI data type")
                require(str(book.iat[labels['series type'], col]).lower() == "original", "Wrong CPI adjustment")
                require(pd.Timestamp(book.iat[labels['series end'], col]) == reference, "CPI vintage reference mismatch")
                values = book.iloc[labels['series id'] + 1:, [0, col]].copy()
                values.columns = ['date', 'index']
                values['date'] = pd.to_datetime(values.date, errors='coerce')
                series = values.dropna(subset=['date']).set_index('date')['index']
                require(series.index.is_unique, "Duplicate CPI reference periods")
                latest, previous = float(series.loc[reference]), float(series.loc[reference - pd.DateOffset(years=1)])
                require(np.isfinite([latest, previous]).all() and min(latest, previous) > 0, "Invalid CPI indexes")
                record.update(value=float(np.log(latest / previous)), latest_index=latest, prior_year_index=previous, status="available")
        rows.append(record)
    return pd.DataFrame(rows)


def cabee_values(payload, extension, year, expected_date):
    """Read all-employment national totals, preserving the workbook's detailed release date."""
    book = pd.ExcelFile(io.BytesIO(payload), engine="xlrd" if extension == '.xls' else 'openpyxl')
    sheet = f"June {year} a"
    candidates = [name for name in book.sheet_names if name.strip() == sheet]
    require(len(candidates) == 1, f"Missing annualised employment sheet {sheet}")
    df = book.parse(candidates[0], header=None)
    header = ' '.join(str(v) for v in df.head(5).to_numpy().ravel() if pd.notna(v))
    require(pd.Timestamp(expected_date).strftime('%d %B %Y').lstrip('0') in header, "Unverified detailed CABEE workbook date")
    require('Industry Class' in header and 'Annualised Employment' in header, "Wrong CABEE table")
    records = []
    for code, description in CABEE_CODES.items():
        rows = df.loc[df.iloc[:, 1].astype(str).str.fullmatch(code + r'(?:\.0)?')
                      & df.iloc[:, 2].astype(str).str.strip().eq('Total ' + description)]
        record = dict(industry_code=code, description=description, reference_month=pd.Timestamp(year, 6, 1),
                      start_count=np.nan, exits=np.nan, value=np.nan, status="mapping_conflict")
        if len(rows) == 1:
            row = rows.iloc[0]
            start, exits = float(row.iloc[7]), float(row.iloc[17])
            require(np.isfinite([start, exits]).all() and start > 0 and 0 <= exits <= start, "Invalid business exit counts")
            record.update(start_count=start, exits=exits, value=exits / start, status="available")
        records.append(record)
    return pd.DataFrame(records), header


def choose_asof(releases, origin):
    """Select only releases published before the end of the historical decision day."""
    end = pd.Timestamp(origin).tz_localize('Australia/Sydney') + pd.Timedelta(days=1)
    valid = releases.loc[releases.published_at.lt(end)].sort_values('published_at')
    require(len(valid) > 0, f"No verified vintage at {origin}")
    return valid.iloc[-1]


def acquire_abs(cache=CACHE, out=OUT, offline=False):
    """Cache required ABS vintages and their source pages with verifiable publication evidence."""
    session = requests.Session()
    releases, vintage = [], []
    for month in pd.date_range('2021-06-01', '2022-06-01', freq='QS-JUN'):
        slug = month.strftime('%b-%Y').lower()
        url = CPI_BASE + slug
        html = cached_get(url, cache / 'cpi' / f'{slug}.html', session, offline)
        r = abs_release(html, url, '640105')
        if slug == 'jun-2021':
            text = BeautifulSoup(html, 'html.parser').get_text(' ', strip=True)
            require('Released 28/07/2021' in text and '19/10/2021 - Released additional information about the graph' in text,
                    'Cannot disambiguate original CPI publication from the later housing-graph update')
            r.update(page_updated_at=r['published_at'], published_at=pd.Timestamp('2021-07-28 11:30', tz='Australia/Sydney'),
                     evidence='Released 28/07/2021; 19/10 update adds housing-graph explanation, not Table 7 values.')
        r.update(source='C', vintage=slug, reference_month=month)
        ext = Path(r['source_url']).suffix
        vals = cpi_values(cached_get(r['source_url'], cache / 'cpi' / f'{slug}{ext}', session, offline), ext, month)
        releases.append(r)
        vintage.append(vals.assign(source='C', vintage=slug, published_at=r['published_at'], source_url=r['source_url']))
        print(f"Verified CPI {slug}: {r['published_at'].date()}", flush=True)
    for slug, year, publication in [('jul2016-jun2020', 2020, '2021-02-16'), ('jul2017-jun2021', 2021, '2021-12-16')]:
        url = CABEE_BASE + slug
        r = abs_release(cached_get(url, cache / 'cabee' / f'{slug}.html', session, offline), url, '816502')
        ext = Path(r['source_url']).suffix
        vals, header = cabee_values(cached_get(r['source_url'], cache / 'cabee' / f'{slug}{ext}', session, offline), ext, year, publication)
        # The overall page is updated after the detailed table publication. Its workbook supplies the date.
        r.update(source='E', vintage=slug, reference_month=pd.Timestamp(year, 6, 1),
                 page_updated_at=r['published_at'], published_at=pd.Timestamp(publication + ' 11:30').tz_localize('Australia/Sydney'),
                 evidence=header)
        releases.append(r)
        vintage.append(vals.assign(source='E', vintage=slug, published_at=r['published_at'], source_url=r['source_url']))
        print(f"Verified business exit vintage {year}: {publication}", flush=True)
    releases, vintage = pd.DataFrame(releases), pd.concat(vintage, ignore_index=True)
    write_table(releases, out, 'external_releases')
    write_table(vintage, out, 'external_vintage_values')
    return releases, vintage


def trends_window(origin):
    """Use the last 26 Sunday-Saturday weeks ending before the decision month begins."""
    last = pd.Timestamp(origin).to_period('M').start_time - pd.Timedelta(days=1)
    end = last - pd.Timedelta(days=(last.weekday() - 5) % 7)
    return end - pd.Timedelta(days=181), end


def trends_requests(mapping):
    """Create outcome-independent official export requests for the frozen queries."""
    rows = []
    for i, item in mapping.reset_index(drop=True).iterrows():
        if not item.trends_query:
            continue
        for origin in ORIGINS:
            start, end = trends_window(origin)
            url = 'https://trends.google.com/trends/explore?' + urlencode(dict(date=f'{start:%Y-%m-%d} {end:%Y-%m-%d}', geo='AU', q=item.trends_query))
            rows.append(dict(category=item.category, query=item.trends_query, origin=origin,
                             start=start, end=end, request_id=f'{i:02d}_{origin:%Y-%m-%d}', url=url))
    return pd.DataFrame(rows)


def parse_trends_csv(payload, start, end, query):
    """Validate official daily or weekly CSV data and aggregate 26 complete calendar weeks."""
    text = payload.decode('utf-8-sig')
    lines = text.splitlines()
    header = next((i for i, x in enumerate(lines) if x.startswith(('Day,', 'Week,'))), None)
    require(header is not None, 'Google export needs a Day or Week column')
    df = pd.read_csv(io.StringIO('\n'.join(lines[header:])))
    require(len(df.columns) == 2 and query.lower() in df.columns[1].lower() and 'Australia' in df.columns[1], 'Wrong Google query or geography')
    dates = pd.to_datetime(df.iloc[:, 0], errors='raise')
    values = pd.to_numeric(df.iloc[:, 1].replace('<1', '0.5'), errors='raise')
    require(dates.is_unique and dates.min() >= start and np.isfinite(values).all() and values.between(0, 100).all(), 'Invalid Trends observations')
    if df.columns[0] == 'Day':
        require(pd.DatetimeIndex(dates).equals(pd.date_range(start, end)), 'Incomplete daily Trends export')
        weekly = values.to_numpy().reshape(26, 7).mean(axis=1)
    else:
        require(pd.DatetimeIndex(dates).equals(pd.date_range(start, periods=26, freq='7D')), 'Incomplete weekly Trends export')
        require(dates.max() + pd.Timedelta(days=6) <= end, 'Partial or future Trends week')
        weekly = values.to_numpy()
    require((weekly > 0).sum() >= 20 and weekly[:13].sum() > 0 and weekly[13:].sum() > 0, 'Insufficient search volume')
    return float(np.log(weekly[13:].sum() / weekly[:13].sum())), weekly


def external_features(mapping, releases, vintages, cache=CACHE, out=OUT):
    """Join one feature per source using each row's own publication cutoff and export window."""
    requests_table = trends_requests(mapping)
    write_table(requests_table, out, 'trends_export_requests')
    records, google_audit = [], []
    for origin in ORIGINS:
        c = choose_asof(releases.loc[releases.source.eq('C')], origin)
        e = choose_asof(releases.loc[releases.source.eq('E')], origin)
        cv = vintages.loc[vintages.source.eq('C') & vintages.vintage.eq(c.vintage)].set_index('series_id')
        ev = vintages.loc[vintages.source.eq('E') & vintages.vintage.eq(e.vintage)].set_index('industry_code')
        for item in mapping.itertuples():
            row = dict(origin=origin, category=item.category, C=np.nan, E=np.nan, G=np.nan,
                       C_vintage=c.vintage, E_vintage=e.vintage, C_published_at=c.published_at,
                       E_published_at=e.published_at, C_source_url=c.source_url, E_source_url=e.source_url)
            if item.cpi_series_id:
                row['C'] = float(cv.loc[item.cpi_series_id, 'value'])
            if item.cabee_codes:
                sub = ev.loc[item.cabee_codes.split('|')]
                if sub.status.eq('available').all():
                    row['E'] = float(sub.exits.sum() / sub.start_count.sum())
            if item.trends_query:
                spec = requests_table.loc[requests_table.origin.eq(origin) & requests_table.category.eq(item.category)].iloc[0]
                estimates, weekly_values, errors = [], [], []
                for replica in [1, 2, 3]:
                    path = cache / 'trends' / f'{spec.request_id}_{replica}.csv'
                    if path.exists():
                        try:
                            val, weekly = parse_trends_csv(path.read_bytes(), spec.start, spec.end, spec.query)
                            estimates.append(val)
                            weekly_values.append(weekly.tolist())
                        except ValueError as exc:
                            errors.append(str(exc))
                status = 'missing_exports'
                if len(estimates) == 3 and not errors:
                    status = 'unstable' if np.ptp(estimates) > 0.20 else 'available_reconstructed'
                    if status == 'available_reconstructed':
                        row['G'] = float(np.median(estimates))
                elif errors:
                    status = 'quality_rejected'
                google_audit.append(dict(origin=origin, category=item.category, query=spec.query, url=spec.url,
                                         start=spec.start, end=spec.end, exports=len(estimates), status=status,
                                         estimates=json.dumps(estimates), weekly_values=json.dumps(weekly_values),
                                         identical_exports=len(estimates) == 3 and np.ptp(estimates) == 0,
                                         errors='; '.join(errors)))
            records.append(row)
    features = pd.DataFrame(records)
    write_table(features, out, 'external_features')
    write_table(pd.DataFrame(google_audit), out, 'trends_quality')
    return features
