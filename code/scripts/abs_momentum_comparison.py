"""Reproducible, point-in-time ABS momentum ablation; run from the CLI or notebook."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import re
import sqlite3
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from source_paths import source_directory
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
from bs4 import BeautifulSoup
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2021-03-01")
END = pd.Timestamp("2022-09-30")
TRAIN_ORIGINS = pd.date_range("2021-08-31", "2022-03-31", freq="ME")
DEV_ORIGINS = pd.date_range("2021-12-31", "2022-03-31", freq="ME")
FINAL_ORIGIN = pd.Timestamp("2022-06-30")
FEATURE_ORIGINS = TRAIN_ORIGINS.union(pd.DatetimeIndex([FINAL_ORIGIN]))
INTERNAL = ["log_recent_sales", "log_own_growth"]
MODEL_PARAMS = dict(power=1.5, alpha=0.1, link="log", fit_intercept=True,
                    solver="lbfgs", max_iter=2000, tol=1e-6)
SERIES = {
    "A3349785A": "Furniture, floor coverings, houseware and textile goods retailing",
    "A3349392L": "Electrical and electronic goods retailing",
    "A3349467V": "Hardware, building and garden supplies retailing",
    "A3349393R": "Footwear and other personal accessory retailing",
    "A3349552L": "Newspaper and book retailing",
    "A3349707V": "Other recreational goods retailing",
    "A3349395V": "Other retailing n.e.c.",
}
BASE_URL = "https://www.abs.gov.au/statistics/industry/retail-and-wholesale-trade/retail-trade-australia/"
TAG_PATTERN = r"(?i)^[\[(]{2}(.*?)[\])]\s*,\s*[\[(]([A-Za-z])[\])]\s*,\s*[\[(]take rate:\s*([0-9]+(?:\.[0-9]+)?)[\])]{2}$"
PACKAGES = ["numpy", "pandas", "pyarrow", "requests", "beautifulsoup4",
            "openpyxl", "xlrd", "scikit-learn", "scipy", "joblib", "threadpoolctl", "nbformat", "nbclient"]


def require(condition, message):
    """Fail a data or experiment contract even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def write_table(frame, directory, name):
    """Save an uncompressed analytical table and a portable CSV view."""
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / f"{name}.parquet", index=False, compression=None)
    frame.to_csv(directory / f"{name}.csv", index=False)


def normalize_category(value):
    """Apply the same formatting-only normalization as the existing ETL notebook."""
    value = re.sub(r"\s+", " ", value.lower().strip())
    value = re.sub(r"\s*([,:])\s*", r"\1 ", value)
    return re.sub(r"\s*-\s*", " - ", value).strip()


def load_merchants(source_dir, mapping_path):
    """Read original merchant metadata and attach the frozen category crosswalk."""
    with zipfile.ZipFile(source_dir / "project-2-bnpl-tables-part1.zip") as archive:
        names = [n for n in archive.namelist() if n.endswith("/tbl_merchants.parquet")]
        require(len(names) == 1, "Expected exactly one merchant metadata file.")
        merchants = pd.read_parquet(io.BytesIO(archive.read(names[0]))).reset_index()
    require(merchants.merchant_abn.notna().all() and merchants.merchant_abn.is_unique,
            "Merchant metadata must have unique non-null ABNs.")
    parsed = merchants.tags.str.extract(TAG_PATTERN)
    require(parsed.notna().all().all(), "Merchant tags failed the existing ETL parser.")
    merchants["category"] = parsed[0].map(normalize_category)
    merchants["letter_group"] = parsed[1].str.lower()
    merchants["take_rate"] = pd.to_numeric(parsed[2])
    mapping = pd.read_csv(mapping_path, keep_default_na=False)
    require(len(mapping) == 25 and mapping.category.is_unique, "Expected 25 unique category mappings.")
    require(set(mapping.category) == set(merchants.category), "Category labels differ from the frozen mapping.")
    require(set(mapping.loc[mapping.series_id.ne(""), "series_id"]) == set(SERIES),
            "Category lookup must use exactly the seven agreed ABS series.")
    merchants = merchants.merge(mapping, on="category", validate="many_to_one")
    merchants["abs_matched"] = merchants.series_id.ne("")
    merchants["valid_take_rate"] = (np.isfinite(merchants.take_rate)
                                    & merchants.take_rate.gt(0) & merchants.take_rate.le(100))
    return merchants.sort_values("merchant_abn").reset_index(drop=True), mapping


def aggregate_transactions(source_dir, output_dir):
    """Scan ZIP partitions once and audit exact global order-ID uniqueness on disk."""
    pieces, daily, dates, snapshot_for_date = [], [], [], {}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="order_audit_", dir=output_dir) as scratch:
        connection = sqlite3.connect(str(Path(scratch) / "order_ids.sqlite"))
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-65536")
            connection.execute("CREATE TABLE orders (id TEXT PRIMARY KEY) WITHOUT ROWID")
            for part in [2, 3, 4]:
                archive_path = source_dir / f"project-2-bnpl-tables-part{part}.zip"
                print(f"Aggregating and checking source archive {part}/4", flush=True)
                with zipfile.ZipFile(archive_path) as archive:
                    members = sorted(n for n in archive.namelist() if n.endswith(".parquet"))
                    require(bool(members), f"No Parquet files in {archive_path.name}")
                    for name in members:
                        match = re.search(r"order_datetime=(\d{4}-\d{2}-\d{2})/", name)
                        require(match is not None, f"Missing transaction date partition: {name}")
                        day = pd.Timestamp(match[1])
                        snapshot = name.split("/order_datetime=")[0]
                        require(day not in snapshot_for_date or snapshot_for_date[day] == snapshot,
                                f"Overlapping transaction snapshots on {day.date()}")
                        snapshot_for_date[day] = snapshot
                        dates.append(day)
                        frame = pq.read_table(io.BytesIO(archive.read(name)),
                                              columns=["order_id", "merchant_abn", "dollar_value"]).to_pandas()
                        require(frame.notna().all().all(), f"Null transaction key or amount: {name}")
                        require(frame.order_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                                f"Blank/non-string order identifier: {name}")
                        require(np.isfinite(frame.dollar_value).all() and frame.dollar_value.gt(0).all(),
                                f"Nonpositive/nonfinite transaction amount: {name}")
                        try:
                            connection.executemany("INSERT INTO orders VALUES (?)", ((x,) for x in frame.order_id))
                            connection.commit()
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(f"Duplicate order ID found while reading {name}") from exc
                        month = day.to_period("M").to_timestamp()
                        grouped = frame.groupby("merchant_abn").dollar_value.agg(sales="sum", orders="size").reset_index()
                        grouped["month"] = month
                        pieces.append(grouped)
                        daily.append(dict(day=day, source_archive=archive_path.name, source_partition=name,
                                          orders=len(frame), sales=float(frame.dollar_value.sum())))
                print(f"Finished archive {part}; {sum(d['orders'] for d in daily):,} orders checked", flush=True)
            unique_orders = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        finally:
            connection.close()
    days = pd.DatetimeIndex(sorted(set(dates)))
    require(days.equals(pd.date_range(days.min(), days.max(), freq="D")), "Missing days in source snapshots.")
    require(set(pd.date_range(START, END)).issubset(set(days)), "Study months are not completely covered.")
    panel = pd.concat(pieces).groupby(["merchant_abn", "month"], as_index=False)[["sales", "orders"]].sum()
    daily = pd.DataFrame(daily)
    require(unique_orders == int(daily.orders.sum()) == int(panel.orders.sum()), "Order-count reconciliation failed.")
    require(np.isclose(panel.sales.sum(), daily.sales.sum(), rtol=1e-12), "Sales reconciliation failed.")
    panel["included_month"] = panel.month.between(START, END)
    audit = dict(source_orders=int(daily.orders.sum()), unique_order_ids=unique_orders,
                 source_sales=float(daily.sales.sum()), source_first_day=str(days.min().date()),
                 source_last_day=str(days.max().date()), complete_days=len(days),
                 complete_month_orders=int(panel.loc[panel.included_month, "orders"].sum()),
                 complete_month_sales=float(panel.loc[panel.included_month, "sales"].sum()),
                 excluded_partial_month_orders=int(panel.loc[~panel.included_month, "orders"].sum()))
    write_table(daily, output_dir, "source_partition_audit")
    return panel, audit


def complete_panel(aggregated, merchants):
    """Zero-fill the verified complete months, preserving unknown-merchant reconciliation."""
    included = aggregated.loc[aggregated.included_month].copy()
    unknown = included.loc[~included.merchant_abn.isin(merchants.merchant_abn)].copy()
    months = pd.date_range(START, END, freq="MS")
    index = pd.MultiIndex.from_product([merchants.merchant_abn, months], names=["merchant_abn", "month"])
    panel = included.set_index(["merchant_abn", "month"])[["sales", "orders"]].reindex(index, fill_value=0).reset_index()
    require(int(panel.orders.sum() + unknown.orders.sum()) == int(included.orders.sum()), "Matched/unknown counts differ.")
    require(np.isclose(panel.sales.sum() + unknown.sales.sum(), included.sales.sum(), rtol=1e-12),
            "Matched/unknown sales differ.")
    return panel, unknown


def cached_get(url, path, session, offline=False):
    """Retain the original historical response and reuse it on subsequent runs."""
    if not path.exists():
        require(not offline, f"Offline mode: missing historical source {path.name}")
        response = session.get(url, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return path.read_bytes()


def parse_release_page(html, reference_month, page_url):
    """Extract the detailed-release timestamp and Table 12 link, not headline timing."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    pattern = r"Release date and time\s*(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2})(am|pm)\s+(AEST|AEDT)"
    match = re.search(pattern, text, re.IGNORECASE)
    require(match is not None, f"Cannot verify detailed publication timestamp: {page_url}")
    published = pd.to_datetime(match[1] + " " + match[2] + match[3], format="%d/%m/%Y %I:%M%p")
    published = published.tz_localize("Australia/Melbourne")
    require(published.tzname() == match[4].upper(), f"Publication timezone mismatch: {page_url}")
    links = {urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True)
             if re.search(r"/8501012\.xlsx?(?:\?|$)", a["href"])}
    require(len(links) == 1, f"Cannot uniquely identify Table 12: {page_url}")
    # From July 2021 onwards, the page must explicitly corroborate the additional release date.
    if pd.Period(reference_month, "M") >= pd.Period("2021-07", "M"):
        evidence = re.search(r"(?:An additional (?:information )?release on|All tables have been made available on) "
                             r"(\d{1,2} \w+ \d{4})", text, re.I)
        require(evidence is not None, f"Missing additional-release evidence: {page_url}")
        require(pd.to_datetime(evidence[1], dayfirst=True).date() == published.date(),
                f"Conflicting detailed-release dates: {page_url}")
    return dict(reference_month=pd.Timestamp(reference_month), published_at=published,
                page_url=page_url, source_url=links.pop(), publication_evidence=match[0])


def parse_workbook(payload, reference_month, extension):
    """Resolve national seasonally adjusted series by ID and validated metadata."""
    book = pd.read_excel(io.BytesIO(payload), sheet_name="Data1", header=None,
                         engine="xlrd" if extension == ".xls" else "openpyxl")
    labels = {str(v).strip().lower(): i for i, v in enumerate(book.iloc[:, 0]) if isinstance(v, str)}
    require(all(k in labels for k in ["series id", "series type", "unit", "series end"]), "Missing ABS workbook metadata.")
    ids = book.iloc[labels["series id"]]
    records = []
    for series_id, description in SERIES.items():
        cols = ids.index[ids.eq(series_id)]
        require(len(cols) == 1, f"Missing/duplicate national ABS series {series_id}")
        col = cols[0]
        header = [x.strip() for x in str(book.iat[0, col]).split(";") if x.strip()]
        require(header == ["Turnover", "Total (State)", description], f"Unexpected series description: {header}")
        require(str(book.iat[labels["series type"], col]).lower() == "seasonally adjusted", "Wrong adjustment type.")
        require(str(book.iat[labels["unit"], col]) == "$ Millions", "Unexpected ABS turnover unit.")
        require(pd.Timestamp(book.iat[labels["series end"], col]) == pd.Timestamp(reference_month),
                "Historical workbook end date differs from its release reference month.")
        values = book.iloc[labels["series id"] + 1:, [0, col]].copy()
        values.columns = ["month", "turnover"]
        values["month"] = pd.to_datetime(values.month, errors="coerce")
        values = values.dropna(subset=["month"])
        values["turnover"] = pd.to_numeric(values.turnover, errors="raise")
        values["series_id"] = series_id
        require(values.month.is_unique and values.month.max() == pd.Timestamp(reference_month), "Invalid ABS month index.")
        require(np.isfinite(values.turnover).all() and values.turnover.gt(0).all(), "Invalid ABS turnover.")
        records.append(values)
    return pd.concat(records, ignore_index=True)


def choose_release(releases, origin):
    """Choose the latest verified vintage available by the end of a decision day."""
    cutoff = pd.Timestamp(origin).tz_localize("Australia/Melbourne") + pd.Timedelta(days=1)
    eligible = releases.loc[releases.published_at.lt(cutoff)].sort_values("published_at")
    require(len(eligible) > 0, f"No historical ABS release available at {origin}")
    return eligible.iloc[-1]


def momentum_from_vintage(values, release, origin):
    """Compute each industry's six-month momentum entirely within one vintage."""
    months = pd.date_range(end=release.reference_month, periods=6, freq="MS")
    records = []
    for series_id in SERIES:
        series = values.loc[values.series_id.eq(series_id)].set_index("month").turnover.reindex(months)
        require(series.notna().all() and series.gt(0).all(), "Six consecutive ABS months are required.")
        records.append(dict(origin=pd.Timestamp(origin), series_id=series_id,
                            abs_momentum=float(series.iloc[3:].sum() / series.iloc[:3].sum() - 1),
                            abs_reference_month=release.reference_month, abs_published_at=release.published_at,
                            abs_source_url=release.source_url, abs_page_url=release.page_url))
    return pd.DataFrame(records)


def load_abs_features(cache_dir, offline=False):
    """Acquire dated release pages and only the workbooks needed at historical origins."""
    session = requests.Session()
    releases = []
    # Include the next reference release to establish that it was still unavailable at June 30.
    for month in pd.date_range("2021-06-01", "2022-05-01", freq="MS"):
        slug = month.strftime("%b-%Y").lower()
        url = BASE_URL + slug
        html = cached_get(url, cache_dir / f"{slug}.html", session, offline)
        releases.append(parse_release_page(html, month, url))
    releases = pd.DataFrame(releases).sort_values("published_at").reset_index(drop=True)
    features, vintage_tables, books = [], [], {}
    for origin in FEATURE_ORIGINS:
        release = choose_release(releases, origin)
        key = release.reference_month
        if key not in books:
            extension = Path(release.source_url).suffix
            path = cache_dir / f"{key:%Y-%m}_8501012{extension}"
            payload = cached_get(release.source_url, path, session, offline)
            books[key] = parse_workbook(payload, key, extension)
            # Retain only the six observations actually used, with their vintage provenance.
            used = books[key].loc[books[key].month.ge(key - pd.DateOffset(months=5))].copy()
            used["reference_month"] = key
            used["published_at"] = release.published_at
            used["source_url"] = release.source_url
            vintage_tables.append(used)
            print(f"Verified ABS reference {key:%Y-%m}, published {release.published_at.date()}", flush=True)
        features.append(momentum_from_vintage(books[key], release, origin))
    require(choose_release(releases, FINAL_ORIGIN).reference_month == pd.Timestamp("2022-04-01"),
            "Final cutoff must use the April 2022 Table 12 vintage.")
    return pd.concat(features, ignore_index=True), releases, pd.concat(vintage_tables, ignore_index=True)


def features_at(panel, merchants, origin, abs_features):
    """Build past-only features and eligibility for all metadata merchants at one origin."""
    origin = pd.Timestamp(origin)
    months = pd.date_range(end=origin.to_period("M").to_timestamp(), periods=6, freq="MS")
    history = panel.loc[panel.month.isin(months)]
    require(not history.duplicated(["merchant_abn", "month"]).any(), "Duplicate merchant-month key.")
    wide = history.pivot(index="merchant_abn", columns="month", values="sales").reindex(index=merchants.merchant_abn, columns=months)
    require(wide.notna().all().all(), "Incomplete merchant history; do not silently invent missing source months.")
    out = merchants.copy().set_index("merchant_abn")
    recent, previous = wide.iloc[:, 3:].sum(axis=1), wide.iloc[:, :3].sum(axis=1)
    out["recent_three_month_sales"] = recent
    out["previous_three_month_sales"] = previous
    out["recent_monthly_sales"] = recent / 3
    out["own_growth"] = recent.div(previous.where(previous.gt(0))) - 1
    out["eligible"] = out.valid_take_rate & recent.gt(0) & previous.gt(0)
    out["eligibility_reason"] = np.select(
        [~out.valid_take_rate, previous.le(0), recent.le(0)],
        ["invalid_take_rate", "no_sales_in_previous_three_months", "no_sales_in_recent_three_months"], default="eligible")
    out["log_recent_sales"] = np.log(out.recent_monthly_sales.where(out.recent_monthly_sales.gt(0)))
    out["log_own_growth"] = np.log(recent.where(recent.gt(0)) / previous.where(previous.gt(0)))
    out["origin"] = origin
    out["target_end"] = origin + pd.offsets.MonthEnd(3)
    out = out.reset_index().merge(abs_features.loc[abs_features.origin.eq(origin)].drop(columns="origin"),
                                  on="series_id", how="left", validate="many_to_one")
    require(out.loc[out.abs_matched, "abs_momentum"].notna().all(), f"Missing mapped ABS momentum at {origin.date()}")
    cutoff = origin.tz_localize("Australia/Melbourne") + pd.Timedelta(days=1)
    require(out.loc[out.abs_matched, "abs_published_at"].lt(cutoff).all(), "Future ABS publication in features.")
    return out


def targets_at(panel, merchant_ids, origin):
    """Read the next three complete months separately from feature construction."""
    months = pd.date_range(pd.Timestamp(origin) + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    target = panel.loc[panel.month.isin(months) & panel.merchant_abn.isin(merchant_ids)]
    require(not target.duplicated(["merchant_abn", "month"]).any(), "Duplicate outcome month.")
    require(len(target) == len(merchant_ids) * 3, "Three complete outcome months are required per merchant.")
    return target.groupby("merchant_abn").sales.sum().reindex(merchant_ids)


def fit_component(training, candidates, columns):
    """Fit the frozen GLM, raising on nonconvergence, and retain its fitted parameters."""
    require(len(training) > 0 and len(candidates) > 0, "Empty model cohort.")
    require(np.isfinite(training[columns]).all().all() and np.isfinite(candidates[columns]).all().all(),
            "Nonfinite model features.")
    model = make_pipeline(StandardScaler(), TweedieRegressor(**MODEL_PARAMS))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(training[columns], training.target_sales / 1000)
    prediction = model.predict(candidates[columns]) * 1000
    require(np.isfinite(prediction).all() and (prediction > 0).all(), "Invalid sales forecast.")
    scaler, glm = model.steps[0][1], model.steps[1][1]
    details = dict(features=columns, coefficients=glm.coef_.tolist(), intercept=float(glm.intercept_),
                   scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                   iterations=int(glm.n_iter_), training_rows=len(training),
                   training_origins=sorted(training.origin.dt.strftime("%Y-%m-%d").unique().tolist()))
    return prediction, details


def predict_at(panel, feature_panel, origin):
    """Fit only matured training targets and rank both policies without accessing future outcomes."""
    origin = pd.Timestamp(origin)
    history = panel.loc[panel.month.le(origin)].copy()
    training = feature_panel.loc[feature_panel.eligible & feature_panel.target_end.le(origin)].copy()
    require(len(training) > 0 and training.origin.lt(origin).all(), "Invalid training origins.")
    training["target_sales"] = np.nan
    for train_origin, group in training.groupby("origin"):
        training.loc[group.index, "target_sales"] = targets_at(history, group.merchant_abn, train_origin).to_numpy()
    candidates = feature_panel.loc[feature_panel.origin.eq(origin) & feature_panel.eligible].copy()
    require(len(candidates) >= 100, "Fewer than 100 eligible merchants.")
    candidates = candidates.sort_values("merchant_abn").reset_index(drop=True)
    matched = candidates.abs_matched
    baseline = np.empty(len(candidates))
    enhanced = np.empty(len(candidates))
    fits = {}
    for label, mask, columns in [
        ("matched_baseline", matched, INTERNAL),
        ("matched_enhanced", matched, INTERNAL + ["abs_momentum"]),
        ("unmatched_shared", ~matched, INTERNAL),
    ]:
        cohort = label != "unmatched_shared"
        pred, fits[label] = fit_component(training.loc[training.abs_matched.eq(cohort)], candidates.loc[mask], columns)
        if label != "matched_enhanced":
            baseline[mask] = pred
        if label != "matched_baseline":
            enhanced[mask] = pred
    require(fits["matched_baseline"]["training_rows"] == fits["matched_enhanced"]["training_rows"], "Mismatched training cohorts.")
    require(np.array_equal(fits["matched_baseline"]["scaler_mean"], fits["matched_enhanced"]["scaler_mean"][:2]),
            "Internal scaling differs between matched methods.")
    require(np.array_equal(baseline[~matched], enhanced[~matched]), "Unmatched forecasts differ.")
    frames = []
    for method, prediction in [("baseline", baseline), ("enhanced", enhanced),
                               ("repeat_sales", candidates.recent_three_month_sales.to_numpy())]:
        frame = candidates.copy()
        frame["method"] = method
        frame["predicted_sales"] = prediction
        frame["predicted_fees"] = prediction * frame.take_rate / 100
        frame = frame.sort_values(["predicted_fees", "merchant_abn"], ascending=[False, True], kind="stable")
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["selected"] = frame["rank"].le(100)
        require(frame.merchant_abn.is_unique and int(frame.selected.sum()) == 100, "Invalid top-100 selection.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), fits


def score_at(predictions, panel, origin):
    """Attach realized fees only after selections exist and compute comparable metrics."""
    actual = targets_at(panel, predictions.merchant_abn.drop_duplicates(), origin).rename("actual_sales")
    scored = predictions.merge(actual, on="merchant_abn", validate="many_to_one")
    scored["actual_fees"] = scored.actual_sales * scored.take_rate / 100
    metrics = []
    for method, group in scored.groupby("method"):
        for cohort, frame in [("full_pool", group), ("matched", group.loc[group.abs_matched])]:
            errors = frame.predicted_fees - frame.actual_fees
            total = frame.actual_fees.sum()
            require(total > 0, "Forecast percentage metrics need positive total observed fees.")
            metrics.append(dict(origin=pd.Timestamp(origin), method=method, cohort=cohort,
                                merchants=len(frame), mae=float(errors.abs().mean()),
                                wape_pct=float(errors.abs().sum() / total * 100),
                                bias_dollars=float(errors.sum()), bias_pct=float(errors.sum() / total * 100)))
    selected = scored.loc[scored.selected]
    totals = selected.groupby("method").actual_fees.sum()
    require(totals.baseline > 0, "Baseline top-100 fees must be positive.")
    a = set(selected.loc[selected.method.eq("baseline"), "merchant_abn"])
    b = set(selected.loc[selected.method.eq("enhanced"), "merchant_abn"])
    changes = scored.loc[scored.method.eq("baseline") & scored.merchant_abn.isin(a ^ b)].copy()
    changes["change"] = np.where(changes.merchant_abn.isin(b), "entered", "removed")
    changes["fee_contribution"] = changes.actual_fees * np.where(changes.change.eq("entered"), 1, -1)
    delta = float(totals.enhanced - totals.baseline)
    require(np.isclose(changes.fee_contribution.sum(), delta, atol=1e-6, rtol=1e-10), "Selection attribution failed.")
    fold = dict(origin=pd.Timestamp(origin), baseline_fees=float(totals.baseline),
                enhanced_fees=float(totals.enhanced), repeat_sales_fees=float(totals.repeat_sales),
                delta_fees=delta, uplift_pct=delta / float(totals.baseline) * 100,
                top100_overlap=len(a & b), replacements=len(b - a),
                largest_absolute_change=float(changes.fee_contribution.abs().max()) if len(changes) else 0.0)
    # Independently recompute selected fees directly from the monthly panel.
    months = pd.date_range(pd.Timestamp(origin) + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    for method, group in selected.groupby("method"):
        monthly = panel.loc[panel.month.isin(months)].merge(group[["merchant_abn", "take_rate"]], on="merchant_abn")
        direct_total = (monthly.sales * monthly.take_rate / 100).sum()
        require(np.isclose(direct_total, totals[method], rtol=1e-12), "Independent selected-fee audit failed.")
    composition = selected.groupby(["origin", "method", "category", "match_quality"], as_index=False).agg(
        merchants=("merchant_abn", "size"), actual_fees=("actual_fees", "sum"), predicted_fees=("predicted_fees", "sum"))
    return scored, pd.DataFrame(metrics), fold, changes, composition


def freeze_protocol(output_dir, mapping):
    """Freeze the agreed experiment before outcomes are scored; reject silent configuration changes."""
    protocol = dict(name="ABS momentum controlled comparison v1", internal_features=INTERNAL,
                    enhanced_features=INTERNAL + ["abs_momentum"], model=MODEL_PARAMS,
                    target="next three complete months GMV / 1000", fee_rate_divisor=100,
                    training_origins=TRAIN_ORIGINS.strftime("%Y-%m-%d").tolist(),
                    development_origins=DEV_ORIGINS.strftime("%Y-%m-%d").tolist(),
                    final_origin=str(FINAL_ORIGIN.date()), top_n=100, tie_break="ascending merchant_abn",
                    eligibility="valid take rate and positive sales in both preceding three-month windows",
                    primary_rule=dict(final_uplift_pct_at_least=1.0, positive_development_folds_at_least=3),
                    mapping=mapping.to_dict("records"))
    path = output_dir / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == protocol,
                "Frozen protocol differs. Do not overwrite the previous experiment or reuse its holdout as unseen.")
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    versions["python"] = sys.version
    (output_dir / "runtime_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")


def retention_decision(folds):
    """Apply the practical threshold without interpreting overlapping folds as independent evidence."""
    require(folds.origin.is_unique and set(folds.origin) == set(DEV_ORIGINS.union(pd.DatetimeIndex([FINAL_ORIGIN]))),
            "The retention rule requires exactly the four agreed development folds and final fold.")
    final = folds.loc[folds.origin.eq(FINAL_ORIGIN)].iloc[0]
    positive = int(folds.loc[folds.origin.isin(DEV_ORIGINS), "uplift_pct"].gt(0).sum())
    return dict(retain_abs_momentum=bool(final.uplift_pct >= 1 and positive >= 3),
                final_uplift_pct=float(final.uplift_pct), final_delta_fees=float(final.delta_fees),
                positive_development_folds=positive)


def write_report(output_dir, folds, changes, audit, coverage, forecast_metrics):
    """Summarize the prespecified retention decision and its practical limitations."""
    final = folds.loc[folds.origin.eq(FINAL_ORIGIN)].iloc[0]
    decision = retention_decision(folds)
    positive, retain = decision["positive_development_folds"], decision["retain_abs_momentum"]
    lines = ["# ABS momentum controlled comparison", "",
             f"**Decision: {'retain ABS momentum for the next project stage' if retain else 'retain Minimal + own growth; insufficient evidence to add ABS momentum'}.**", "",
             f"Final July-September 2022 uplift: **{final.uplift_pct:+.4f}% (A${final.delta_fees:+,.2f})**. "
             f"Positive earlier tests: **{positive}/4**. Required: final uplift >=1% and at least 3/4 earlier tests positive.", "",
             "| Selection date | Baseline fees | With ABS fees | Repeat-sales fees | Uplift | Shared selections |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in folds.itertuples():
        lines.append(f"| {row.origin:%Y-%m-%d} | {row.baseline_fees:,.2f} | {row.enhanced_fees:,.2f} | "
                     f"{row.repeat_sales_fees:,.2f} | {row.uplift_pct:+.4f}% | {row.top100_overlap}/100 |")
    accuracy = forecast_metrics.loc[forecast_metrics.origin.eq(FINAL_ORIGIN)
                                   & forecast_metrics.cohort.eq("full_pool")].set_index("method")
    lines += ["", "## Forecast accuracy and benchmark", "",
              f"Final full-pool WAPE: baseline {accuracy.loc['baseline', 'wape_pct']:.2f}%, "
              f"with ABS {accuracy.loc['enhanced', 'wape_pct']:.2f}%, "
              f"repeat-sales benchmark {accuracy.loc['repeat_sales', 'wape_pct']:.2f}% (lower is better). "
              "An improvement in average forecast accuracy need not change the top 100.", "",
              f"The repeat-sales selection earned A${final.repeat_sales_fees - final.baseline_fees:+,.2f} "
              f"({(final.repeat_sales_fees / final.baseline_fees - 1) * 100:+.4f}%) relative to the fitted baseline. "
              "Retaining the simpler feature set does not establish that this fixed regression is the best ranking model."]
    last_changes = changes.loc[changes.origin.eq(FINAL_ORIGIN)]
    attribution = last_changes.groupby(["category", "match_quality"]).fee_contribution.sum().sort_values(key=abs, ascending=False)
    lines += ["", "## What changed in the final selection", "",
              f"ABS replaced {int(final.replacements)} merchants. Entrants and removals reconcile exactly to the fee difference.", ""]
    for (category, quality), amount in attribution.items():
        lines.append(f"- {category} ({quality}): A${amount:+,.2f} net contribution.")
    if len(last_changes):
        largest = last_changes.loc[last_changes.fee_contribution.abs().idxmax()]
        lines += ["", f"Largest individual change: merchant {largest.merchant_abn} ({largest.change}), "
                  f"A${largest.fee_contribution:+,.2f}. These are descriptive contributions, not independent treatment effects."]
    final_coverage = coverage.loc[coverage.origin.eq(FINAL_ORIGIN)]
    lines += ["", "## Data and interpretation", "",
              f"Audited {audit['source_orders']:,} unique orders across {audit['complete_days']} source days. "
              f"Used {audit['complete_month_orders']:,} orders in complete months before metadata exclusions; "
              f"{audit['unknown_complete_month_orders']:,} of those belong to merchants without metadata.", "",
              "The mapping covers 13/25 categories (2,121/4,026 metadata merchants before activity eligibility). "
              "Coverage by origin and recent fees is in coverage.csv. Category matches include documented proxies.", "",
              f"At the final cutoff, {int(final_coverage.merchants.sum()):,} merchants were eligible, "
              f"including {int(final_coverage.loc[final_coverage.abs_matched, 'merchants'].sum()):,} with ABS matches.", "",
              "Gross fees are recorded sales multiplied by the assumed percentage take rate. They exclude defaults, "
              "funding costs and operating costs. Categories and rates are assumed stable and known throughout the study; "
              "historical recorded sales stand in for outcomes of hypothetical merchant selection.", "",
              "The final June 30 selection used the April 2022 ABS release. Historical workbooks and release pages are "
              "cached with publication provenance. They are the currently accessible dated ABS release files; "
              "independent captures of the exact original publication bytes were not available.", "",
              "Only 19 complete merchant months are available. Earlier evaluation windows overlap; the initial fit has "
              "two distinct training origins and the final fit eight. Thousands of merchants do not create thousands "
              "of independent macroeconomic observations. Results support a pilot decision, not statistical significance, "
              "causal impact, or stability across economic cycles. Synthetic merchant demand may not track real ABS spending.", "",
              "The repeat-sales benchmark is a sanity check. Forecast MAE/WAPE/bias, category composition and mapping "
              "attribution are supporting diagnostics; they do not replace the frozen top-100 decision rule.", "",
              "No model tuning, category changes, risk penalties or extra predictors were introduced after evaluation.", ""]
    (output_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def run(source_dir=None, output_dir=None, offline=False, reuse_panel=False, prepare_only=False):
    """Run the frozen comparison end to end, preserving source data and detailed audits."""
    source_dir = source_directory(source_dir)
    output_dir = Path(output_dir) if output_dir else ROOT / "tables/analysis/abs_momentum_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    merchants, mapping = load_merchants(source_dir, ROOT / "tables/reference/abs_category_mapping.csv")
    freeze_protocol(output_dir, mapping)
    write_table(mapping, output_dir, "category_mapping")
    cache_dir = ROOT / "tables/landing/abs_retail_trade"
    abs_features, releases, vintages = load_abs_features(cache_dir, offline)
    write_table(releases, output_dir, "abs_releases")
    write_table(vintages, output_dir, "abs_vintage_values")
    write_table(abs_features, output_dir, "abs_features")
    if reuse_panel:
        panel = pd.read_parquet(output_dir / "merchant_month.parquet")
        audit = json.loads((output_dir / "data_audit.json").read_text(encoding="utf-8"))
        require(audit.get("global_order_id_check") == "passed", "Cannot reuse a panel without an exact ID audit.")
    else:
        aggregated, audit = aggregate_transactions(source_dir, output_dir)
        panel, unknown = complete_panel(aggregated, merchants)
        audit.update(unknown_complete_month_orders=int(unknown.orders.sum()),
                     unknown_complete_month_sales=float(unknown.sales.sum()),
                     unknown_complete_month_merchants=int(unknown.merchant_abn.nunique()),
                     metadata_merchants=len(merchants), global_order_id_check="passed")
        write_table(panel, output_dir, "merchant_month")
        write_table(unknown, output_dir, "unknown_merchants_monthly")
        (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    require(not panel.duplicated(["merchant_abn", "month"]).any(), "Duplicate panel keys.")
    require(len(panel) == len(merchants) * 19 and np.isfinite(panel.sales).all() and panel.sales.ge(0).all(),
            "Invalid complete-month panel.")
    # No outcome columns enter this feature table, including at the final decision origin.
    features = pd.concat([features_at(panel.loc[panel.month.le(origin)], merchants, origin, abs_features)
                          for origin in FEATURE_ORIGINS], ignore_index=True)
    write_table(features, output_dir, "merchant_features")
    eligibility = features.groupby(["origin", "eligibility_reason"], as_index=False).size()
    write_table(eligibility, output_dir, "eligibility")
    eligible = features.loc[features.eligible].copy()
    eligible["recent_fees"] = eligible.recent_three_month_sales * eligible.take_rate / 100
    coverage = eligible.groupby(["origin", "abs_matched"], as_index=False).agg(
        merchants=("merchant_abn", "size"), recent_fees=("recent_fees", "sum"))
    coverage["merchant_share_pct"] = coverage.merchants / coverage.groupby("origin").merchants.transform("sum") * 100
    coverage["recent_fee_share_pct"] = coverage.recent_fees / coverage.groupby("origin").recent_fees.transform("sum") * 100
    write_table(coverage, output_dir, "coverage")
    if prepare_only:
        print("Preparation complete; no models fitted and no future outcomes scored.", flush=True)
        return dict(output_dir=output_dir)
    all_scored, all_metrics, all_folds, all_changes, all_composition, model_details = [], [], [], [], [], {}
    development_panel = panel.loc[panel.month.le(FINAL_ORIGIN)].copy()
    for origin in list(DEV_ORIGINS) + [FINAL_ORIGIN]:
        print(f"Fitting frozen models at {origin:%Y-%m-%d}", flush=True)
        prediction, fits = predict_at(development_panel, features, origin)
        # Save unscored selections before reading the final held-out outcome window.
        write_table(prediction, output_dir / "unscored_predictions", f"{origin:%Y-%m-%d}")
        model_details[str(origin.date())] = fits
        scoring_panel = panel if origin == FINAL_ORIGIN else development_panel
        scored, metrics, fold, changes, composition = score_at(prediction, scoring_panel, origin)
        all_scored.append(scored)
        all_metrics.append(metrics)
        all_folds.append(fold)
        all_changes.append(changes)
        all_composition.append(composition)
        print(f"Scored {origin:%Y-%m-%d}: ABS uplift {fold['uplift_pct']:+.4f}%", flush=True)
    predictions = pd.concat(all_scored, ignore_index=True)
    folds = pd.DataFrame(all_folds)
    changes = pd.concat(all_changes, ignore_index=True)
    write_table(predictions, output_dir, "predictions")
    forecast_metrics = pd.concat(all_metrics, ignore_index=True)
    write_table(forecast_metrics, output_dir, "forecast_metrics")
    write_table(folds, output_dir, "fold_metrics")
    write_table(changes, output_dir, "selection_changes")
    write_table(pd.concat(all_composition, ignore_index=True), output_dir, "category_composition")
    final_selected = predictions.loc[predictions.origin.eq(FINAL_ORIGIN) & predictions.selected]
    write_table(final_selected, output_dir, "final_selected_merchants")
    (output_dir / "fitted_models.json").write_text(json.dumps(model_details, indent=2), encoding="utf-8")
    decision = write_report(output_dir, folds, changes, audit, coverage, forecast_metrics)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    return dict(output_dir=output_dir, folds=folds, decision=decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true", help="Require cached ABS sources; make no HTTP requests.")
    parser.add_argument("--reuse-panel", action="store_true", help="Reuse the previously audited panel; caller must ensure source archives are unchanged.")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and audit inputs without fitting or evaluating models.")
    args = parser.parse_args()
    run(args.source_dir, args.output_dir, args.offline, args.reuse_panel, args.prepare_only)
