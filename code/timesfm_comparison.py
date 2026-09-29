"""Local, zero-shot TimesFM 3.0 daily forecasts versus fixed historical baselines."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from abs_momentum_comparison import require, write_table
from external_data_sources import archive_signature, merchants_and_mapping, save_json
from forecast_comparison import ORIGINS, attach_outcomes, build_features, digest, make_predictions, summarize

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "tables/analysis/external_data_comparison"
PREVIOUS = ROOT / "tables/analysis/forecast_comparison"
OUT = ROOT / "tables/analysis/timesfm_comparison"
CHECKPOINT = ROOT / "tables/landing/timesfm3/checkpoint"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
QUANTILES = np.arange(1, 10) / 10
MEAN_WEIGHTS = np.array([.15, .1, .1, .1, .1, .1, .1, .1, .15])
BATCH_SIZE = 32  # Chosen using synthetic throughput/memory checks, before merchant outcomes.


def validated_inputs():
    source = ROOT.parent / "project 2 materials"
    audit = json.loads((PREPARED / "merchant_audit.json").read_text())
    require(audit["archives"] == archive_signature(source), "Original archives changed")
    merchants = pd.read_parquet(PREPARED / "merchants.parquet")
    reparsed, _ = merchants_and_mapping(source)
    pd.testing.assert_frame_equal(merchants, reparsed)
    panel = pd.read_parquet(PREPARED / "merchant_month.parquet")
    previous_protocol = json.loads((PREVIOUS / "protocol.json").read_text())
    require(digest(panel) == previous_protocol["inputs"]["panel"], "Monthly panel changed")
    return source, panel, merchants, audit


def prepare_daily():
    """Aggregate source partitions, retaining their existing audited coverage/ID contract."""
    source, monthly, merchants, audit = validated_inputs()
    path = OUT / "merchant_day.parquet"
    if path.exists():
        saved = json.loads((OUT / "daily_audit.json").read_text())
        require(saved["archives"] == audit["archives"] and saved["monthly_digest"] == digest(monthly),
                "Daily source fingerprint changed")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == saved["daily_sha256"], "Daily panel changed")
        return pd.read_parquet(path)
    pieces, partition_rows = [], []
    ids = set(merchants.merchant_abn)
    for part in [2, 3, 4]:
        print(f"Aggregating merchant-day sales: archive {part}/4", flush=True)
        with zipfile.ZipFile(source / f"project-2-bnpl-tables-part{part}.zip") as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith(".parquet")):
                match = re.search(r"order_datetime=(\d{4}-\d{2}-\d{2})/", name)
                require(match is not None, "Missing partition date")
                day = pd.Timestamp(match[1])
                if not pd.Timestamp("2021-03-01") <= day <= pd.Timestamp("2022-09-30"):
                    continue
                count, total = 0, 0.
                reader = pq.ParquetFile(io.BytesIO(archive.read(name)))
                for batch in reader.iter_batches(batch_size=100_000, columns=["merchant_abn", "dollar_value"]):
                    raw = batch.to_pandas()
                    require(raw.notna().all().all() and np.isfinite(raw.dollar_value).all()
                            and raw.dollar_value.gt(0).all(), "Invalid source values")
                    count += len(raw)
                    total += float(raw.dollar_value.sum())
                    known = raw.loc[raw.merchant_abn.isin(ids)]
                    grouped = known.groupby("merchant_abn").dollar_value.agg(sales="sum", orders="size").reset_index()
                    pieces.append(grouped.assign(day=day))
                partition_rows.append(dict(source_partition=name, day=day, orders=count, sales=total))
    actual = pd.DataFrame(partition_rows).sort_values("source_partition").reset_index(drop=True)
    expected = pd.read_parquet(PREPARED / "source_partition_audit.parquet")
    expected = expected.loc[expected.day.ge("2021-03-01"), actual.columns].sort_values("source_partition").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-7)
    dates = pd.date_range("2021-03-01", "2022-09-30")
    require(set(actual.day) == set(dates), "Incomplete source-day coverage")
    daily = pd.concat(pieces, ignore_index=True).groupby(["merchant_abn", "day"])[["sales", "orders"]].sum()
    complete = pd.MultiIndex.from_product([merchants.merchant_abn, dates], names=["merchant_abn", "day"])
    daily = daily.reindex(complete, fill_value=0).reset_index()
    recomputed = daily.assign(month=daily.day.dt.to_period("M").dt.to_timestamp()).groupby(
        ["merchant_abn", "month"])[["sales", "orders"]].sum().sort_index()
    expected_monthly = monthly.set_index(["merchant_abn", "month"])[["sales", "orders"]].sort_index()
    pd.testing.assert_frame_equal(recomputed, expected_monthly, check_dtype=False, rtol=1e-12, atol=1e-7)
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(path, index=False, compression=None)
    save_json(OUT / "daily_audit.json", dict(archives=audit["archives"], monthly_digest=digest(monthly),
        daily_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(daily), days=len(dates),
        merchants=len(merchants), known_orders=int(daily.orders.sum()), known_sales=float(daily.sales.sum()),
        original_global_order_id_audit_reused=True, source_partitions_reconciled=True,
        all_merchant_months_reconciled=True))
    return daily


def past_contexts(wide, ids, origin):
    """Extract past values before any model or reference forecast can access outcomes."""
    dates = pd.date_range("2021-03-01", origin)
    data = wide.reindex(index=ids, columns=dates).to_numpy(dtype=np.float32)
    require(np.isfinite(data).all() and (data >= 0).all(), "Invalid daily history")
    return data


def seasonal_naive(contexts, horizon):
    """Repeat the last seven observed days over the exact future calendar horizon."""
    require(contexts.shape[1] >= 7, "Insufficient weekly history")
    return contexts[:, -7:][:, np.arange(horizon) % 7].sum(axis=1, dtype=np.float64)


def quantile_totals(quantiles):
    """A bounded-tail quantile-integral proxy for the mean, plus the native point line."""
    q = np.asarray(quantiles, dtype=np.float64)
    require(q.ndim == 3 and q.shape[-1] == 9 and np.isfinite(q).all(), "Invalid forecast quantiles")
    require((q >= 0).all() and (np.diff(q, axis=-1) >= -1e-6).all(), "Nonmonotone/negative quantiles")
    # Integrate piecewise-linear Q(p), holding Q(.1) and Q(.9) flat in the tails.
    # This is an explicit approximation, not a learned/calibrated conditional mean.
    return (q @ MEAN_WEIGHTS).sum(axis=1), q[:, :, 4].sum(axis=1)


def load_model(batch_size=BATCH_SIZE):
    import torch
    from timesfm3 import TimesFM3Forecaster
    torch.manual_seed(2026)
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), "This local experiment requires the verified CUDA GPU")
    model = TimesFM3Forecaster.from_pretrained(str(CHECKPOINT), device="cuda",
        per_core_batch_size=batch_size, local_files_only=True)
    return model


def predict_quantiles(model, contexts, horizon, ids):
    outputs = list(model.predict_batch(contexts=list(contexts), horizon=horizon,
        ts_ids=[str(i) for i in ids], return_quantiles=True, make_positive=True,
        sort_quantiles=True, use_symmetric_averaging=False))
    require([x.ts_id for x in outputs] == [str(i) for i in ids], "Forecast identifier mismatch")
    q = np.stack([x.quantiles for x in outputs])
    require(q.shape == (len(ids), horizon, 9), "Wrong forecast shape")
    return q


def smoke():
    import torch
    model = load_model()
    contexts = np.stack([100 + i + np.sin(np.arange(488) * 2 * np.pi / 7) * 20 for i in range(16)]).astype(np.float32)
    start = time.perf_counter()
    q = predict_quantiles(model, contexts, 92, range(16))
    elapsed = time.perf_counter() - start
    quantile_totals(q)
    again = predict_quantiles(model, contexts[:4], 92, range(4))
    require(np.allclose(q[:4], again, rtol=1e-5, atol=1e-5), "Nonreproducible smoke forecast")
    info = dict(seconds=elapsed, merchants_per_second=16 / elapsed,
        max_allocated_GB=torch.cuda.max_memory_allocated()/1024**3,
        gpu=torch.cuda.get_device_name(), quantile_shape=list(q.shape), reproducible=True)
    save_json(OUT / "smoke.json", info)
    print(json.dumps(info, indent=2), flush=True)


def run():
    import torch
    _, panel, merchants, _ = validated_inputs()
    daily = prepare_daily()
    wide = daily.pivot(index="merchant_abn", columns="day", values="sales")
    features = build_features(panel, merchants)
    previous = pd.read_parquet(PREVIOUS / "predictions.parquet")
    model = load_model()
    protocol = dict(model="google/timesfm-3.0-pytorch", checkpoint_revision=REVISION,
        model_sha256=hashlib.sha256((CHECKPOINT / "model.safetensors").read_bytes()).hexdigest(),
        model_config=dataclasses.asdict(model.config), seed=2026, batch_size=BATCH_SIZE, mode="zero-shot, univariate",
        input="Daily merchant sales from March 2021 through each origin; no future covariates or fine-tuning",
        target="Sum forecasts over the next three complete calendar months, then multiply by take rate / 100",
        primary_timesfm="Quantile-integral mean proxy, linearly interpolate p=.1 to .9 and hold endpoints in tails",
        diagnostic_timesfm="Sum of daily medians, not the median of the three-month total",
        primary_metric="Three-month fee WAPE and bias; selected-100 realized fees reported separately",
        calibration="No fitted recalibration; quantiles and tails are not assumed calibrated",
        transforms=dict(make_positive=True, sort_quantiles=True, use_symmetric_averaging=False),
        origins=ORIGINS.strftime("%Y-%m-%d").tolist(), baseline="Identical saved repeat/minimal/pooled_lags/damped_trend",
        additional_reference="Repeat last seven daily sales over future horizon",
        eligibility="Existing valid rate and positive sales in each of two preceding three-month windows",
        selection="100 highest predicted fees; ties by ascending numeric ABN",
        inputs=dict(panel=digest(panel), merchants=digest(merchants), features=digest(features),
                    daily=json.loads((OUT / "daily_audit.json").read_text())["daily_sha256"]),
        versions={p: importlib.metadata.version(p) for p in ["timesfm", "torch", "numpy", "pandas", "pyarrow", "huggingface_hub"]},
        retrospective=True, untouched_holdout=False,
        pretraining="Model released in 2026 and pretraining includes data after study; not an ex-ante 2021 deployment test")
    path = OUT / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text()) == protocol, "Changed protocol; use a new experiment directory")
    save_json(path, protocol)
    frames, checks = [], []
    started = time.perf_counter()
    for origin in ORIGINS:
        candidates = features.loc[features.origin.eq(origin) & features.eligible].sort_values("merchant_abn").reset_index(drop=True)
        ids = candidates.merchant_abn.to_numpy()
        contexts = past_contexts(wide, ids, origin)
        horizon = (origin + pd.offsets.MonthEnd(3) - origin).days
        output = OUT / "unscored" / (origin.strftime("%Y-%m-%d") + ".parquet")
        if output.exists():
            prediction = pd.read_parquet(output)
        else:
            chunks = []
            for offset in range(0, len(ids), 128):
                batch_ids = ids[offset:offset+128]
                q = predict_quantiles(model, contexts[offset:offset+128], horizon, batch_ids)
                mean_proxy, median_sum = quantile_totals(q)
                chunk = candidates.iloc[offset:offset+128][["merchant_abn"]].copy()
                chunk["mean_proxy"], chunk["median_sum"] = mean_proxy, median_sum
                chunks.append(chunk)
                # Save daily quantiles for traceability; they do not define joint portfolio risk.
                qpath = OUT / "daily_quantiles" / origin.strftime("%Y-%m-%d")
                qpath.mkdir(parents=True, exist_ok=True)
                qframe = pd.DataFrame(q.reshape(-1, 9), columns=[f"q{int(p*100):02}" for p in QUANTILES])
                qframe.insert(0, "day", np.tile(pd.date_range(origin + pd.Timedelta(days=1), periods=horizon), len(batch_ids)))
                qframe.insert(0, "merchant_abn", np.repeat(batch_ids, horizon))
                qframe.to_parquet(qpath / f"{offset:05}.parquet", index=False, compression=None)
                print(f"{origin.date()}: {min(offset+128,len(ids))}/{len(ids)} merchants; elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)
            new = pd.concat(chunks, ignore_index=True)
            require(np.array_equal(new.merchant_abn, ids), "Prediction order changed")
            prediction = make_predictions(candidates, dict(timesfm_mean_proxy=new.mean_proxy.to_numpy(),
                timesfm_daily_median_sum=new.median_sum.to_numpy(), seasonal_naive=seasonal_naive(contexts, horizon)))
            write_table(prediction, OUT / "unscored", origin.strftime("%Y-%m-%d"))
        require(prediction.groupby("method").selected.sum().eq(100).all(), "Invalid selection size")
        frames.append(prediction)
        # Perturb future data and verify exact history plus numerically identical repeated forecasts.
        altered = wide.copy()
        altered.loc[:, altered.columns > origin] *= 1000
        require(np.array_equal(contexts, past_contexts(altered, ids, origin)), "Future leakage in daily inputs")
        check_count = min(128, len(ids))
        repeated = predict_quantiles(model, contexts[:check_count], horizon, ids[:check_count])
        mean_proxy, median_sum = quantile_totals(repeated)
        for method, values in [("timesfm_mean_proxy", mean_proxy), ("timesfm_daily_median_sum", median_sum)]:
            saved = prediction.loc[prediction.method.eq(method)].set_index("merchant_abn").loc[ids[:check_count], "predicted_sales"]
            require(np.allclose(saved, values, rtol=1e-5, atol=1e-3), "Repeated real-data forecasts differ")
        checks.append(dict(origin=str(origin.date()), future_inputs_identical=True, repeat_forecasts_checked=check_count))
    new_predictions = pd.concat(frames, ignore_index=True)
    save_json(OUT / "forecast_freeze.json", dict(saved_at=pd.Timestamp.now(tz="UTC"), digest=digest(new_predictions)))
    scored = attach_outcomes(new_predictions, panel)
    all_predictions = pd.concat([previous, scored], ignore_index=True)
    shared = all_predictions.loc[all_predictions.method.isin(previous.method.unique()), previous.columns]
    pd.testing.assert_frame_equal(shared.reset_index(drop=True), previous.reset_index(drop=True))
    selection_checks = 0
    for _, group in all_predictions.groupby(["origin", "method"]):
        ordered = group.sort_values(["predicted_fees", "merchant_abn"], ascending=[False, True])
        require(group.merchant_abn.is_unique and ordered["rank"].tolist() == list(range(1, len(group) + 1)),
                "Independent ranking check failed")
        require(set(ordered.head(100).merchant_abn) == set(group.loc[group.selected, "merchant_abn"]),
                "Independent selection check failed")
        selection_checks += 1
    folds, changes = summarize(all_predictions)
    summary = folds.groupby("method", as_index=False).agg(mean_wape_pct=("wape_pct", "mean"),
        mean_bias_pct=("bias_pct", "mean"), mean_mae=("mae", "mean"), mean_deviance=("deviance", "mean"),
        mean_selected_fees=("selected_fees", "mean"), mean_selection_uplift_pct=("uplift_vs_repeat_pct", "mean"),
        fee_wins=("delta_vs_repeat", lambda x: int(x.gt(0).sum())))
    for (origin, method), selected in scored.loc[scored.selected].groupby(["origin", "method"]):
        future_months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        independent = panel.loc[panel.month.isin(future_months) & panel.merchant_abn.isin(selected.merchant_abn)]
        independent = independent.merge(merchants[["merchant_abn", "take_rate"]], on="merchant_abn", validate="many_to_one")
        require(np.isclose((independent.sales * independent.take_rate / 100).sum(), selected.actual_fees.sum(), rtol=1e-12),
                "Selected actual fees failed independent recomputation")
    for frame, name in [(all_predictions, "predictions"), (folds, "fold_metrics"), (changes, "selection_changes"), (summary, "summary")]:
        write_table(frame, OUT, name)
    save_json(OUT / "acceptance_checks.json", dict(source_daily_monthly_reconciliation=True,
        frozen_before_scoring=True, all_selections_unique_100=True, independent_realized_fees=True,
        unchanged_previous_baselines_verified=True, independent_rank_and_selection_checks=selection_checks,
        numerical_reproduction_and_future_perturbation=checks, seconds=time.perf_counter()-started,
        max_allocated_gpu_GB=torch.cuda.max_memory_allocated()/1024**3))
    report(summary, folds)
    print(summary.round(4).to_string(index=False), flush=True)


def report(summary, folds):
    indexed = summary.set_index("method")
    repeat = indexed.loc["repeat"]
    proxy = indexed.loc["timesfm_mean_proxy"]
    native = indexed.loc["timesfm_daily_median_sum"]
    wape = folds.pivot(index="origin", columns="method", values="wape_pct")
    wins = int(wape.timesfm_mean_proxy.lt(wape.repeat).sum())
    lines = ["# TimesFM 3.0 versus repeat-sales", "",
        f"**Result: this zero-shot setup does not justify replacing repeat-sales.** Across eight origins, "
        f"repeat-sales has mean fee WAPE {repeat.mean_wape_pct:.2f}%, versus {proxy.mean_wape_pct:.2f}% "
        f"for the TimesFM mean proxy and {native.mean_wape_pct:.2f}% for its daily-median sum. "
        f"The mean proxy improves WAPE in {wins}/8 periods, but its top-100 selection earns "
        f"{proxy.mean_selection_uplift_pct:+.2f}% versus repeat on average. The native daily-median sum earns "
        f"{native.mean_selection_uplift_pct:+.2f}%. Both earn less selected gross fees in all eight periods. "
        "This does not rule out different covariates or fine-tuning; neither was tested or chosen after scoring.", "",
        "Retrospective zero-shot daily forecasting on a local RTX 3050 4 GB, with no cloud infrastructure or paid API. "
        "The same eight origins, eligible merchants and next-three-calendar-month outcomes are used as in the previous comparison. "
        "The official checkpoint is frozen in protocol.json. There is no fine-tuning or outcome-driven parameter search.", "",
        "The native point forecast is a daily median. Summing daily medians does not produce an expected three-month total "
        "or necessarily its median. The primary mean proxy integrates the nine nonnegative, sorted quantiles, using linear "
        "interpolation and constant tails beyond p=.1 and .9. This is a bounded-tail approximation; extreme-loss/revenue tails "
        "are not identified. Neither output is automatically calibrated. Daily quantiles are not joint portfolio scenarios.", "",
        "| Method | Mean fee WAPE | Mean fee bias | Mean selected fees | Selection uplift vs repeat | Fee wins / 8 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | {r.mean_wape_pct:.3f}% | {r.mean_bias_pct:+.3f}% | A${r.mean_selected_fees:,.2f} | "
                     f"{r.mean_selection_uplift_pct:+.3f}% | {r.fee_wins}/8 |")
    lines += ["", "These are equal-origin averages; overlapping folds are not independent. WAPE measures errors in total "
        "future fees, not daily fit. Better point accuracy need not improve the selected 100. "
        "The prior hindsight ceiling remains relevant: repeat-sales already captures 99.82% of maximum fees on average. "
        "No 1% selection-gain threshold is imposed because it exceeds the available historical headroom. "
        "Daily TimesFM and the prior monthly models use different temporal resolutions; this compares complete forecasting "
        "methods and does not isolate model architecture alone.", "",
        "| Origin | Method | Fee WAPE | Fee bias | Selection uplift vs repeat | Top-100 overlap with repeat |",
        "|---|---|---:|---:|---:|---:|"]
    for r in folds.loc[folds.method.isin(["repeat", "timesfm_mean_proxy", "timesfm_daily_median_sum"])].itertuples():
        lines.append(f"| {r.origin:%Y-%m-%d} | {r.method} | {r.wape_pct:.3f}% | {r.bias_pct:+.3f}% | "
                     f"{r.uplift_vs_repeat_pct:+.3f}% | {r.overlap_with_repeat}/100 |")
    lines += ["", "The model was released in 2026 and its pretraining includes data later than the 2021-22 study. "
        "Therefore this is a retrospective transfer-learning benchmark, not proof of a strategy deployable in 2021. "
        "Exact pretraining independence from the unknown upstream synthetic merchant data cannot be established. "
        "All merchant outcomes were already examined in previous research; there is no untouched holdout. "
        "Take rates and categories are assumed historically known and stable. Gross fees exclude fraud/default and other costs.", "",
        "See code/TIMESFM_COMPARISON.md for reproduction. Generated data remain ignored and Parquet is uncompressed."]
    (OUT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "run"])
    args = parser.parse_args()
    {"prepare": prepare_daily, "smoke": smoke, "run": run}[args.stage]()
