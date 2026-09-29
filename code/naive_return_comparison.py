"""Fixed six-month, blended, exponential and annual-seasonal revenue baselines."""

import json

import numpy as np
import pandas as pd

from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from external_data_models import forecast_metrics
from forecast_comparison import ORIGINS, attach_outcomes, build_features, digest, make_predictions
from simple_return_comparison import OUT as PREVIOUS_SIMPLE, sales_forecasts
from timesfm_comparison import validated_inputs

OUT = PREVIOUS_SIMPLE.parent / "naive_return_comparison"
HALFLIFE = 3.0
LABELS = dict(repeat="Recent 3 months", first_month="March only", expanding_mean="All-history mean",
              six_month_mean="Six-month mean", blend_50_50="50/50 blend", ewma_3m="Exponentially weighted mean",
              seasonal_naive_annual="Same months last year")


def naive_forecasts(panel, ids, origin):
    """Predict three-month totals; never read outcomes to select weights or windows."""
    forecasts = sales_forecasts(panel, ids, origin)
    history = panel.loc[panel.month.le(origin)]
    months = pd.date_range("2021-03-01", origin, freq="MS")
    wide = history.pivot(index="merchant_abn", columns="month", values="sales").reindex(index=ids, columns=months)
    require(len(months) >= 6 and wide.notna().all().all(), "Incomplete six-month history")
    forecasts["six_month_mean"] = 3 * wide.iloc[:, -6:].mean(axis=1).to_numpy()
    forecasts["blend_50_50"] = .5 * forecasts["repeat"] + .5 * forecasts["expanding_mean"]
    weights = np.exp2(-np.arange(len(months)-1, -1, -1) / HALFLIFE)
    forecasts["ewma_3m"] = 3 * (wide.to_numpy() @ weights) / weights.sum()
    prior_year = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3, freq="MS") - pd.DateOffset(years=1)
    if set(prior_year).issubset(set(months)):
        forecasts["seasonal_naive_annual"] = wide.loc[:, prior_year].sum(axis=1).to_numpy()
    return forecasts


def summarize(metrics):
    rows = []
    for method, group in metrics.groupby("method"):
        controls = metrics.loc[metrics.method.isin(["repeat", "expanding_mean"]) & metrics.origin.isin(group.origin)]
        controls = controls.pivot(index="origin", columns="method", values="wape_pct")
        errors = group.set_index("origin").wape_pct
        rows.append(dict(method=method, origins=len(group), mean_wape_pct=float(group.wape_pct.mean()),
            mean_bias_pct=float(group.bias_pct.mean()), mean_absolute_bias_pct=float(group.bias_pct.abs().mean()),
            mean_mae=float(group.mae.mean()), mean_deviance=float(group.deviance.mean()),
            mean_selected_fees=float(group.selected_fees.mean()),
            mean_fee_uplift_vs_repeat_pct=float(group.uplift_vs_repeat_pct.mean()),
            mean_fee_uplift_vs_mean_pct=float(group.uplift_vs_mean_pct.mean()),
            mean_fee_delta_vs_repeat=float(group.delta_vs_repeat.mean()),
            mean_overlap_with_mean=float(group.overlap_with_mean.mean()),
            wape_wins_vs_repeat=int(errors.lt(controls.repeat).sum()),
            wape_wins_vs_mean=int(errors.lt(controls.expanding_mean).sum())))
    return pd.DataFrame(rows).sort_values("mean_wape_pct").reset_index(drop=True)


def run():
    _, panel, merchants, audit = validated_inputs()
    features = build_features(panel, merchants)
    previous = pd.read_parquet(PREVIOUS_SIMPLE / "predictions.parquet")
    protocol = dict(origins=ORIGINS.strftime("%Y-%m-%d").tolist(), horizon="next three complete calendar months",
        new_rules=dict(six_month_mean="Latest six monthly sales averaged, multiplied by three",
            blend_50_50="Equal blend of repeat-sales and expanding-mean three-month forecasts",
            ewma_3m="All past monthly sales weighted by 2^(-age_in_months/3), normalized, then multiplied by three",
            seasonal_naive_annual="Sum the same three target calendar months from the previous year"),
        exponential_half_life_months=HALFLIFE, blend_weights=[.5,.5], tuning=False,
        seasonal_missing="Disable at cutoffs without all three complete prior-year months; never fill partial months",
        eligibility="Same existing valid-rate and positive-sales-in-both-prior-three-month-windows rule",
        selection="Exactly 100; predicted fees descending then numeric ABN ascending", fees="sales * take_rate / 100",
        zero_sales_months_included=True, retrospective=True, untouched_holdout=False,
        source_archives=audit["archives"], inputs=dict(panel=digest(panel), merchants=digest(merchants), features=digest(features)))
    path = OUT / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text()) == protocol, "Protocol changed")
    save_json(path, protocol)
    frames, availability = [], []
    for origin in ORIGINS:
        candidates = features.loc[features.origin.eq(origin) & features.eligible].sort_values("merchant_abn").reset_index(drop=True)
        forecasts = naive_forecasts(panel, candidates.merchant_abn, origin)
        changed = panel.copy()
        changed.loc[changed.month.gt(origin), "sales"] = changed.loc[changed.month.gt(origin), "sales"] * 1000 + 1
        perturbed = naive_forecasts(changed, candidates.merchant_abn, origin)
        require(forecasts.keys() == perturbed.keys() and all(np.array_equal(forecasts[k], perturbed[k]) for k in forecasts),
                "Future perturbation changed forecasts")
        prediction = make_predictions(candidates, forecasts)
        saved = previous.loc[previous.origin.eq(origin)]
        new = prediction.loc[prediction.method.isin(saved.method.unique())]
        cols = ["origin", "method", "merchant_abn", "predicted_sales", "predicted_fees", "rank", "selected"]
        pd.testing.assert_frame_equal(new[cols].reset_index(drop=True), saved[cols].reset_index(drop=True))
        write_table(prediction, OUT / "unscored", origin.strftime("%Y-%m-%d"))
        frames.append(prediction)
        for method in LABELS:
            availability.append(dict(origin=origin, method=method, available=method in forecasts,
                reason="available" if method in forecasts else "Incomplete prior-year three-month window"))
    frozen = pd.concat(frames, ignore_index=True)
    save_json(OUT / "forecast_freeze.json", dict(saved_at=pd.Timestamp.now(tz="UTC"), digest=digest(frozen)))
    scored = attach_outcomes(frozen, panel)
    records = []
    for origin, fold in scored.groupby("origin"):
        base_fees = {m: float(fold.loc[fold.method.eq(m) & fold.selected, "actual_fees"].sum())
                     for m in ["repeat", "expanding_mean"]}
        mean_ids = set(fold.loc[fold.method.eq("expanding_mean") & fold.selected, "merchant_abn"])
        candidate_pool = set(fold.loc[fold.method.eq("repeat"), "merchant_abn"])
        for method, frame in fold.groupby("method"):
            ordered = frame.sort_values(["predicted_fees", "merchant_abn"], ascending=[False, True])
            chosen = frame.loc[frame.selected]
            require(frame.merchant_abn.is_unique and set(frame.merchant_abn) == candidate_pool, "Changed candidate pool")
            require(ordered["rank"].tolist() == list(range(1,len(frame)+1))
                    and chosen.merchant_abn.nunique() == 100
                    and set(chosen.merchant_abn) == set(ordered.head(100).merchant_abn), "Invalid selection")
            months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3, freq="MS")
            direct = panel.loc[panel.month.isin(months) & panel.merchant_abn.isin(chosen.merchant_abn)]
            direct = direct.merge(merchants[["merchant_abn", "take_rate"]], on="merchant_abn", validate="many_to_one")
            fees = float(chosen.actual_fees.sum())
            require(len(direct) == 300 and np.isclose((direct.sales * direct.take_rate / 100).sum(), fees, rtol=1e-12),
                    "Independent realized fees differ")
            records.append(dict(origin=origin, method=method, selected_fees=fees,
                delta_vs_repeat=fees-base_fees["repeat"],
                uplift_vs_repeat_pct=100*(fees/base_fees["repeat"]-1),
                uplift_vs_mean_pct=100*(fees/base_fees["expanding_mean"]-1),
                overlap_with_mean=len(mean_ids & set(chosen.merchant_abn)),
                **forecast_metrics(frame.actual_fees, frame.predicted_fees)))
    metrics = pd.DataFrame(records)
    seasonal_origins = metrics.loc[metrics.method.eq("seasonal_naive_annual"), "origin"]
    complete = summarize(metrics.loc[~metrics.method.eq("seasonal_naive_annual")])
    common = summarize(metrics.loc[metrics.origin.isin(seasonal_origins)])
    schedules = []
    for schedule in range(3):
        data = metrics.loc[metrics.origin.isin(ORIGINS[schedule::3])]
        schedules.append(summarize(data).assign(schedule=schedule))
    for frame, name in [(scored, "predictions"), (metrics, "fold_metrics"), (pd.DataFrame(availability), "availability"),
                         (complete, "summary_all_eight"), (common, "summary_seasonal_common_five"),
                         (pd.concat(schedules), "nonoverlapping_schedules")]:
        write_table(frame, OUT, name)
    save_json(OUT / "acceptance_checks.json", dict(source_and_metadata_unchanged=True,
        reproduced_existing_baseline_forecasts_and_selections=True, future_perturbation_passed_all_origins=True,
        deterministic_top100_and_pool_checks=len(metrics), independent_realized_fee_checks=len(metrics),
        seasonal_origins=seasonal_origins.dt.strftime("%Y-%m-%d").tolist(), frozen_before_scoring=True))
    write_report(complete, common, metrics)
    print("All eight origins:", flush=True)
    print(complete.round(5).to_string(index=False), flush=True)
    print("Same five origins with prior-year history:", flush=True)
    print(common.round(5).to_string(index=False), flush=True)


def write_report(complete, common, metrics):
    lines = ["# Four additional simple revenue forecasts", "",
        "The six-month average, 50/50 blend, exponentially weighted average and annual seasonal naive were specified "
        "before scoring. The exponential half-life is fixed at three months with normalized weights over all available "
        "past months; no decay-rate search was run. Annual seasonal naive requires the exact three target calendar "
        "months from the preceding year. All forecasts retain zero-sales months, predict three-month sales totals "
        "and convert to gross fees using take rate / 100.", ""]
    for title, table in [("All eight cutoffs: November 2021-June 2022", complete),
                          ("Matched seasonal comparison: February-June 2022 (five cutoffs)", common)]:
        lines += ["## " + title, "", "| Method | Mean fee WAPE | Mean fee bias | Fee uplift vs all-history mean | WAPE wins vs mean |",
                  "|---|---:|---:|---:|---:|"]
        for r in table.itertuples():
            lines.append(f"| {LABELS[r.method]} | {r.mean_wape_pct:.3f}% | {r.mean_bias_pct:+.3f}% | "
                         f"{r.mean_fee_uplift_vs_mean_pct:+.3f}% | {r.wape_wins_vs_mean}/{r.origins} |")
    lines += ["", "WAPE is absolute forecast fee error divided by actual fees. These summaries weight each cutoff "
        "equally. Do not compare a five-cutoff seasonal result with another method's eight-cutoff average. "
        "Revenue forecast errors and actual selected-100 fees answer different questions. The baseline methods "
        "exactly reproduce their earlier forecasts and selections; all methods use the same eligible merchants per cutoff.", "",
        "## Every cutoff", "", "| Origin | Method | WAPE | Bias | Fee uplift vs historical mean | Overlap with historical mean |",
        "|---|---|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {r.origin:%Y-%m-%d} | {LABELS[r.method]} | {r.wape_pct:.3f}% | {r.bias_pct:+.3f}% | "
                     f"{r.uplift_vs_mean_pct:+.3f}% | {r.overlap_with_mean}/100 |")
    lines += ["", "The first three seasonal comparisons are unavailable because the source begins with a partial "
        "February 2021 and complete March 2021; no missing prior-year months are imputed. The five later windows still "
        "overlap and reflect only one prior annual cycle. Non-overlapping schedules are exported separately. "
        "All outcomes have previously been examined: these are retrospective development comparisons, with no untouched "
        "holdout or statistical-significance claim. Metadata and take rates are assumed historically known and stable. "
        "Historical recorded sales represent hypothetical selection outcomes; gross fees exclude fraud/default and other costs.", "",
        "Reproduce: `python code/naive_return_comparison.py`. Outputs are ignored CSV and uncompressed Parquet under "
        "`tables/analysis/naive_return_comparison/`. Checks verify original inputs, baseline reproduction, future perturbations, "
        "deterministic eligible top-100 selections and independently recomputed realized fees."]
    (OUT / "findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    run()
