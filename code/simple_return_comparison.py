"""Compare March-only, expanding mean and recent-three-month fee forecasts."""

import json

import numpy as np
import pandas as pd

from abs_momentum_comparison import require, write_table
from external_data_sources import save_json
from external_data_models import forecast_metrics
from forecast_comparison import ORIGINS, attach_outcomes, build_features, digest, make_predictions
from timesfm_comparison import OUT as TIMESFM_OUT, PREVIOUS, validated_inputs

OUT = TIMESFM_OUT.parent / "simple_return_comparison"


def sales_forecasts(panel, ids, origin):
    """All methods use past complete months; three-month monthly mean equals repeat."""
    history = panel.loc[panel.month.le(origin)]
    require(not history.duplicated(["merchant_abn", "month"]).any(), "Duplicate history")
    months = pd.date_range("2021-03-01", origin, freq="MS")
    wide = history.pivot(index="merchant_abn", columns="month", values="sales").reindex(index=ids, columns=months)
    require(wide.notna().all().all() and len(months) >= 3, "Incomplete history")
    return dict(repeat=wide.iloc[:, -3:].sum(axis=1).to_numpy(),
                first_month=3 * wide.iloc[:, 0].to_numpy(),
                expanding_mean=3 * wide.mean(axis=1).to_numpy())


def run():
    _, panel, merchants, audit = validated_inputs()
    features = build_features(panel, merchants)
    old = pd.read_parquet(PREVIOUS / "predictions.parquet")
    protocol = dict(origins=ORIGINS.strftime("%Y-%m-%d").tolist(), horizon="next three complete months",
        methods=dict(repeat="Sum latest three complete months", first_month="March 2021 sales multiplied by three",
                     expanding_mean="Mean of every complete month from March 2021 through cutoff, multiplied by three"),
        units="Predicted sales times fixed take rate / 100 = predicted gross fees",
        eligibility="Identical previous baseline pool at each origin; no future-activity filter",
        ranking="100 highest predicted fees; numeric ABN ascending for ties", retrospective=True,
        tuning=False, zero_sales_months_included_in_mean=True,
        inputs=dict(panel=digest(panel), merchants=digest(merchants), features=digest(features)), archives=audit["archives"])
    path = OUT / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text()) == protocol, "Changed protocol")
    save_json(path, protocol)
    frames = []
    for origin in ORIGINS:
        candidates = features.loc[features.origin.eq(origin) & features.eligible].sort_values("merchant_abn").reset_index(drop=True)
        forecasts = sales_forecasts(panel, candidates.merchant_abn, origin)
        changed = panel.copy()
        changed.loc[changed.month.gt(origin), "sales"] *= 1000
        perturbed = sales_forecasts(changed, candidates.merchant_abn, origin)
        require(all(np.array_equal(forecasts[k], perturbed[k]) for k in forecasts), "Future leakage")
        prediction = make_predictions(candidates, forecasts)
        repeat = prediction.loc[prediction.method.eq("repeat")].set_index("merchant_abn")
        saved = old.loc[old.origin.eq(origin) & old.method.eq("repeat")].set_index("merchant_abn").sort_index()
        require(repeat.index.equals(saved.index), "Candidate pool differs")
        require(np.allclose(repeat.predicted_fees, saved.predicted_fees, rtol=1e-12, atol=1e-7)
                and repeat["rank"].equals(saved["rank"]), "Repeat benchmark differs")
        write_table(prediction, OUT / "unscored", origin.strftime("%Y-%m-%d"))
        frames.append(prediction)
    frozen = pd.concat(frames, ignore_index=True)
    save_json(OUT / "forecast_freeze.json", dict(digest=digest(frozen), saved_at=pd.Timestamp.now(tz="UTC")))
    scored = attach_outcomes(frozen, panel)
    rows = []
    for origin, fold in scored.groupby("origin"):
        repeat = fold.loc[fold.method.eq("repeat")]
        repeat_fees = repeat.loc[repeat.selected, "actual_fees"].sum()
        repeat_ids = set(repeat.loc[repeat.selected, "merchant_abn"])
        for method, frame in fold.groupby("method"):
            chosen = frame.loc[frame.selected]
            ranked = frame.sort_values(["predicted_fees", "merchant_abn"], ascending=[False, True])
            require(frame.merchant_abn.is_unique and ranked["rank"].tolist() == list(range(1, len(frame)+1)), "Invalid ranking")
            require(set(ranked.head(100).merchant_abn) == set(chosen.merchant_abn), "Invalid selected set")
            future_months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3, freq="MS")
            direct = panel.loc[panel.month.isin(future_months) & panel.merchant_abn.isin(chosen.merchant_abn)]
            direct = direct.merge(merchants[["merchant_abn", "take_rate"]], on="merchant_abn", validate="many_to_one")
            fees = float(chosen.actual_fees.sum())
            require(np.isclose((direct.sales * direct.take_rate / 100).sum(), fees, rtol=1e-12), "Independent fees differ")
            rows.append(dict(origin=origin, method=method, selected_fees=fees,
                delta_vs_repeat=fees-repeat_fees, uplift_vs_repeat_pct=100*(fees/repeat_fees-1),
                overlap_with_repeat=len(repeat_ids & set(chosen.merchant_abn)),
                **forecast_metrics(frame.actual_fees, frame.predicted_fees)))
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby("method", as_index=False).agg(mean_wape_pct=("wape_pct", "mean"),
        mean_bias_pct=("bias_pct", "mean"), mean_mae=("mae", "mean"), mean_deviance=("deviance", "mean"),
        mean_selected_fees=("selected_fees", "mean"), mean_uplift_pct=("uplift_vs_repeat_pct", "mean"),
        mean_delta_dollars=("delta_vs_repeat", "mean"), mean_overlap=("overlap_with_repeat", "mean"))
    wide = metrics.pivot(index="origin", columns="method", values="wape_pct")
    summary["wape_wins_vs_repeat"] = summary.method.map(lambda m: int(wide[m].lt(wide.repeat).sum()))
    for frame, name in [(scored, "predictions"), (metrics, "fold_metrics"), (summary, "summary")]:
        write_table(frame, OUT, name)
    save_json(OUT / "acceptance_checks.json", dict(source_and_metadata_unchanged=True, past_only_mean=True,
        future_perturbation_passed_at_all_origins=True, identical_repeat_forecasts_and_ranks=True,
        independent_rank_and_top100_checks=24, independent_realized_fees=True))
    lines = ["# First month versus historical mean versus recent three months", "",
        "All three estimators are compared at the same eight cutoffs, with the same eligible merchants and "
        "next-three-complete-month outcomes. March-only predicts three times March 2021 sales. The expanding mean "
        "uses all complete months from March through the decision date, including zero-sales months, then multiplies "
        "by three. Repeat-sales sums the last three months; this is exactly the last-three-month monthly mean times three. "
        "All are converted to gross fees with the fixed take rate / 100. No future months enter an average.", "",
        "| Method | Mean fee WAPE | Mean fee bias | Mean selected fees | Fee difference vs repeat | WAPE wins / 8 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"| {r.method} | {r.mean_wape_pct:.3f}% | {r.mean_bias_pct:+.3f}% | A${r.mean_selected_fees:,.2f} | "
                     f"{r.mean_uplift_pct:+.3f}% | {r.wape_wins_vs_repeat}/8 |")
    lines += ["", "WAPE is absolute fee error divided by actual fees; table entries are equal-origin averages. "
        "Selection fees and revenue prediction accuracy are separate outcomes. The previous first-month experiment "
        "scored single future months using all 4,026 merchants; its WAPE cannot be directly compared with this "
        "three-month-horizon, eligibility-matched experiment.", "",
        "| Origin | Method | Fee WAPE | Fee bias | Selected-fee difference vs repeat | Overlap with repeat |",
        "|---|---|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        lines.append(f"| {r.origin:%Y-%m-%d} | {r.method} | {r.wape_pct:.3f}% | {r.bias_pct:+.3f}% | "
                     f"{r.uplift_vs_repeat_pct:+.3f}% | {r.overlap_with_repeat}/100 |")
    lines += ["", "These are retrospective descriptive results on synthetic data and previously examined outcomes. "
        "The eight periods overlap and are not independent; there is no fresh holdout or claim of statistical significance. "
        "Metadata and take rates are assumed historically known and stable. Gross fees exclude fraud/default and other costs. "
        "Checks validate source fingerprints, past-only inputs, future perturbations, benchmark reproduction, deterministic "
        "top-100 selections and independently recomputed realized fees.", "",
        "Reproduce: `python code/simple_return_comparison.py`. Outputs use ignored data directories and uncompressed Parquet."]
    (OUT / "findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(summary.round(5).to_string(index=False))
    print(metrics[["origin", "method", "wape_pct", "bias_pct", "uplift_vs_repeat_pct"]].round(4).to_string(index=False))


if __name__ == "__main__":
    run()
