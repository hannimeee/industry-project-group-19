"""Measure hindsight selection headroom; this is a diagnostic, never a forecast."""

from pathlib import Path

import numpy as np
import pandas as pd

from abs_momentum_comparison import require, write_table
from external_data_sources import save_json

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "tables/analysis/external_data_comparison"
FORECASTS = ROOT / "tables/analysis/forecast_comparison"
OUT = ROOT / "tables/analysis/selection_headroom"


def run():
    panel = pd.read_parquet(PREPARED / "merchant_month.parquet")
    merchants = pd.read_parquet(PREPARED / "merchants.parquet")
    predictions = pd.read_parquet(FORECASTS / "predictions.parquet")
    fold_rows, ceilings, selected_sets, oracle_sets, independent = [], [], [], [], []
    for origin, all_methods in predictions.groupby("origin"):
        base = all_methods.loc[all_methods.method.eq("repeat")].copy()
        require(base.merchant_abn.is_unique, "Duplicate merchant in candidate pool")
        # Recompute realized fees independently from the audited monthly panel.
        months = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=3, freq="MS")
        future = panel.loc[panel.month.isin(months) & panel.merchant_abn.isin(base.merchant_abn)]
        require(len(future) == 3 * len(base), "Incomplete outcome window")
        actual = future.groupby("merchant_abn", as_index=False).sales.sum().merge(
            merchants[["merchant_abn", "take_rate"]], on="merchant_abn", validate="one_to_one")
        actual["recomputed_fees"] = actual.sales * actual.take_rate / 100
        base = base.merge(actual[["merchant_abn", "recomputed_fees"]], on="merchant_abn", validate="one_to_one")
        require(np.allclose(base.actual_fees, base.recomputed_fees, rtol=1e-12, atol=1e-8),
                "Saved and independently recomputed outcomes differ")
        oracle = base.sort_values(["recomputed_fees", "merchant_abn"], ascending=[False, True]).head(100)
        require(oracle.merchant_abn.nunique() == 100, "Invalid hindsight selection")
        chosen = base.loc[base.selected]
        best, obtained = float(oracle.recomputed_fees.sum()), float(chosen.recomputed_fees.sum())
        require(best >= obtained - 1e-6, "Hindsight result below attainable benchmark")
        actual_ids, selected_ids = set(oracle.merchant_abn), set(chosen.merchant_abn)
        selected_sets.append(selected_ids)
        oracle_sets.append(actual_ids)
        cutoff = float(base.loc[base["rank"].eq(100), "predicted_fees"].iloc[0])
        fold_rows.append(dict(origin=origin, eligible_merchants=len(base), repeat_fees=obtained,
                              hindsight_best_fees=best, headroom_dollars=best - obtained,
                              maximum_uplift_pct=(best / obtained - 1) * 100,
                              captured_hindsight_fees_pct=obtained / best * 100,
                              overlap_with_hindsight=len(actual_ids & selected_ids),
                              selected_pool_share_pct=len(chosen) / len(base) * 100,
                              selected_fee_share_pct=obtained / base.recomputed_fees.sum() * 100,
                              merchants_within_10pct_of_cutoff=int(base.predicted_fees.between(cutoff * .9, cutoff * 1.1).sum()),
                              fee_rank_spearman=float(base.predicted_fees.corr(base.recomputed_fees, method="spearman"))))
        for method, f in all_methods.groupby("method"):
            selected = f.loc[f.selected, "actual_fees"].sum()
            require(selected <= best + 1e-6, "Saved selection exceeds independent hindsight maximum")
            ceilings.append(dict(origin=origin, method=method, selected_fees=selected,
                                 maximum_possible_uplift_pct=(best / selected - 1) * 100))
        independent.append(oracle.assign(origin=origin, hindsight_only=True))
    folds = pd.DataFrame(fold_rows)
    bounds = pd.DataFrame(ceilings)
    overview = dict(origins=len(folds), retrospective=True, hindsight_not_deployable=True,
                    mean_capture_pct=float(folds.captured_hindsight_fees_pct.mean()),
                    mean_maximum_uplift_pct=float(folds.maximum_uplift_pct.mean()),
                    minimum_maximum_uplift_pct=float(folds.maximum_uplift_pct.min()),
                    maximum_maximum_uplift_pct=float(folds.maximum_uplift_pct.max()),
                    mean_headroom_dollars=float(folds.headroom_dollars.mean()),
                    repeat_merchants_selected_at_every_origin=len(set.intersection(*selected_sets)),
                    repeat_union_of_selected_merchants=len(set.union(*selected_sets)),
                    hindsight_merchants_selected_at_every_origin=len(set.intersection(*oracle_sets)),
                    mean_selected_pool_share_pct=float(folds.selected_pool_share_pct.mean()),
                    mean_selected_fee_share_pct=float(folds.selected_fee_share_pct.mean()),
                    one_percent_uplift_possible_in_any_fold=bool(folds.maximum_uplift_pct.ge(1).any()),
                    one_percent_uplift_vs_minimal_possible_in_any_fold=bool(
                        bounds.loc[bounds.method.eq("minimal"), "maximum_possible_uplift_pct"].ge(1).any()),
                    independent_outcomes_verified=True)
    write_table(folds, OUT, "fold_headroom")
    write_table(bounds, OUT, "method_headroom")
    write_table(pd.concat(independent, ignore_index=True), OUT, "hindsight_selections")
    save_json(OUT / "summary.json", overview)
    lines = ["# Why forecasting barely changes merchant selection", "",
             "This diagnostic independently recomputes actual fees for every eligible merchant from the audited monthly panel. "
             "For each historical date, it compares repeat sales with the 100 merchants that would have earned the most "
             "over the following three months. This hindsight selection uses future outcomes and is not a deployable method.", "",
             f"Repeat sales captures **{overview['mean_capture_pct']:.2f}%** of the hindsight maximum on average. "
             f"Even perfect foresight can improve fees by only **{overview['minimum_maximum_uplift_pct']:.3f}% to "
             f"{overview['maximum_maximum_uplift_pct']:.3f}%** across these eight periods. "
             "Therefore a 1% improvement requirement against repeat sales is unattainable in every observed fold. "
             "It is also unattainable against the original minimal regression in every fold. "
             "The earlier 1% screens should not be read as evidence that useful features cannot exist: their threshold exceeded "
             "the available improvement for this particular top-100 gross-fee objective.", "",
             "| Selection date | Repeat fees | Hindsight maximum | Maximum possible gain | Shared merchants |",
             "|---|---:|---:|---:|---:|"]
    for r in folds.itertuples():
        lines.append(f"| {r.origin:%Y-%m-%d} | A${r.repeat_fees:,.2f} | A${r.hindsight_best_fees:,.2f} | "
                     f"{r.maximum_uplift_pct:.3f}% | {r.overlap_with_hindsight}/100 |")
    lines += ["", "## Observed structure", "",
              f"- {overview['repeat_merchants_selected_at_every_origin']} merchants appear in every repeat-sales selection; "
              f"only {overview['repeat_union_of_selected_merchants']} different merchants appear across all eight selections.",
              f"- The selected 100 represent approximately {overview['mean_selected_pool_share_pct']:.2f}% of eligible merchants "
              f"but generate {overview['mean_selected_fee_share_pct']:.2f}% of their subsequent gross fees on average.",
              f"- Mean remaining dollar opportunity is A${overview['mean_headroom_dollars']:,.2f} per three-month selection. "
              "Whether that opportunity is economically worthwhile depends on implementation costs, which are unobserved.",
              "- An accurate forecast of each merchant's fee level and an accurate ranking are different objectives. "
              "A common forecast bias can leave the ranking unchanged. This is a possible explanation; the ceiling itself "
              "does not identify the data-generating process or establish a causal explanation.", "",
              "## What this changes", "",
              "Keep the prior experiments as executed, but correct the interpretation of their practical-success thresholds. "
              "Do not lower thresholds after observing outcomes and describe the result as a fresh confirmation. "
              "Use the hindsight gap as a descriptive ranking diagnostic, alongside absolute fees and forecast errors. "
              "The gap can be tiny, so percentage-of-gap metrics alone can exaggerate small dollar changes.", "",
              "A substantive next question under the portfolio objective is whether customer concentration, shared customer "
              "exposure or revenue uncertainty can be reduced at a small, explicitly measured fee cost. This requires "
              "separate customer-level analysis and risk validation; no benefit is established by the present calculation. "
              "Risk preferences must reflect a stated business objective, not be selected just to make a complex method win.", "",
              "All results use the same historically eligible pool, fixed take rates and three-month horizon. Gross fees exclude "
              "funding costs, defaults and operating costs. Data are synthetic; periods overlap and have already been examined. "
              "These exact retrospective accounting results do not prove future rankings will be stable or that any method "
              "is statistically superior."]
    (OUT / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(folds.round(4).to_string(index=False))
    print(overview)


if __name__ == "__main__":
    run()
