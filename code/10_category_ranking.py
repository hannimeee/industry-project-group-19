"""Build five segment-specific top-100 merchant portfolio dashboards."""

from __future__ import annotations

import json
import multiprocessing
import os
import platform
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Keep each numerical worker single-threaded. Independent segments are run in
# parallel instead, which is more predictable on an Apple Silicon laptop.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "handoff"
FINAL = HANDOFF / "final"
INPUT = HANDOFF / "input_snapshot"
SEGMENTS_PATH = ROOT / "tables" / "curated" / "category_segments.parquet"

if str(HANDOFF) not in sys.path:
    sys.path.insert(0, str(HANDOFF))

from portfolio_optimizer.cli import optimize_snapshot
from portfolio_optimizer.data import build_snapshot, read_table
from portfolio_optimizer.export import merge_dashboard


RETENTIONS = [1.0, 0.995, 0.99, 0.98, 0.95]
COUNT = 100
WEIGHT = 0.5

SEGMENT_FILES = {
    "Technology, Digital & Communications": "tech_portfolio_dashboard",
    "Home, Garden & General Supplies": "home_portfolio_dashboard",
    "Fashion, Jewellery & Personal Care": "fashion_portfolio_dashboard",
    "Arts, Crafts, Gifts & Entertainment": "arts_portfolio_dashboard",
    "Automotive, Outdoor & Recreational Equipment": "auto_portfolio_dashboard",
}


def load_segment_categories() -> dict[str, list[str]]:
    """Load and validate the five previously defined category segments."""
    frame = pd.read_parquet(SEGMENTS_PATH)
    if list(frame.columns) != ["segments", "categories"]:
        raise ValueError("category_segments.parquet must contain segments and categories columns.")
    if frame["segments"].duplicated().any() or set(frame["segments"]) != set(SEGMENT_FILES):
        raise ValueError("The saved category segments do not match the required five segments.")

    result = {row.segments: list(row.categories) for row in frame.itertuples(index=False)}
    flattened = [category for categories in result.values() for category in categories]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A merchant category is assigned to more than one segment.")
    return result


def build_segment(segment: str, categories: list[str], output_stem: str) -> dict[str, object]:
    """Optimize all five retention levels for one restricted merchant segment."""
    merchants = read_table(INPUT / "merchants.parquet")
    weekly_fees = read_table(INPUT / "weekly_fees.parquet")
    customer_fees = read_table(INPUT / "customer_fees.parquet")
    metadata = json.loads((INPUT / "snapshot.json").read_text(encoding="utf-8"))

    missing_categories = sorted(set(categories).difference(merchants["category"].dropna().unique()))
    if missing_categories:
        raise ValueError(f"{segment} contains unknown categories: {missing_categories}")

    # Preserve the original eligibility rules and additionally restrict the
    # optimizer's complete candidate pool to this segment.
    merchants = merchants.copy()
    merchants["eligible"] = merchants["eligible"].astype(bool) & merchants["category"].isin(categories)
    eligible_count = int(merchants["eligible"].sum())
    if eligible_count < COUNT:
        raise ValueError(f"{segment} has only {eligible_count} eligible merchants; {COUNT} are required.")

    with threadpool_limits(limits=1):
        snapshot = build_snapshot(
            merchants,
            weekly_fees,
            customer_fees,
            metadata,
            count=COUNT,
        )
        with tempfile.TemporaryDirectory(prefix=f"{output_stem}_", dir=None) as temporary:
            metrics, selections = optimize_snapshot(
                snapshot,
                Path(temporary),
                RETENTIONS,
                WEIGHT,
            )
        dashboard = merge_dashboard(metrics, selections)

    csv_path = FINAL / f"{output_stem}.csv"
    parquet_path = FINAL / f"{output_stem}.parquet"
    dashboard.to_csv(csv_path, index=False)
    dashboard.to_parquet(parquet_path, index=False, compression=None)

    if len(dashboard) != COUNT * len(RETENTIONS):
        raise ValueError(f"Unexpected dashboard row count for {segment}: {len(dashboard)}")
    if set(dashboard["retention"]) != set(RETENTIONS):
        raise ValueError(f"Missing retention output for {segment}.")
    if not dashboard.groupby("retention")["merchant_abn"].nunique().eq(COUNT).all():
        raise ValueError(f"A {segment} portfolio does not contain 100 unique merchants.")
    if not dashboard["merchant_category"].isin(categories).all():
        raise ValueError(f"A merchant outside {segment} was selected.")

    return {
        "segment": segment,
        "eligible_merchants": eligible_count,
        "portfolios": len(RETENTIONS),
        "rows": len(dashboard),
        "csv": str(csv_path.relative_to(ROOT)),
        "parquet": str(parquet_path.relative_to(ROOT)),
    }


def main() -> None:
    """Run independent segment optimizations and print the generated outputs."""
    FINAL.mkdir(parents=True, exist_ok=True)
    segment_categories = load_segment_categories()

    # Two processes use the M-series performance cores without making five
    # simultaneous copies of the 3.3-million-row customer exposure table.
    workers = 2 if platform.machine().lower() in {"arm64", "aarch64"} else 1
    print(f"Running {len(SEGMENT_FILES)} segment jobs with {workers} parallel worker(s).", flush=True)

    jobs = [
        (segment, segment_categories[segment], output_stem)
        for segment, output_stem in SEGMENT_FILES.items()
    ]
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(build_segment, segment, categories, output_stem): segment
            for segment, categories, output_stem in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed {result['segment']}: {result['eligible_merchants']} eligible, "
                f"{result['rows']} output rows.",
                flush=True,
            )

    order = {segment: position for position, segment in enumerate(SEGMENT_FILES)}
    summary = pd.DataFrame(sorted(results, key=lambda row: order[row["segment"]]))
    print("\nGenerated segment dashboards:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
