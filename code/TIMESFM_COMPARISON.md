# Local TimesFM 3.0 benchmark

This tests a pretrained forecasting model on daily merchant sales, using the existing eight cutoffs from November 2021 through June 2022 and forecasting the next three complete calendar months (89-92 days). It requires no cloud deployment or API key. Run from the repository root with the isolated environment `.venv/timesfm`, which inherits the existing audited experiment dependencies and CUDA-enabled PyTorch 2.6.0+cu124. TimesFM 3.0.1 was installed into that environment without changing system packages.

```powershell
& '.venv/timesfm/Scripts/python.exe' code/timesfm_comparison.py prepare
& '.venv/timesfm/Scripts/python.exe' code/timesfm_comparison.py smoke
& '.venv/timesfm/Scripts/python.exe' code/timesfm_comparison.py run
python -m pytest -p no:capture code/tests/test_timesfm_comparison.py -q
```

For a new machine, install the pinned packages in `requirements-timesfm.txt` into a virtual environment and install a compatible CUDA build of PyTorch if using a GPU. The tested machine has an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM. `smoke` verifies local execution and numerical reproduction before the full experiment. The code intentionally requires CUDA for this run; it does not silently switch to a slow CPU or a paid service.

Download the official `google/timesfm-3.0-pytorch` checkpoint at revision `43046b85ec22d584a13f8098c2ed39c889e129c2` to `tables/landing/timesfm3/checkpoint/`. Required files are `config.json` and `model.safetensors`; retain `LICENSE` and `README.md`. The checkpoint is approximately 1.32 GB. The initial download was verified against Hugging Face's published LFS SHA-256; the run records the actual hash. Its weights have a non-commercial, non-production license; this experiment is academic. Long workspace paths can exceed Windows download-cache limits, so the original download used direct official Hugging Face resolve URLs rather than the default nested cache path.

## Frozen methods

- **TimesFM mean proxy (primary candidate):** use all daily history from March 2021 through the cutoff, independently per merchant, with no fine-tuning or added covariates. Request the nine quantiles at probabilities .1-.9. Sort quantiles and clip negative sales to zero using the official API. Approximate the mean by integrating the piecewise-linear quantile curve and holding the endpoints constant outside .1-.9: weights .15 on the two endpoint quantiles and .1 on each interior quantile. Sum the resulting daily mean proxies and multiply by take rate / 100. These tail assumptions are explicit; this is not a claim of an exactly estimated or calibrated expected return.
- **TimesFM daily-median sum (diagnostic):** sum its native daily point forecasts. The native point is a median, not a mean, and summing daily medians does not generally equal the median of the total. Report this to distinguish model output from the mean approximation, not to select a winning postprocessing rule after scoring.
- **Weekly seasonal naive:** repeat the last seven observed daily sales for the exact forecast horizon and convert to fees.
- Reuse the prior **repeat-sales, Minimal + own growth, pooled lags and damped Holt** predictions unchanged on identical eligible pools.

Settings are the checkpoint defaults, float32 inference, batch size 32, no symmetric averaging, no fine-tuning, seed 2026. Batch size was chosen on synthetic input for hardware throughput/memory, before merchant outcomes were scored; the hardware check is saved separately. The effective loaded model configuration, software versions and hashes are saved before forecasting. All methods select exactly 100 merchants, breaking ties by numeric ABN. The complete three-month horizon is scored, not just the accuracy of individual daily forecasts. Inputs are extracted by cutoff before calling the model, and each forecast is saved before new outcomes are attached. Daily quantiles are exported for audit; they are not joint samples and cannot directly establish three-month portfolio variance or covariance.

## Evaluation and limits

Report three-month fee WAPE, MAE, aggregate bias and Tweedie deviance, plus realized selected fees, overlap and changed selections. Summaries give equal weight to each origin; overlapping windows are not independent and their fees must not be added as separate economic earnings. The earlier 1% fee-uplift requirement is inapplicable because perfect hindsight cannot attain it against repeat-sales on these folds. Retain a distinction between improved forecast amounts and improved top-100 selection.

The model was released in 2026 and pretraining contains data after 2022. This is a retrospective transfer-learning experiment, not a simulation of technology available in 2021. The unknown original source of the synthetic merchant data prevents establishing exact independence from pretraining. Outcomes have already been examined in earlier experiments, so none is a new holdout. Fixed take rates and categories are assumed available throughout. Gross fees exclude fraud losses, defaults and operating/funding costs.

## Data checks and outputs

The preparation command verifies original archive size/timestamp signatures, reparses merchant metadata, and checks the previous monthly-panel fingerprint. It incrementally aggregates transaction partitions, reconciles every source partition's transaction count and sales against the existing audit, verifies every required calendar day, and reconciles all merchant-month sums. Only then are transaction-free days filled with zero. Unknown-metadata merchants remain outside selection. Existing global order-ID checks are reused for the unchanged archives.

Generated files live under ignored `tables/analysis/timesfm_comparison/`: the daily panel, protocol, daily audit, smoke results, daily quantiles, unscored forecasts, scored predictions, fold metrics, summary, selection changes and findings. Parquet output is uncompressed. Tests check the quantile integral by hand, calendar horizons, weekly-naive calculation, missing inputs and future-data exclusion. Each real fold also checks future perturbation, repeated forecasts, 100 selections and independently recomputed realized fees.

References: [Google's TimesFM-3 research announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [official source](https://github.com/google-research/timesfm), [official checkpoint and license](https://huggingface.co/google/timesfm-3.0-pytorch). Google reports leading average ranks on GIFT-Eval, FEV-Bench and TIME; this is a reason to test the model, not proof it is best on these merchants. At setup, OpenRouter did not list TimesFM or Chronos; the inspected TSFM.ai catalog offered TimesFM 2.0/2.5 but not 3.0, and the Hugging Face 3.0 card listed no hosted inference provider.
