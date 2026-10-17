"""Run customer_concentration stages with one numerical worker.

Example: python code/scripts/customer_concentration_serial.py all
Accepts the same stages as customer_concentration.py. Estimators, search budgets,
inputs and frozen scientific settings are unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

# Much tighter than the optimizer's 0.1% gap or any economic decision threshold.
ROUNDING_RTOL = 1e-12
ROUNDING_ATOL = 1e-15
# The original run used 20-thread BLAS reductions outside optimize(), which
# explicitly limits its own numerical loop to one thread. Changing the outer
# reduction order can alter cutting-plane paths even when selections agree.
ORIGINAL_BLAS_THREADS = 20
# These two original cases were completed by the first serial recovery run.
# Evidence: serial_oracle.log; the final-oracle diagnostic reproduces the
# one-thread objective and initial solver trace without changing tolerances.
ORIGINAL_SERIAL_ORACLES = {'2022-05-31': .95, '2022-06-30': .95}


def reproduction_threads():
    """Restore original reduction ordering; the Windows job still caps resources."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=ORIGINAL_BLAS_THREADS, user_api='blas')


def install_original_oracle_threads(study):
    """Preserve the recorded execution settings of the two recovered cases."""
    from contextvars import ContextVar
    from functools import wraps
    from threadpoolctl import threadpool_limits

    if getattr(study, '_original_oracle_threads_installed', False):
        return
    current_date = ContextVar('original_oracle_date', default=None)
    original_stage, original_optimize = study.oracle_stage, study.optimize

    @wraps(original_stage)
    def oracle_stage(verify_only=False, only_date=None):
        token = current_date.set(str(only_date)[:10] if only_date is not None else None)
        try:
            return original_stage(verify_only=verify_only, only_date=only_date)
        finally:
            current_date.reset(token)

    @wraps(original_optimize)
    def optimize(*args, **kwargs):
        retention = ORIGINAL_SERIAL_ORACLES.get(current_date.get())
        if (retention is not None and kwargs.get('retention') == retention
                and kwargs.get('actual_floor') is not None):
            with threadpool_limits(limits=1):
                return original_optimize(*args, **kwargs)
        return original_optimize(*args, **kwargs)

    study.oracle_stage, study.optimize = oracle_stage, optimize
    study._original_oracle_threads_installed = True


def assert_reproduced_frame(left, right, exact_assert=None):
    """Require exact structure/identities and numerical equality of float results."""
    import numpy as np
    import pandas as pd

    exact_assert = exact_assert or pd.testing.assert_frame_equal
    try:
        exact_assert(left, right, check_exact=True)
        return []
    except AssertionError:
        pass
    # Preserve schema, dtypes, row order, identifiers, decisions and missingness.
    exact_assert(left.iloc[:0], right.iloc[:0], check_exact=True)
    exact_names = {'merchant_abn', 'as_of', 'window', 'method', 'retention', 'k', 'severity',
                   'probability', 'replicate', 'draw', 'iteration', 'iterations', 'budget_tier'}
    floating = [name for name in left if pd.api.types.is_float_dtype(left[name].dtype)
                and name not in exact_names and not str(name).endswith('_id')
                and 'rank' not in str(name)]
    exact_assert(left.drop(columns=floating), right.drop(columns=floating), check_exact=True)
    differences = []
    for name in floating:
        a = left[name].to_numpy(dtype=float, na_value=np.nan)
        b = right[name].to_numpy(dtype=float, na_value=np.nan)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a, b, rtol=ROUNDING_RTOL, atol=ROUNDING_ATOL, equal_nan=True,
                                   err_msg=f'Non-rounding difference in {name}')
        changed = np.isfinite(a) & np.isfinite(b) & (a != b)
        if changed.any():
            absolute = np.abs(a[changed]-b[changed])
            denominator = np.maximum(np.abs(a[changed]), np.abs(b[changed]))
            differences.append(dict(column=name, changed=int(changed.sum()),
                                    maximum_absolute_difference=float(absolute.max()),
                                    maximum_relative_difference=float((absolute/denominator).max())))
    return differences


def install_reproduction_checks(study, log_path):
    """Compare reruns without rewriting frozen outputs when only rounding differs."""
    import datetime
    import inspect
    import pandas as pd

    log_path = Path(log_path)
    original_assert = pd.testing.assert_frame_equal
    original_save = study.save_checked

    def record(event):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, default=str)+'\n')

    record(dict(event='comparison_policy', rtol=ROUNDING_RTOL, atol=ROUNDING_ATOL,
                non_float_fields='exact', frozen_outputs='preserved',
                original_blas_threads=ORIGINAL_BLAS_THREADS, optimizer_threads=1,
                original_serial_oracles=ORIGINAL_SERIAL_ORACLES,
                scientific_source_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))

    def compare(left, right, *args, **kwargs):
        if args or kwargs != {'check_exact': True}:
            return original_assert(left, right, *args, **kwargs)
        try:
            differences = assert_reproduced_frame(left, right, original_assert)
        except AssertionError:
            # Keep the already-computed group for diagnosis if any substantive
            # mismatch occurs; do not replace the frozen scientific artifacts.
            context = inspect.currentframe().f_back.f_locals
            group = context.get('tables', context.get('t', {}))
            folder = log_path.parent / 'verification_forensics' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            folder.mkdir(parents=True, exist_ok=True)
            if isinstance(group, dict):
                for name, frame in group.items():
                    if isinstance(frame, pd.DataFrame):
                        frame.to_parquet(folder/(name+'.parquet'), index=False, compression=None)
            record(dict(event='mismatch', forensic_directory=str(folder)))
            raise
        if differences:
            record(dict(event='rounding_equivalent', rows=len(left), differences=differences))

    def save_preserving_frozen(tables, stage, directory=None):
        directory = study.OUT if directory is None else Path(directory)
        marker = directory/(stage+'_checks.json')
        if not marker.exists():
            return original_save(tables, stage, directory)
        saved = json.loads(marker.read_text(encoding='utf-8'))['digests']
        if set(saved) != set(tables):
            raise AssertionError('Recomputed table names changed')
        for name, frame in tables.items():
            existing = pd.read_parquet(directory/(name+'.parquet'))
            if study.digest(existing) != saved[name]:
                # The original audit hashes the in-memory frame before writing
                # index=False Parquet. Parquet can normalize object-bool columns
                # and discard a non-default index. Recover that original schema
                # only if its exact original digest AND lossless round trip pass.
                import pyarrow as pa
                if list(frame.columns) != list(existing.columns) or len(frame) != len(existing):
                    raise AssertionError('Frozen table checksum changed: '+name)
                try:
                    restored = existing.astype(frame.dtypes.to_dict())
                    restored.index = frame.index.copy()
                    if study.digest(restored) != saved[name]:
                        raise AssertionError('Frozen table checksum changed: '+name)
                    serialized = pa.Table.from_pandas(restored, preserve_index=False).to_pandas()
                    original_assert(serialized, existing, check_exact=True)
                except (TypeError, ValueError) as exc:
                    raise AssertionError('Frozen table schema cannot be restored: '+name) from exc
                record(dict(event='serialization_schema_restored', table=name,
                            original_digest_matched=True, stored_values_exact=True,
                            index_restored=not existing.index.equals(restored.index),
                            dtype_columns=[c for c in frame if existing[c].dtype != restored[c].dtype]))
                existing = restored
            compare(frame, existing, check_exact=True)
        record(dict(event='stage_reproduced', stage=stage, tables=len(tables), frozen_outputs_preserved=True))
        if stage == 'evaluation':
            (study.OUT/'evaluation_rerun.json').write_text(json.dumps(dict(
                matched=True, rtol=ROUNDING_RTOL, atol=ROUNDING_ATOL,
                non_float_fields_exact=True, frozen_outputs_preserved=True), indent=2), encoding='utf-8')

    pd.testing.assert_frame_equal = compare
    study.save_checked = save_preserving_frozen


class SerialDateExecutor:
    """Execute the study's date map synchronously, preserving its input order."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, arguments):
        return map(function, arguments)


def main():
    stop_marker = (Path(__file__).resolve().parents[2] / 'tables/analysis/'
                   'customer_concentration/local_execution_blocked.json')
    if stop_marker.exists() and not any(arg in ('-h', '--help') for arg in sys.argv[1:]):
        raise SystemExit('Local study execution is stopped after a second laptop freeze. '
                         'Review local_execution_blocked.json before any restart; '
                         'existing results remain available.')
    # Set limits before importing NumPy/SciPy through the scientific module.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    from customer_concentration_resources import apply_worker_limits
    resource_limits = apply_worker_limits()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise ctypes.WinError(ctypes.get_last_error())

    import customer_concentration as study

    study.ProcessPoolExecutor = SerialDateExecutor
    install_original_oracle_threads(study)
    if not any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        study.OUT.mkdir(parents=True, exist_ok=True)
        settings = dict(
            maximum_numerical_workers=1, blas_threads=ORIGINAL_BLAS_THREADS,
            optimizer_threads=1, other_thread_limits=1,
            original_serial_oracles=ORIGINAL_SERIAL_ORACLES,
            priority="below_normal" if os.name == "nt" else "unchanged",
            requested_stage=sys.argv[1:] or ["all"],
            overrides_default_date_pool=True, numerical_settings_changed=False,
            scientific_source_sha256=hashlib.sha256(
                Path(study.__file__).read_bytes()).hexdigest(),
            resource_limits=resource_limits,
        )
        (study.OUT / "single_worker_execution.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8")
        install_reproduction_checks(study, study.OUT/'single_worker_comparisons.jsonl')
        if any(arg in ('verify-select', 'verify-oracle') for arg in sys.argv[1:]):
            from customer_concentration_checkpoints import install_solver_checkpoints
            install_solver_checkpoints(study, study.OUT/'independent_solver_steps')
    with reproduction_threads():
        study.main()


if __name__ == "__main__":
    main()
