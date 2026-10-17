"""Numerical reproducibility tolerates rounding, never changed portfolio decisions."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "scripts")))
from customer_concentration_serial import (
    assert_reproduced_frame, install_reproduction_checks, reproduction_threads,
    install_original_oracle_threads,
)


@pytest.fixture
def detectable_blas_backend():
    """Only test thread restoration when threadpoolctl can observe the backend.

    Apple Accelerate can perform the numerical work without exposing a BLAS
    backend to this API. In that case the runtime limits are not verified or
    enforced through threadpoolctl; an empty list must not pass via all([]).
    """
    from threadpoolctl import threadpool_info
    backends = [pool for pool in threadpool_info() if pool['user_api'] == 'blas']
    if not backends:
        pytest.skip(
            'No BLAS backend detectable by threadpoolctl (e.g. Apple Accelerate); '
            'runtime BLAS thread limits/restoration cannot be verified or enforced through this API.'
        )
    return backends


@pytest.mark.parametrize('date,retention,actual_floor,expected', [
    ('2022-05-31', .95, 1., 1), ('2022-06-30', .95, 1., 1),
    ('2022-06-30', .98, 1., 20), ('2022-04-30', .95, 1., 20),
    ('2022-06-30', .95, None, 20),
])
def test_original_recovery_settings_do_not_leak_to_other_cases(date, retention, actual_floor, expected, detectable_blas_backend):
    from types import SimpleNamespace
    from threadpoolctl import threadpool_info, threadpool_limits
    def counts():
        return [p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
    study=SimpleNamespace()
    study.optimize=lambda *args, **kwargs: counts()
    study.oracle_stage=lambda **kwargs: study.optimize(retention=retention, actual_floor=actual_floor)
    install_original_oracle_threads(study)
    with threadpool_limits(limits=20,user_api='blas'):
        assert all(n==expected for n in study.oracle_stage(only_date=date))
        assert all(n==20 for n in counts())
        assert all(n==20 for n in study.optimize(retention=.95,actual_floor=1.))


def test_original_recovery_setting_restored_after_error(detectable_blas_backend):
    from types import SimpleNamespace
    from threadpoolctl import threadpool_info, threadpool_limits
    study=SimpleNamespace()
    def failing(*args,**kwargs):
        assert all(p['num_threads']==1 for p in threadpool_info() if p['user_api']=='blas')
        raise ValueError('expected test failure')
    study.optimize=failing
    study.oracle_stage=lambda **kwargs: study.optimize(retention=.95,actual_floor=1.)
    install_original_oracle_threads(study)
    with threadpool_limits(limits=20,user_api='blas'):
        with pytest.raises(ValueError,match='expected test failure'):
            study.oracle_stage(only_date='2022-06-30')
        assert all(p['num_threads']==20 for p in threadpool_info() if p['user_api']=='blas')


def test_original_reduction_order_and_nested_optimizer_limit_are_restored(detectable_blas_backend):
    from threadpoolctl import threadpool_info, threadpool_limits
    values = np.linspace(.1, 1., 500_000)
    def blas_counts():
        return [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
    with threadpool_limits(limits=20, user_api='blas'):
        expected = values @ values
    with threadpool_limits(limits=1):
        assert blas_counts() and all(n == 1 for n in blas_counts())
        with reproduction_threads():
            assert all(n == 20 for n in blas_counts())
            assert values @ values == expected
            with threadpool_limits(limits=1):
                assert all(n == 1 for n in blas_counts())
            assert all(n == 20 for n in blas_counts())
        assert all(n == 1 for n in blas_counts())


def example():
    return pd.DataFrame(dict(merchant_abn=[12345678901, 12345678902], rank=[1, 2],
                             selected=[True, False], retention=[.99, .99],
                             hhi=[4.648257805243025e-5, np.nan]))


def test_accepts_documented_thread_reduction_roundoff():
    saved = example()
    recomputed = saved.copy()
    recomputed.loc[0, 'hhi'] = 4.648257805243023e-5
    result = assert_reproduced_frame(recomputed, saved)
    assert result[0]['maximum_absolute_difference'] < 3e-20


@pytest.mark.parametrize('column,value', [
    ('merchant_abn', 12345678903), ('rank', 2), ('selected', False),
    ('retention', np.nextafter(.99, 1)), ('hhi', 4.648257805243025e-5 + 1e-10),
    ('hhi', np.nan),
])
def test_rejects_changed_decisions_values_or_missingness(column, value):
    saved = example()
    recomputed = saved.copy()
    recomputed.loc[0, column] = value
    with pytest.raises(AssertionError):
        assert_reproduced_frame(recomputed, saved)


def test_rejects_changed_order_or_schema():
    saved = example()
    with pytest.raises(AssertionError):
        assert_reproduced_frame(saved.iloc[::-1], saved)
    with pytest.raises(AssertionError):
        assert_reproduced_frame(saved.rename(columns={'hhi': 'HHI'}), saved)


def test_cached_evaluation_is_compared_without_overwriting(tmp_path, monkeypatch):
    import customer_concentration as study
    saved = example()
    monkeypatch.setattr(study, 'OUT', tmp_path)
    study.save_checked({'metrics': saved}, 'evaluation')
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original_assert, original_save = pd.testing.assert_frame_equal, study.save_checked
    try:
        install_reproduction_checks(study, tmp_path/'comparisons.jsonl')
        candidate = saved.copy()
        candidate.loc[0, 'hhi'] = np.nextafter(candidate.loc[0, 'hhi'], 0)
        study.save_checked({'metrics': candidate}, 'evaluation')
        assert all((tmp_path/name).read_bytes() == contents for name, contents in before.items())
        assert (tmp_path/'evaluation_rerun.json').exists()
        candidate.loc[0, 'merchant_abn'] += 1
        with pytest.raises(AssertionError):
            study.save_checked({'metrics': candidate}, 'evaluation')
    finally:
        pd.testing.assert_frame_equal, study.save_checked = original_assert, original_save


def test_parquet_schema_restoration_requires_original_hash_and_unchanged_values(tmp_path, monkeypatch):
    import customer_concentration as study
    monkeypatch.setattr(study, 'OUT', tmp_path)
    saved = example().set_axis([20, 22])
    saved['concentration_benefit'] = pd.Series([False, True], index=saved.index, dtype=object)
    study.save_checked({'summary': saved}, 'evaluation')
    stored = pd.read_parquet(tmp_path/'summary.parquet')
    assert stored.concentration_benefit.dtype == bool
    assert not stored.index.equals(saved.index)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original_assert, original_save = pd.testing.assert_frame_equal, study.save_checked
    try:
        install_reproduction_checks(study, tmp_path/'comparisons.jsonl')
        study.save_checked({'summary': saved.copy()}, 'evaluation')
        assert all((tmp_path/name).read_bytes() == content for name, content in before.items())
        # Different in-memory semantics cannot be concealed as serialization.
        changed_schema = saved.astype({'concentration_benefit': bool})
        with pytest.raises(AssertionError, match='checksum changed'):
            study.save_checked({'summary': changed_schema}, 'evaluation')
        # A changed saved value must still fail the original exact audit hash.
        stored.loc[0, 'merchant_abn'] += 1
        stored.to_parquet(tmp_path/'summary.parquet', index=False, compression=None)
        with pytest.raises(AssertionError, match='checksum changed'):
            study.save_checked({'summary': saved}, 'evaluation')
    finally:
        pd.testing.assert_frame_equal, study.save_checked = original_assert, original_save


def test_solver_step_checkpoints_reproduce_and_reject_corruption(tmp_path, monkeypatch):
    from scipy import sparse
    import customer_concentration as study
    from customer_concentration_checkpoints import install_solver_checkpoints
    calls = []
    original = study.milp
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(study, 'milp', counted)
    install_solver_checkpoints(study, tmp_path)
    exposure = sparse.csc_matrix([[6., 2., 1., 0.], [0., 2., 4., 2.], [1., 0., 0., 4.]])
    means = np.asarray(exposure.sum(axis=0)).ravel()
    budgets = [dict(max_cuts=8, node_limit=200, max_swaps=20, relative_gap=.001)]
    first = study.optimize(exposure, means, 'shared', count=2, retention=.8, budgets=budgets)
    solved = len(calls)
    assert solved > 0
    second = study.optimize(exposure, means, 'shared', count=2, retention=.8, budgets=budgets)
    assert len(calls) == solved
    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]
    pd.testing.assert_frame_equal(first[2], second[2], check_exact=True)
    # A changed numerical input cannot reuse a previous problem's answer.
    study.optimize(exposure, means, 'shared', count=2, retention=.9, budgets=budgets)
    assert len(calls) > solved
    import json
    marker = next(tmp_path.glob('*/complete.json'))
    record = json.loads(marker.read_text())
    assert record['provenance']['purpose'] == 'independent_rerun_only'
    # Corrupt every completed cache; any attempted replay must fail closed.
    for marker in tmp_path.glob('*/complete.json'):
        record = json.loads(marker.read_text())
        for name in record['files']:
            with (marker.parent/name).open('ab') as stream:
                stream.write(b'corrupt')
    with pytest.raises(AssertionError, match='checkpoint corrupted'):
        study.optimize(exposure, means, 'shared', count=2, retention=.8, budgets=budgets)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows native resource enforcement')
def test_windows_job_limit_denies_oversized_allocation():
    import json
    import subprocess
    result = subprocess.run([sys.executable, str((Path(__file__).resolve().parents[1] / "scripts")/
                            'customer_concentration_resources.py')], capture_output=True,
                            text=True, timeout=15, check=True)
    record = json.loads(result.stdout)
    assert record['native_limits_verified'] and record['oversized_allocation_rejected']
    assert record['memory_limit_bytes'] == 128*2**20
    assert record['windows_allocation_error'] in (8, 14, 1455, 1816)
