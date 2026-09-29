"""Replay completed MILP steps from independent reruns, never original results.

The optimizer itself is unchanged. Every exact numerical solver input, solver
version and scientific source hash contributes to the checkpoint identity.
Results commit atomically only after the solver returns an accepted status.
"""
import hashlib
import json
import pickle
from pathlib import Path


def install_solver_checkpoints(study, directory):
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.optimize import OptimizeResult

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    original = study.milp
    provenance = dict(format=1, purpose='independent_rerun_only', scipy=scipy.__version__,
                      numpy=np.__version__, scientific_source_sha256=hashlib.sha256(
                          Path(study.__file__).read_bytes()).hexdigest())

    def cached(*args, **kwargs):
        # Pickle is used for an in-memory digest only; no pickle is loaded.
        key = hashlib.sha256(pickle.dumps((provenance, args, kwargs), protocol=5)).hexdigest()
        folder = directory/key
        marker = folder/'complete.json'
        if marker.exists():
            record = json.loads(marker.read_text(encoding='utf-8'))
            assert record['provenance'] == provenance
            assert record['input_digest'] == key
            for name, expected in record['files'].items():
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected, 'Solver checkpoint corrupted'
            result = OptimizeResult(record['result'])
            if record['has_x']:
                result.x = pd.read_parquet(folder/'x.parquet')['x'].to_numpy()
            else:
                result.x = None
            print('    Reused independently recomputed solver step '+key[:12], flush=True)
            return result
        result = original(*args, **kwargs)
        accepted = result.status in [0, 1] or ('HiGHS Status 16:' in result.message
                                              and 'Solution limit reached' in result.message)
        if not accepted:
            return result
        folder.mkdir(exist_ok=True)
        values = dict(result); x = values.pop('x', None)
        # Solver scalar values retain their full binary precision in JSON.
        values = {k: v.item() if isinstance(v, np.generic) else v for k, v in values.items()}
        files = {}
        if x is not None:
            pending = folder/'x.pending.parquet'
            pd.DataFrame({'x': x}).to_parquet(pending, index=False, compression=None)
            pending.replace(folder/'x.parquet')
            files['x.parquet'] = hashlib.sha256((folder/'x.parquet').read_bytes()).hexdigest()
        record = dict(provenance=provenance, input_digest=key, has_x=x is not None,
                      result=values, files=files)
        pending = folder/'complete.pending.json'
        pending.write_text(json.dumps(record, indent=2), encoding='utf-8')
        pending.replace(marker)
        return result

    study.milp = cached
