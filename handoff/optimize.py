"""Run `python optimize.py --help` from this portable handoff directory."""
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[variable] = '1'

from portfolio_optimizer.cli import main

if __name__ == '__main__':
    main()
