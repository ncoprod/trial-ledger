"""A backtest script an agent might write. Each configuration is reported on stdout with
``emit`` so the shell hook (or ``trial-ledger exec``) logs one trial per configuration,
not one per script run.

    trial-ledger exec -- python examples/backtest_script.py
"""

import numpy as np

from trial_ledger.backtest import Market, run_backtest
from trial_ledger.hooks import emit

rng = np.random.default_rng(0)
market = Market("demo", rng.standard_t(4, 1500) * 0.02, periods_per_year=365)

for fast in (5, 10, 20):
    for slow in (50, 100, 200):
        spec = {"family": "ma_crossover", "fast": fast, "slow": slow}
        res = run_backtest(spec, market)
        emit(res.sharpe, res.n_obs, spec, periods_per_year=365)
