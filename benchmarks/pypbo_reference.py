"""Print pypbo's outputs on the fixed inputs used in tests/test_pypbo_reference.py.

uv run python benchmarks/pypbo_reference.py
"""

from __future__ import annotations

import json

import numpy as np

from benchmarks.bench_pbo import fetch_pypbo, sharpe_cols


def fixed_matrix() -> np.ndarray:
    rng = np.random.default_rng(42)
    m = rng.standard_t(5, (483, 12)) * 0.01
    m[:, 3] += 0.0015
    return m


def main() -> None:
    pp = fetch_pypbo()
    m = fixed_matrix()
    out = {}
    for s in (4, 8):
        r = pp.pbo(m, S=s, metric_func=sharpe_cols, threshold=0, n_jobs=1)
        out[f"pbo_S{s}"] = float(r.pbo)
        out[f"logits_S{s}_first5"] = [float(x) for x in r.logits[:5]]
        out[f"prob_oos_loss_S{s}"] = float(r.prob_oos_loss)
    out["psr"] = float(pp.psr(sharpe=0.08, T=750, skew=-0.4, kurtosis=6.0, target_sharpe=0.02))
    out["dsr"] = float(
        pp.dsr(test_sharpe=0.1581, sharpe_std=0.04472, N=100, T=1250, skew=-3, kurtosis=10)
    )
    out["expected_max_100"] = float(pp.expected_max(100))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
