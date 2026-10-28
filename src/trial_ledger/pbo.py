"""PBO dispatcher: uses the Rust core (``trial_ledger_core``) when installed, else NumPy."""

from __future__ import annotations

import numpy as np

from .stats import PBOResult, pbo_cscv

try:  # pragma: no cover - depends on whether the optional extension is built
    import trial_ledger_core as _core

    HAVE_RUST = True
except ImportError:  # pragma: no cover
    _core = None
    HAVE_RUST = False


def pbo(returns_matrix: np.ndarray, n_partitions: int = 16, backend: str = "auto") -> PBOResult:
    """PBO via CSCV with the Sharpe ratio as performance metric.

    ``backend`` is ``"auto"`` (Rust if available), ``"rust"`` or ``"python"``.
    """
    if backend not in {"auto", "rust", "python"}:
        raise ValueError(f"unknown backend {backend!r}")
    use_rust = backend == "rust" or (backend == "auto" and HAVE_RUST)
    if not use_rust:
        return pbo_cscv(returns_matrix, n_partitions)
    if not HAVE_RUST:
        raise ImportError("trial_ledger_core is not installed; build it with maturin")
    m = np.ascontiguousarray(np.asarray(returns_matrix, dtype=np.float64))
    logits, is_best, oos_best = _core.pbo_cscv_sharpe(m, int(n_partitions))
    logits = np.asarray(logits)
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        is_best_perf=np.asarray(is_best),
        oos_perf_of_best=np.asarray(oos_best),
        n_splits=logits.size,
        n_partitions=n_partitions,
        n_trials=m.shape[1],
        meta={"backend": "rust"},
    )
