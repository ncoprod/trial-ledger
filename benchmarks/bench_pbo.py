"""Benchmark CSCV PBO: pypbo vs the NumPy reference vs the Rust core.

pypbo (https://github.com/esvhd/pypbo, AGPL-3.0) is not on PyPI; this script downloads its two
source files at a pinned commit into ``benchmarks/_vendor/`` (git-ignored, not redistributed)
and imports them. Run with::

    uv sync --all-extras --group bench
    uv run python benchmarks/bench_pbo.py

Writes ``benchmarks/results/pbo_benchmark.json`` and ``.md``.
All implementations use the same metric (per-column Sharpe, ddof=1) and the same CSCV
splits; the script checks that they return the same PBO and logits before timing them.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from trial_ledger.stats import pbo_cscv

HERE = Path(__file__).parent
VENDOR = HERE / "_vendor"
OUT = HERE / "results"
PYPBO_COMMIT = "4d723f06498267a2a6280cb9d7d5649348e961d1"
PYPBO_FILES = [
    "pypbo/__init__.py",
    "pypbo/pbo.py",
    "pypbo/perf/__init__.py",
    "pypbo/perf/metrics.py",
]
CASES = [  # (T, N, S)
    (1000, 50, 8),
    (1000, 10, 16),
    (2190, 100, 10),
    (2190, 500, 8),
    (2190, 1000, 16),
]
# pypbo keeps every split's IS and OOS matrices in memory: ~C(S, S/2) * T * N * 8 bytes.
PYPBO_MAX_BYTES = 1.5e9


def pypbo_bytes(t: int, n: int, s: int) -> float:
    from math import comb

    return comb(s, s // 2) * (t - t % s) * n * 8.0


def sharpe_cols(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) / x.std(axis=0, ddof=1)


def fetch_pypbo():
    for f in PYPBO_FILES:
        dst = VENDOR / PYPBO_COMMIT[:12] / f
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/esvhd/pypbo/{PYPBO_COMMIT}/{f}"
            with urllib.request.urlopen(url, timeout=60) as r:
                dst.write_bytes(r.read())
    sys.path.insert(0, str(VENDOR / PYPBO_COMMIT[:12]))
    import matplotlib

    matplotlib.use("Agg")
    return importlib.import_module("pypbo.pbo")


def timeit(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


RUST_1T = """
import sys, time, numpy as np, trial_ledger_core as c
t, n, s, seed, rep = map(int, sys.argv[1:])
m = np.random.default_rng(seed).standard_t(4, (t, n)) * 0.01
best = 1e9
for _ in range(rep):
    t0 = time.perf_counter(); c.pbo_cscv_sharpe(m, s); best = min(best, time.perf_counter() - t0)
print(best, c.num_threads())
"""


def rust_single_thread(t: int, n: int, s: int, seed: int, repeat: int) -> float:
    env = {**os.environ, "RAYON_NUM_THREADS": "1"}
    out = subprocess.run(
        [sys.executable, "-c", RUST_1T, str(t), str(n), str(s), str(seed), str(repeat)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    secs, threads = out.stdout.split()
    assert int(threads) == 1
    return float(secs)


def main() -> None:
    import trial_ledger_core as core

    from trial_ledger.pbo import pbo

    pypbo = fetch_pypbo()
    rows = []
    for i, (t, n, s) in enumerate(CASES):
        seed = 100 + i
        m = np.random.default_rng(seed).standard_t(4, (t, n)) * 0.01

        ref = pbo_cscv(m, s)
        rust = pbo(m, s, backend="rust")
        np.testing.assert_allclose(rust.logits, ref.logits, atol=1e-12)
        assert rust.pbo == ref.pbo
        run_pypbo = pypbo_bytes(t, n, s) <= PYPBO_MAX_BYTES
        t_pypbo = None
        if run_pypbo:
            res_pypbo = pypbo.pbo(m, S=s, metric_func=sharpe_cols, threshold=0, n_jobs=1)
            np.testing.assert_allclose(np.asarray(res_pypbo.logits), ref.logits, atol=1e-9)
            assert abs(res_pypbo.pbo - ref.pbo) < 1e-12  # pypbo sums phi/C, we take a mean
            del res_pypbo
            t_pypbo = timeit(
                lambda: pypbo.pbo(m, S=s, metric_func=sharpe_cols, threshold=0, n_jobs=1),
                repeat=1,
            )
        t_generic = timeit(lambda: pbo_cscv(m, s, metric=sharpe_cols), repeat=1)
        t_numpy = timeit(lambda: pbo_cscv(m, s), repeat=3)
        t_rust1 = rust_single_thread(t, n, s, seed, repeat=5)
        t_rust = timeit(lambda: core.pbo_cscv_sharpe(m, s), repeat=5)
        row = {
            "T": t, "N": n, "S": s, "splits": ref.n_splits, "pbo": ref.pbo,
            "pypbo_s": t_pypbo,
            "pypbo_est_memory_gb": pypbo_bytes(t, n, s) / 1e9,
            "numpy_generic_s": t_generic,
            "numpy_blockstats_s": t_numpy,
            "rust_1thread_s": t_rust1,
            "rust_threads": core.num_threads(),
            "rust_parallel_s": t_rust,
        }  # fmt: skip
        row["speedup_rust_vs_pypbo"] = t_pypbo / t_rust if t_pypbo else None
        row["speedup_rust1_vs_pypbo"] = t_pypbo / t_rust1 if t_pypbo else None
        row["speedup_rust1_vs_numpy_blockstats"] = t_numpy / t_rust1
        row["speedup_rust_vs_numpy_blockstats"] = t_numpy / t_rust
        rows.append(row)
        print(json.dumps(row), flush=True)

    meta = {
        "machine": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pypbo_commit": PYPBO_COMMIT,
        "timing": "best of N wall-clock runs (pypbo and numpy_generic: 1 run)",
        "pypbo_skip_rule": f"skipped when its estimated memory exceeds {PYPBO_MAX_BYTES:.1e} B",
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "pbo_benchmark.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    lines = [
        f"Machine: {meta['machine']}, {meta['cpu_count']} logical CPUs, Python "
        f"{meta['python']}, NumPy {meta['numpy']}; pypbo @ {PYPBO_COMMIT[:7]}.",
        "",
        "| T | N | S | splits | pypbo | NumPy generic | NumPy block-stats | Rust 1 thread "
        "| Rust parallel | Rust ∥ vs pypbo | Rust 1t vs NumPy block-stats |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["pypbo_s"] is None:
            pyp = f"not run (~{r['pypbo_est_memory_gb']:.0f} GB)"
            sp = "n/a"
        else:
            pyp = f"{r['pypbo_s']:.2f} s"
            sp = f"{r['speedup_rust_vs_pypbo']:,.0f}x"
        lines.append(
            f"| {r['T']} | {r['N']} | {r['S']} | {r['splits']} | {pyp} "
            f"| {r['numpy_generic_s']:.2f} s | {r['numpy_blockstats_s'] * 1e3:.1f} ms "
            f"| {r['rust_1thread_s'] * 1e3:.2f} ms | {r['rust_parallel_s'] * 1e3:.2f} ms "
            f"| {sp} "
            f"| {r['speedup_rust1_vs_numpy_blockstats']:.1f}x |"
        )
    (OUT / "pbo_benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
