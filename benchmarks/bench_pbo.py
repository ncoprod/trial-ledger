"""Benchmark CSCV PBO: pypbo vs the NumPy reference vs the Rust core.

pypbo (https://github.com/esvhd/pypbo, AGPL-3.0) is not on PyPI; this script downloads its two
source files at a pinned commit into ``benchmarks/_vendor/`` (git-ignored, not redistributed)
and imports them. Run with::

    uv sync --all-extras --group bench
    uv run python -m benchmarks.bench_pbo

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

from experiments.provenance import provenance
from trial_ledger.hashing import hash_array, hash_file
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


def timeit(fn, repeat: int) -> list[float]:
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


RUST_1T = """
import sys, time, json, numpy as np, trial_ledger_core as c
from trial_ledger.hashing import hash_array
t, n, s, seed, rep = map(int, sys.argv[1:])
m = np.random.default_rng(seed).standard_t(4, (t, n)) * 0.01
samples = []
for _ in range(rep):
    t0 = time.perf_counter(); c.pbo_cscv_sharpe(m, s); samples.append(time.perf_counter() - t0)
outputs = c.pbo_cscv_sharpe(m, s)
print(json.dumps({'samples':samples, 'threads':c.num_threads(),
                  'hashes':[hash_array(x) for x in outputs]}))
"""


def rust_single_thread(t: int, n: int, s: int, seed: int, repeat: int, hashes) -> list[float]:
    env = {**os.environ, "RAYON_NUM_THREADS": "1"}
    out = subprocess.run(
        [sys.executable, "-c", RUST_1T, str(t), str(n), str(s), str(seed), str(repeat)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout)
    assert result["threads"] == 1
    assert result["hashes"] == hashes
    return result["samples"]


def main() -> None:
    import trial_ledger_core as core

    from trial_ledger.pbo import pbo

    pypbo = fetch_pypbo()
    receipt = provenance(
        {
            "cases": CASES,
            "rayon_threads": core.num_threads(),
            "numpy_repeats": 7,
            "rust_repeats": 25,
            "pypbo_repeats": 3,
            "generic_repeats": 3,
        }
    )
    benchmark_sha = hash_file(__file__)
    rows = []
    for i, (t, n, s) in enumerate(CASES):
        seed = 100 + i
        m = np.random.default_rng(seed).standard_t(4, (t, n)) * 0.01

        ref = pbo_cscv(m, s)
        rust = pbo(m, s, backend="rust")
        np.testing.assert_allclose(rust.logits, ref.logits, atol=1e-12, rtol=0)
        assert rust.pbo == ref.pbo
        outputs = core.pbo_cscv_sharpe(m, s)
        hashes = [hash_array(x) for x in outputs]
        run_pypbo = pypbo_bytes(t, n, s) <= PYPBO_MAX_BYTES
        samples_pypbo = None
        if run_pypbo:
            res_pypbo = pypbo.pbo(m, S=s, metric_func=sharpe_cols, threshold=0, n_jobs=1)
            np.testing.assert_allclose(np.asarray(res_pypbo.logits), ref.logits, atol=1e-9, rtol=0)
            assert abs(res_pypbo.pbo - ref.pbo) < 1e-12  # pypbo sums phi/C, we take a mean
            del res_pypbo
            samples_pypbo = timeit(
                lambda: pypbo.pbo(m, S=s, metric_func=sharpe_cols, threshold=0, n_jobs=1),
                repeat=3,
            )
        samples = {
            "pypbo": samples_pypbo,
            "numpy_generic": timeit(lambda: pbo_cscv(m, s, metric=sharpe_cols), repeat=3),
            "numpy_blockstats": timeit(lambda: pbo_cscv(m, s), repeat=7),
            "rust_1thread": rust_single_thread(t, n, s, seed, repeat=25, hashes=hashes),
            "rust_parallel": timeit(lambda: core.pbo_cscv_sharpe(m, s), repeat=25),
        }
        medians = {k: float(np.median(v)) if v else None for k, v in samples.items()}
        t_pypbo, t_rust = medians["pypbo"], medians["rust_parallel"]
        t_rust1, t_numpy = medians["rust_1thread"], medians["numpy_blockstats"]
        row = {
            "T": t, "N": n, "S": s, "splits": ref.n_splits, "pbo": ref.pbo,
            "seed": seed, "input_sha256": hash_array(m), "rust_output_sha256": hashes,
            "checks": {"numpy_pbo_exact": True, "numpy_logits_atol": 1e-12,
                       "rust_threads_byte_identical": True, "pypbo_checked": run_pypbo},
            "pypbo_s": t_pypbo,
            "pypbo_est_memory_gb": pypbo_bytes(t, n, s) / 1e9,
            "samples_seconds": samples,
            "numpy_generic_s": medians["numpy_generic"],
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
        "provenance": receipt,
        "benchmark_source_sha256": benchmark_sha,
        "rust_source_sha256": hash_file("rust/src/lib.rs"),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pypbo_commit": PYPBO_COMMIT,
        "timing": "median wall-clock runs; all samples recorded, shared machine",
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
