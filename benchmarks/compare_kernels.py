"""Interleave fixed-input native builds to reduce shared-machine timing confounding."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from experiments.provenance import provenance
from trial_ledger.hashing import hash_array, hash_file

from .bench_pbo import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("builds", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modules, builds = {}, {}
    for path in sorted(args.builds.iterdir()):
        binary = next((path / "trial_ledger_core").glob("*.so"))
        spec = importlib.util.spec_from_file_location(f"{path.name}.trial_ledger_core", binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path.name] = module
        builds[path.name] = json.loads((path / "metadata.json").read_text())
    rng = np.random.default_rng(77)
    rows = []
    for i, (t, n, s) in enumerate(CASES):
        matrix = np.random.default_rng(100 + i).standard_t(4, (t, n)) * 0.01
        hashes = {
            name: [hash_array(a) for a in module.pbo_cscv_sharpe(matrix, s)]
            for name, module in modules.items()
        }
        assert all(h == next(iter(hashes.values())) for h in hashes.values())
        samples = {name: [] for name in modules}
        orders = []
        for _ in range(25):
            order = rng.permutation(list(modules)).tolist()
            orders.append(order)
            for name in order:
                start = time.perf_counter()
                modules[name].pbo_cscv_sharpe(matrix, s)
                samples[name].append(time.perf_counter() - start)
        medians = {name: float(np.median(v)) for name, v in samples.items()}
        row = {
            "T": t,
            "N": n,
            "S": s,
            "seed": 100 + i,
            "input_sha256": hash_array(matrix),
            "output_sha256": hashes,
            "byte_identical": True,
            "sample_orders": orders,
            "samples_seconds": samples,
            "medians_seconds": medians,
        }
        rows.append(row)
        print(t, n, s, medians, flush=True)
    result = {
        "provenance": provenance(
            {
                "order_seed": 77,
                "repeats": 25,
                "threads_per_build": {k: v.num_threads() for k, v in modules.items()},
            }
        ),
        "benchmark_source_sha256": hash_file(__file__),
        "builds": builds,
        "rows": rows,
        "caveat": "Interleaved on a shared machine; no isolated-latency guarantee.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
