"""Fixed-input kernel timings with byte-identical output checks across revisions."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import trial_ledger_core as core

from experiments.provenance import provenance
from trial_ledger.hashing import hash_array, hash_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    rows = []
    for i, (t, n, s) in enumerate(
        ((1000, 50, 8), (1000, 10, 16), (2190, 100, 10), (2190, 500, 8), (2190, 1000, 16))
    ):
        matrix = np.random.default_rng(100 + i).standard_t(4, (t, n)) * 0.01
        outputs = core.pbo_cscv_sharpe(matrix, s)
        hashes = [hash_array(x) for x in outputs]
        if baseline:
            assert hashes == baseline["rows"][i]["output_sha256"]
        samples = []
        for _ in range(25):
            start = time.perf_counter()
            core.pbo_cscv_sharpe(matrix, s)
            samples.append(time.perf_counter() - start)
        rows.append(
            {
                "T": t,
                "N": n,
                "S": s,
                "seed": 100 + i,
                "input_sha256": hash_array(matrix),
                "output_sha256": hashes,
                "samples_seconds": samples,
                "median_seconds": float(np.median(samples)),
                "best_seconds": min(samples),
                "identical_to_baseline": bool(baseline),
            }
        )
    result = {
        "provenance": provenance(
            {
                "repeats": 25,
                "threads": core.num_threads(),
                "rayon_env": os.environ.get("RAYON_NUM_THREADS"),
            }
        ),
        "rust_source_sha256": hash_file("rust/src/lib.rs"),
        "package_wrapper_sha256": hash_file(core.__file__),
        "extension_sha256": hash_file(next(Path(core.__file__).parent.glob("*.so"))),
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for r in rows:
        print(r["T"], r["N"], r["S"], r["median_seconds"], flush=True)


if __name__ == "__main__":
    main()
