"""Archive GARCH inputs once so cache replay preserves bytes across CPU architectures."""

import argparse
import json
import os

import numpy as np

from trial_ledger.hashing import hash_array, hash_file

from .markets import DATA, generate_garch_returns
from .provenance import provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    path = DATA.with_name("garch_markets.npz")
    with np.load(path, allow_pickle=False) as archive:
        series = {k: archive[k] for k in archive.files}
    for seed in range(args.count):
        key = f"seed_{seed}"
        if key not in series:
            series[key] = generate_garch_returns(seed)
    temporary = path.with_suffix(".new.npz")
    np.savez_compressed(temporary, **series)
    os.replace(temporary, path)
    receipt = provenance({"seeds": list(range(args.count)), "n": 2190, "burn": 500})
    receipt["archive_sha256"] = hash_file(path)
    receipt["series_sha256"] = {k: hash_array(v) for k, v in series.items()}
    receipt["note"] = "Existing archived inputs are preserved, never refitted or regenerated."
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
