"""Build the recorded CSCV revisions for an interleaved comparison."""

import argparse
import json
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

from trial_ledger.hashing import hash_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = Path("rust/src/lib.rs")
    original = source.read_bytes()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    refs = {"original": "4162d54", "fused": "59a2d766", "branchless": head}
    try:
        for name, ref in refs.items():
            source.write_bytes(subprocess.check_output(["git", "show", f"{ref}:rust/src/lib.rs"]))
            subprocess.run(
                ["uv", "pip", "install", "--reinstall", "--no-deps", "./rust"],
                check=True,
                env={**os.environ, "CARGO_BUILD_JOBS": "2"},
            )
            out = args.output / name
            package = Path(sysconfig.get_path("purelib")) / "trial_ledger_core"
            shutil.copytree(package, out / "trial_ledger_core", dirs_exist_ok=True)
            shutil.copyfile(source, out / "lib.rs")
            metadata = {
                "source_ref": f"{ref}:rust/src/lib.rs",
                "source_sha256": hash_file(source),
                "native_sha256": hash_file(next((out / "trial_ledger_core").glob("*.so"))),
                "build_command": "CARGO_BUILD_JOBS=2 uv pip install --reinstall --no-deps ./rust",
                "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
            }
            (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        source.write_bytes(original)


if __name__ == "__main__":
    main()
