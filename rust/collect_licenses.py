"""Copy exact upstream notices from the locked crates.io source archives."""

import hashlib
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "THIRD_PARTY_LICENSES"


def main():
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--locked",
                "--features",
                "python",
                "--format-version",
                "1",
                "--manifest-path",
                str(ROOT / "Cargo.toml"),
            ]
        )
    )
    locked = tomllib.loads((ROOT / "Cargo.lock").read_text())["package"]
    checksums = {(p["name"], p["version"]): p.get("checksum") for p in locked}
    packages = []
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if not package["source"]:
            continue
        name, version = package["name"], package["version"]
        source = Path(package["manifest_path"]).parent
        archive_path = source.parents[2] / "cache" / source.parent.name / f"{name}-{version}.crate"
        crate_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        assert crate_sha256 == checksums[name, version]
        archive = tarfile.open(archive_path)
        notices = sorted(
            (
                m
                for m in archive.getmembers()
                if m.isfile()
                and len(Path(m.name).parts) == 2
                and Path(m.name)
                .name.upper()
                .startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE"))
            ),
            key=lambda m: m.name,
        )
        assert notices, f"missing upstream licence: {name} {version}"
        output = BUNDLE / f"{name}-{version}"
        output.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for member in notices:
            filename = Path(member.name).name
            data = archive.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            (output / filename).write_bytes(data)
            hashes[filename] = digest
        archive.close()
        packages.append(
            dict(
                name=name,
                version=version,
                license=package["license"],
                repository=package["repository"],
                authors=package["authors"],
                crate_sha256=crate_sha256,
                notice_sha256=hashes,
            )
        )
    BUNDLE.mkdir(exist_ok=True)
    (BUNDLE / "manifest.json").write_text(
        json.dumps(
            {
                "command": "python rust/collect_licenses.py",
                "source": "Crates.io archives; SHA-256 verified against Cargo.lock",
                "packages": packages,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Collected full notices for {len(packages)} locked native dependencies")


if __name__ == "__main__":
    main()
