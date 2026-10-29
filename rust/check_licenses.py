"""Verify every bundled upstream notice in wheel and source archives."""

import hashlib
import json
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent / "THIRD_PARTY_LICENSES"


def main(paths):
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    locked = tomllib.loads((BUNDLE.parent / "Cargo.lock").read_text())["package"]
    assert {(p["name"], p["version"], p["checksum"]) for p in locked if p.get("source")} == {
        (p["name"], p["version"], p["crate_sha256"]) for p in manifest["packages"]
    }, "notice manifest differs from Cargo.lock; rerun the collector"
    expected = {"THIRD_PARTY_LICENSES/manifest.json": (BUNDLE / "manifest.json").read_bytes()}
    for package in manifest["packages"]:
        for filename, digest in package["notice_sha256"].items():
            name = f"THIRD_PARTY_LICENSES/{package['name']}-{package['version']}/{filename}"
            data = (BUNDLE.parent / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, name
            expected[name] = data
    assert paths, "provide a wheel or source archive"
    for path in paths:
        if path.endswith(".whl"):
            with zipfile.ZipFile(path) as archive:
                files = {name: archive.read(name) for name in archive.namelist()}
        else:
            with tarfile.open(path) as archive:
                files = {
                    member.name: archive.extractfile(member).read()
                    for member in archive.getmembers()
                    if member.isfile()
                }
        for name, data in expected.items():
            matches = [value for key, value in files.items() if key.endswith("/" + name)]
            assert matches and all(value == data for value in matches), (path, name)
        print(f"PASS {path}: {len(expected) - 1} complete upstream notice files")


if __name__ == "__main__":
    main(sys.argv[1:])
