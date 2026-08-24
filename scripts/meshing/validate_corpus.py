#!/usr/bin/env python3
"""Validate the immutable geometry-corpus inventory and its executable evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .catalog_support import (
        CatalogError,
        object_value,
        read_document,
        relative_path,
        require_test_anchor,
        sha256,
        string_value,
    )
else:
    from catalog_support import (
        CatalogError,
        object_value,
        read_document,
        relative_path,
        require_test_anchor,
        sha256,
        string_value,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "verification/meshing/corpus.json"
REQUIRED_EXPECTATIONS = {"outcome", "topology", "mass_properties", "regions", "mesh_error_bounds"}
SUPPORTED_FORMATS = {"brep", "iges", "step"}
SUPPORTED_TIERS = {"small", "medium", "extended"}


CorpusError = CatalogError


def validate_manifest(manifest_path: Path, repo_root: Path = REPO_ROOT) -> int:
    document = read_document(manifest_path, "corpus manifest")

    if document.get("schema_version") != 1:
        raise CorpusError("schema_version must equal 1")
    revision = document.get("corpus_revision")
    if not isinstance(revision, int) or revision < 1:
        raise CorpusError("corpus_revision must be a positive integer")

    fixture_root = repo_root / relative_path(document.get("fixture_root"), "fixture_root")
    if not fixture_root.is_dir():
        raise CorpusError(f"fixture_root does not exist: {fixture_root}")
    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CorpusError("entries must be a non-empty array")

    ids: set[str] = set()
    paths: set[Path] = set()
    for index, raw_entry in enumerate(entries):
        context = f"entries[{index}]"
        entry = object_value(raw_entry, context)
        entry_id = string_value(entry.get("id"), f"{context}.id")
        if entry_id in ids:
            raise CorpusError(f"duplicate corpus id: {entry_id}")
        ids.add(entry_id)

        entry_path = relative_path(entry.get("path"), f"{context}.path")
        if entry_path in paths:
            raise CorpusError(f"duplicate corpus path: {entry_path}")
        paths.add(entry_path)
        fixture = fixture_root / entry_path
        if not fixture.is_file():
            raise CorpusError(f"missing corpus fixture: {entry_path}")

        expected_format = fixture.suffix.lower().lstrip(".")
        if expected_format == "igs":
            expected_format = "iges"
        file_format = entry.get("format")
        if file_format not in SUPPORTED_FORMATS or file_format != expected_format:
            raise CorpusError(f"{entry_id}: format does not match the file extension")
        if entry.get("tier") not in SUPPORTED_TIERS:
            raise CorpusError(f"{entry_id}: invalid verification tier")

        recorded_digest = entry.get("sha256")
        if recorded_digest != sha256(fixture):
            raise CorpusError(f"{entry_id}: SHA-256 does not match {entry_path}")

        provenance = object_value(entry.get("provenance"), f"{entry_id}.provenance")
        for field in ("origin", "exporter", "exporter_version", "license"):
            string_value(provenance.get(field), f"{entry_id}.provenance.{field}")
        features = entry.get("features")
        if not isinstance(features, list) or not features or any(
            not isinstance(feature, str) or not feature for feature in features
        ):
            raise CorpusError(f"{entry_id}.features must contain non-empty strings")

        supported = entry.get("supported_input")
        if not isinstance(supported, bool):
            raise CorpusError(f"{entry_id}.supported_input must be boolean")
        expected = object_value(entry.get("expected"), f"{entry_id}.expected")
        if set(expected) != REQUIRED_EXPECTATIONS:
            raise CorpusError(f"{entry_id}.expected must contain exactly {sorted(REQUIRED_EXPECTATIONS)}")
        outcome = string_value(expected.get("outcome"), f"{entry_id}.expected.outcome")
        if supported and outcome != "success":
            raise CorpusError(f"{entry_id}: supported inputs cannot declare an expected failure")
        string_value(expected.get("topology"), f"{entry_id}.expected.topology")
        string_value(expected.get("mass_properties"), f"{entry_id}.expected.mass_properties")
        string_value(expected.get("mesh_error_bounds"), f"{entry_id}.expected.mesh_error_bounds")
        if not isinstance(expected.get("regions"), int) or expected["regions"] < 0:
            raise CorpusError(f"{entry_id}.expected.regions must be a non-negative integer")

        require_test_anchor(repo_root, entry.get("test"), f"{entry_id}.test")

    fixture_paths = {
        path.relative_to(fixture_root)
        for path in fixture_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".brep", ".iges", ".igs", ".step", ".stp"}
    }
    if paths != fixture_paths:
        missing = sorted(str(path) for path in fixture_paths - paths)
        stale = sorted(str(path) for path in paths - fixture_paths)
        raise CorpusError(f"corpus inventory mismatch; unregistered={missing}, stale={stale}")
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args(argv)
    try:
        count = validate_manifest(arguments.manifest)
    except CorpusError as error:
        print(f"meshing corpus validation failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {count} immutable meshing corpus entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
