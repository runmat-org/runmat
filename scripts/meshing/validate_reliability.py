#!/usr/bin/env python3
"""Validate executable meshing budget, cancellation, and reliability coverage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .catalog_support import (
        CatalogError,
        object_value,
        read_document,
        require_test_anchor,
        string_array,
        string_value,
    )
else:
    from catalog_support import (
        CatalogError,
        object_value,
        read_document,
        require_test_anchor,
        string_array,
        string_value,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "verification/meshing/reliability.json"
OWNERS = {
    "geometry-io",
    "meshing-curve",
    "meshing-surface",
    "meshing-tetrahedron",
    "meshing-execution",
    "execution-artifact",
    "execution-runner",
    "native-execution",
}
ReliabilityError = CatalogError


def validate_catalog(catalog_path: Path, repo_root: Path = REPO_ROOT) -> tuple[int, int]:
    document = read_document(catalog_path, "reliability catalog")
    if document.get("schema_version") != 1:
        raise ReliabilityError("schema_version must equal 1")
    revision = document.get("catalog_revision")
    if not isinstance(revision, int) or revision < 1:
        raise ReliabilityError("catalog_revision must be a positive integer")
    required = set(string_array(document.get("required_controls"), "required_controls"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ReliabilityError("cases must be a non-empty array")

    identities: set[str] = set()
    covered: set[str] = set()
    for index, raw_case in enumerate(cases):
        context = f"cases[{index}]"
        case = object_value(raw_case, context)
        identity = string_value(case.get("id"), f"{context}.id")
        if identity in identities:
            raise ReliabilityError(f"duplicate reliability case: {identity}")
        identities.add(identity)
        if case.get("owner") not in OWNERS:
            raise ReliabilityError(f"{identity}: invalid domain owner")
        controls = set(string_array(case.get("controls"), f"{identity}.controls"))
        unknown = controls - required
        if unknown:
            raise ReliabilityError(f"{identity}: unknown controls {sorted(unknown)}")
        covered.update(controls)
        string_value(case.get("expected"), f"{identity}.expected")
        require_test_anchor(repo_root, case.get("test"), f"{identity}.test")

    missing = required - covered
    if missing:
        raise ReliabilityError(f"required reliability controls lack executable coverage: {sorted(missing)}")
    return len(cases), len(required)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="?", type=Path, default=DEFAULT_CATALOG)
    arguments = parser.parse_args(argv)
    try:
        case_count, control_count = validate_catalog(arguments.catalog)
    except ReliabilityError as error:
        print(f"meshing reliability validation failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {case_count} reliability cases covering {control_count} required controls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
