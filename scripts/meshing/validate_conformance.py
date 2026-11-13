#!/usr/bin/env python3
"""Validate versioned difficult-geometry conformance coverage and executable anchors."""

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
DEFAULT_CATALOG = REPO_ROOT / "verification/meshing/conformance.json"
EXPECTED_FIELDS = {"outcome", "topology", "mass_properties", "regions", "error_bounds"}
SUPPORTED_TIERS = {"small", "medium", "extended"}


ConformanceError = CatalogError


def validate_catalog(catalog_path: Path, repo_root: Path = REPO_ROOT) -> tuple[int, int]:
    document = read_document(catalog_path, "conformance catalog")
    if document.get("schema_version") != 1:
        raise ConformanceError("schema_version must equal 1")
    revision = document.get("catalog_revision")
    if not isinstance(revision, int) or revision < 1:
        raise ConformanceError("catalog_revision must be a positive integer")
    required = set(string_array(document.get("required_features"), "required_features"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ConformanceError("cases must be a non-empty array")

    ids: set[str] = set()
    covered: set[str] = set()
    for index, raw_case in enumerate(cases):
        context = f"cases[{index}]"
        case = object_value(raw_case, context)
        case_id = string_value(case.get("id"), f"{context}.id")
        if case_id in ids:
            raise ConformanceError(f"duplicate conformance id: {case_id}")
        ids.add(case_id)
        if case.get("tier") not in SUPPORTED_TIERS:
            raise ConformanceError(f"{case_id}: invalid verification tier")
        features = set(string_array(case.get("features"), f"{case_id}.features"))
        unknown = features - required
        if unknown:
            raise ConformanceError(f"{case_id}: unknown features {sorted(unknown)}")
        covered.update(features)

        supported = case.get("supported_input")
        if not isinstance(supported, bool):
            raise ConformanceError(f"{case_id}.supported_input must be boolean")
        expected = object_value(case.get("expected"), f"{case_id}.expected")
        if set(expected) != EXPECTED_FIELDS:
            raise ConformanceError(f"{case_id}.expected must contain exactly {sorted(EXPECTED_FIELDS)}")
        for field in EXPECTED_FIELDS:
            string_value(expected.get(field), f"{case_id}.expected.{field}")
        if supported and expected["outcome"] != "success":
            raise ConformanceError(f"{case_id}: supported inputs cannot declare an expected failure")

        require_test_anchor(repo_root, case.get("test"), f"{case_id}.test")

    missing = required - covered
    if missing:
        raise ConformanceError(f"required conformance features lack executable coverage: {sorted(missing)}")
    return len(cases), len(required)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="?", type=Path, default=DEFAULT_CATALOG)
    arguments = parser.parse_args(argv)
    try:
        case_count, feature_count = validate_catalog(arguments.catalog)
    except ConformanceError as error:
        print(f"meshing conformance validation failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {case_count} meshing conformance cases covering {feature_count} required features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
