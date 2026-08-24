#!/usr/bin/env python3
"""Validate versioned difficult-geometry conformance coverage and executable anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "verification/meshing/conformance.json"
EXPECTED_FIELDS = {"outcome", "topology", "mass_properties", "regions", "error_bounds"}
SUPPORTED_TIERS = {"small", "medium", "extended"}


class ConformanceError(ValueError):
    """Raised when required meshing conformance evidence is incomplete."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConformanceError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConformanceError(f"{context} must be a non-empty string")
    return value


def _path(value: Any, context: str) -> Path:
    path = Path(_string(value, context))
    if path.is_absolute() or ".." in path.parts:
        raise ConformanceError(f"{context} must stay below the repository root")
    return path


def _strings(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConformanceError(f"{context} must be a non-empty array")
    strings = [_string(item, f"{context}[]") for item in value]
    if len(strings) != len(set(strings)):
        raise ConformanceError(f"{context} must be unique")
    return strings


def validate_catalog(catalog_path: Path, repo_root: Path = REPO_ROOT) -> tuple[int, int]:
    try:
        document = _object(json.loads(catalog_path.read_text()), "catalog")
    except (OSError, json.JSONDecodeError) as error:
        raise ConformanceError(f"cannot read conformance catalog: {error}") from error
    if document.get("schema_version") != 1:
        raise ConformanceError("schema_version must equal 1")
    revision = document.get("catalog_revision")
    if not isinstance(revision, int) or revision < 1:
        raise ConformanceError("catalog_revision must be a positive integer")
    required = set(_strings(document.get("required_features"), "required_features"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ConformanceError("cases must be a non-empty array")

    ids: set[str] = set()
    covered: set[str] = set()
    for index, raw_case in enumerate(cases):
        context = f"cases[{index}]"
        case = _object(raw_case, context)
        case_id = _string(case.get("id"), f"{context}.id")
        if case_id in ids:
            raise ConformanceError(f"duplicate conformance id: {case_id}")
        ids.add(case_id)
        if case.get("tier") not in SUPPORTED_TIERS:
            raise ConformanceError(f"{case_id}: invalid verification tier")
        features = set(_strings(case.get("features"), f"{case_id}.features"))
        unknown = features - required
        if unknown:
            raise ConformanceError(f"{case_id}: unknown features {sorted(unknown)}")
        covered.update(features)

        supported = case.get("supported_input")
        if not isinstance(supported, bool):
            raise ConformanceError(f"{case_id}.supported_input must be boolean")
        expected = _object(case.get("expected"), f"{case_id}.expected")
        if set(expected) != EXPECTED_FIELDS:
            raise ConformanceError(f"{case_id}.expected must contain exactly {sorted(EXPECTED_FIELDS)}")
        for field in EXPECTED_FIELDS:
            _string(expected.get(field), f"{case_id}.expected.{field}")
        if supported and expected["outcome"] != "success":
            raise ConformanceError(f"{case_id}: supported inputs cannot declare an expected failure")

        test = _object(case.get("test"), f"{case_id}.test")
        source = repo_root / _path(test.get("source"), f"{case_id}.test.source")
        name = _string(test.get("name"), f"{case_id}.test.name")
        if not source.is_file():
            raise ConformanceError(f"{case_id}: test source does not exist")
        if f"fn {name}(" not in source.read_text():
            raise ConformanceError(f"{case_id}: test anchor {name} does not exist")

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
