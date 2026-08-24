#!/usr/bin/env python3
"""Validate independent-mesher comparisons and mismatch dispositions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "verification/meshing/differential.json"
DISPOSITIONS = {"accepted", "fixed", "reference-corrected"}


class DifferentialError(ValueError):
    """Raised when differential evidence or its disposition is incomplete."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DifferentialError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DifferentialError(f"{context} must be a non-empty string")
    return value


def _strings(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DifferentialError(f"{context} must be a non-empty array")
    strings = [_string(item, f"{context}[]") for item in value]
    if len(strings) != len(set(strings)):
        raise DifferentialError(f"{context} must be unique")
    return strings


def _path(value: Any, context: str) -> Path:
    path = Path(_string(value, context))
    if path.is_absolute() or ".." in path.parts:
        raise DifferentialError(f"{context} must stay below the repository root")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_catalog(catalog_path: Path, repo_root: Path = REPO_ROOT) -> int:
    try:
        document = _object(json.loads(catalog_path.read_text()), "catalog")
    except (OSError, json.JSONDecodeError) as error:
        raise DifferentialError(f"cannot read differential catalog: {error}") from error
    if document.get("schema_version") != 1:
        raise DifferentialError("schema_version must equal 1")
    revision = document.get("catalog_revision")
    if not isinstance(revision, int) or revision < 1:
        raise DifferentialError("catalog_revision must be a positive integer")
    _string(document.get("revision_explanation"), "revision_explanation")
    required = set(_strings(document.get("required_comparisons"), "required_comparisons"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DifferentialError("cases must be a non-empty array")

    identities: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _object(raw_case, f"cases[{index}]")
        identity = _string(case.get("id"), f"cases[{index}].id")
        if identity in identities:
            raise DifferentialError(f"duplicate differential case: {identity}")
        identities.add(identity)
        fixture = repo_root / _path(case.get("fixture"), f"{identity}.fixture")
        digest = _string(case.get("fixture_sha256"), f"{identity}.fixture_sha256")
        if not fixture.is_file() or _sha256(fixture) != digest:
            raise DifferentialError(f"{identity}: fixture is missing or its digest changed")

        meshers = case.get("reference_meshers")
        if not isinstance(meshers, list) or not meshers:
            raise DifferentialError(f"{identity}: reference_meshers must be a non-empty array")
        for mesher_index, raw_mesher in enumerate(meshers):
            mesher = _object(raw_mesher, f"{identity}.reference_meshers[{mesher_index}]")
            _string(mesher.get("name"), f"{identity}.reference_meshers[].name")
            _string(mesher.get("implementation"), f"{identity}.reference_meshers[].implementation")
            if mesher.get("trusted") is not True or mesher.get("independent_of_runmat_generator") is not True:
                raise DifferentialError(f"{identity}: every reference mesher must be trusted and independent")

        comparisons = _object(case.get("comparisons"), f"{identity}.comparisons")
        if set(comparisons) != required:
            raise DifferentialError(f"{identity}: comparison inventory differs from required policy")
        for comparison, explanation in comparisons.items():
            _string(explanation, f"{identity}.comparisons.{comparison}")

        mismatches = case.get("mismatches")
        if not isinstance(mismatches, list):
            raise DifferentialError(f"{identity}.mismatches must be an array")
        metrics: set[str] = set()
        for mismatch_index, raw_mismatch in enumerate(mismatches):
            mismatch = _object(raw_mismatch, f"{identity}.mismatches[{mismatch_index}]")
            metric = _string(mismatch.get("metric"), f"{identity}.mismatches[].metric")
            if metric in metrics:
                raise DifferentialError(f"{identity}: duplicate mismatch metric {metric}")
            metrics.add(metric)
            _string(mismatch.get("runmat"), f"{identity}.{metric}.runmat")
            _string(mismatch.get("reference"), f"{identity}.{metric}.reference")
            if mismatch.get("disposition") not in DISPOSITIONS:
                raise DifferentialError(f"{identity}: mismatch {metric} lacks a reviewed disposition")
            _string(mismatch.get("explanation"), f"{identity}.{metric}.explanation")

        test = _object(case.get("test"), f"{identity}.test")
        source = repo_root / _path(test.get("source"), f"{identity}.test.source")
        name = _string(test.get("name"), f"{identity}.test.name")
        if not source.is_file() or f"fn {name}(" not in source.read_text():
            raise DifferentialError(f"{identity}: executable test anchor is missing")
    return len(cases)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", nargs="?", type=Path, default=DEFAULT_CATALOG)
    arguments = parser.parse_args(argv)
    try:
        count = validate_catalog(arguments.catalog)
    except DifferentialError as error:
        print(f"meshing differential validation failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {count} independently meshed differential cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
