#!/usr/bin/env python3
"""Validate independent-mesher comparisons and mismatch dispositions."""

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
        string_array,
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
        string_array,
        string_value,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "verification/meshing/differential.json"
DISPOSITIONS = {"accepted", "fixed", "reference-corrected"}


DifferentialError = CatalogError


def validate_catalog(catalog_path: Path, repo_root: Path = REPO_ROOT) -> int:
    document = read_document(catalog_path, "differential catalog")
    if document.get("schema_version") != 1:
        raise DifferentialError("schema_version must equal 1")
    revision = document.get("catalog_revision")
    if not isinstance(revision, int) or revision < 1:
        raise DifferentialError("catalog_revision must be a positive integer")
    string_value(document.get("revision_explanation"), "revision_explanation")
    required = set(string_array(document.get("required_comparisons"), "required_comparisons"))
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DifferentialError("cases must be a non-empty array")

    identities: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = object_value(raw_case, f"cases[{index}]")
        identity = string_value(case.get("id"), f"cases[{index}].id")
        if identity in identities:
            raise DifferentialError(f"duplicate differential case: {identity}")
        identities.add(identity)
        fixture = repo_root / relative_path(case.get("fixture"), f"{identity}.fixture")
        digest = string_value(case.get("fixture_sha256"), f"{identity}.fixture_sha256")
        if not fixture.is_file() or sha256(fixture) != digest:
            raise DifferentialError(f"{identity}: fixture is missing or its digest changed")

        meshers = case.get("reference_meshers")
        if not isinstance(meshers, list) or not meshers:
            raise DifferentialError(f"{identity}: reference_meshers must be a non-empty array")
        for mesher_index, raw_mesher in enumerate(meshers):
            mesher = object_value(raw_mesher, f"{identity}.reference_meshers[{mesher_index}]")
            string_value(mesher.get("name"), f"{identity}.reference_meshers[].name")
            string_value(mesher.get("implementation"), f"{identity}.reference_meshers[].implementation")
            if mesher.get("trusted") is not True or mesher.get("independent_of_runmat_generator") is not True:
                raise DifferentialError(f"{identity}: every reference mesher must be trusted and independent")

        comparisons = object_value(case.get("comparisons"), f"{identity}.comparisons")
        if set(comparisons) != required:
            raise DifferentialError(f"{identity}: comparison inventory differs from required policy")
        for comparison, explanation in comparisons.items():
            string_value(explanation, f"{identity}.comparisons.{comparison}")

        mismatches = case.get("mismatches")
        if not isinstance(mismatches, list):
            raise DifferentialError(f"{identity}.mismatches must be an array")
        metrics: set[str] = set()
        for mismatch_index, raw_mismatch in enumerate(mismatches):
            mismatch = object_value(raw_mismatch, f"{identity}.mismatches[{mismatch_index}]")
            metric = string_value(mismatch.get("metric"), f"{identity}.mismatches[].metric")
            if metric in metrics:
                raise DifferentialError(f"{identity}: duplicate mismatch metric {metric}")
            metrics.add(metric)
            string_value(mismatch.get("runmat"), f"{identity}.{metric}.runmat")
            string_value(mismatch.get("reference"), f"{identity}.{metric}.reference")
            if mismatch.get("disposition") not in DISPOSITIONS:
                raise DifferentialError(f"{identity}: mismatch {metric} lacks a reviewed disposition")
            string_value(mismatch.get("explanation"), f"{identity}.{metric}.explanation")

        require_test_anchor(repo_root, case.get("test"), f"{identity}.test")
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
