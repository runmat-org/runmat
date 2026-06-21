#!/usr/bin/env python3
"""Validate public FEA field-id literals.

This gate is deliberately scoped to field contract sources. Some legacy fixture
ids still contain "proxy"; those are not public result field ids and should be
renamed separately from this contract guard.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT_PATHS = (
    REPO_ROOT / "crates/runmat-analysis/fea/src/contracts/mod.rs",
)
FIELD_NAMESPACE_RE = re.compile(
    r'"((?:structural|modal|acoustic|thermal|transient|nonlinear|'
    r"thermo_mechanical|electro_thermal|em|cfd|cht|fsi)\.[^\"]+)\""
)
FORBIDDEN_TOKENS = ("proxy", "placeholder")


def public_field_ids(paths: list[Path]) -> list[tuple[Path, int, str]]:
    field_ids: list[tuple[Path, int, str]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in FIELD_NAMESPACE_RE.finditer(line):
                field_ids.append((path, line_number, match.group(1)))
    return field_ids


def validate(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path, line_number, field_id in public_field_ids(paths):
        lowered = field_id.lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            try:
                display_path = path.relative_to(REPO_ROOT)
            except ValueError:
                display_path = path
            errors.append(
                f"{display_path}:{line_number}: forbidden public field id '{field_id}'"
            )
    return errors


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(arg) for arg in args] if args else list(DEFAULT_CONTRACT_PATHS)
    errors = validate(paths)
    if errors:
        print("public FEA field-id validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("public FEA field-id validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
