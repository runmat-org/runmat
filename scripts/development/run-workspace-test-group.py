#!/usr/bin/env python3
"""Run one complete, non-overlapping group of workspace package tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


GROUPS = {
    "language": (
        "runmat-accelerate-api",
        "runmat-async",
        "runmat-builtins",
        "runmat-canonical-codec",
        "runmat-gc-api",
        "runmat-hir",
        "runmat-lexer",
        "runmat-macros",
        "runmat-mir",
        "runmat-parser",
        "runmat-test",
        "runmat-thread-local",
        "runmat-types",
        "runmat-value",
    ),
    "runtime": (
        "runmat-accelerate",
        "runmat-aot",
        "runmat-aot-runtime",
        "runmat-core",
        "runmat-filesystem",
        "runmat-gc",
        "runmat-gc-miri-tests",
        "runmat-jit",
        "runmat-native-codegen",
        "runmat-native-executor",
        "runmat-plot",
        "runmat-runtime",
        "runmat-runtime-integration-tests",
        "runmat-static-analysis",
        "runmat-telemetry",
        "runmat-time",
        "runmat-vm",
    ),
    "geometry-analysis": (
        "runmat-analysis-core",
        "runmat-analysis-fea",
        "runmat-geometry-core",
        "runmat-geometry-fixtures",
        "runmat-geometry-io",
        "runmat-geometry-ops",
        "runmat-meshing",
        "runmat-meshing-cad",
        "runmat-meshing-core",
        "runmat-meshing-curve",
        "runmat-meshing-evidence",
        "runmat-meshing-execution",
        "runmat-meshing-opt",
        "runmat-meshing-plc",
        "runmat-meshing-size",
        "runmat-meshing-surface",
        "runmat-meshing-tetrahedron",
    ),
    "tooling-distributed": (
        "runmat",
        "runmat-config",
        "runmat-execution",
        "runmat-execution-artifact",
        "runmat-execution-runner",
        "runmat-execution-runner-native",
        "runmat-execution-transport-native",
        "runmat-logging",
        "runmat-lsp",
        "runmat-node-agent",
        "runmat-package",
        "runmat-package-cache",
        "runmat-package-cache-native",
        "runmat-process-host",
        "runmat-server-client",
        "runmat-test-runner",
        "runmat-test-runner-execution",
        "runmat-test-runner-native",
        "runmat-wasm",
    ),
}


def workspace_packages(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    return {package["name"] for package in metadata["packages"]}


def validate_groups(actual: set[str]) -> None:
    configured = [package for packages in GROUPS.values() for package in packages]
    duplicates = sorted({package for package in configured if configured.count(package) > 1})
    missing = sorted(actual - set(configured))
    unknown = sorted(set(configured) - actual)
    if duplicates or missing or unknown:
        details = []
        if duplicates:
            details.append(f"packages assigned more than once: {', '.join(duplicates)}")
        if missing:
            details.append(f"workspace packages without a test group: {', '.join(missing)}")
        if unknown:
            details.append(f"test-group packages absent from the workspace: {', '.join(unknown)}")
        raise SystemExit("invalid workspace test groups: " + "; ".join(details))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", choices=GROUPS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that every workspace package belongs to exactly one group",
    )
    args = parser.parse_args()
    if not args.check and args.group is None:
        parser.error("group is required unless --check is used")

    repo_root = Path(__file__).resolve().parents[2]
    validate_groups(workspace_packages(repo_root))
    if args.check:
        print(f"{sum(map(len, GROUPS.values()))} workspace packages assigned across {len(GROUPS)} groups")
        return 0

    command = [
        "cargo",
        "test",
        "--all-targets",
        "--all-features",
        *(argument for package in GROUPS[args.group] for argument in ("--package", package)),
    ]
    return subprocess.run(command, cwd=repo_root).returncode


if __name__ == "__main__":
    sys.exit(main())
