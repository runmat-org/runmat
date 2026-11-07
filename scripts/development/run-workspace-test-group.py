#!/usr/bin/env python3
"""Run one complete, non-overlapping group of workspace package tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


# Match the native CLI command/Tokio-worker stack reserve. Unoptimized native
# continuations can exceed libtest's default 2-MiB stack. This is a per-thread
# virtual-address reserve; operating systems commit pages as they are used.
NATIVE_TEST_STACK_BYTES = 64 * 1024 * 1024


def test_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["RUST_MIN_STACK"] = str(NATIVE_TEST_STACK_BYTES)
    return environment


GROUPS = {
    "language": (
        "runmat-accelerate-api",
        "runmat-async",
        "runmat-builtins",
        "runmat-build-identity",
        "runmat-canonical-codec",
        "runmat-extension-abi",
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
        "runmat-workspace-contracts",
    ),
    "runtime": (
        "runmat-accelerate",
        "runmat-aot",
        "runmat-aot-runtime",
        "runmat-core",
        "runmat-cpu-execution",
        "runmat-simulation-usage",
        "runmat-clarabel",
        "runmat-filesystem",
        "runmat-gc",
        "runmat-gc-miri-tests",
        "runmat-jit",
        "runmat-java",
        "runmat-mex",
        "runmat-native-ffi",
        "runmat-native-codegen",
        "runmat-native-executor",
        "runmat-plot",
        "runmat-python",
        "runmat-runtime",
        "runmat-runtime-integration-tests",
        "runmat-static-analysis",
        "runmat-sparse-accelerate",
        "runmat-sparse-solvers",
        "runmat-telemetry",
        "runmat-time",
        "runmat-vm",
    ),
    "geometry-analysis": (
        "runmat-analysis-core",
        "runmat-analysis-distributed",
        "runmat-analysis-execution",
        "runmat-analysis-fea",
        "runmat-analysis-language",
        "runmat-analysis-qualification",
        "runmat-fea-source",
        "runmat-fea-syntax",
        "runmat-geometry-core",
        "runmat-geometry-fixtures",
        "runmat-geometry-io",
        "runmat-geometry-ops",
        "runmat-geometry-wasm-host",
        "runmat-meshing",
        "runmat-meshing-core",
        "runmat-meshing-curve",
        "runmat-meshing-execution",
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
        "runmat-fea-admission-native",
        "runmat-test-runner",
        "runmat-test-runner-execution",
        "runmat-test-runner-native",
        "runmat-wasm",
    ),
}

def scientific_filters(group: str) -> tuple[str, ...]:
    """Route the declared long cohorts separately; never change their assertions."""
    policy_path = Path(__file__).resolve().parents[2] / "qualification/scientific/policy.json"
    policy = json.loads(policy_path.read_text())
    if policy.get("schema_version") != 1:
        raise SystemExit("unsupported scientific qualification policy")
    return tuple(pattern for cohort in policy["cohorts"] if cohort.get("group") == group
                 for pattern in cohort["ordinary_skip"])


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


# The vendored solver retains upstream integrations, but RunMat depends on its
# pure-Rust core with default features disabled. --all-features would instead
# enable every external BLAS backend and Python/Julia binding simultaneously.
PACKAGE_TEST_FEATURE_PROFILES = {"runmat-clarabel": ((), ("serde",))}

# These harness-free programs parse positional arguments, not libtest filters.
# The release compiler census still checks every target; these bounded runs
# exercise their result assertions without collecting performance evidence.
STANDALONE_BENCHMARK_SMOKES = (
    ("runmat-accelerate", "direct_call_observation", "100", "3"),
    ("runmat-execution", "cancellation", "1000", "3"),
)


def benchmark_commands(group: str) -> list[list[str]]:
    # Keep the same package feature selection as the ordinary group. Selecting
    # only the benchmark's owner gives its dependencies a different feature
    # graph and can rebuild the large runtime solely for this bounded smoke.
    packages = [package for package in GROUPS[group]
                if package not in PACKAGE_TEST_FEATURE_PROFILES]
    return [
        ["cargo", "test", "--all-features",
         *(argument for member in packages for argument in ("--package", member)),
         "--bench", target, "--", iterations, samples]
        for package, target, iterations, samples in STANDALONE_BENCHMARK_SMOKES
        if package in GROUPS[group]
    ]


def test_commands(group: str, include_extended: bool) -> list[list[str]]:
    packages = GROUPS[group]
    common = ["cargo", "test", "--lib", "--bins", "--tests", "--examples", "--no-fail-fast"]
    ordinary = [package for package in packages if package not in PACKAGE_TEST_FEATURE_PROFILES]
    commands = [
        [*common, "--all-features", *(argument for package in ordinary for argument in ("--package", package))]
    ] if ordinary else []
    for package in packages:
        for features in PACKAGE_TEST_FEATURE_PROFILES.get(package, ()):
            command = [*common, "--no-default-features", "--package", package]
            if features:
                command.extend(("--features", ",".join(features)))
            commands.append(command)
    if not include_extended:
        filters = scientific_filters(group)
        if filters:
            for command in commands:
                command.append("--")
                for test_filter in filters:
                    command.extend(("--skip", test_filter))
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", choices=GROUPS)
    parser.add_argument(
        "--include-extended",
        action="store_true",
        help="include all declared scientific qualification cohorts in the selected group",
    )
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

    failure = 0
    for command in [*test_commands(args.group, args.include_extended), *benchmark_commands(args.group)]:
        code = subprocess.run(command, cwd=repo_root, env=test_environment()).returncode
        if code and not failure:
            failure = code
    return failure


if __name__ == "__main__":
    sys.exit(main())
