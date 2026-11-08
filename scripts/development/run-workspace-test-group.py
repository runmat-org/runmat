#!/usr/bin/env python3
"""Run workspace package-test groups or their explicit, disjoint command shards."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile


# Match the native CLI command/Tokio-worker stack reserve. Unoptimized native
# continuations can exceed libtest's default 2-MiB stack. This is a per-thread
# virtual-address reserve; operating systems commit pages as they are used.
NATIVE_TEST_STACK_BYTES = 64 * 1024 * 1024


def test_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["RUST_MIN_STACK"] = str(NATIVE_TEST_STACK_BYTES)
    return environment


@contextmanager
def test_runtime_environment():
    environment = test_environment()
    if sys.platform != "linux":
        yield environment
        return
    # Native GPU discovery also runs in child processes. Keep their physical
    # runtime directory valid for the entire test group, including cold reopen.
    with tempfile.TemporaryDirectory(prefix="runmat-test-xdg-") as directory:
        os.chmod(directory, 0o700)
        environment["XDG_RUNTIME_DIR"] = directory
        yield environment


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
    return {package["name"] for package in workspace_metadata(repo_root)["packages"]}


def workspace_metadata(repo_root: Path) -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    metadata = json.loads(result.stdout)
    return metadata


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


def test_commands(group: str, include_extended: bool, metadata: dict | None = None) -> list[list[str]]:
    packages = GROUPS[group]
    common = ["cargo", "test", "--no-fail-fast"]
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
    if metadata is None:
        selections = [["--lib", "--bins", "--tests", "--examples"]]
    else:
        # Keep the complete package/feature union for every invocation. Only the
        # linked root targets vary; dependencies remain reusable between batches.
        targets = [target for package in metadata["packages"] if package["name"] in packages
                   and package["name"] not in PACKAGE_TEST_FEATURE_PROFILES
                   for target in package["targets"]]
        selections = []
        library_kinds = {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}
        if any(library_kinds.intersection(target["kind"]) for target in targets):
            selections.append(["--lib"])
        if any("bin" in target["kind"] for target in targets):
            selections.append(["--bins"])
        for kind, flag in (("test", "--test"), ("example", "--example")):
            names = Counter(target["name"] for target in targets if kind in target["kind"])
            batch, count = [], 0
            for name, roots in sorted(names.items()):
                # Cargo selects a shared target name in every selected package.
                # Keep those roots together while preserving the feature union.
                if batch and count + roots > 4:
                    selections.append(batch)
                    batch, count = [], 0
                batch.extend((flag, name))
                count += roots
            if batch:
                selections.append(batch)
    ordinary_count = 1 if ordinary else 0
    commands = [commands[0] + selection for selection in selections] + [
        command + ["--lib", "--bins", "--tests", "--examples"]
        for command in commands[ordinary_count:]
    ] if ordinary else [command + ["--lib", "--bins", "--tests", "--examples"] for command in commands]
    if not include_extended:
        filters = scientific_filters(group)
        if filters:
            for command in commands:
                command.append("--")
                for test_filter in filters:
                    command.extend(("--skip", test_filter))
    return commands


def shard_commands(commands: list[list[str]], index: int, count: int) -> list[list[str]]:
    """Partition complete Cargo invocations; preserve every package/feature union."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index must be in [0, shard count), with a positive count")
    if count > len(commands):
        raise ValueError("shard count exceeds the number of Cargo invocations")
    return commands[index::count]


# A single observed Windows batch added 48 GiB. Compact retained archives
# before that working set can exhaust the volume, while keeping runner logs.
WINDOWS_ARCHIVE_FREE_SPACE = 64 * 1024 ** 3
WINDOWS_ARCHIVE_MIN_BYTES = 16 * 1024 ** 2


def compact_windows_archives(target_directory: Path) -> int:
    """Compress completed dependency archives under disk pressure; retain their bytes."""
    if os.name != "nt" or not target_directory.is_dir():
        return 0
    if shutil.disk_usage(target_directory).free >= WINDOWS_ARCHIVE_FREE_SPACE:
        return 0
    dependencies = target_directory / "debug" / "deps"
    if not dependencies.is_dir():
        return 0
    if any(directory.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
           for directory in (target_directory, dependencies.parent, dependencies)):
        raise RuntimeError("dependency archive compression refuses a redirected directory")
    candidates = []
    with os.scandir(dependencies) as entries:
        for entry in entries:
            if Path(entry.name).suffix.lower() not in {".rlib", ".rmeta", ".lib", ".dll", ".pdb"}:
                continue
            info = entry.stat(follow_symlinks=False)
            if (not stat.S_ISREG(info.st_mode) or info.st_size < WINDOWS_ARCHIVE_MIN_BYTES
                    or info.st_file_attributes & (stat.FILE_ATTRIBUTE_COMPRESSED
                                                   | stat.FILE_ATTRIBUTE_REPARSE_POINT)):
                continue
            candidates.append((info.st_size, entry.path))
    before = shutil.disk_usage(target_directory).free
    compressed = 0
    compact = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "compact.exe")
    for _, archive in sorted(candidates, reverse=True):
        if shutil.disk_usage(target_directory).free >= WINDOWS_ARCHIVE_FREE_SPACE:
            break
        result = subprocess.run([compact, "/C", "/I", "/Q", archive],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", check=False)
        if (result.returncode or not Path(archive).stat().st_file_attributes
                & stat.FILE_ATTRIBUTE_COMPRESSED):
            print(f"Dependency archive compression failed for {archive}: {result.stdout}", flush=True)
        else:
            compressed += 1
    after = shutil.disk_usage(target_directory).free
    print(f"Windows dependency archive compression: {compressed} files; "
          f"{before // (1024 ** 3)} -> {after // (1024 ** 3)} GiB free", flush=True)
    return compressed


def run_batch(command: list[str], repo_root: Path, target_directory: Path, reclaim: bool,
              environment: dict[str, str] | None = None) -> int:
    """Retain dependencies; release only completed root test executables in CI."""
    command = [*command[:2], "--message-format=json-render-diagnostics", *command[2:]]
    print("+ " + " ".join(command), flush=True)
    if reclaim and os.name == "nt":
        compact_windows_archives(target_directory)
        checked_volumes: set[str] = set()
        for directory in (repo_root, target_directory):
            existing = directory.resolve()
            while not existing.exists():
                if existing.parent == existing:
                    raise RuntimeError(f"build storage is unavailable: {directory}")
                existing = existing.parent
            if existing.anchor in checked_volumes:
                continue
            checked_volumes.add(existing.anchor)
            storage = shutil.disk_usage(existing)
            print(f"Windows build storage {existing.anchor}: {storage.free // (1024 ** 3)} GiB free "
                  f"of {storage.total // (1024 ** 3)} GiB", flush=True)
            if storage.free < 10 * 1024 ** 3:
                raise RuntimeError("less than 10 GiB remains for the next build batch and runner logs")
    executables: set[Path] = set()
    with subprocess.Popen(command, cwd=repo_root, env=environment if environment is not None else test_environment(), encoding="utf-8", errors="replace",
                          stdout=subprocess.PIPE) as process:
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                continue
            if not isinstance(message, dict) or "reason" not in message:
                print(line, end="", flush=True)
                continue
            if message["reason"] not in {"compiler-message", "compiler-artifact", "build-script-executed", "build-finished"}:
                print(line, end="", flush=True)
                continue
            if message["reason"] == "compiler-message":
                print(message["message"].get("rendered") or "", end="", flush=True)
            if (message["reason"] == "compiler-artifact" and message.get("executable")
                    and (message.get("profile", {}).get("test")
                         or "example" in message.get("target", {}).get("kind", []))):
                executable = Path(message["executable"]).resolve()
                if executable.is_relative_to(target_directory.resolve()):
                    executables.add(executable)
        code = process.wait()
    print(f"Batch exit: {code}", flush=True)
    if reclaim:
        for executable in executables:
            executable.unlink(missing_ok=True)
            # MSVC emits a matching root PDB; dependency PDBs stay in the cache.
            if os.name == "nt":
                executable.with_suffix(".pdb").unlink(missing_ok=True)
        compact_windows_archives(target_directory)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", nargs="?", choices=GROUPS)
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based whole-command shard")
    parser.add_argument("--shard-count", type=int, default=1, help="number of disjoint command shards")
    parser.add_argument(
        "--reclaim-completed-test-binaries", action="store_true",
        help="remove completed root test executables from this exclusive CI target directory",
    )
    parser.add_argument(
        "--benchmark-smokes", action="store_true",
        help="also build and execute the standalone benchmark programs (manual qualification)",
    )
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
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count), with a positive count")
    if not args.check and args.group is None:
        parser.error("group is required unless --check is used")

    repo_root = Path(__file__).resolve().parents[2]
    metadata = workspace_metadata(repo_root)
    validate_groups({package["name"] for package in metadata["packages"]})
    if args.check:
        print(f"{sum(map(len, GROUPS.values()))} workspace packages assigned across {len(GROUPS)} groups")
        return 0

    failure = 0
    commands = test_commands(args.group, args.include_extended, metadata)
    if args.benchmark_smokes:
        commands.extend(benchmark_commands(args.group))
    commands = shard_commands(commands, args.shard_index, args.shard_count)
    with test_runtime_environment() as environment:
        for command in commands:
            code = run_batch(command, repo_root, Path(metadata["target_directory"]),
                             args.reclaim_completed_test_binaries, environment)
            if code and not failure:
                failure = code
    return failure


if __name__ == "__main__":
    sys.exit(main())
