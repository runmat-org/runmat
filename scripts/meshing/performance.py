#!/usr/bin/env python3
"""Generate and validate measured native meshing performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "verification/meshing/mesh-slo.toml"
DEFAULT_REPORT = REPO_ROOT / "target/runmat-meshing-verification/performance.json"


class PerformanceError(ValueError):
    """Raised when measured performance evidence is incomplete or outside policy."""


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PerformanceError(f"cannot read {context}: {error}") from error
    if not isinstance(value, dict):
        raise PerformanceError(f"{context} must be an object")
    return value


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PerformanceError(f"cannot read performance policy: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PerformanceError("performance policy schema_version must equal 1")
    for field in (
        "approved_at",
        "approval_basis",
        "workload_id",
        "fixture",
        "fixture_sha256",
        "expected_canonical_digest",
    ):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise PerformanceError(f"performance policy {field} must be a non-empty string")
    for field in ("policy_revision", "minimum_elements", "maximum_report_age_hours"):
        if not isinstance(value.get(field), int) or value[field] <= 0:
            raise PerformanceError(f"performance policy {field} must be a positive integer")
    required_stages = value.get("required_stages")
    if (
        not isinstance(required_stages, list)
        or not required_stages
        or not all(isinstance(stage, str) and stage for stage in required_stages)
        or len(required_stages) != len(set(required_stages))
    ):
        raise PerformanceError("performance policy required_stages must be a unique string array")
    profiles = value.get("profiles")
    limits = value.get("limits")
    if not isinstance(profiles, dict) or not isinstance(limits, dict):
        raise PerformanceError("performance policy requires profiles and limits tables")
    for profile in ("smoke", "stable"):
        profile_policy = profiles.get(profile)
        if not isinstance(profile_policy, dict):
            raise PerformanceError(f"performance policy is missing the {profile} profile")
        for field in ("minimum_samples", "warmup_samples"):
            setting = profile_policy.get(field)
            if not isinstance(setting, int) or setting < (1 if field == "minimum_samples" else 0):
                raise PerformanceError(f"performance profile {profile}.{field} is invalid")
    required_limits = {
        "wall_time_p50_ms",
        "wall_time_p95_ms",
        "wall_time_p99_ms",
        "peak_rss_p99_bytes",
        "allocation_count_p99",
        "allocated_bytes_p99",
        "peak_live_bytes_p99",
        "minimum_element_throughput_per_second_p50",
        "maximum_wall_time_tail_ratio",
    }
    if set(limits) != required_limits or any(
        isinstance(limits[field], bool)
        or not isinstance(limits[field], (int, float))
        or not math.isfinite(limits[field])
        or limits[field] <= 0
        for field in required_limits
    ):
        raise PerformanceError("performance policy limits are incomplete or nonpositive")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise PerformanceError("cannot calculate a percentile without samples")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _maximum_rss_bytes(rusage: Any) -> int:
    maximum = int(rusage.ru_maxrss)
    return maximum if sys.platform == "darwin" else maximum * 1024


def _run_process(command: list[str], environment: dict[str, str]) -> tuple[int, str, str, int]:
    if not hasattr(os, "wait4"):
        raise PerformanceError("peak-RSS verification requires a platform with wait4 resource usage")
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment)
        _, status, rusage = os.wait4(process.pid, 0)
        return_code = os.waitstatus_to_exitcode(status)
        process.returncode = return_code
        maximum_rss = _maximum_rss_bytes(rusage)
        stdout.seek(0)
        stderr.seek(0)
        return return_code, stdout.read().decode(), stderr.read().decode(), maximum_rss


def _allocation_totals(directory: Path) -> dict[str, int]:
    records = [_load_json(path, "allocation record") for path in sorted(directory.glob("process-*.json"))]
    if not records:
        raise PerformanceError("instrumented run emitted no process allocation records")
    process_ids: set[int] = set()
    for record in records:
        if record.get("schema_version") != 1:
            raise PerformanceError("allocation record schema_version must equal 1")
        process_id = record.get("process_id")
        if not isinstance(process_id, int) or process_id in process_ids:
            raise PerformanceError("allocation process identities must be unique integers")
        process_ids.add(process_id)
        for field in ("allocation_count", "allocated_bytes", "peak_live_bytes", "peak_rss_bytes"):
            if not isinstance(record.get(field), int) or record[field] < 0:
                raise PerformanceError(f"allocation record {field} must be a nonnegative integer")
    return {
        "process_count": len(records),
        "allocation_count": sum(record["allocation_count"] for record in records),
        "allocated_bytes": sum(record["allocated_bytes"] for record in records),
        "peak_live_bytes": max(record["peak_live_bytes"] for record in records),
        "peak_rss_bytes": max(record["peak_rss_bytes"] for record in records),
    }


def _sample(binary: Path, policy: dict[str, Any], root: Path) -> dict[str, Any]:
    allocation_directory = root / "allocations"
    allocation_directory.mkdir()
    artifact = root / "mesh.cbor"
    evidence = root / "evidence.cbor"
    fixture = REPO_ROOT / str(policy["fixture"])
    environment = dict(os.environ)
    environment["RUNMAT_MESH_ALLOCATION_REPORT_DIR"] = str(allocation_directory)
    command = [
        str(binary),
        str(fixture),
        "--output",
        str(artifact),
        "--evidence",
        str(evidence),
        "--target-size",
        "10",
        "--deviation",
        "0.1",
        "--max-elements",
        "10000",
    ]
    started = time.perf_counter_ns()
    return_code, stdout, stderr, maximum_rss = _run_process(command, environment)
    wall_time_ms = (time.perf_counter_ns() - started) / 1_000_000
    if return_code != 0:
        raise PerformanceError(f"meshing benchmark failed ({return_code}): {stderr[-2000:]}")
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise PerformanceError(f"meshing benchmark emitted invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise PerformanceError("meshing benchmark result must be an object")
    allocations = _allocation_totals(allocation_directory)
    stages: dict[str, dict[str, int]] = {}
    for stage in result.get("stages", []):
        name = stage.get("stage")
        if not isinstance(name, str):
            raise PerformanceError("stage evidence is missing a stage name")
        aggregate = stages.setdefault(name, {"elapsed_time_ms": 0, "peak_memory_bytes": 0})
        aggregate["elapsed_time_ms"] += int(stage["elapsed_time_ms"])
        aggregate["peak_memory_bytes"] = max(
            aggregate["peak_memory_bytes"], int(stage["peak_memory_bytes"])
        )
    element_count = int(result["element_count"])
    return {
        "wall_time_ms": wall_time_ms,
        "peak_rss_bytes": max(maximum_rss, allocations["peak_rss_bytes"]),
        "allocation_count": allocations["allocation_count"],
        "allocated_bytes": allocations["allocated_bytes"],
        "peak_live_bytes": allocations["peak_live_bytes"],
        "process_count": allocations["process_count"],
        "element_count": element_count,
        "element_throughput_per_second": element_count * 1000.0 / wall_time_ms,
        "canonical_digest": result.get("canonical_digest"),
        "stages": stages,
    }


def _summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "wall_time_ms",
        "peak_rss_bytes",
        "allocation_count",
        "allocated_bytes",
        "peak_live_bytes",
        "element_throughput_per_second",
    )
    summary = {field: _distribution([float(sample[field]) for sample in samples]) for field in fields}
    stage_names = sorted({name for sample in samples for name in sample["stages"]})
    summary["stages"] = {
        name: {
            "elapsed_time_ms": _distribution(
                [float(sample["stages"][name]["elapsed_time_ms"]) for sample in samples]
            ),
            "peak_memory_bytes": max(
                sample["stages"][name]["peak_memory_bytes"] for sample in samples
            ),
        }
        for name in stage_names
    }
    return summary


def generate_report(binary: Path, policy_path: Path, profile: str, samples: int) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    profile_policy = policy.get("profiles", {}).get(profile)
    if not isinstance(profile_policy, dict):
        raise PerformanceError(f"unknown performance profile: {profile}")
    minimum_samples = profile_policy.get("minimum_samples")
    if not isinstance(minimum_samples, int) or samples < minimum_samples:
        raise PerformanceError(f"profile {profile} requires at least {minimum_samples} samples")
    warmup_samples = profile_policy.get("warmup_samples")
    if not isinstance(warmup_samples, int) or warmup_samples < 0:
        raise PerformanceError(f"profile {profile} has an invalid warmup sample count")
    if not binary.is_file():
        raise PerformanceError(f"benchmark binary does not exist: {binary}")
    fixture = REPO_ROOT / str(policy["fixture"])
    measured: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="runmat-meshing-performance-") as temporary:
        root = Path(temporary)
        for index in range(warmup_samples):
            sample_root = root / f"warmup-{index:03}"
            sample_root.mkdir()
            _sample(binary, policy, sample_root)
        for index in range(samples):
            sample_root = root / f"sample-{index:03}"
            sample_root.mkdir()
            measured.append(_sample(binary, policy, sample_root))
    return {
        "schema_version": 1,
        "policy_revision": policy["policy_revision"],
        "workload_id": policy["workload_id"],
        "profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {"os": platform.system().lower(), "architecture": platform.machine().lower()},
        "binary_sha256": _sha256(binary),
        "fixture_sha256": _sha256(fixture),
        "samples": measured,
        "summary": _summarize(measured),
    }


def validate_report(report: dict[str, Any], policy: dict[str, Any], profile: str) -> None:
    if report.get("schema_version") != 1 or report.get("policy_revision") != policy.get("policy_revision"):
        raise PerformanceError("performance report schema or policy revision is stale")
    if report.get("workload_id") != policy.get("workload_id") or report.get("profile") != profile:
        raise PerformanceError("performance report workload or profile does not match policy")
    fixture = REPO_ROOT / str(policy["fixture"])
    if _sha256(fixture) != policy["fixture_sha256"]:
        raise PerformanceError("performance policy fixture digest is stale")
    if report.get("fixture_sha256") != policy["fixture_sha256"]:
        raise PerformanceError("performance report fixture digest does not match policy")
    binary_digest = report.get("binary_sha256")
    if (
        not isinstance(binary_digest, str)
        or len(binary_digest) != 64
        or any(character not in "0123456789abcdef" for character in binary_digest)
    ):
        raise PerformanceError("performance report binary digest is invalid")
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise PerformanceError("performance report samples must be an array")
    minimum = policy.get("profiles", {}).get(profile, {}).get("minimum_samples")
    if not isinstance(minimum, int) or len(samples) < minimum:
        raise PerformanceError(f"performance report requires at least {minimum} samples")
    generated = report.get("generated_at")
    try:
        generated_at = datetime.fromisoformat(generated)
    except (TypeError, ValueError) as error:
        raise PerformanceError("performance report generated_at is invalid") from error
    if generated_at.utcoffset() is None:
        raise PerformanceError("performance report generated_at must include a timezone")
    maximum_age = float(policy["maximum_report_age_hours"]) * 3600
    age = (datetime.now(timezone.utc) - generated_at).total_seconds()
    if age < -300 or age > maximum_age:
        raise PerformanceError("performance report is future-dated or stale")
    digest = policy["expected_canonical_digest"]
    required_stages = set(policy["required_stages"])
    for sample in samples:
        if not isinstance(sample, dict):
            raise PerformanceError("every performance sample must be an object")
        for field in (
            "wall_time_ms",
            "peak_rss_bytes",
            "allocation_count",
            "allocated_bytes",
            "peak_live_bytes",
            "process_count",
            "element_count",
            "element_throughput_per_second",
        ):
            measurement = sample.get(field)
            if (
                isinstance(measurement, bool)
                or not isinstance(measurement, (int, float))
                or not math.isfinite(measurement)
                or measurement <= 0
            ):
                raise PerformanceError(f"performance sample {field} must be positive")
        if sample.get("canonical_digest") != digest:
            raise PerformanceError("performance sample canonical digest changed")
        if sample.get("element_count", 0) < policy["minimum_elements"]:
            raise PerformanceError("performance sample element inventory is incomplete")
        stages = sample.get("stages")
        if not isinstance(stages, dict) or set(stages) != required_stages:
            raise PerformanceError("performance sample stage inventory differs from policy")
        for name, stage in stages.items():
            if not isinstance(name, str) or not isinstance(stage, dict):
                raise PerformanceError("performance stage evidence must be an object by name")
            for field in ("elapsed_time_ms", "peak_memory_bytes"):
                if not isinstance(stage.get(field), int) or stage[field] < 0:
                    raise PerformanceError(f"performance stage {name}.{field} is invalid")
    recomputed = _summarize(samples)
    if report.get("summary") != recomputed:
        raise PerformanceError("performance summary does not match raw samples")
    limits = policy["limits"]
    checks = (
        (recomputed["wall_time_ms"]["p50"], limits["wall_time_p50_ms"], "wall-time p50"),
        (recomputed["wall_time_ms"]["p95"], limits["wall_time_p95_ms"], "wall-time p95"),
        (recomputed["wall_time_ms"]["p99"], limits["wall_time_p99_ms"], "wall-time p99"),
        (recomputed["peak_rss_bytes"]["p99"], limits["peak_rss_p99_bytes"], "peak RSS p99"),
        (recomputed["allocation_count"]["p99"], limits["allocation_count_p99"], "allocation-count p99"),
        (recomputed["allocated_bytes"]["p99"], limits["allocated_bytes_p99"], "allocated-bytes p99"),
        (recomputed["peak_live_bytes"]["p99"], limits["peak_live_bytes_p99"], "peak-live-bytes p99"),
    )
    for measured, limit, label in checks:
        if measured > limit:
            raise PerformanceError(f"{label} {measured:.3f} exceeds SLO {limit}")
    throughput = recomputed["element_throughput_per_second"]["p50"]
    if throughput < limits["minimum_element_throughput_per_second_p50"]:
        raise PerformanceError("element-throughput p50 is below its SLO")
    tail_ratio = recomputed["wall_time_ms"]["p99"] / recomputed["wall_time_ms"]["p50"]
    if tail_ratio > limits["maximum_wall_time_tail_ratio"]:
        raise PerformanceError("wall-time distribution is not statistically stable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--profile", choices=("smoke", "stable"), default="smoke")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        policy = _load_policy(arguments.policy)
        if arguments.validate_only:
            report = _load_json(arguments.report, "performance report")
        else:
            if arguments.binary is None or arguments.samples is None:
                raise PerformanceError("generation requires --binary and --samples")
            report = generate_report(arguments.binary, arguments.policy, arguments.profile, arguments.samples)
            arguments.report.parent.mkdir(parents=True, exist_ok=True)
            arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        validate_report(report, policy, arguments.profile)
    except PerformanceError as error:
        print(f"meshing performance verification failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {len(report['samples'])} {arguments.profile} meshing performance samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
