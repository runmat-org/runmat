#!/usr/bin/env python3
import json
import os
from pathlib import Path


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    path = Path(
        os.getenv(
            "RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH",
            "target/runmat-analysis-artifacts/external_reference_benchmark.json",
        )
    )
    enforce = is_true(os.getenv("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE", "false"))
    require_pass = is_true(
        os.getenv("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_REQUIRE_PASS", "false")
    )

    if not path.exists():
        print(f"external reference benchmark artifact missing: {path}")
        return 1 if enforce else 0

    payload = json.loads(path.read_text())
    errors = []

    if payload.get("schema_version") != "external-reference-benchmark/v1":
        errors.append("schema_version must be external-reference-benchmark/v1")
    if not isinstance(payload.get("scenario_id"), str):
        errors.append("scenario_id missing or invalid")
    if not isinstance(payload.get("reference_source"), dict):
        errors.append("reference_source missing or invalid")
    if not isinstance(payload.get("generated_at"), str):
        errors.append("generated_at missing or invalid")

    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("metrics must be a non-empty list")
    else:
        for idx, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"metrics[{idx}] must be an object")
                continue
            if not isinstance(metric.get("name"), str):
                errors.append(f"metrics[{idx}].name missing or invalid")
            if "source" in metric and metric.get("source") not in {
                "field",
                "threshold_assertion",
            }:
                errors.append(f"metrics[{idx}].source invalid")
            if not isinstance(metric.get("observed"), (int, float)):
                errors.append(f"metrics[{idx}].observed missing or invalid")
            if not isinstance(metric.get("reference"), (int, float)):
                errors.append(f"metrics[{idx}].reference missing or invalid")
            if not isinstance(metric.get("pass"), bool):
                errors.append(f"metrics[{idx}].pass missing or invalid")

    if require_pass and isinstance(metrics, list):
        failing = [m.get("name", f"index_{i}") for i, m in enumerate(metrics) if isinstance(m, dict) and m.get("pass") is False]
        if failing:
            errors.append(
                "reference comparison failed for metrics: " + ", ".join(str(item) for item in failing)
            )

    if errors:
        print("external reference benchmark validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1 if enforce else 0

    print("external reference benchmark validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
