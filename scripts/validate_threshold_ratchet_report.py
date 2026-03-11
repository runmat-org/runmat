#!/usr/bin/env python3
import json
import os
from pathlib import Path


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    report_path = Path(
        os.getenv(
            "RUNMAT_THRESHOLD_RATCHET_REPORT",
            "target/runmat-analysis-artifacts/threshold_ratchet_report.json",
        )
    )
    enforce = _is_true(os.getenv("RUNMAT_VALIDATE_THRESHOLD_RATCHET_ENFORCE", "false"))
    require_observed = _is_true(
        os.getenv("RUNMAT_VALIDATE_THRESHOLD_RATCHET_REQUIRE_OBSERVED", "false")
    )

    if not report_path.exists():
        print(f"threshold ratchet report missing: {report_path}")
        return 1 if enforce else 0

    report = json.loads(report_path.read_text())
    profile = report.get("governance_profile", "unknown")
    entries = report.get("entries", [])
    rationale = report.get("rationale")

    errors = []
    if not isinstance(entries, list) or not entries:
        errors.append("ratchet report has no entries")

    if profile in {"development", "feature"}:
        if rationale != "rolling_median_reference_fixtures":
            errors.append(
                "non-release profile ratchet rationale must be rolling_median_reference_fixtures"
            )
        for entry in entries:
            status = entry.get("status")
            old = entry.get("old")
            new = entry.get("new")
            key = entry.get("threshold_key", "<unknown>")
            if status != "ratcheted":
                errors.append(f"{key}: expected ratcheted status, got {status}")
            if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
                errors.append(f"{key}: old/new must be numeric")
            elif float(new) >= float(old):
                errors.append(f"{key}: expected new < old for non-release ratchet")
    elif profile == "release":
        for entry in entries:
            old = entry.get("old")
            new = entry.get("new")
            key = entry.get("threshold_key", "<unknown>")
            if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
                errors.append(f"{key}: old/new must be numeric")
            elif float(new) > float(old):
                errors.append(
                    f"{key}: release profile must be non-regressive (new <= old)"
                )

    if require_observed:
        for entry in entries:
            key = entry.get("threshold_key", "<unknown>")
            observed = entry.get("observed")
            if not isinstance(observed, (int, float)):
                errors.append(f"{key}: observed metric required but missing/non-numeric")

    if errors:
        print("threshold ratchet validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1 if enforce else 0

    print("threshold ratchet report validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
