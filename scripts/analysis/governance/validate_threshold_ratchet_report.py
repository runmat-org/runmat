#!/usr/bin/env python3
import json
import os
from pathlib import Path

REQUIRED_THRESHOLD_KEYS = {
    "RUNMAT_RELEASE_READINESS_PLASTIC_MAX_TREND_RATIO",
    "RUNMAT_RELEASE_READINESS_PLASTIC_REFERENCE_MAX_TREND_RATIO",
    "RUNMAT_RELEASE_READINESS_CONTACT_MAX_TREND_RATIO",
    "RUNMAT_RELEASE_READINESS_CONTACT_REFERENCE_MAX_TREND_RATIO",
    "RUNMAT_RELEASE_READINESS_THERMAL_MAX_SPREAD_TREND_RATIO",
}


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
    rolling_count = report.get("rolling_report_count")
    trusted_rolling_count = report.get("rolling_trusted_report_count", rolling_count)

    errors = []
    if report.get("schema_version") != "threshold-ratchet-report/v1":
        errors.append("schema_version must be threshold-ratchet-report/v1")
    if profile not in {"release", "development", "feature"}:
        errors.append(
            "governance_profile must be one of: release, development, feature"
        )
    if not isinstance(entries, list) or not entries:
        errors.append("ratchet report has no entries")
        entries = []
    if not isinstance(rolling_count, int) or rolling_count < 0:
        errors.append("rolling_report_count must be non-negative int")
    if not isinstance(trusted_rolling_count, int) or trusted_rolling_count < 0:
        errors.append("rolling_trusted_report_count must be non-negative int")
    elif isinstance(rolling_count, int) and trusted_rolling_count > rolling_count:
        errors.append(
            "rolling_trusted_report_count must be less than or equal to rolling_report_count"
        )

    threshold_key_counts = {}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("entries must contain only objects")
            continue
        key = entry.get("threshold_key")
        if not isinstance(key, str) or not key:
            errors.append("entries[].threshold_key must be non-empty string")
            continue
        threshold_key_counts[key] = threshold_key_counts.get(key, 0) + 1

    present_threshold_keys = set(threshold_key_counts.keys())
    missing_threshold_keys = sorted(REQUIRED_THRESHOLD_KEYS - present_threshold_keys)
    unexpected_threshold_keys = sorted(present_threshold_keys - REQUIRED_THRESHOLD_KEYS)
    duplicate_threshold_keys = sorted(
        key for key, count in threshold_key_counts.items() if count > 1
    )

    if missing_threshold_keys:
        errors.append(
            "missing required threshold keys: " + ", ".join(missing_threshold_keys)
        )
    if unexpected_threshold_keys:
        errors.append(
            "unexpected threshold keys: " + ", ".join(unexpected_threshold_keys)
        )
    if duplicate_threshold_keys:
        errors.append(
            "duplicate threshold keys: " + ", ".join(duplicate_threshold_keys)
        )

    if profile in {"development", "feature"}:
        if rationale != "rolling_median_reference_fixtures":
            errors.append(
                "non-release profile ratchet rationale must be rolling_median_reference_fixtures"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
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
            if not isinstance(entry, dict):
                continue
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
            if not isinstance(entry, dict):
                continue
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
