#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone
from pathlib import Path


PLASTIC_REFERENCE_FIXTURES = {
    "nonlinear_plastic_hardening_reference_gpu_provider",
    "nonlinear_plastic_hardening_reference_complex_gpu_provider",
}

CONTACT_REFERENCE_FIXTURES = {
    "nonlinear_contact_frictionless_reference_gpu_provider",
    "nonlinear_contact_frictionless_reference_complex_gpu_provider",
}


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def report_records(report: dict):
    out = []
    for rec in report.get("records", []):
        if isinstance(rec, dict):
            out.append(rec)
    return out


def blocker_count(report: dict, fixture_ids: set[str], field: str) -> int:
    values = []
    for rec in report_records(report):
        if rec.get("fixture_id") not in fixture_ids:
            continue
        raw = rec.get(field)
        if isinstance(raw, (int, float)):
            values.append(float(raw))
    count = 0
    if len(values) < 2:
        count += 1
    if not values:
        count += 1
    elif any(v > 0.75 for v in values):
        count += 1
    return count


def p75_int(values: list[int]) -> int:
    if not values:
        return 1
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * 0.75))
    return int(ordered[idx])


def p90_positive_int(values: list[int]) -> int:
    positives = [v for v in values if v > 0]
    if not positives:
        return 0
    ordered = sorted(positives)
    idx = int(round((len(ordered) - 1) * 0.9))
    return int(ordered[idx])


def calibrate_profile(rolling_reports: list[dict], static_budget: int, static_regression: int):
    plastic = [
        blocker_count(report, PLASTIC_REFERENCE_FIXTURES, "plastic_nonlinear_severity")
        for report in rolling_reports
    ]
    contact = [
        blocker_count(report, CONTACT_REFERENCE_FIXTURES, "contact_nonlinear_severity")
        for report in rolling_reports
    ]
    plastic_budget = min(static_budget, p75_int(plastic))
    contact_budget = min(static_budget, p75_int(contact))

    regressions = []
    for seq in (plastic, contact):
        for i in range(1, len(seq)):
            regressions.append(seq[i] - seq[i - 1])
    regression = min(static_regression, p90_positive_int(regressions))

    return {
        "plastic_promotion_max_blockers": int(plastic_budget),
        "contact_promotion_max_blockers": int(contact_budget),
        "promotion_max_blocker_regression": int(regression),
        "rolling_report_count": len(rolling_reports),
    }


def main() -> int:
    rolling_dir = Path(
        os.getenv("RUNMAT_ANALYSIS_BASELINE_DIR", "target/runmat-analysis-artifacts/rolling")
    )
    out = Path(
        os.getenv(
            "RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_OUTPUT",
            "target/runmat-analysis-artifacts/promotion_threshold_calibration.json",
        )
    )

    reports = []
    if rolling_dir.exists():
        for path in sorted(rolling_dir.glob("analysis_benchmark_report_rolling_*.json")):
            parsed = load_json(path)
            if isinstance(parsed, dict):
                reports.append(parsed)

    by_profile = {
        "release": calibrate_profile(reports, static_budget=0, static_regression=0),
        "development": calibrate_profile(reports, static_budget=1, static_regression=0),
        "feature": calibrate_profile(reports, static_budget=2, static_regression=1),
    }

    payload = {
        "schema_version": "promotion-threshold-calibration/v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rationale": "rolling_median_reference_fixtures",
        "source_report_count": len(reports),
        "cadence_days": {"release": 7, "development": 14, "feature": 30},
        "max_missed_cycles_allowed": 1,
        "by_profile": by_profile,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote promotion calibration: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
