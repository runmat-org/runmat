#!/usr/bin/env python3
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def governance_profile_name() -> str:
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_name in {"main", "master"} or ref_name.startswith("release/"):
        return "release"
    if ref_name in {"develop", "dev"}:
        return "development"
    return "feature"


def parse_iso8601_utc(raw: str):
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    path = Path(
        os.getenv(
            "RUNMAT_PROMOTION_THRESHOLD_CALIBRATION_INPUT",
            "target/runmat-analysis-artifacts/promotion_threshold_calibration.json",
        )
    )
    enforce = is_true(os.getenv("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_ENFORCE", "false"))
    min_source_reports = int(
        os.getenv("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MIN_SOURCE_REPORTS", "0")
    )
    min_trusted_source_reports = int(
        os.getenv(
            "RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MIN_TRUSTED_SOURCE_REPORTS",
            str(min_source_reports),
        )
    )
    max_age_override = os.getenv("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_MAX_AGE_DAYS")
    require_cadence = is_true(
        os.getenv("RUNMAT_VALIDATE_PROMOTION_CALIBRATION_REQUIRE_CADENCE", "false")
    )

    if not path.exists():
        print(f"promotion calibration artifact missing: {path}")
        return 1 if enforce else 0

    payload = json.loads(path.read_text())
    errors = []

    if payload.get("schema_version") != "promotion-threshold-calibration/v1":
        errors.append("schema_version must be promotion-threshold-calibration/v1")

    by_profile = payload.get("by_profile")
    if not isinstance(by_profile, dict):
        errors.append("by_profile missing or invalid")
        by_profile = {}

    for profile in ("release", "development", "feature"):
        entry = by_profile.get(profile)
        if not isinstance(entry, dict):
            errors.append(f"missing profile calibration entry: {profile}")
            continue
        for key in (
            "plastic_promotion_max_blockers",
            "contact_promotion_max_blockers",
            "promotion_max_blocker_regression",
        ):
            value = entry.get(key)
            if not isinstance(value, int) or value < 0:
                errors.append(f"{profile}.{key} must be non-negative int")
        rolling_count = entry.get("rolling_report_count")
        trusted_rolling_count = entry.get("rolling_trusted_report_count")
        if not isinstance(rolling_count, int) or rolling_count < 0:
            errors.append(f"{profile}.rolling_report_count must be non-negative int")
        if not isinstance(trusted_rolling_count, int) or trusted_rolling_count < 0:
            errors.append(
                f"{profile}.rolling_trusted_report_count must be non-negative int"
            )
        elif isinstance(rolling_count, int) and trusted_rolling_count > rolling_count:
            errors.append(
                f"{profile}.rolling_trusted_report_count must be less than or equal to "
                f"{profile}.rolling_report_count"
            )

    source_count = payload.get("source_report_count")
    if not isinstance(source_count, int) or source_count < 0:
        errors.append("source_report_count must be non-negative int")
    elif source_count < min_source_reports:
        errors.append(
            f"source_report_count {source_count} below minimum {min_source_reports}"
        )
    source_trusted_count = payload.get("source_trusted_report_count")
    if not isinstance(source_trusted_count, int) or source_trusted_count < 0:
        errors.append("source_trusted_report_count must be non-negative int")
    elif isinstance(source_count, int) and source_trusted_count > source_count:
        errors.append(
            "source_trusted_report_count must be less than or equal to source_report_count"
        )
    elif source_trusted_count < min_trusted_source_reports:
        errors.append(
            "source_trusted_report_count "
            f"{source_trusted_count} below minimum {min_trusted_source_reports}"
        )

    generated_at_raw = payload.get("generated_at")
    generated_at = None
    if isinstance(generated_at_raw, str):
        generated_at = parse_iso8601_utc(generated_at_raw)
    if generated_at is None:
        errors.append("generated_at missing or invalid")
    else:
        age_days = (datetime.now(timezone.utc) - generated_at).total_seconds() / 86400.0
        if max_age_override is not None:
            max_age_days = float(max_age_override)
        else:
            cadence_days = payload.get("cadence_days")
            profile = governance_profile_name()
            if require_cadence:
                if not isinstance(cadence_days, dict) or not isinstance(
                    cadence_days.get(profile), (int, float)
                ):
                    errors.append("cadence_days missing or invalid for active profile")
                    max_age_days = 0.0
                else:
                    missed_cycles = payload.get("max_missed_cycles_allowed", 1)
                    if not isinstance(missed_cycles, int) or missed_cycles < 0:
                        errors.append("max_missed_cycles_allowed must be non-negative int")
                        missed_cycles = 1
                    max_age_days = float(cadence_days.get(profile)) * (1 + missed_cycles)
            else:
                max_age_days = 30.0
        if age_days > max_age_days:
            errors.append(
                f"calibration age {age_days:.2f}d exceeds max allowed {max_age_days:.2f}d"
            )

    if errors:
        print("promotion calibration validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1 if enforce else 0

    print("promotion calibration validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
