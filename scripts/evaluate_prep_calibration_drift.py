#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path


def load_evidence(path: Path):
    if not path.exists():
        return None
    parsed = json.loads(path.read_text())
    if not isinstance(parsed, dict):
        return None
    fixtures = parsed.get("fixtures")
    if not isinstance(fixtures, dict):
        return None
    return parsed


def _parse_iso8601(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def validate_evidence(evidence: dict, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    errors = []
    warnings = []
    if not isinstance(evidence, dict):
        return {
            "valid": False,
            "stale": True,
            "errors": ["evidence payload must be a JSON object"],
            "warnings": [],
            "age_days": None,
            "max_age_days": None,
        }

    schema = evidence.get("schema_version")
    if schema != "prep-calibration-evidence/v1":
        warnings.append("unexpected schema_version; expected prep-calibration-evidence/v1")

    state = evidence.get("state")
    if state is not None and state not in {"candidate", "approved"}:
        errors.append("state must be either 'candidate' or 'approved'")

    fixtures = evidence.get("fixtures")
    if not isinstance(fixtures, dict) or not fixtures:
        errors.append("fixtures map is missing or empty")
    else:
        for fixture_id, spec in fixtures.items():
            if not isinstance(spec, dict):
                errors.append(f"fixture '{fixture_id}' spec must be an object")
                continue
            profiles = spec.get("profiles")
            if not isinstance(profiles, dict) or not profiles:
                errors.append(f"fixture '{fixture_id}' profiles must be a non-empty object")
                continue
            for profile_name, envelope in profiles.items():
                if not isinstance(envelope, dict):
                    errors.append(
                        f"fixture '{fixture_id}' profile '{profile_name}' envelope must be object"
                    )
                    continue
                min_score = envelope.get("acceptance_score_min")
                max_score = envelope.get("acceptance_score_max")
                if not isinstance(min_score, (int, float)) or not isinstance(
                    max_score, (int, float)
                ):
                    errors.append(
                        f"fixture '{fixture_id}' profile '{profile_name}' requires numeric acceptance_score_min/max"
                    )

    generated_at = _parse_iso8601(evidence.get("generated_at"))
    max_age_days = evidence.get("max_age_days")
    age_days = None
    stale = False
    if generated_at is not None and isinstance(max_age_days, (int, float)):
        age_days = max((now - generated_at).total_seconds(), 0.0) / 86400.0
        stale = age_days > float(max_age_days)
    else:
        warnings.append("generated_at/max_age_days missing; staleness cannot be evaluated")

    return {
        "valid": not errors,
        "stale": stale,
        "state": state,
        "errors": errors,
        "warnings": warnings,
        "age_days": age_days,
        "max_age_days": float(max_age_days) if isinstance(max_age_days, (int, float)) else None,
    }


def _profile_expectation(fixture_spec: dict, profile: str | None):
    profiles = fixture_spec.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        return None
    requested = profile or fixture_spec.get("default_profile")
    if requested in profiles and isinstance(profiles[requested], dict):
        return profiles[requested]
    default_profile = fixture_spec.get("default_profile")
    if default_profile in profiles and isinstance(profiles[default_profile], dict):
        return profiles[default_profile]
    first = next(iter(profiles.values()))
    return first if isinstance(first, dict) else None


def evaluate_record_drift(record: dict, fixture_spec: dict):
    profile = record.get("prep_calibration_profile")
    score = record.get("prep_acceptance_score")
    if not isinstance(score, (int, float)):
        return None
    expectation = _profile_expectation(fixture_spec, profile)
    if expectation is None:
        return None
    min_score = expectation.get("acceptance_score_min")
    max_score = expectation.get("acceptance_score_max")
    if not isinstance(min_score, (int, float)) or not isinstance(max_score, (int, float)):
        return None
    if min_score > max_score:
        min_score, max_score = max_score, min_score

    if min_score <= score <= max_score:
        drift_ratio = 0.0
    elif score < min_score:
        drift_ratio = (min_score - score) / max(min_score, 1.0e-9)
    else:
        drift_ratio = (score - max_score) / max(max_score, 1.0e-9)

    return {
        "profile": profile,
        "acceptance_score": float(score),
        "expected_min": float(min_score),
        "expected_max": float(max_score),
        "drift_ratio": float(max(drift_ratio, 0.0)),
    }


def evaluate_report_drift(report: dict, evidence: dict):
    fixtures = evidence.get("fixtures", {}) if isinstance(evidence, dict) else {}
    if not isinstance(fixtures, dict):
        return []
    results = []
    for record in report.get("records", []):
        if not isinstance(record, dict):
            continue
        fixture_id = record.get("fixture_id")
        if fixture_id not in fixtures:
            continue
        evaluated = evaluate_record_drift(record, fixtures[fixture_id])
        if evaluated is None:
            continue
        evaluated["fixture_id"] = fixture_id
        results.append(evaluated)
    return results


def evaluate_rolling_drift(rolling_reports: list[dict], evidence: dict):
    series = {}
    for report in rolling_reports:
        for row in evaluate_report_drift(report, evidence):
            fixture_id = row["fixture_id"]
            series.setdefault(fixture_id, []).append(float(row.get("drift_ratio", 0.0)))
    slopes = {}
    for fixture_id, values in series.items():
        if len(values) <= 1:
            slopes[fixture_id] = 0.0
        else:
            slopes[fixture_id] = (values[-1] - values[0]) / (len(values) - 1)
    return slopes


def recommend_profile_shifts(
    latest_report: dict,
    rolling_reports: list[dict],
    evidence: dict,
    drift_trigger: float = 0.1,
):
    recommendations = []
    latest_rows = evaluate_report_drift(latest_report, evidence)
    slopes = evaluate_rolling_drift(rolling_reports, evidence)
    profile_rank = {"fast": 0, "balanced": 1, "conservative": 2}
    rank_profile = {value: key for key, value in profile_rank.items()}

    for row in latest_rows:
        fixture_id = row["fixture_id"]
        drift_ratio = float(row.get("drift_ratio", 0.0))
        slope = float(slopes.get(fixture_id, 0.0))
        profile = row.get("profile") or "balanced"
        rank = profile_rank.get(profile, 1)
        suggested = rank
        rationale = []
        if drift_ratio >= drift_trigger and slope >= 0.0:
            suggested = min(rank + 1, 2)
            rationale.append("drift_above_trigger")
            if slope > 0.01:
                rationale.append("drift_trending_worse")
        elif drift_ratio == 0.0 and slope <= -0.01:
            suggested = max(rank - 1, 0)
            rationale.append("drift_stable_improving")
        if suggested == rank:
            continue
        confidence = min(1.0, max(0.0, drift_ratio * 1.5 + max(slope, 0.0) * 8.0))
        recommendations.append(
            {
                "fixture_id": fixture_id,
                "current_profile": profile,
                "suggested_profile": rank_profile[suggested],
                "suggested_profile_shift": suggested - rank,
                "drift_ratio": drift_ratio,
                "drift_slope": slope,
                "confidence": confidence,
                "rationale": rationale,
            }
        )
    return sorted(recommendations, key=lambda item: (item["fixture_id"], item["suggested_profile"]))


def build_recommendation_artifact(
    latest_report: dict,
    rolling_reports: list[dict],
    evidence: dict,
    drift_trigger: float = 0.1,
):
    recommendations = recommend_profile_shifts(
        latest_report,
        rolling_reports,
        evidence,
        drift_trigger=drift_trigger,
    )
    max_drift_ratio = 0.0
    drift_rows = evaluate_report_drift(latest_report, evidence)
    if drift_rows:
        max_drift_ratio = max(row.get("drift_ratio", 0.0) for row in drift_rows)
    return {
        "schema_version": "prep-calibration-recommendations/v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "drift_trigger": float(drift_trigger),
        "rolling_window_count": len(rolling_reports),
        "max_drift_ratio": float(max_drift_ratio),
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
    }


def validate_recommendation_artifact(artifact: dict, now: datetime | None = None):
    now = now or datetime.now(timezone.utc)
    errors = []
    warnings = []
    if not isinstance(artifact, dict):
        return {
            "valid": False,
            "stale": True,
            "errors": ["recommendation artifact must be a JSON object"],
            "warnings": [],
            "age_days": None,
        }
    if artifact.get("schema_version") != "prep-calibration-recommendations/v1":
        errors.append("schema_version must be prep-calibration-recommendations/v1")
    generated_at = _parse_iso8601(artifact.get("generated_at"))
    if generated_at is None:
        errors.append("generated_at missing or invalid")
        age_days = None
    else:
        age_days = max((now - generated_at).total_seconds(), 0.0) / 86400.0
    recs = artifact.get("recommendations")
    if not isinstance(recs, list):
        errors.append("recommendations must be a list")
        recs = []
    for idx, rec in enumerate(recs):
        if not isinstance(rec, dict):
            errors.append(f"recommendation index {idx} must be object")
            continue
        for key in ["fixture_id", "current_profile", "suggested_profile"]:
            if not isinstance(rec.get(key), str):
                errors.append(f"recommendation index {idx} missing string field '{key}'")
        if not isinstance(rec.get("suggested_profile_shift"), int):
            errors.append(f"recommendation index {idx} missing int field 'suggested_profile_shift'")

    stale_days = artifact.get("max_age_days", 7)
    stale = False
    if age_days is not None and isinstance(stale_days, (int, float)):
        stale = age_days > float(stale_days)
    else:
        warnings.append("max_age_days invalid; stale evaluation disabled")

    return {
        "valid": not errors,
        "stale": stale,
        "errors": errors,
        "warnings": warnings,
        "age_days": age_days,
    }
