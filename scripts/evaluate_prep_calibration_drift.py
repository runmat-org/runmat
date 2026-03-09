#!/usr/bin/env python3
import json
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
