#!/usr/bin/env python3
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from scripts.evaluate_prep_calibration_drift import load_evidence, evaluate_report_drift


NONLINEAR_FIXTURES = {
    "nonlinear_assembly_gpu_provider",
    "nonlinear_assembly_stress_gpu_provider",
    "nonlinear_softening_proxy_gpu_provider",
    "nonlinear_load_path_mix_gpu_provider",
}


@dataclass
class Reason:
    code: str
    severity: str
    detail: str


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_protected_branch() -> bool:
    if not is_true(os.getenv("RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED", "false")):
        return False
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    protected = {
        name.strip()
        for name in os.getenv("RUNMAT_ANALYSIS_PROTECTED_BRANCHES", "main,master,release").split(",")
        if name.strip()
    }
    if ref_name in protected:
        return True
    return any(ref_name.startswith(f"{name}/") for name in protected)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def rolling_reports(rolling_dir: Path) -> List[dict]:
    if not rolling_dir.exists():
        return []
    reports = []
    for path in sorted(rolling_dir.glob("analysis_benchmark_report_rolling_*.json")):
        parsed = load_json(path)
        if isinstance(parsed, dict):
            reports.append(parsed)
    return reports


def prep_artifacts(prep_root: Path) -> List[dict]:
    prep_dir = prep_root / "prep"
    if not prep_dir.exists():
        return []
    artifacts = []
    for path in sorted(prep_dir.glob("*.json")):
        parsed = load_json(path)
        if isinstance(parsed, dict):
            artifacts.append(parsed)
    return artifacts


def prep_health_from_artifacts(artifacts: List[dict]) -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ages = []
    for artifact in artifacts:
        created = artifact.get("created_at")
        if isinstance(created, str):
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                ages.append(max((now - dt).total_seconds(), 0.0))
            except Exception:
                continue
    ages.sort()
    if not ages:
        p95_age = 0.0
    elif len(ages) == 1:
        p95_age = ages[0]
    else:
        idx = round((len(ages) - 1) * 0.95)
        p95_age = ages[int(idx)]
    return {
        "artifact_count": len(artifacts),
        "p95_age_seconds": p95_age,
    }


def env_u64(name: str, default: int = 0) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def nonlinear_records(report: dict) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    for rec in report.get("records", []):
        if not isinstance(rec, dict):
            continue
        fixture_id = rec.get("fixture_id")
        if isinstance(fixture_id, str) and fixture_id in NONLINEAR_FIXTURES:
            records[fixture_id] = rec
    return records


def evaluate_release_readiness(
    latest: dict,
    rolling: List[dict],
    protected: bool,
    prep_health: dict | None = None,
    calibration_evidence: dict | None = None,
) -> dict:
    reasons: List[Reason] = []
    latest_passed = bool(latest.get("passed", False))
    if not latest_passed:
        reasons.append(
            Reason(
                code="CONFORMANCE_FAILED",
                severity="fail",
                detail="latest nonlinear conformance report is not passed",
            )
        )

    records = nonlinear_records(latest)
    missing = sorted(NONLINEAR_FIXTURES.difference(records.keys()))
    if missing:
        reasons.append(
            Reason(
                code="NONLINEAR_FIXTURE_MISSING",
                severity="fail" if protected else "warn",
                detail=f"missing nonlinear fixture records: {', '.join(missing)}",
            )
        )

    unpublished = sorted(
        fixture
        for fixture, rec in records.items()
        if rec.get("publishable") is False
    )
    if unpublished:
        reasons.append(
            Reason(
                code="NONLINEAR_FIXTURE_UNPUBLISHABLE",
                severity="fail" if protected else "warn",
                detail=f"non-publishable nonlinear fixtures: {', '.join(unpublished)}",
            )
        )

    max_slowdown_ratio = float(
        os.getenv("RUNMAT_RELEASE_READINESS_MAX_SLOWDOWN_RATIO", "1.25")
    )
    require_trends = is_true(os.getenv("RUNMAT_RELEASE_READINESS_REQUIRE_TRENDS", "false"))
    if not rolling:
        if protected or require_trends:
            reasons.append(
                Reason(
                    code="TREND_DATA_MISSING",
                    severity="warn",
                    detail="no rolling reports available for trend comparison",
                )
            )
    else:
        hist = {fixture: [] for fixture in NONLINEAR_FIXTURES}
        for report in rolling:
            for fixture, rec in nonlinear_records(report).items():
                value = rec.get("gpu_run_ms")
                if isinstance(value, (int, float)) and value > 0:
                    hist[fixture].append(float(value))
        for fixture, rec in records.items():
            current = rec.get("gpu_run_ms")
            history = hist.get(fixture, [])
            if not isinstance(current, (int, float)) or current <= 0 or not history:
                continue
            baseline = statistics.median(history)
            if baseline <= 0:
                continue
            ratio = float(current) / baseline
            if ratio > max_slowdown_ratio:
                reasons.append(
                    Reason(
                        code="NONLINEAR_TREND_SLOWDOWN",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"{fixture} gpu_run_ms slowdown ratio {ratio:.3f} exceeds "
                            f"{max_slowdown_ratio:.3f}"
                        ),
                    )
                )

    replay_ok = is_true(os.getenv("RUNMAT_RELEASE_READINESS_ARTIFACT_REPLAY_OK", "false"))
    compat_ok = is_true(os.getenv("RUNMAT_RELEASE_READINESS_ARTIFACT_COMPAT_OK", "false"))
    if not replay_ok:
        reasons.append(
            Reason(
                code="ARTIFACT_REPLAY_UNVERIFIED",
                severity="fail" if protected else "warn",
                detail="artifact replay verification did not run or failed",
            )
        )
    if not compat_ok:
        reasons.append(
            Reason(
                code="ARTIFACT_COMPAT_UNVERIFIED",
                severity="fail" if protected else "warn",
                detail="artifact compatibility verification did not run or failed",
            )
        )

    prep_acceptance_min_rate = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_MIN_RATE", "0.9")
    )
    prep_acceptance_require = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_ACCEPTANCE_REQUIRE", "false")
    )
    acceptance_flags = []
    for rec in records.values():
        accepted = rec.get("prep_acceptance_passed")
        if isinstance(accepted, bool):
            acceptance_flags.append(accepted)
    acceptance_rate = None
    if not acceptance_flags:
        if protected or prep_acceptance_require:
            reasons.append(
                Reason(
                    code="PREP_ACCEPTANCE_MISSING",
                    severity="warn",
                    detail="prep acceptance metrics missing from nonlinear fixture records",
                )
            )
    else:
        acceptance_rate = sum(1 for flag in acceptance_flags if flag) / len(acceptance_flags)
        if acceptance_rate < prep_acceptance_min_rate:
            reasons.append(
                Reason(
                    code="PREP_ACCEPTANCE_RATE_LOW",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep acceptance pass rate {acceptance_rate:.3f} below "
                        f"minimum {prep_acceptance_min_rate:.3f}"
                    ),
                )
            )

    prep_warn_count = env_u64("RUNMAT_RELEASE_READINESS_PREP_WARN_ARTIFACT_COUNT", 64)
    prep_fail_count = env_u64("RUNMAT_RELEASE_READINESS_PREP_FAIL_ARTIFACT_COUNT", 128)
    prep_warn_p95_age = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_WARN_P95_AGE_SECONDS", "604800")
    )
    prep_fail_p95_age = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_FAIL_P95_AGE_SECONDS", "1209600")
    )
    prep_require_health = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_REQUIRE_HEALTH", "false")
    )

    if prep_health is None:
        if protected or prep_require_health:
            reasons.append(
                Reason(
                    code="PREP_HEALTH_MISSING",
                    severity="warn",
                    detail="prep artifact health data missing for release readiness evaluation",
                )
            )
    else:
        artifact_count = int(prep_health.get("artifact_count", 0))
        p95_age = float(prep_health.get("p95_age_seconds", 0.0))

        if artifact_count >= prep_fail_count:
            reasons.append(
                Reason(
                    code="PREP_SLO_COUNT_EXCEEDED",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep artifact count {artifact_count} exceeds "
                        f"fail threshold {prep_fail_count}"
                    ),
                )
            )
        elif artifact_count >= prep_warn_count:
            reasons.append(
                Reason(
                    code="PREP_SLO_COUNT_EXCEEDED",
                    severity="warn",
                    detail=(
                        f"prep artifact count {artifact_count} exceeds "
                        f"warn threshold {prep_warn_count}"
                    ),
                )
            )

        if p95_age >= prep_fail_p95_age:
            reasons.append(
                Reason(
                    code="PREP_SLO_AGE_EXCEEDED",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep artifact p95 age {p95_age:.1f}s exceeds "
                        f"fail threshold {prep_fail_p95_age:.1f}s"
                    ),
                )
            )
        elif p95_age >= prep_warn_p95_age:
            reasons.append(
                Reason(
                    code="PREP_SLO_AGE_EXCEEDED",
                    severity="warn",
                    detail=(
                        f"prep artifact p95 age {p95_age:.1f}s exceeds "
                        f"warn threshold {prep_warn_p95_age:.1f}s"
                    ),
                )
            )

    created_count = env_u64("RUNMAT_PREP_CREATED_COUNT", 0)
    stale_reject_count = env_u64("RUNMAT_PREP_STALE_REJECT_COUNT", 0)
    mismatch_reject_count = env_u64("RUNMAT_PREP_MISMATCH_REJECT_COUNT", 0)
    reject_rate_threshold = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_MAX_REJECT_RATE", "0.25")
    )
    if created_count > 0:
        reject_rate = (stale_reject_count + mismatch_reject_count) / created_count
        if reject_rate > reject_rate_threshold:
            reasons.append(
                Reason(
                    code="PREP_REJECT_RATE_HIGH",
                    severity="fail" if protected else "warn",
                    detail=(
                        f"prep reject rate {reject_rate:.3f} exceeds "
                        f"threshold {reject_rate_threshold:.3f}"
                    ),
                )
            )

    prep_calibration_max_drift = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_MAX_DRIFT", "0.2")
    )
    prep_calibration_require_evidence = is_true(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_REQUIRE_EVIDENCE", "false")
    )
    prep_calibration_max_observed_drift = None
    if calibration_evidence is None:
        if protected or prep_calibration_require_evidence:
            reasons.append(
                Reason(
                    code="PREP_CALIBRATION_EVIDENCE_MISSING",
                    severity="warn",
                    detail="prep calibration evidence artifact missing for drift evaluation",
                )
            )
    else:
        drift_rows = evaluate_report_drift(latest, calibration_evidence)
        if drift_rows:
            prep_calibration_max_observed_drift = max(
                row.get("drift_ratio", 0.0) for row in drift_rows
            )
            if prep_calibration_max_observed_drift > prep_calibration_max_drift:
                offending = [
                    row["fixture_id"]
                    for row in drift_rows
                    if row.get("drift_ratio", 0.0) > prep_calibration_max_drift
                ]
                reasons.append(
                    Reason(
                        code="PREP_CALIBRATION_DRIFT_HIGH",
                        severity="fail" if protected else "warn",
                        detail=(
                            f"max calibration drift {prep_calibration_max_observed_drift:.3f} exceeds "
                            f"threshold {prep_calibration_max_drift:.3f}; fixtures: {', '.join(offending)}"
                        ),
                    )
                )

    if any(reason.severity == "fail" for reason in reasons):
        verdict = "fail"
    elif reasons:
        verdict = "warn"
    else:
        verdict = "pass"

    return {
        "schema_version": "analysis-release-readiness/v1",
        "verdict": verdict,
        "protected_branch": protected,
        "reasons": [reason.__dict__ for reason in reasons],
        "latest_report_passed": latest_passed,
        "nonlinear_fixture_count": len(records),
        "prep_acceptance_rate": acceptance_rate,
        "prep_calibration_max_observed_drift": prep_calibration_max_observed_drift,
    }


def markdown_summary(result: dict) -> str:
    lines = ["## Nonlinear Release Readiness", ""]
    lines.append(f"Verdict: **{result['verdict']}**")
    lines.append(f"Protected branch enforcement: **{result['protected_branch']}**")
    lines.append("")
    if result["reasons"]:
        lines.append("### Reasons")
        for reason in result["reasons"]:
            lines.append(
                f"- `{reason['severity']}` `{reason['code']}`: {reason['detail']}"
            )
    else:
        lines.append("- No release blockers or warnings")
    return "\n".join(lines)


def main() -> int:
    latest_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_REPORT_PATH",
            "target/runmat-analysis-artifacts/analysis_benchmark_report.json",
        )
    )
    rolling_dir = Path(
        os.getenv("RUNMAT_ANALYSIS_BASELINE_DIR", "target/runmat-analysis-artifacts/rolling")
    )
    prep_root = Path(
        os.getenv("RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT", "target/runmat-prep-artifacts")
    )
    calibration_evidence_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_EVIDENCE",
            "scripts/prep_calibration_evidence.json",
        )
    )
    output_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_OUTPUT",
            "target/runmat-analysis-artifacts/nonlinear_release_readiness.json",
        )
    )

    latest = load_json(latest_path)
    if not isinstance(latest, dict):
        print("release readiness failed: latest conformance report missing or invalid")
        return 1

    prep_health = prep_health_from_artifacts(prep_artifacts(prep_root))
    result = evaluate_release_readiness(
        latest,
        rolling_reports(rolling_dir),
        is_protected_branch(),
        prep_health=prep_health,
        calibration_evidence=load_evidence(calibration_evidence_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(markdown_summary(result))
    return 1 if result["verdict"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
