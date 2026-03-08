#!/usr/bin/env python3
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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


def nonlinear_records(report: dict) -> Dict[str, dict]:
    return {
        rec.get("fixture_id"): rec
        for rec in report.get("records", [])
        if isinstance(rec, dict) and rec.get("fixture_id") in NONLINEAR_FIXTURES
    }


def evaluate_release_readiness(latest: dict, rolling: List[dict], protected: bool) -> dict:
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
    if not rolling:
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

    result = evaluate_release_readiness(latest, rolling_reports(rolling_dir), is_protected_branch())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))

    print(markdown_summary(result))
    return 1 if result["verdict"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
