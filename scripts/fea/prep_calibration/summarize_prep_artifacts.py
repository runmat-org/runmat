#!/usr/bin/env python3
import json
import os
import statistics
from pathlib import Path


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def protected_branch() -> bool:
    if not parse_bool(os.getenv("RUNMAT_ANALYSIS_ENFORCE_BASELINE_ON_PROTECTED", "false")):
        return False
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    protected = {
        item.strip()
        for item in os.getenv("RUNMAT_ANALYSIS_PROTECTED_BRANCHES", "main,master,release").split(",")
        if item.strip()
    }
    if ref_name in protected:
        return True
    return any(ref_name.startswith(f"{item}/") for item in protected)


def load_prep_artifacts(root: Path):
    prep_dir = root / "prep"
    if not prep_dir.exists():
        return []
    artifacts = []
    for path in sorted(prep_dir.glob("*.json")):
        try:
            artifacts.append(json.loads(path.read_text()))
        except Exception:
            continue
    return artifacts


def main() -> int:
    root = Path(
        os.getenv("RUNMAT_GEOMETRY_PREP_ARTIFACT_ROOT", "target/runmat-prep-artifacts")
    )
    artifacts = load_prep_artifacts(root)

    warn_count = int(os.getenv("RUNMAT_PREP_SLO_WARN_ARTIFACT_COUNT", "64"))
    fail_count = int(os.getenv("RUNMAT_PREP_SLO_FAIL_ARTIFACT_COUNT", "128"))
    warn_p95_age = float(os.getenv("RUNMAT_PREP_SLO_WARN_P95_AGE_SECONDS", "604800"))
    fail_p95_age = float(os.getenv("RUNMAT_PREP_SLO_FAIL_P95_AGE_SECONDS", "1209600"))

    now = None
    try:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
    except Exception:
        now = None

    ages = []
    per_geometry = {}
    if now is not None:
        from datetime import datetime

        for artifact in artifacts:
            created = artifact.get("created_at")
            if isinstance(created, str):
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    age = max((now - dt).total_seconds(), 0.0)
                    ages.append(age)
                except Exception:
                    pass
            geometry_id = artifact.get("source_geometry_id")
            if isinstance(geometry_id, str):
                per_geometry[geometry_id] = per_geometry.get(geometry_id, 0) + 1

    p95_age = statistics.quantiles(ages, n=20)[-1] if len(ages) >= 20 else (max(ages) if ages else 0.0)
    verdict = "pass"
    reasons = []
    if len(artifacts) >= fail_count:
        verdict = "fail"
        reasons.append(
            f"artifact_count={len(artifacts)} exceeds fail threshold {fail_count}"
        )
    elif len(artifacts) >= warn_count:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"artifact_count={len(artifacts)} exceeds warn threshold {warn_count}"
        )

    if p95_age >= fail_p95_age:
        verdict = "fail"
        reasons.append(
            f"p95_age_seconds={p95_age:.1f} exceeds fail threshold {fail_p95_age:.1f}"
        )
    elif p95_age >= warn_p95_age:
        if verdict != "fail":
            verdict = "warn"
        reasons.append(
            f"p95_age_seconds={p95_age:.1f} exceeds warn threshold {warn_p95_age:.1f}"
        )

    print("## Prep Artifact SLO")
    print()
    print(f"Verdict: **{verdict}**")
    print(f"Artifact root: `{root}`")
    print(f"Current artifact count: `{len(artifacts)}`")
    print(f"P95 artifact age seconds: `{p95_age:.1f}`")
    print(f"Distinct geometry ids: `{len(per_geometry)}`")
    print()
    if reasons:
        print("### Reasons")
        for reason in reasons:
            print(f"- {reason}")
    else:
        print("- No SLO warnings")

    should_fail = verdict == "fail" and protected_branch()
    if should_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
