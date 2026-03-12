#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.analysis.prep_calibration.evaluate_prep_calibration_drift import (
    load_evidence,
    validate_evidence,
    validate_recommendation_artifact,
)


def main() -> int:
    candidate_path = Path(
        os.getenv(
            "RUNMAT_PREP_CALIBRATION_EVIDENCE_CANDIDATE",
            "scripts/analysis/prep_calibration/evidence/prep_calibration_evidence.json",
        )
    )
    approved_path = Path(
        os.getenv(
            "RUNMAT_PREP_CALIBRATION_EVIDENCE_APPROVED",
            "target/runmat-analysis-artifacts/prep_calibration_evidence_approved.json",
        )
    )
    recommendations_path = Path(
        os.getenv(
            "RUNMAT_PREP_CALIBRATION_RECOMMENDATIONS_INPUT",
            "target/runmat-analysis-artifacts/prep_calibration_recommendations.json",
        )
    )
    max_recommendations = int(
        os.getenv("RUNMAT_PREP_CALIBRATION_PROMOTION_MAX_RECOMMENDATIONS", "0")
    )

    candidate = load_evidence(candidate_path)
    status = validate_evidence(candidate)
    if not status.get("valid", False):
        print("candidate evidence is invalid; refusing promotion")
        for error in status.get("errors", []):
            print(f"- error: {error}")
        return 1

    if not recommendations_path.exists():
        print("recommendation artifact missing; refusing promotion")
        return 1
    recommendations = json.loads(recommendations_path.read_text())
    recommendation_status = validate_recommendation_artifact(recommendations)
    if not recommendation_status.get("valid", False):
        print("recommendation artifact invalid; refusing promotion")
        for error in recommendation_status.get("errors", []):
            print(f"- error: {error}")
        return 1
    recommendation_count = int(recommendations.get("recommendation_count", 0))
    if recommendation_count > max_recommendations:
        print(
            f"recommendation_count {recommendation_count} exceeds promotion max {max_recommendations}; refusing promotion"
        )
        return 1

    promoted = dict(candidate)
    promoted["state"] = "approved"
    promoted["promoted_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    promoted["promotion"] = {
        "source_candidate": str(candidate_path),
        "recommendation_artifact": str(recommendations_path),
        "recommendation_count": recommendation_count,
    }

    approved_path.parent.mkdir(parents=True, exist_ok=True)
    approved_path.write_text(json.dumps(promoted, indent=2))
    print(f"promoted evidence written to {approved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
