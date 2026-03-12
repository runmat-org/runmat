#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.analysis.prep_calibration.evaluate_prep_calibration_drift import (
    load_evidence,
    validate_evidence,
)


def main() -> int:
    evidence_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_EVIDENCE",
            "scripts/analysis/prep_calibration/evidence/prep_calibration_evidence.json",
        )
    )
    output_path = Path(
        os.getenv(
            "RUNMAT_PREP_CALIBRATION_EVIDENCE_VALIDATION_OUTPUT",
            "target/runmat-analysis-artifacts/prep_calibration_evidence_validation.json",
        )
    )
    evidence = load_evidence(evidence_path)
    status = validate_evidence(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(status, indent=2))

    if status.get("errors"):
        print("prep calibration evidence validation failed")
        for error in status["errors"]:
            print(f"- error: {error}")
        return 1

    print("prep calibration evidence validation passed")
    for warning in status.get("warnings", []):
        print(f"- warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
