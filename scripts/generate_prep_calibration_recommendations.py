#!/usr/bin/env python3
import json
import os
from pathlib import Path

from scripts.evaluate_prep_calibration_drift import (
    build_recommendation_artifact,
    load_evidence,
)
from scripts.release_readiness_nonlinear import load_json, rolling_reports


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
    evidence_path = Path(
        os.getenv(
            "RUNMAT_RELEASE_READINESS_PREP_CALIBRATION_EVIDENCE",
            "scripts/prep_calibration_evidence.json",
        )
    )
    output_path = Path(
        os.getenv(
            "RUNMAT_PREP_CALIBRATION_RECOMMENDATIONS_OUTPUT",
            "target/runmat-analysis-artifacts/prep_calibration_recommendations.json",
        )
    )
    drift_trigger = float(
        os.getenv("RUNMAT_RELEASE_READINESS_PREP_RETRAIN_TRIGGER_DRIFT", "0.1")
    )

    latest = load_json(latest_path)
    if not isinstance(latest, dict):
        print("failed to load latest benchmark report")
        return 1
    evidence = load_evidence(evidence_path)
    if not isinstance(evidence, dict):
        print("failed to load calibration evidence")
        return 1

    artifact = build_recommendation_artifact(
        latest,
        rolling_reports(rolling_dir),
        evidence,
        drift_trigger=drift_trigger,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    print(
        f"generated recommendation artifact with {artifact['recommendation_count']} recommendations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
