#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from scripts.fea.thermo_artifacts.validate_thermo_field_artifact import (
        validate,
        compute_payload_hash,
        compute_signature,
    )
except ModuleNotFoundError:
    from validate_thermo_field_artifact import (
        validate,
        compute_payload_hash,
        compute_signature,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote thermo field artifact candidate")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--approved-dir", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--max-age-days", type=float, default=30.0)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--trend-drift-ratio", type=float, default=0.0)
    parser.add_argument("--max-trend-drift-ratio", type=float, default=1.2)
    parser.add_argument("--override-token", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    candidate_path = Path(args.candidate_dir) / f"{args.artifact_id}.json"
    if not candidate_path.exists():
        raise SystemExit(f"candidate artifact not found: {candidate_path}")

    errors = validate(candidate_path)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)

    payload = json.loads(candidate_path.read_text())
    reasons = []
    created_at_raw = payload.get("created_at")
    if isinstance(created_at_raw, str):
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0
        if age_days > args.max_age_days:
            reasons.append(
                f"THERMO_FIELD_PROMOTION_CANDIDATE_STALE:{age_days:.1f}d>{args.max_age_days:.1f}d"
            )

    if args.trend_drift_ratio > args.max_trend_drift_ratio:
        required_override = os.getenv("RUNMAT_THERMO_FIELD_PROMOTION_OVERRIDE_TOKEN", "")
        if not required_override or args.override_token != required_override:
            reasons.append(
                "THERMO_FIELD_PROMOTION_TREND_DRIFT_BLOCKED:"
                f"{args.trend_drift_ratio:.3f}>{args.max_trend_drift_ratio:.3f}"
            )

    report = {
        "artifact_id": args.artifact_id,
        "trend_drift_ratio": args.trend_drift_ratio,
        "max_trend_drift_ratio": args.max_trend_drift_ratio,
        "blocked": bool(reasons),
        "reasons": reasons,
    }
    if args.report_out:
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if reasons:
        for reason in reasons:
            print(reason)
        raise SystemExit(1)

    payload["artifact_status"] = "approved"
    payload["approved_by"] = args.approved_by
    payload["payload_hash"] = compute_payload_hash(payload)
    signing_key = os.getenv("RUNMAT_THERMO_FIELD_SIGNING_KEY", "runmat-dev-thermo-signing-key")
    payload["signature"] = compute_signature(payload["payload_hash"], args.approved_by, signing_key)
    payload["approved_at"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    approved_dir = Path(args.approved_dir)
    approved_dir.mkdir(parents=True, exist_ok=True)
    approved_path = approved_dir / f"{args.artifact_id}.json"
    approved_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"promoted thermo field artifact: {approved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
