#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.validate_thermo_field_artifact import validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote thermo field artifact candidate")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--approved-dir", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--max-age-days", type=float, default=30.0)
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
    created_at_raw = payload.get("created_at")
    if isinstance(created_at_raw, str):
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0
        if age_days > args.max_age_days:
            raise SystemExit(
                f"candidate artifact is stale ({age_days:.1f} days > {args.max_age_days:.1f})"
            )

    payload["artifact_status"] = "approved"
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
