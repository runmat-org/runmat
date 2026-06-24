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
        compute_payload_hash,
        compute_signature,
    )
except ModuleNotFoundError:
    from validate_thermo_field_artifact import compute_payload_hash, compute_signature


def parse_key_value_pairs(values, key_name, value_name, key_numeric=False):
    parsed = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"invalid {key_name} pair '{raw}', expected {key_name}={value_name}")
        key, value = raw.split("=", 1)
        key_parsed = float(key.strip()) if key_numeric else key.strip()
        parsed.append((key_parsed, float(value)))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thermo field artifact JSON")
    parser.add_argument("--output", required=True, help="output artifact path")
    parser.add_argument("--geometry-id", required=True)
    parser.add_argument("--geometry-revision", required=True, type=int)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--interpolation", choices=["linear", "step"], default="linear")
    parser.add_argument("--expected-region", action="append", default=[])
    parser.add_argument("--region-delta", action="append", default=[])
    parser.add_argument("--time-point", action="append", default=[])
    parser.add_argument("--status", choices=["candidate", "approved"], default="candidate")
    parser.add_argument("--approved-by", default="")
    args = parser.parse_args()

    region_pairs = parse_key_value_pairs(args.region_delta, "region", "delta_k")
    time_pairs = parse_key_value_pairs(args.time_point, "t", "scale", key_numeric=True)
    time_pairs.sort(key=lambda pair: pair[0])

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    artifact = {
        "schema_version": "analysis_thermo_field_artifact/v1",
        "source_geometry_id": args.geometry_id,
        "source_geometry_revision": args.geometry_revision,
        "artifact_status": args.status,
        "created_at": now,
        "approved_at": now if args.status == "approved" else None,
        "approved_by": args.approved_by if args.status == "approved" else None,
        "field_source": {
            "source_id": args.source_id,
            "revision": args.revision,
            "interpolation_mode": args.interpolation,
            "expected_region_ids": args.expected_region,
        },
        "region_temperature_deltas": [
            {"region_id": region_id, "temperature_delta_k": delta_k}
            for region_id, delta_k in region_pairs
        ],
        "time_profile": [
            {"normalized_time": float(t), "scale": float(scale)} for t, scale in time_pairs
        ],
    }
    artifact["payload_hash"] = compute_payload_hash(artifact)
    signing_key = os.getenv("RUNMAT_THERMO_FIELD_SIGNING_KEY", "runmat-dev-thermo-signing-key")
    if args.status == "approved" and args.approved_by:
        artifact["signature"] = compute_signature(
            artifact["payload_hash"],
            args.approved_by,
            signing_key,
        )
    else:
        artifact["signature"] = None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote thermo field artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
