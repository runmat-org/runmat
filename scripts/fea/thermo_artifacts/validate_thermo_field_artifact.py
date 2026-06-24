#!/usr/bin/env python3
import json
import math
import hashlib
import struct
import os
import sys
from pathlib import Path


def compute_payload_hash(payload: dict) -> str:
    source_obj = payload.get("field_source")
    source = source_obj if isinstance(source_obj, dict) else {}
    source_id = source.get("source_id", "")
    source_revision = int(source.get("revision", 0))
    interpolation = source.get("interpolation_mode") or ""
    expected_regions = source.get("expected_region_ids") or []
    if not isinstance(expected_regions, list):
        expected_regions = []
    expected_regions_term = ",".join(str(item) for item in expected_regions)

    regions = payload.get("region_temperature_deltas") or []
    region_terms = []
    for delta in regions:
        if not isinstance(delta, dict):
            continue
        region_id = str(delta.get("region_id", ""))
        value = float(delta.get("temperature_delta_k", 0.0))
        bits = int.from_bytes(bytearray(struct.pack("!d", value)), "big")
        region_terms.append(f"{region_id}:{bits:016x}")

    profile = payload.get("time_profile") or []
    time_terms = []
    for point in profile:
        if not isinstance(point, dict):
            continue
        t_bits = int.from_bytes(bytearray(struct.pack("!d", float(point.get("normalized_time", 0.0)))), "big")
        s_bits = int.from_bytes(bytearray(struct.pack("!d", float(point.get("scale", 0.0)))), "big")
        time_terms.append(f"{t_bits:016x}:{s_bits:016x}")

    canonical = "|".join(
        [
            str(payload.get("schema_version", "")),
            str(payload.get("source_geometry_id", "")),
            str(payload.get("source_geometry_revision", "")),
            str(source_id),
            str(source_revision),
            str(interpolation),
            expected_regions_term,
            ",".join(region_terms),
            ",".join(time_terms),
        ]
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_signature(payload_hash: str, approved_by: str, signing_key: str) -> str:
    digest = hashlib.sha256(f"{payload_hash}:{approved_by}:{signing_key}".encode("utf-8")).hexdigest()
    return f"sigv1:sha256:{digest}"


def validate(path: Path) -> list[str]:
    errors = []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return [f"invalid JSON: {exc}"]

    if payload.get("schema_version") != "analysis_thermo_field_artifact/v1":
        errors.append("schema_version must be analysis_thermo_field_artifact/v1")

    source = payload.get("field_source")
    if not isinstance(source, dict):
        errors.append("field_source is required")
    else:
        if not isinstance(source.get("source_id"), str) or not source.get("source_id", "").strip():
            errors.append("field_source.source_id must be non-empty string")
        if source.get("interpolation_mode") not in {"linear", "step", None}:
            errors.append("field_source.interpolation_mode must be linear or step")

    profile = payload.get("time_profile", [])
    if not isinstance(profile, list):
        errors.append("time_profile must be an array")
    else:
        last_t = -1.0
        for idx, point in enumerate(profile):
            if not isinstance(point, dict):
                errors.append(f"time_profile[{idx}] must be object")
                continue
            t = point.get("normalized_time")
            scale = point.get("scale")
            if not isinstance(t, (int, float)) or not math.isfinite(float(t)) or float(t) < 0.0 or float(t) > 1.0:
                errors.append(f"time_profile[{idx}].normalized_time must be finite and within [0,1]")
                continue
            if float(t) + 1.0e-12 < last_t:
                errors.append("time_profile.normalized_time must be monotonic non-decreasing")
            last_t = float(t)
            if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
                errors.append(f"time_profile[{idx}].scale must be finite")

    deltas = payload.get("region_temperature_deltas", [])
    if not isinstance(deltas, list):
        errors.append("region_temperature_deltas must be an array")
    else:
        for idx, delta in enumerate(deltas):
            if not isinstance(delta, dict):
                errors.append(f"region_temperature_deltas[{idx}] must be object")
                continue
            if not isinstance(delta.get("region_id"), str) or not delta.get("region_id", "").strip():
                errors.append(f"region_temperature_deltas[{idx}].region_id must be non-empty string")
            v = delta.get("temperature_delta_k")
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                errors.append(f"region_temperature_deltas[{idx}].temperature_delta_k must be finite")

    payload_hash = payload.get("payload_hash")
    computed_hash = compute_payload_hash(payload)
    if not isinstance(payload_hash, str) or payload_hash != computed_hash:
        errors.append("payload_hash must match computed thermo field payload hash")

    status = payload.get("artifact_status")
    if status == "approved":
        approved_by = payload.get("approved_by")
        signature = payload.get("signature")
        if not isinstance(approved_by, str) or not approved_by.strip():
            errors.append("approved artifacts require approved_by")
        signing_key = (
            os.getenv("RUNMAT_THERMO_FIELD_SIGNING_KEY", "runmat-dev-thermo-signing-key")
        )
        expected_signature = (
            compute_signature(computed_hash, approved_by, signing_key)
            if isinstance(approved_by, str)
            else None
        )
        if not isinstance(signature, str) or signature != expected_signature:
            errors.append("approved artifact signature is missing or invalid")

    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_thermo_field_artifact.py <artifact.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"artifact missing: {path}", file=sys.stderr)
        return 1
    errors = validate(path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"thermo field artifact checks passed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
