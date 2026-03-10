#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path


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
