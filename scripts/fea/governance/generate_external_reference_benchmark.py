#!/usr/bin/env python3
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def threshold_assertion_observed(record: dict, assertion_name: str):
    assertions = record.get("threshold_assertions")
    if not isinstance(assertions, list):
        return None
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        if assertion.get("name") != assertion_name:
            continue
        observed = assertion.get("observed")
        if isinstance(observed, (int, float)):
            return float(observed)
        return None
    return None


def metric_pass(observed: float, reference: float, tol_abs, tol_rel) -> bool:
    abs_ok = True if tol_abs is None else abs(observed - reference) <= float(tol_abs)
    rel_ok = True if tol_rel is None else abs(observed - reference) <= abs(reference) * float(tol_rel)
    return abs_ok and rel_ok


def main() -> int:
    report_path = Path(
        os.getenv(
            "RUNMAT_EXTERNAL_REFERENCE_REPORT_PATH",
            "target/runmat-analysis-artifacts/analysis_benchmark_report.json",
        )
    )
    baseline_path = Path(
        os.getenv(
            "RUNMAT_EXTERNAL_REFERENCE_BASELINE_PATH",
            "scripts/fea/reference_data/m6_external_reference_baseline.json",
        )
    )
    output_path = Path(
        os.getenv(
            "RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH",
            "target/runmat-analysis-artifacts/external_reference_benchmark.json",
        )
    )
    allow_partial = os.getenv("RUNMAT_EXTERNAL_REFERENCE_ALLOW_PARTIAL", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    report = load_json(report_path)
    baseline = load_json(baseline_path)
    if not isinstance(report, dict):
        print(f"analysis report missing or invalid: {report_path}")
        return 1
    if not isinstance(baseline, dict):
        print(f"baseline reference missing or invalid: {baseline_path}")
        return 1

    records = {
        rec.get("fixture_id"): rec
        for rec in report.get("records", [])
        if isinstance(rec, dict) and isinstance(rec.get("fixture_id"), str)
    }

    errors = []
    out_metrics = []
    for idx, spec in enumerate(baseline.get("metrics", [])):
        if not isinstance(spec, dict):
            errors.append(f"baseline.metrics[{idx}] invalid")
            continue
        fixture_id = spec.get("fixture_id")
        field = spec.get("field")
        name = spec.get("name")
        source = spec.get("source", "field")
        assertion_name = spec.get("assertion_name")
        reference = spec.get("reference")
        if not all(isinstance(v, str) for v in (fixture_id, field, name)):
            errors.append(f"baseline.metrics[{idx}] missing name/fixture_id/field")
            continue
        if source not in {"field", "threshold_assertion"}:
            errors.append(f"baseline.metrics[{idx}] has unsupported source={source}")
            continue
        if source == "threshold_assertion" and assertion_name is not None and not isinstance(
            assertion_name, str
        ):
            errors.append(f"baseline.metrics[{idx}] has invalid assertion_name")
            continue
        if not isinstance(reference, (int, float)):
            errors.append(f"baseline.metrics[{idx}] missing numeric reference")
            continue

        rec = records.get(fixture_id)
        if rec is None:
            errors.append(f"missing fixture record: {fixture_id}")
            continue
        if source == "field":
            raw_observed = rec.get(field)
            if not isinstance(raw_observed, (int, float)):
                errors.append(f"missing numeric observed value for {fixture_id}.{field}")
                continue
            observed = float(raw_observed)
        else:
            assertion_key = assertion_name or field
            observed = threshold_assertion_observed(rec, assertion_key)
            if observed is None:
                errors.append(
                    f"missing threshold assertion observed value for {fixture_id}.{assertion_key}"
                )
                continue
        if not math.isfinite(observed):
            errors.append(f"non-finite observed value for {fixture_id}.{field} (source={source})")
            continue

        tol_abs = spec.get("tolerance_abs")
        tol_rel = spec.get("tolerance_rel")
        passed = metric_pass(observed, float(reference), tol_abs, tol_rel)
        out_metrics.append(
            {
                "name": name,
                "fixture_id": fixture_id,
                "field": field,
                "source": source,
                "observed": observed,
                "reference": float(reference),
                "tolerance_abs": tol_abs,
                "tolerance_rel": tol_rel,
                "pass": bool(passed),
            }
        )

    if errors and not allow_partial:
        print("external reference benchmark generation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    payload = {
        "schema_version": "external-reference-benchmark/v1",
        "scenario_id": baseline.get("scenario_id", "m6_elastoplastic_contact_bracket_v1"),
        "reference_source": baseline.get("reference_source", {}),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metrics": out_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote external reference benchmark: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
