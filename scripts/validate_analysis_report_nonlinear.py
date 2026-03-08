#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REQUIRED_FIXTURES = {
    "nonlinear_assembly_gpu_provider": {
        "nonlinear_total_increments",
        "nonlinear_failed_increments",
        "nonlinear_iteration_spike_count",
    },
    "nonlinear_assembly_stress_gpu_provider": {
        "nonlinear_stress_total_increments",
        "nonlinear_stress_stall_count",
        "nonlinear_stress_iteration_spike_count",
    },
    "nonlinear_softening_proxy_gpu_provider": {
        "nonlinear_softening_total_increments",
        "nonlinear_softening_spike_count",
        "nonlinear_softening_backtrack_bursts",
    },
    "nonlinear_load_path_mix_gpu_provider": {
        "nonlinear_path_mix_total_increments",
        "nonlinear_path_mix_max_backtracks_per_increment",
        "nonlinear_path_mix_backtrack_bursts",
    },
}


def main() -> int:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("target/runmat-analysis-artifacts/analysis_benchmark_report.json")
    )
    if not path.exists():
        print(f"report missing: {path}", file=sys.stderr)
        return 1

    report = json.loads(path.read_text())
    records = {
        record.get("fixture_id"): record
        for record in report.get("records", [])
        if isinstance(record, dict)
    }

    errors = []
    for fixture_id, required in REQUIRED_FIXTURES.items():
        record = records.get(fixture_id)
        if record is None:
            errors.append(f"missing fixture record: {fixture_id}")
            continue
        names = {
            item.get("name")
            for item in record.get("threshold_assertions", [])
            if isinstance(item, dict)
        }
        missing = sorted(name for name in required if name not in names)
        if missing:
            errors.append(
                f"fixture {fixture_id} missing threshold assertions: {', '.join(missing)}"
            )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("nonlinear analysis report schema checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
