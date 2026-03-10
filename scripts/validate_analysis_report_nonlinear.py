#!/usr/bin/env python3
import json
import os
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
        "nonlinear_path_mix_effective_modulus_scale",
        "nonlinear_path_mix_material_spread_ratio",
        "thermo_nonlinear_severity",
    },
    "thermo_mech_kickoff_gpu_provider": {
        "thermo_mech_thermal_strain_scale",
        "thermo_mech_thermal_load_scale",
        "thermo_mech_effective_modulus_scale",
        "thermo_mech_material_spread_ratio",
        "thermo_mech_assignment_heterogeneity_index",
        "thermo_mech_transient_severity",
        "thermo_mech_transient_time_scale_mean",
    },
    "thermo_gradient_benign_gpu_provider": {
        "thermo_gradient_benign_spread_ratio",
        "thermo_gradient_benign_heterogeneity",
    },
    "thermo_gradient_pathological_gpu_provider": {
        "thermo_gradient_pathological_spread_ratio",
        "thermo_gradient_pathological_heterogeneity",
        "thermo_gradient_pathological_temporal_variation",
    },
    "thermo_ramp_smooth_gpu_provider": {
        "thermo_ramp_smooth_temporal_variation",
        "thermo_ramp_smooth_spatial_gradient_index",
        "thermo_ramp_smooth_spatial_coverage_ratio",
        "thermo_ramp_smooth_field_extrapolation_ratio",
    },
    "thermo_ramp_smooth_field_artifact_gpu_provider": {
        "thermo_ramp_smooth_temporal_variation",
        "thermo_ramp_smooth_spatial_gradient_index",
        "thermo_ramp_smooth_spatial_coverage_ratio",
        "thermo_ramp_smooth_field_extrapolation_ratio",
    },
    "thermo_shock_oscillatory_gpu_provider": {
        "thermo_shock_oscillatory_temporal_variation",
        "thermo_shock_oscillatory_spatial_gradient_index",
        "thermo_shock_oscillatory_spatial_coverage_ratio",
        "thermo_shock_oscillatory_field_extrapolation_ratio",
    },
    "thermo_shock_oscillatory_field_artifact_gpu_provider": {
        "thermo_shock_oscillatory_temporal_variation",
        "thermo_shock_oscillatory_spatial_gradient_index",
        "thermo_shock_oscillatory_spatial_coverage_ratio",
        "thermo_shock_oscillatory_field_extrapolation_ratio",
    },
}

THERMO_REQUIRED_FIELDS = {
    "thermo_coupling_enabled",
    "thermo_coupling_fingerprint",
    "thermo_constitutive_temperature_factor",
    "thermo_effective_modulus_scale",
    "thermo_constitutive_material_spread_ratio",
    "thermo_assignment_heterogeneity_index",
    "thermo_region_delta_count",
    "thermo_spatial_coverage_ratio",
    "thermo_field_extrapolation_ratio",
    "thermo_transient_severity",
    "thermo_nonlinear_severity",
}


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    require_thermo_summary = is_true(
        os.getenv("RUNMAT_VALIDATE_REQUIRE_THERMO_SUMMARY", "false")
    )
    require_thermo_field_summary = is_true(
        os.getenv("RUNMAT_VALIDATE_REQUIRE_THERMO_FIELD_SUMMARY", "false")
    )
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

        if fixture_id in {
            "thermo_mech_kickoff_gpu_provider",
            "thermo_gradient_benign_gpu_provider",
            "thermo_gradient_pathological_gpu_provider",
            "thermo_ramp_smooth_gpu_provider",
            "thermo_ramp_smooth_field_artifact_gpu_provider",
            "thermo_shock_oscillatory_gpu_provider",
            "thermo_shock_oscillatory_field_artifact_gpu_provider",
            "nonlinear_load_path_mix_gpu_provider",
        }:
            missing_fields = sorted(
                field for field in THERMO_REQUIRED_FIELDS if field not in record
            )
            if missing_fields:
                errors.append(
                    f"fixture {fixture_id} missing thermo summary fields: {', '.join(missing_fields)}"
                )

    if require_thermo_summary:
        thermo_records = [
            record
            for record in records.values()
            if isinstance(record.get("thermo_coupling_enabled"), bool)
            or isinstance(record.get("thermo_transient_severity"), (int, float))
            or isinstance(record.get("thermo_nonlinear_severity"), (int, float))
        ]
        if not thermo_records:
            errors.append(
                "thermo summary fields missing across all records while RUNMAT_VALIDATE_REQUIRE_THERMO_SUMMARY=true"
            )

    if require_thermo_field_summary:
        thermo_field_records = [
            record
            for record in records.values()
            if isinstance(record.get("thermo_region_delta_count"), (int, float))
            and float(record.get("thermo_region_delta_count", 0.0)) > 0.0
        ]
        if not thermo_field_records:
            errors.append(
                "thermo field summary records missing while RUNMAT_VALIDATE_REQUIRE_THERMO_FIELD_SUMMARY=true"
            )
        else:
            if not any(
                isinstance(record.get("thermo_spatial_coverage_ratio"), (int, float))
                for record in thermo_field_records
            ):
                errors.append(
                    "thermo_spatial_coverage_ratio missing across thermo field records while RUNMAT_VALIDATE_REQUIRE_THERMO_FIELD_SUMMARY=true"
                )
            if not any(
                isinstance(record.get("thermo_field_extrapolation_ratio"), (int, float))
                for record in thermo_field_records
            ):
                errors.append(
                    "thermo_field_extrapolation_ratio missing across thermo field records while RUNMAT_VALIDATE_REQUIRE_THERMO_FIELD_SUMMARY=true"
                )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("nonlinear analysis report schema checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
