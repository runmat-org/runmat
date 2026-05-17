#!/usr/bin/env python3
import json
import os
from pathlib import Path


REQUIRED_METRICS_BY_FIXTURE = {
    "electromagnetic_reference_homogeneous_gpu_provider": {
        "em_homogeneous_boundary_energy_ratio",
        "em_homogeneous_sigma_omega_response_coverage_ratio",
        "em_homogeneous_sigma_omega_scale_mean",
        "em_homogeneous_dispersive_loss_scale_mean",
        "em_homogeneous_dispersive_phase_attenuation_mean",
        "em_homogeneous_dispersive_coupling_ratio",
        "em_homogeneous_dispersive_phase_conductivity_attenuation_ratio",
        "em_homogeneous_flux_divergence_proxy",
        "em_homogeneous_source_material_alignment_ratio",
        "em_homogeneous_source_region_coverage_ratio",
        "em_homogeneous_boundary_anchor_ratio",
    },
    "electromagnetic_reference_heterogeneous_gpu_provider": {
        "em_heterogeneous_sigma_omega_response_coverage_ratio",
        "em_heterogeneous_source_realization_ratio",
        "em_heterogeneous_sigma_omega_scale_spread_ratio",
        "em_heterogeneous_dispersive_loss_scale_mean",
        "em_heterogeneous_dispersive_phase_attenuation_mean",
        "em_heterogeneous_dispersive_coupling_ratio",
        "em_heterogeneous_dispersive_phase_conductivity_attenuation_ratio",
        "em_heterogeneous_region_contrast_index",
        "em_heterogeneous_source_material_alignment_ratio",
        "em_heterogeneous_source_region_coverage_ratio",
        "em_heterogeneous_boundary_anchor_ratio",
    },
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider": {
        "em_boundary_penalty_anchor_ratio",
        "em_boundary_penalty_conditioning_contribution",
    },
    "electromagnetic_reference_multi_region_phased_source_gpu_provider": {
        "em_phased_source_energy_consistency_ratio",
        "em_phased_source_region_coverage_ratio",
    },
    "electromagnetic_reference_sparse_assignments_gpu_provider": {
        "em_sparse_fallback_coefficient_ratio",
    },
    "electromagnetic_reference_fallback_heavy_gpu_provider": {
        "em_fallback_heavy_fallback_coefficient_ratio",
    },
    "electromagnetic_reference_overlap_interference_gpu_provider": {
        "em_overlap_source_interference_index",
    },
    "electromagnetic_reference_boundary_kernel_gpu_provider": {
        "em_boundary_kernel_boundary_localization_ratio",
    },
    "cfd_steady_gpu_provider": {
        "cfd_reference_density_kg_per_m3",
        "cfd_reynolds_proxy",
    },
    "cht_coupled_gpu_provider": {
        "cht_applied_temperature_delta_k",
        "cht_reynolds_proxy",
    },
    "fsi_coupled_gpu_provider": {
        "fsi_reynolds_proxy",
        "fsi_structural_step_count",
    },
    "thermo_gradient_pathological_gpu_provider": {
        "thermo_gradient_pathological_spread_ratio",
    },
    "thermo_shock_oscillatory_gpu_provider": {
        "thermo_shock_oscillatory_temporal_variation",
    },
    "electro_thermal_joule_pathological_gpu_provider": {
        "electro_thermal_pathological_conductivity_spread_ratio",
    },
    "nonlinear_plastic_hardening_reference_complex_gpu_provider": {
        "plasticity_hardening_reference_complex_load_realization_ratio",
    },
    "nonlinear_contact_frictionless_reference_complex_gpu_provider": {
        "contact_frictionless_complex_load_amplification_ratio",
    },
    "acoustic_harmonic_gpu_provider": {
        "acoustic_max_m_orthogonality_offdiag",
        "acoustic_min_relative_frequency_separation",
        "acoustic_mode_count",
        "acoustic_residual_warn_threshold",
    },
}


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    path = Path(
        os.getenv(
            "RUNMAT_EXTERNAL_REFERENCE_BENCHMARK_PATH",
            "target/runmat-analysis-artifacts/external_reference_benchmark.json",
        )
    )
    enforce = is_true(os.getenv("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_ENFORCE", "false"))
    require_pass = is_true(
        os.getenv("RUNMAT_VALIDATE_EXTERNAL_REFERENCE_REQUIRE_PASS", "false")
    )

    if not path.exists():
        print(f"external reference benchmark artifact missing: {path}")
        return 1 if enforce else 0

    payload = json.loads(path.read_text())
    errors = []

    if payload.get("schema_version") != "external-reference-benchmark/v1":
        errors.append("schema_version must be external-reference-benchmark/v1")
    if not isinstance(payload.get("scenario_id"), str):
        errors.append("scenario_id missing or invalid")
    if not isinstance(payload.get("reference_source"), dict):
        errors.append("reference_source missing or invalid")
    if not isinstance(payload.get("generated_at"), str):
        errors.append("generated_at missing or invalid")

    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("metrics must be a non-empty list")
    else:
        metric_names_by_fixture = {}
        for idx, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                errors.append(f"metrics[{idx}] must be an object")
                continue
            if not isinstance(metric.get("name"), str):
                errors.append(f"metrics[{idx}].name missing or invalid")
                continue
            if not isinstance(metric.get("fixture_id"), str):
                errors.append(f"metrics[{idx}].fixture_id missing or invalid")
            if "source" in metric and metric.get("source") not in {
                "field",
                "threshold_assertion",
            }:
                errors.append(f"metrics[{idx}].source invalid")
            if not isinstance(metric.get("observed"), (int, float)):
                errors.append(f"metrics[{idx}].observed missing or invalid")
            if not isinstance(metric.get("reference"), (int, float)):
                errors.append(f"metrics[{idx}].reference missing or invalid")
            if not isinstance(metric.get("pass"), bool):
                errors.append(f"metrics[{idx}].pass missing or invalid")
            fixture_id = metric.get("fixture_id")
            if isinstance(fixture_id, str):
                metric_names_by_fixture.setdefault(fixture_id, set()).add(metric["name"])

        for fixture_id, required_names in REQUIRED_METRICS_BY_FIXTURE.items():
            observed_names = metric_names_by_fixture.get(fixture_id, set())
            missing = sorted(name for name in required_names if name not in observed_names)
            if missing:
                errors.append(
                    f"fixture {fixture_id} missing required external-reference metrics: "
                    + ", ".join(missing)
                )

    if require_pass and isinstance(metrics, list):
        failing = [m.get("name", f"index_{i}") for i, m in enumerate(metrics) if isinstance(m, dict) and m.get("pass") is False]
        if failing:
            errors.append(
                "reference comparison failed for metrics: " + ", ".join(str(item) for item in failing)
            )

    if errors:
        print("external reference benchmark validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1 if enforce else 0

    print("external reference benchmark validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
