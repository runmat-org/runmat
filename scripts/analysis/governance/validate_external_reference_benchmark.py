#!/usr/bin/env python3
import json
import os
from pathlib import Path


REQUIRED_METRICS_BY_FIXTURE = {
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
        "electro_nonlinear_joule_heating_scale",
        "electro_nonlinear_conductivity_spread_ratio",
        "electro_nonlinear_severity_peak",
    },
    "nonlinear_plasticity_proxy_gpu_provider": {
        "plasticity_nonlinear_severity_peak",
        "plasticity_nonlinear_severity_mean",
    },
    "nonlinear_contact_proxy_gpu_provider": {
        "contact_nonlinear_severity_peak",
        "contact_nonlinear_severity_mean",
    },
    "nonlinear_contact_frictionless_reference_gpu_provider": {
        "contact_frictionless_severity_peak",
        "contact_frictionless_severity_mean",
    },
    "nonlinear_plastic_hardening_reference_gpu_provider": {
        "plasticity_hardening_reference_severity_peak",
        "plasticity_hardening_reference_severity_mean",
    },
    "thermo_gradient_benign_gpu_provider": {
        "thermo_gradient_benign_spread_ratio",
        "thermo_gradient_benign_heterogeneity",
    },
    "electromagnetic_reference_homogeneous_gpu_provider": {
        "em_homogeneous_boundary_energy_ratio",
        "em_homogeneous_sigma_omega_response_coverage_ratio",
        "em_homogeneous_sigma_omega_scale_mean",
        "em_homogeneous_sigma_omega_scale_spread_ratio",
        "electromagnetic_applied_current_a",
        "electromagnetic_source_region_energy_consistency_ratio",
        "electromagnetic_source_localization_ratio",
        "electromagnetic_boundary_condition_localization_ratio",
        "electromagnetic_ground_anchor_effectiveness_ratio",
        "electromagnetic_source_interference_index",
        "electromagnetic_source_realization_ratio",
        "electromagnetic_source_region_coverage_ratio",
        "electromagnetic_source_material_alignment_ratio",
        "electromagnetic_assignment_coverage_ratio",
        "electromagnetic_fallback_coefficient_ratio",
        "electromagnetic_boundary_anchor_ratio",
        "electromagnetic_conductivity_spread_ratio",
        "electromagnetic_relative_permittivity_spread_ratio",
        "electromagnetic_relative_permeability_spread_ratio",
        "electromagnetic_material_heterogeneity_index",
        "electromagnetic_region_coefficient_contrast_index",
        "electromagnetic_boundary_energy_ratio",
        "electromagnetic_boundary_penalty_conditioning_contribution",
        "electromagnetic_source_overlap_ratio",
        "electromagnetic_insulation_leakage_proxy",
        "electromagnetic_placeholder_quality",
        "electromagnetic_energy_imbalance_ratio",
        "electromagnetic_flux_divergence_proxy",
        "electromagnetic_real_residual_norm",
        "electromagnetic_imag_residual_norm",
        "electromagnetic_solver_conditioning_proxy",
        "electromagnetic_reference_frequency_hz",
        "electromagnetic_sweep_count",
        "electromagnetic_resonance_peak_frequency_hz",
        "electromagnetic_resonance_peak_flux_density",
        "electromagnetic_resonance_bandwidth_hz",
        "electromagnetic_resonance_q_proxy",
        "electromagnetic_resonance_flux_gain",
        "em_homogeneous_relative_permittivity_frequency_scale_mean",
        "em_homogeneous_relative_permittivity_frequency_scale_spread_ratio",
        "em_homogeneous_relative_permittivity_frequency_response_coverage_ratio",
        "em_homogeneous_relative_permeability_frequency_scale_mean",
        "em_homogeneous_relative_permeability_frequency_scale_spread_ratio",
        "em_homogeneous_relative_permeability_frequency_response_coverage_ratio",
        "em_homogeneous_dispersive_loss_scale_mean",
        "em_homogeneous_dispersive_loss_scale_spread_ratio",
        "em_homogeneous_dispersive_phase_attenuation_mean",
        "em_homogeneous_dispersive_phase_attenuation_spread_ratio",
        "em_homogeneous_dispersive_coupling_ratio",
        "em_homogeneous_dispersive_phase_conductivity_attenuation_ratio",
        "em_homogeneous_material_heterogeneity_index",
        "em_homogeneous_conductivity_spread_ratio",
        "em_homogeneous_relative_permittivity_spread_ratio",
        "em_homogeneous_relative_permeability_spread_ratio",
        "em_homogeneous_assignment_coverage_ratio",
        "em_homogeneous_fallback_coefficient_ratio",
        "em_homogeneous_flux_divergence_proxy",
        "em_homogeneous_energy_imbalance_ratio",
        "em_homogeneous_source_realization_ratio",
        "em_homogeneous_source_material_alignment_ratio",
        "em_homogeneous_source_region_coverage_ratio",
        "em_homogeneous_boundary_anchor_ratio",
    },
    "electromagnetic_reference_heterogeneous_gpu_provider": {
        "em_heterogeneous_sigma_omega_scale_mean",
        "em_heterogeneous_sigma_omega_response_coverage_ratio",
        "em_heterogeneous_source_realization_ratio",
        "em_heterogeneous_sigma_omega_scale_spread_ratio",
        "electromagnetic_applied_current_a",
        "electromagnetic_source_region_energy_consistency_ratio",
        "electromagnetic_source_localization_ratio",
        "electromagnetic_boundary_condition_localization_ratio",
        "electromagnetic_ground_anchor_effectiveness_ratio",
        "electromagnetic_source_interference_index",
        "electromagnetic_source_realization_ratio",
        "electromagnetic_source_region_coverage_ratio",
        "electromagnetic_source_material_alignment_ratio",
        "electromagnetic_assignment_coverage_ratio",
        "electromagnetic_fallback_coefficient_ratio",
        "electromagnetic_boundary_anchor_ratio",
        "electromagnetic_conductivity_spread_ratio",
        "electromagnetic_relative_permittivity_spread_ratio",
        "electromagnetic_relative_permeability_spread_ratio",
        "electromagnetic_material_heterogeneity_index",
        "electromagnetic_region_coefficient_contrast_index",
        "electromagnetic_boundary_energy_ratio",
        "electromagnetic_boundary_penalty_conditioning_contribution",
        "electromagnetic_source_overlap_ratio",
        "electromagnetic_insulation_leakage_proxy",
        "electromagnetic_placeholder_quality",
        "electromagnetic_energy_imbalance_ratio",
        "electromagnetic_flux_divergence_proxy",
        "electromagnetic_real_residual_norm",
        "electromagnetic_imag_residual_norm",
        "electromagnetic_solver_conditioning_proxy",
        "electromagnetic_reference_frequency_hz",
        "electromagnetic_sweep_count",
        "electromagnetic_resonance_peak_frequency_hz",
        "electromagnetic_resonance_peak_flux_density",
        "electromagnetic_resonance_bandwidth_hz",
        "electromagnetic_resonance_q_proxy",
        "electromagnetic_resonance_flux_gain",
        "em_heterogeneous_relative_permittivity_frequency_scale_mean",
        "em_heterogeneous_relative_permittivity_frequency_scale_spread_ratio",
        "em_heterogeneous_relative_permittivity_frequency_response_coverage_ratio",
        "em_heterogeneous_relative_permeability_frequency_scale_mean",
        "em_heterogeneous_relative_permeability_frequency_scale_spread_ratio",
        "em_heterogeneous_relative_permeability_frequency_response_coverage_ratio",
        "em_heterogeneous_dispersive_loss_scale_mean",
        "em_heterogeneous_dispersive_loss_scale_spread_ratio",
        "em_heterogeneous_dispersive_phase_attenuation_mean",
        "em_heterogeneous_dispersive_phase_attenuation_spread_ratio",
        "em_heterogeneous_dispersive_coupling_ratio",
        "em_heterogeneous_dispersive_phase_conductivity_attenuation_ratio",
        "em_heterogeneous_material_heterogeneity_index",
        "em_heterogeneous_conductivity_spread_ratio",
        "em_heterogeneous_relative_permittivity_spread_ratio",
        "em_heterogeneous_relative_permeability_spread_ratio",
        "em_heterogeneous_region_contrast_index",
        "em_heterogeneous_assignment_coverage_ratio",
        "em_heterogeneous_source_material_alignment_ratio",
        "em_heterogeneous_source_region_coverage_ratio",
        "em_heterogeneous_flux_divergence_proxy",
        "em_heterogeneous_energy_imbalance_ratio",
        "em_heterogeneous_boundary_anchor_ratio",
    },
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider": {
        "em_boundary_penalty_anchor_ratio",
        "em_boundary_penalty_conditioning_contribution",
        "em_boundary_penalty_real_residual_norm",
        "em_boundary_penalty_imag_residual_norm",
        "electromagnetic_placeholder_quality",
    },
    "electromagnetic_reference_multi_region_phased_source_gpu_provider": {
        "em_phased_source_energy_consistency_ratio",
        "em_phased_source_region_coverage_ratio",
        "em_phased_source_overlap_ratio",
        "em_phased_source_interference_index",
        "electromagnetic_placeholder_quality",
    },
    "electromagnetic_reference_sparse_assignments_gpu_provider": {
        "em_sparse_fallback_coefficient_ratio",
        "em_sparse_assignment_coverage_ratio",
        "em_sparse_source_realization_ratio",
        "em_sparse_source_region_coverage_ratio",
        "em_sparse_source_material_alignment_ratio",
        "em_sparse_boundary_anchor_ratio",
        "em_sparse_energy_imbalance_ratio",
        "electromagnetic_placeholder_quality",
    },
    "electromagnetic_reference_fallback_heavy_gpu_provider": {
        "em_fallback_heavy_fallback_coefficient_ratio",
        "em_fallback_heavy_assignment_coverage_ratio",
        "em_fallback_heavy_source_realization_ratio",
        "em_fallback_heavy_source_region_coverage_ratio",
        "em_fallback_heavy_source_material_alignment_ratio",
        "em_fallback_heavy_boundary_anchor_ratio",
        "em_fallback_heavy_energy_imbalance_ratio",
        "electromagnetic_placeholder_quality",
    },
    "electromagnetic_reference_overlap_interference_gpu_provider": {
        "em_overlap_source_interference_index",
        "em_overlap_source_overlap_ratio",
        "em_overlap_source_region_coverage_ratio",
        "em_overlap_source_material_alignment_ratio",
        "electromagnetic_placeholder_quality",
    },
    "electromagnetic_reference_boundary_kernel_gpu_provider": {
        "em_boundary_kernel_boundary_localization_ratio",
        "em_boundary_kernel_ground_anchor_effectiveness_ratio",
        "em_boundary_kernel_insulation_leakage_proxy",
        "electromagnetic_placeholder_quality",
    },
    "cfd_steady_gpu_provider": {
        "cfd_reference_density_kg_per_m3",
        "cfd_dynamic_viscosity_pa_s",
        "cfd_inlet_velocity_m_per_s",
        "cfd_turbulence_intensity",
        "cfd_reynolds_proxy",
        "cfd_profile_point_count",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "cht_coupled_gpu_provider": {
        "cht_reference_density_kg_per_m3",
        "cht_dynamic_viscosity_pa_s",
        "cht_inlet_velocity_m_per_s",
        "cht_turbulence_intensity",
        "cht_applied_temperature_delta_k",
        "cht_reynolds_proxy",
        "cht_profile_point_count",
        "cht_step_count",
        "cht_time_step_s",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "fsi_coupled_gpu_provider": {
        "fsi_reference_density_kg_per_m3",
        "fsi_dynamic_viscosity_pa_s",
        "fsi_inlet_velocity_m_per_s",
        "fsi_turbulence_intensity",
        "fsi_reynolds_proxy",
        "fsi_profile_point_count",
        "fsi_step_count",
        "fsi_time_step_s",
        "fsi_structural_step_count",
        "fsi_cfd_profile_point_count",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
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
    "thermo_ramp_smooth_gpu_provider": {
        "thermo_ramp_smooth_temporal_variation",
        "thermo_ramp_smooth_spatial_gradient_index",
        "thermo_ramp_smooth_spatial_coverage_ratio",
        "thermo_ramp_smooth_field_extrapolation_ratio",
        "thermo_ramp_smooth_field_clamp_ratio",
    },
    "thermo_ramp_smooth_field_artifact_gpu_provider": {
        "thermo_ramp_smooth_temporal_variation",
        "thermo_ramp_smooth_spatial_gradient_index",
        "thermo_ramp_smooth_spatial_coverage_ratio",
        "thermo_ramp_smooth_field_extrapolation_ratio",
        "thermo_ramp_smooth_field_clamp_ratio",
    },
    "thermo_gradient_pathological_gpu_provider": {
        "thermo_gradient_pathological_spread_ratio",
        "thermo_gradient_pathological_heterogeneity",
        "thermo_gradient_pathological_temporal_variation",
    },
    "thermo_shock_oscillatory_gpu_provider": {
        "thermo_shock_oscillatory_temporal_variation",
        "thermo_shock_oscillatory_spatial_gradient_index",
        "thermo_shock_oscillatory_spatial_coverage_ratio",
        "thermo_shock_oscillatory_field_extrapolation_ratio",
        "thermo_shock_oscillatory_field_clamp_ratio",
    },
    "thermo_shock_oscillatory_field_artifact_gpu_provider": {
        "thermo_shock_oscillatory_temporal_variation",
        "thermo_shock_oscillatory_spatial_gradient_index",
        "thermo_shock_oscillatory_spatial_coverage_ratio",
        "thermo_shock_oscillatory_field_extrapolation_ratio",
        "thermo_shock_oscillatory_field_clamp_ratio",
    },
    "thermal_standalone_ramp_gpu_provider": {
        "thermal_standalone_max_residual_norm",
        "thermal_standalone_min_temperature_k",
        "thermal_standalone_max_temperature_k",
        "thermal_standalone_conductivity_spread_ratio",
        "thermal_standalone_heat_capacity_spread_ratio",
        "thermal_standalone_spatial_gradient_index",
        "thermal_standalone_monotonic_response_fraction",
        "thermal_standalone_response_realization_ratio",
    },
    "electro_thermal_joule_benign_gpu_provider": {
        "electro_thermal_benign_joule_heating_scale",
        "electro_thermal_benign_conductivity_spread_ratio",
        "electro_thermal_benign_transient_severity_peak",
        "electro_thermal_benign_temporal_variation",
        "electro_thermal_benign_time_scale_mean",
    },
    "electro_thermal_joule_pathological_gpu_provider": {
        "electro_thermal_pathological_joule_heating_scale",
        "electro_thermal_pathological_conductivity_spread_ratio",
        "electro_thermal_pathological_transient_severity_peak",
        "electro_thermal_pathological_temporal_variation",
        "electro_thermal_pathological_time_scale_mean",
    },
    "nonlinear_plastic_hardening_reference_complex_gpu_provider": {
        "plasticity_hardening_reference_complex_severity_peak",
        "plasticity_hardening_reference_complex_severity_mean",
        "plasticity_hardening_reference_complex_load_realization_ratio",
    },
    "nonlinear_contact_frictionless_reference_complex_gpu_provider": {
        "contact_frictionless_complex_severity_peak",
        "contact_frictionless_complex_severity_mean",
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
