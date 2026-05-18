#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path


REQUIRED_FIXTURES = {
    "nonlinear_assembly_gpu_provider": {
        "nonlinear_total_increments",
        "nonlinear_failed_increments",
        "nonlinear_iteration_spike_count",
        "nonlinear_converged_increments",
        "nonlinear_line_search_backtracks",
        "nonlinear_max_increment_norm",
    },
    "nonlinear_assembly_stress_gpu_provider": {
        "nonlinear_stress_total_increments",
        "nonlinear_stress_stall_count",
        "nonlinear_stress_iteration_spike_count",
        "nonlinear_stress_converged_increments",
        "nonlinear_stress_failed_increments",
        "nonlinear_stress_line_search_backtracks",
        "nonlinear_stress_max_increment_norm",
        "nonlinear_stress_max_residual_norm",
        "nonlinear_stress_tangent_rebuild_count",
    },
    "nonlinear_softening_proxy_gpu_provider": {
        "nonlinear_softening_total_increments",
        "nonlinear_softening_spike_count",
        "nonlinear_softening_backtrack_bursts",
        "nonlinear_softening_failed_increments",
        "nonlinear_softening_stall_count",
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
        "plasticity_nonlinear_load_amplification_ratio",
        "plasticity_nonlinear_load_realization_ratio",
    },
    "nonlinear_contact_proxy_gpu_provider": {
        "contact_nonlinear_severity_peak",
        "contact_nonlinear_severity_mean",
        "contact_nonlinear_load_amplification_ratio",
        "contact_nonlinear_load_realization_ratio",
    },
    "nonlinear_contact_frictionless_reference_gpu_provider": {
        "contact_frictionless_severity_peak",
        "contact_frictionless_severity_mean",
        "contact_frictionless_load_amplification_ratio",
        "contact_frictionless_load_realization_ratio",
    },
    "nonlinear_contact_frictionless_reference_complex_gpu_provider": {
        "contact_frictionless_complex_severity_peak",
        "contact_frictionless_complex_severity_mean",
        "contact_frictionless_complex_load_amplification_ratio",
        "contact_frictionless_complex_load_realization_ratio",
    },
    "nonlinear_plastic_hardening_reference_gpu_provider": {
        "plasticity_hardening_reference_severity_peak",
        "plasticity_hardening_reference_severity_mean",
        "plasticity_hardening_reference_load_amplification_ratio",
        "plasticity_hardening_reference_load_realization_ratio",
    },
    "nonlinear_plastic_hardening_reference_complex_gpu_provider": {
        "plasticity_hardening_reference_complex_severity_peak",
        "plasticity_hardening_reference_complex_severity_mean",
        "plasticity_hardening_reference_complex_load_realization_ratio",
        "plasticity_hardening_reference_complex_load_amplification_ratio",
    },
    "thermo_mech_kickoff_gpu_provider": {
        "thermo_mech_thermal_strain_scale",
        "thermo_mech_thermal_load_scale",
        "thermo_mech_effective_modulus_scale",
        "thermo_mech_material_spread_ratio",
        "thermo_mech_assignment_heterogeneity_index",
        "thermo_mech_transient_severity",
        "thermo_mech_transient_time_scale_mean",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
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
        "thermo_ramp_smooth_field_clamp_ratio",
    },
    "thermo_ramp_smooth_field_artifact_gpu_provider": {
        "thermo_ramp_smooth_temporal_variation",
        "thermo_ramp_smooth_spatial_gradient_index",
        "thermo_ramp_smooth_spatial_coverage_ratio",
        "thermo_ramp_smooth_field_extrapolation_ratio",
        "thermo_ramp_smooth_field_clamp_ratio",
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
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "electro_thermal_joule_pathological_gpu_provider": {
        "electro_thermal_pathological_joule_heating_scale",
        "electro_thermal_pathological_conductivity_spread_ratio",
        "electro_thermal_pathological_transient_severity_peak",
        "electro_thermal_pathological_temporal_variation",
        "electro_thermal_pathological_time_scale_mean",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "electromagnetic_reference_homogeneous_gpu_provider": {
        "em_homogeneous_sigma_omega_scale_mean",
        "em_homogeneous_sigma_omega_scale_spread_ratio",
        "em_homogeneous_sigma_omega_response_coverage_ratio",
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
        "em_homogeneous_source_realization_ratio",
        "em_homogeneous_flux_divergence_proxy",
        "em_homogeneous_energy_imbalance_ratio",
        "em_homogeneous_boundary_energy_ratio",
        "em_homogeneous_source_material_alignment_ratio",
        "em_homogeneous_source_region_coverage_ratio",
        "em_homogeneous_boundary_anchor_ratio",
    },
    "electromagnetic_reference_heterogeneous_gpu_provider": {
        "em_heterogeneous_sigma_omega_scale_mean",
        "em_heterogeneous_sigma_omega_scale_spread_ratio",
        "em_heterogeneous_sigma_omega_response_coverage_ratio",
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
        "em_heterogeneous_source_realization_ratio",
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
    },
    "electromagnetic_reference_multi_region_phased_source_gpu_provider": {
        "em_phased_source_energy_consistency_ratio",
        "em_phased_source_interference_index",
        "em_phased_source_overlap_ratio",
        "em_phased_source_region_coverage_ratio",
    },
    "electromagnetic_reference_sparse_assignments_gpu_provider": {
        "em_sparse_assignment_coverage_ratio",
        "em_sparse_fallback_coefficient_ratio",
        "em_sparse_source_realization_ratio",
        "em_sparse_source_region_coverage_ratio",
        "em_sparse_source_material_alignment_ratio",
        "em_sparse_boundary_anchor_ratio",
        "em_sparse_energy_imbalance_ratio",
    },
    "electromagnetic_reference_fallback_heavy_gpu_provider": {
        "em_fallback_heavy_assignment_coverage_ratio",
        "em_fallback_heavy_fallback_coefficient_ratio",
        "em_fallback_heavy_source_realization_ratio",
        "em_fallback_heavy_source_region_coverage_ratio",
        "em_fallback_heavy_source_material_alignment_ratio",
        "em_fallback_heavy_boundary_anchor_ratio",
        "em_fallback_heavy_energy_imbalance_ratio",
    },
    "electromagnetic_reference_overlap_interference_gpu_provider": {
        "em_overlap_source_interference_index",
        "em_overlap_source_region_coverage_ratio",
        "em_overlap_source_overlap_ratio",
        "em_overlap_source_material_alignment_ratio",
    },
    "electromagnetic_reference_boundary_kernel_gpu_provider": {
        "em_boundary_kernel_boundary_localization_ratio",
        "em_boundary_kernel_ground_anchor_effectiveness_ratio",
        "em_boundary_kernel_insulation_leakage_proxy",
    },
    "acoustic_harmonic_gpu_provider": {
        "acoustic_max_m_orthogonality_offdiag",
        "acoustic_min_relative_frequency_separation",
        "acoustic_mode_count",
        "acoustic_residual_warn_threshold",
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
        "cht_reynolds_proxy",
        "cht_profile_point_count",
        "cht_applied_temperature_delta_k",
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

ELECTRO_REQUIRED_FIELDS = {
    "electro_thermal_coupling_enabled",
    "electro_thermal_coupling_fingerprint",
    "electro_joule_heating_scale",
    "electro_conductivity_spread_ratio",
    "electro_transient_severity",
    "electro_nonlinear_severity",
}

PLASTIC_REQUIRED_FIELDS = {
    "plastic_nonlinear_severity",
}

CONTACT_REQUIRED_FIELDS = {
    "contact_nonlinear_severity",
}

EM_SWEEP_RESONANCE_REQUIRED_FIELDS = {
    "electromagnetic_sweep_count",
    "electromagnetic_resonance_peak_frequency_hz",
    "electromagnetic_resonance_peak_flux_density",
    "electromagnetic_resonance_bandwidth_hz",
    "electromagnetic_resonance_q_proxy",
    "electromagnetic_resonance_flux_gain",
}

EM_FREQUENCY_REQUIRED_FIELDS = {
    "electromagnetic_reference_frequency_hz",
}

EM_CONDITIONING_REQUIRED_FIELDS = {
    "electromagnetic_solver_conditioning_proxy",
}

EM_APPLIED_CURRENT_REQUIRED_FIELDS = {
    "electromagnetic_applied_current_a",
}

EM_SOURCE_ENERGY_CONSISTENCY_REQUIRED_FIELDS = {
    "electromagnetic_source_region_energy_consistency_ratio",
}

EM_SOURCE_LOCALIZATION_REQUIRED_FIELDS = {
    "electromagnetic_source_localization_ratio",
}

EM_BOUNDARY_CONDITION_LOCALIZATION_REQUIRED_FIELDS = {
    "electromagnetic_boundary_condition_localization_ratio",
}

EM_GROUND_ANCHOR_EFFECTIVENESS_REQUIRED_FIELDS = {
    "electromagnetic_ground_anchor_effectiveness_ratio",
}

EM_SOURCE_INTERFERENCE_REQUIRED_FIELDS = {
    "electromagnetic_source_interference_index",
}

EM_SOURCE_FIDELITY_REQUIRED_FIELDS = {
    "electromagnetic_source_realization_ratio",
    "electromagnetic_source_region_coverage_ratio",
    "electromagnetic_source_material_alignment_ratio",
}

EM_CORE_ASSIGNMENT_REQUIRED_FIELDS = {
    "electromagnetic_assignment_coverage_ratio",
    "electromagnetic_fallback_coefficient_ratio",
    "electromagnetic_boundary_anchor_ratio",
}

EM_CONSTITUTIVE_REQUIRED_FIELDS = {
    "electromagnetic_conductivity_spread_ratio",
    "electromagnetic_relative_permittivity_spread_ratio",
    "electromagnetic_relative_permeability_spread_ratio",
    "electromagnetic_material_heterogeneity_index",
    "electromagnetic_region_coefficient_contrast_index",
}

EM_BOUNDARY_SOURCE_REQUIRED_FIELDS = {
    "electromagnetic_boundary_energy_ratio",
    "electromagnetic_boundary_penalty_conditioning_contribution",
    "electromagnetic_source_overlap_ratio",
    "electromagnetic_insulation_leakage_proxy",
}

EM_PLACEHOLDER_QUALITY_REQUIRED_FIELDS = {
    "electromagnetic_placeholder_quality",
}

EM_RESIDUAL_REQUIRED_FIELDS = {
    "electromagnetic_real_residual_norm",
    "electromagnetic_imag_residual_norm",
}

EM_BALANCE_REQUIRED_FIELDS = {
    "electromagnetic_energy_imbalance_ratio",
    "electromagnetic_flux_divergence_proxy",
}

PERFORMANCE_REQUIRED_FIELDS = {
    "nonlinear_assembly_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_assembly_stress_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_softening_proxy_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_load_path_mix_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_plasticity_proxy_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_contact_proxy_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_contact_frictionless_reference_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_contact_frictionless_reference_complex_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_plastic_hardening_reference_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_plastic_hardening_reference_complex_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_mech_kickoff_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_gradient_benign_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_gradient_pathological_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_ramp_smooth_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_ramp_smooth_field_artifact_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_shock_oscillatory_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermo_shock_oscillatory_field_artifact_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "thermal_standalone_ramp_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electro_thermal_joule_benign_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electro_thermal_joule_pathological_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_homogeneous_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_heterogeneous_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_multi_region_phased_source_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_sparse_assignments_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_fallback_heavy_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_overlap_interference_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "electromagnetic_reference_boundary_kernel_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "acoustic_harmonic_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cfd_steady_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cht_coupled_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "fsi_coupled_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
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
    require_electro_summary = is_true(
        os.getenv("RUNMAT_VALIDATE_REQUIRE_ELECTRO_SUMMARY", "false")
    )
    require_plastic_summary = is_true(
        os.getenv("RUNMAT_VALIDATE_REQUIRE_PLASTIC_SUMMARY", "false")
    )
    require_contact_summary = is_true(
        os.getenv("RUNMAT_VALIDATE_REQUIRE_CONTACT_SUMMARY", "false")
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
            if "field_artifact" in fixture_id:
                for required_field in (
                    "thermo_field_artifact_id",
                    "thermo_field_artifact_approved",
                    "thermo_field_artifact_age_days",
                    "thermo_field_artifact_provenance_valid",
                ):
                    if required_field not in record:
                        errors.append(
                            f"fixture {fixture_id} missing thermo artifact field: {required_field}"
                        )

        if fixture_id in {
            "electro_thermal_joule_benign_gpu_provider",
            "electro_thermal_joule_pathological_gpu_provider",
            "nonlinear_load_path_mix_gpu_provider",
            "nonlinear_plasticity_proxy_gpu_provider",
        }:
            missing_fields = sorted(
                field for field in ELECTRO_REQUIRED_FIELDS if field not in record
            )
            if missing_fields:
                errors.append(
                    f"fixture {fixture_id} missing electro summary fields: {', '.join(missing_fields)}"
                )

        if fixture_id in {
            "nonlinear_plasticity_proxy_gpu_provider",
        }:
            missing_fields = sorted(
                field for field in PLASTIC_REQUIRED_FIELDS if field not in record
            )
            if missing_fields:
                errors.append(
                    f"fixture {fixture_id} missing plastic summary fields: {', '.join(missing_fields)}"
                )

        if fixture_id in {
            "nonlinear_contact_proxy_gpu_provider",
            "nonlinear_contact_frictionless_reference_gpu_provider",
            "nonlinear_contact_frictionless_reference_complex_gpu_provider",
        }:
            missing_fields = sorted(
                field for field in CONTACT_REQUIRED_FIELDS if field not in record
            )
            if missing_fields:
                errors.append(
                    f"fixture {fixture_id} missing contact summary fields: {', '.join(missing_fields)}"
                )

        if fixture_id.startswith("electromagnetic_reference_"):
            missing_fields = []
            for field in sorted(EM_PLACEHOLDER_QUALITY_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM activation-quality fields: "
                    + ", ".join(missing_fields)
                )
            if record.get("electromagnetic_enabled") is not True:
                errors.append(
                    f"fixture {fixture_id} missing true EM enabled flag: electromagnetic_enabled"
                )
            missing_fields = []
            for field in sorted(EM_BALANCE_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM balance fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_RESIDUAL_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM residual fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_FREQUENCY_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM reference-frequency fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_CONDITIONING_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM conditioning fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_APPLIED_CURRENT_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM applied-current fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_SOURCE_ENERGY_CONSISTENCY_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM source-energy-consistency fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_SOURCE_LOCALIZATION_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM source-localization fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_SOURCE_INTERFERENCE_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM source-interference fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_BOUNDARY_CONDITION_LOCALIZATION_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM boundary-condition-localization fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_GROUND_ANCHOR_EFFECTIVENESS_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM ground-anchor-effectiveness fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_SOURCE_FIDELITY_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM source-fidelity fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_CORE_ASSIGNMENT_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM core-assignment fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_CONSTITUTIVE_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM constitutive fields: "
                    + ", ".join(missing_fields)
                )
            missing_fields = []
            for field in sorted(EM_BOUNDARY_SOURCE_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM boundary/source fields: "
                    + ", ".join(missing_fields)
                )

        if fixture_id in {
            "electromagnetic_reference_homogeneous_gpu_provider",
            "electromagnetic_reference_heterogeneous_gpu_provider",
        }:
            missing_fields = []
            for field in sorted(EM_SWEEP_RESONANCE_REQUIRED_FIELDS):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    missing_fields.append(field)
            if missing_fields:
                errors.append(
                    "fixture "
                    f"{fixture_id} missing finite EM sweep/resonance fields: "
                    + ", ".join(missing_fields)
                )
        if fixture_id in PERFORMANCE_REQUIRED_FIELDS:
            for field in sorted(PERFORMANCE_REQUIRED_FIELDS[fixture_id]):
                value = record.get(field)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    errors.append(
                        f"fixture {fixture_id} missing finite performance field: {field}"
                    )
                    continue
                if float(value) < 0.0:
                    errors.append(
                        f"fixture {fixture_id} has negative performance field: {field}"
                    )
            backend = record.get("gpu_solver_backend")
            if not isinstance(backend, str) or not backend.strip():
                errors.append(
                    f"fixture {fixture_id} missing non-empty performance field: gpu_solver_backend"
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

    if require_electro_summary:
        electro_records = [
            record
            for record in records.values()
            if isinstance(record.get("electro_thermal_coupling_enabled"), bool)
            or isinstance(record.get("electro_transient_severity"), (int, float))
            or isinstance(record.get("electro_nonlinear_severity"), (int, float))
        ]
        if not electro_records:
            errors.append(
                "electro summary fields missing across all records while RUNMAT_VALIDATE_REQUIRE_ELECTRO_SUMMARY=true"
            )

    if require_plastic_summary:
        plastic_records = [
            record
            for record in records.values()
            if isinstance(record.get("plastic_nonlinear_severity"), (int, float))
        ]
        if not plastic_records:
            errors.append(
                "plastic summary fields missing across all records while RUNMAT_VALIDATE_REQUIRE_PLASTIC_SUMMARY=true"
            )

    if require_contact_summary:
        contact_records = [
            record
            for record in records.values()
            if isinstance(record.get("contact_nonlinear_severity"), (int, float))
        ]
        if not contact_records:
            errors.append(
                "contact summary fields missing across all records while RUNMAT_VALIDATE_REQUIRE_CONTACT_SUMMARY=true"
            )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("nonlinear analysis report schema checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
