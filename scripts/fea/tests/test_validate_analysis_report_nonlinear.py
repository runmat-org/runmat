import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.fea.governance.validate_analysis_report_nonlinear import main


TRANSIENT_ENERGY_BALANCE_ASSERTIONS = {
    "transient_initial_total_energy",
    "transient_final_total_energy",
    "transient_max_total_energy",
    "transient_energy_balance_growth_ratio",
    "transient_max_step_energy_jump_ratio",
}

PLASTIC_HARDENING_KNOWN_ANSWER_ASSERTIONS = {
    "plasticity_hardening_reference_known_monotonic_equivalent_plastic_strain_fraction",
    "plasticity_hardening_reference_known_active_element_coverage_ratio",
    "plasticity_hardening_reference_known_final_to_peak_equivalent_plastic_strain_ratio",
    "plasticity_hardening_reference_known_known_answer_coverage_ratio",
}

PLASTIC_HARDENING_COMPLEX_KNOWN_ANSWER_ASSERTIONS = {
    "plasticity_hardening_reference_complex_known_monotonic_equivalent_plastic_strain_fraction",
    "plasticity_hardening_reference_complex_known_active_element_coverage_ratio",
    "plasticity_hardening_reference_complex_known_final_to_peak_equivalent_plastic_strain_ratio",
    "plasticity_hardening_reference_complex_known_known_answer_coverage_ratio",
}

CONTACT_FRICTIONLESS_KNOWN_ANSWER_ASSERTIONS = {
    "contact_frictionless_known_pressure_gap_consistency_residual",
    "contact_frictionless_known_active_entity_coverage_ratio",
    "contact_frictionless_known_nonpenetration_gap_min",
    "contact_frictionless_known_friction_coefficient",
    "contact_frictionless_known_known_answer_coverage_ratio",
}

CONTACT_FRICTIONLESS_COMPLEX_KNOWN_ANSWER_ASSERTIONS = {
    "contact_frictionless_complex_known_pressure_gap_consistency_residual",
    "contact_frictionless_complex_known_active_entity_coverage_ratio",
    "contact_frictionless_complex_known_nonpenetration_gap_min",
    "contact_frictionless_complex_known_friction_coefficient",
    "contact_frictionless_complex_known_known_answer_coverage_ratio",
}


def _record(fixture_id: str, assertion_names: set[str]) -> dict:
    record = {
        "fixture_id": fixture_id,
        "threshold_assertions": [
            {"name": name, "observed": 1.0, "passed": True}
            for name in sorted(assertion_names)
        ],
        "gpu_speedup_ratio": 0.25,
        "gpu_solver_solve_ms": 1.5,
        "gpu_solver_backend": "runtime_tensor",
    }
    if fixture_id.startswith("electromagnetic_reference_"):
        record["gpu_solver_fallback_apply_count"] = 0.0
        record["electromagnetic_solve_quality"] = 1.0
        record["electromagnetic_enabled"] = True
        record["electromagnetic_energy_imbalance_ratio"] = 0.0
        record["electromagnetic_flux_divergence_ratio"] = 0.0
        record["electromagnetic_real_residual_norm"] = 0.0
        record["electromagnetic_imag_residual_norm"] = 0.0
        record["electromagnetic_source_region_energy_consistency_ratio"] = 1.0
        record["electromagnetic_source_realization_ratio"] = 1.0
        record["electromagnetic_source_region_coverage_ratio"] = 1.0
        record["electromagnetic_source_material_alignment_ratio"] = 1.0
        record["electromagnetic_source_localization_ratio"] = 1.0
        record["electromagnetic_source_interference_index"] = 0.0
        record["electromagnetic_boundary_condition_localization_ratio"] = 1.0
        record["electromagnetic_ground_anchor_effectiveness_ratio"] = 1.0
        record["electromagnetic_applied_current_a"] = 1.0
        record["electromagnetic_condition_number_estimate"] = 1.0
        record["electromagnetic_reference_frequency_hz"] = 1.0
        record["electromagnetic_assignment_coverage_ratio"] = 1.0
        record["electromagnetic_fallback_coefficient_ratio"] = 0.0
        record["electromagnetic_boundary_anchor_ratio"] = 1.0
        record["electromagnetic_conductivity_spread_ratio"] = 1.0
        record["electromagnetic_relative_permittivity_spread_ratio"] = 1.0
        record["electromagnetic_relative_permeability_spread_ratio"] = 1.0
        record["electromagnetic_material_heterogeneity_index"] = 1.0
        record["electromagnetic_region_coefficient_contrast_index"] = 1.0
        record["electromagnetic_boundary_energy_ratio"] = 0.5
        record["electromagnetic_boundary_penalty_conditioning_contribution"] = 0.5
        record["electromagnetic_source_overlap_ratio"] = 0.0
        record["electromagnetic_insulation_leakage_ratio"] = 1.0
    return record


def _error_record(fixture_id: str, run_error_code: str) -> dict:
    return {
        "fixture_id": fixture_id,
        "validate_ok": True,
        "validate_error_code": None,
        "run_ok": False,
        "run_error_code": run_error_code,
        "threshold_assertions": [],
        "failures": [],
    }


class ValidateAnalysisReportNonlinearTests(unittest.TestCase):
    def _base_records(self) -> list[dict]:
        return [
            _record(
                "cantilever_gpu_provider",
                {
                    "structural_normalized_residual_norm",
                    "structural_total_strain_energy",
                    "structural_work_energy_ratio",
                    "structural_work_energy_residual_ratio",
                    "structural_known_answer_coverage_ratio",
                    "structural_active_stiffness_edge_count",
                    "structural_recovery_element_count",
                    "structural_max_edge_displacement_jump",
                    "structural_von_mises_peak_pa",
                    "structural_stress_tensor_peak_pa",
                },
            ),
            _record(
                "cantilever_gpu_fallback",
                {
                    "structural_normalized_residual_norm",
                    "structural_total_strain_energy",
                    "structural_work_energy_ratio",
                    "structural_work_energy_residual_ratio",
                    "structural_known_answer_coverage_ratio",
                    "structural_active_stiffness_edge_count",
                    "structural_recovery_element_count",
                    "structural_max_edge_displacement_jump",
                    "structural_von_mises_peak_pa",
                    "structural_stress_tensor_peak_pa",
                },
            ),
            _record(
                "cantilever_load_sweep_gpu_provider",
                {
                    "structural_normalized_residual_norm",
                    "structural_total_strain_energy",
                    "structural_work_energy_ratio",
                    "structural_work_energy_residual_ratio",
                    "structural_known_answer_coverage_ratio",
                    "structural_active_stiffness_edge_count",
                    "structural_recovery_element_count",
                    "structural_max_edge_displacement_jump",
                    "structural_von_mises_peak_pa",
                    "structural_stress_tensor_peak_pa",
                },
            ),
            _record(
                "cantilever_large_load_sweep_gpu_provider",
                {
                    "structural_normalized_residual_norm",
                    "structural_total_strain_energy",
                    "structural_work_energy_ratio",
                    "structural_work_energy_residual_ratio",
                    "structural_known_answer_coverage_ratio",
                    "structural_active_stiffness_edge_count",
                    "structural_recovery_element_count",
                    "structural_max_edge_displacement_jump",
                    "structural_von_mises_peak_pa",
                    "structural_stress_tensor_peak_pa",
                },
            ),
            _record(
                "nonlinear_assembly_gpu_provider",
                {
                    "nonlinear_total_increments",
                    "nonlinear_failed_increments",
                    "nonlinear_iteration_spike_count",
                    "nonlinear_converged_increments",
                    "nonlinear_line_search_backtracks",
                    "nonlinear_max_increment_norm",
                },
            ),
            _record(
                "nonlinear_assembly_stress_gpu_provider",
                {
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
            ),
            _record(
                "nonlinear_softening_benchmark_gpu_provider",
                {
                    "nonlinear_softening_total_increments",
                    "nonlinear_softening_spike_count",
                    "nonlinear_softening_backtrack_bursts",
                    "nonlinear_softening_failed_increments",
                    "nonlinear_softening_stall_count",
                },
            ),
            _record(
                "nonlinear_load_path_mix_gpu_provider",
                {
                    "nonlinear_path_mix_total_increments",
                    "nonlinear_path_mix_max_backtracks_per_increment",
                    "nonlinear_path_mix_backtrack_bursts",
                    "nonlinear_path_mix_effective_modulus_scale",
                    "nonlinear_path_mix_material_spread_ratio",
                    "nonlinear_path_mix_spike_count",
                    "thermo_nonlinear_severity",
                    "thermo_nonlinear_field_clamp_ratio",
                    "thermo_nonlinear_time_scale_mean",
                    "electro_nonlinear_joule_heating_scale",
                    "electro_nonlinear_conductivity_spread_ratio",
                    "electro_nonlinear_severity_peak",
                    "electro_nonlinear_severity_mean",
                    "electro_nonlinear_temporal_variation",
                    "electro_nonlinear_time_scale_mean",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
                "electro_thermal_coupling_enabled": True,
                "electro_thermal_coupling_fingerprint": 1,
                "electro_joule_heating_scale": 1.0,
                "electro_conductivity_spread_ratio": 1.0,
                "electro_transient_severity": 0.1,
                "electro_nonlinear_severity": 0.1,
            },
            _record(
                "nonlinear_plasticity_benchmark_gpu_provider",
                {
                    "plasticity_nonlinear_severity_peak",
                    "plasticity_nonlinear_severity_mean",
                    "plasticity_nonlinear_load_amplification_ratio",
                    "plasticity_nonlinear_load_realization_ratio",
                    "plasticity_nonlinear_state_active_element_count",
                    "plasticity_nonlinear_state_max_equivalent_plastic_strain",
                },
            )
            | {
                "electro_thermal_coupling_enabled": True,
                "electro_thermal_coupling_fingerprint": 1,
                "electro_joule_heating_scale": 1.0,
                "electro_conductivity_spread_ratio": 1.0,
                "electro_transient_severity": 0.1,
                "electro_nonlinear_severity": 0.1,
                "plastic_nonlinear_severity": 0.1,
            },
            _record(
                "nonlinear_contact_benchmark_gpu_provider",
                {
                    "contact_nonlinear_severity_peak",
                    "contact_nonlinear_severity_mean",
                    "contact_nonlinear_load_amplification_ratio",
                    "contact_nonlinear_load_realization_ratio",
                    "contact_nonlinear_state_active_entity_count",
                    "contact_nonlinear_state_max_contact_pressure",
                    "contact_nonlinear_state_min_contact_gap",
                },
            )
            | {"contact_nonlinear_severity": 0.1},
            _record(
                "nonlinear_contact_frictionless_reference_gpu_provider",
                {
                    "contact_frictionless_severity_peak",
                    "contact_frictionless_severity_mean",
                    "contact_frictionless_load_amplification_ratio",
                    "contact_frictionless_load_realization_ratio",
                    "contact_frictionless_state_active_entity_count",
                    "contact_frictionless_state_max_contact_pressure",
                    "contact_frictionless_state_min_contact_gap",
                }
                | CONTACT_FRICTIONLESS_KNOWN_ANSWER_ASSERTIONS,
            )
            | {"contact_nonlinear_severity": 0.1},
            _record(
                "nonlinear_contact_frictionless_reference_complex_gpu_provider",
                {
                    "contact_frictionless_complex_severity_peak",
                    "contact_frictionless_complex_severity_mean",
                    "contact_frictionless_complex_load_amplification_ratio",
                    "contact_frictionless_complex_load_realization_ratio",
                    "contact_frictionless_complex_state_active_entity_count",
                    "contact_frictionless_complex_state_max_contact_pressure",
                    "contact_frictionless_complex_state_min_contact_gap",
                }
                | CONTACT_FRICTIONLESS_COMPLEX_KNOWN_ANSWER_ASSERTIONS,
            )
            | {"contact_nonlinear_severity": 0.1},
            _record(
                "nonlinear_plastic_hardening_reference_gpu_provider",
                {
                    "plasticity_hardening_reference_severity_peak",
                    "plasticity_hardening_reference_severity_mean",
                    "plasticity_hardening_reference_load_amplification_ratio",
                    "plasticity_hardening_reference_load_realization_ratio",
                    "plasticity_hardening_reference_state_active_element_count",
                    "plasticity_hardening_reference_state_max_equivalent_plastic_strain",
                }
                | PLASTIC_HARDENING_KNOWN_ANSWER_ASSERTIONS,
            ),
            _record(
                "nonlinear_plastic_hardening_reference_complex_gpu_provider",
                {
                    "plasticity_hardening_reference_complex_severity_peak",
                    "plasticity_hardening_reference_complex_severity_mean",
                    "plasticity_hardening_reference_complex_load_realization_ratio",
                    "plasticity_hardening_reference_complex_load_amplification_ratio",
                    "plasticity_hardening_reference_complex_state_active_element_count",
                    "plasticity_hardening_reference_complex_state_max_equivalent_plastic_strain",
                }
                | PLASTIC_HARDENING_COMPLEX_KNOWN_ANSWER_ASSERTIONS,
            ),
            _record(
                "thermo_mech_kickoff_gpu_provider",
                {
                    "thermo_mech_thermal_strain_scale",
                    "thermo_mech_thermal_load_scale",
                    "thermo_mech_effective_modulus_scale",
                    "thermo_mech_material_spread_ratio",
                    "thermo_mech_assignment_heterogeneity_index",
                    "thermo_mech_transient_severity",
                    "thermo_mech_transient_time_scale_mean",
                    "thermo_mech_constitutive_residual_ratio",
                    "thermo_mech_thermal_strain_energy_density_mean",
                    "thermo_mech_consistency_coverage_ratio",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
            },
            _record(
                "thermo_gradient_benign_gpu_provider",
                {
                    "thermo_gradient_benign_spread_ratio",
                    "thermo_gradient_benign_heterogeneity",
                    "thermo_gradient_benign_temporal_variation",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
            },
            _record(
                "thermo_gradient_pathological_gpu_provider",
                {
                    "thermo_gradient_pathological_spread_ratio",
                    "thermo_gradient_pathological_heterogeneity",
                    "thermo_gradient_pathological_temporal_variation",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
            },
            _record(
                "thermo_ramp_smooth_gpu_provider",
                {
                    "thermo_ramp_smooth_temporal_variation",
                    "thermo_ramp_smooth_spatial_gradient_index",
                    "thermo_ramp_smooth_spatial_coverage_ratio",
                    "thermo_ramp_smooth_field_extrapolation_ratio",
                    "thermo_ramp_smooth_field_clamp_ratio",
                    "thermo_ramp_smooth_constitutive_temperature_factor",
                    "thermo_ramp_smooth_effective_modulus_scale",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
            },
            _record(
                "thermo_ramp_smooth_field_artifact_gpu_provider",
                {
                    "thermo_ramp_smooth_temporal_variation",
                    "thermo_ramp_smooth_spatial_gradient_index",
                    "thermo_ramp_smooth_spatial_coverage_ratio",
                    "thermo_ramp_smooth_field_extrapolation_ratio",
                    "thermo_ramp_smooth_field_clamp_ratio",
                    "thermo_ramp_smooth_constitutive_temperature_factor",
                    "thermo_ramp_smooth_effective_modulus_scale",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
                "thermo_field_artifact_id": "artifact_1",
                "thermo_field_artifact_approved": True,
                "thermo_field_artifact_age_days": 0.1,
                "thermo_field_artifact_provenance_valid": True,
            },
            _record(
                "thermo_shock_oscillatory_gpu_provider",
                {
                    "thermo_shock_oscillatory_temporal_variation",
                    "thermo_shock_oscillatory_spatial_gradient_index",
                    "thermo_shock_oscillatory_spatial_coverage_ratio",
                    "thermo_shock_oscillatory_field_extrapolation_ratio",
                    "thermo_shock_oscillatory_field_clamp_ratio",
                    "thermo_shock_constitutive_temperature_factor",
                    "thermo_shock_effective_modulus_scale",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
            },
            _record(
                "thermo_shock_oscillatory_field_artifact_gpu_provider",
                {
                    "thermo_shock_oscillatory_temporal_variation",
                    "thermo_shock_oscillatory_spatial_gradient_index",
                    "thermo_shock_oscillatory_spatial_coverage_ratio",
                    "thermo_shock_oscillatory_field_extrapolation_ratio",
                    "thermo_shock_oscillatory_field_clamp_ratio",
                    "thermo_shock_constitutive_temperature_factor",
                    "thermo_shock_effective_modulus_scale",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "thermo_coupling_enabled": True,
                "thermo_coupling_fingerprint": 1,
                "thermo_constitutive_temperature_factor": 1.0,
                "thermo_effective_modulus_scale": 1.0,
                "thermo_constitutive_material_spread_ratio": 1.0,
                "thermo_assignment_heterogeneity_index": 1.0,
                "thermo_region_delta_count": 1.0,
                "thermo_spatial_coverage_ratio": 1.0,
                "thermo_field_extrapolation_ratio": 0.0,
                "thermo_transient_severity": 0.1,
                "thermo_nonlinear_severity": 0.1,
                "thermo_field_artifact_id": "artifact_2",
                "thermo_field_artifact_approved": True,
                "thermo_field_artifact_age_days": 0.1,
                "thermo_field_artifact_provenance_valid": True,
            },
            *[
                _record(
                    fixture_id,
                    {
                        "thermal_standalone_max_residual_norm",
                        "thermal_standalone_min_temperature_k",
                        "thermal_standalone_max_temperature_k",
                        "thermal_standalone_heat_balance_residual_ratio",
                        "thermal_standalone_conductivity_spread_ratio",
                        "thermal_standalone_heat_capacity_spread_ratio",
                        "thermal_standalone_spatial_gradient_index",
                        "thermal_standalone_monotonic_response_fraction",
                        "thermal_standalone_response_realization_ratio",
                        "thermal_standalone_slab_linear_profile_rms_ratio",
                        "thermal_standalone_slab_monotonic_edge_fraction",
                        "thermal_standalone_lumped_response_error_ratio",
                        "thermal_standalone_source_response_sign_alignment",
                        "thermal_standalone_source_coverage_ratio",
                        "thermal_standalone_boundary_coverage_ratio",
                        "thermal_standalone_prescribed_temperature_count",
                        "thermal_standalone_heat_flux_boundary_count",
                        "thermal_standalone_convection_boundary_count",
                    },
                )
                for fixture_id in (
                    "thermal_standalone_ramp_cpu",
                    "thermal_standalone_ramp_gpu_provider",
                    "thermal_standalone_ramp_gpu_fallback",
                )
            ],
            _error_record(
                "thermal_standalone_ramp_invalid_material",
                "RM.FEA.RUN_THERMAL.INVALID_THERMAL_MATERIAL",
            ),
            _error_record(
                "thermal_standalone_ramp_invalid_source",
                "RM.FEA.RUN_THERMAL.INVALID_THERMAL_SOURCE",
            ),
            _error_record(
                "thermal_standalone_ramp_invalid_boundary",
                "RM.FEA.RUN_THERMAL.INVALID_THERMAL_BOUNDARY",
            ),
            _record(
                "electro_thermal_joule_benign_gpu_provider",
                {
                    "electro_thermal_benign_joule_heating_scale",
                    "electro_thermal_benign_conductivity_spread_ratio",
                    "electro_thermal_benign_transient_severity_peak",
                    "electro_thermal_benign_temporal_variation",
                    "electro_thermal_benign_time_scale_mean",
                    "electro_thermal_benign_potential_residual_norm",
                    "electro_thermal_benign_current_balance_residual",
                    "electro_thermal_benign_potential_span_v",
                    "electro_thermal_benign_ohms_law_residual_ratio",
                    "electro_thermal_benign_joule_heat_balance_ratio",
                    "electro_thermal_benign_potential_monotonic_edge_fraction",
                    "electro_thermal_benign_resistor_known_answer_coverage_ratio",
                    "electro_thermal_benign_joule_heat_realization_ratio",
                    "electro_thermal_benign_joule_source_coverage_ratio",
                    "electro_thermal_benign_thermal_temperature_source_alignment",
                    "electro_thermal_benign_thermal_source_residual_ratio",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "electro_thermal_coupling_enabled": True,
                "electro_thermal_coupling_fingerprint": 1,
                "electro_joule_heating_scale": 1.0,
                "electro_conductivity_spread_ratio": 1.0,
                "electro_transient_severity": 0.1,
                "electro_nonlinear_severity": 0.1,
            },
            _record(
                "electro_thermal_joule_pathological_gpu_provider",
                {
                    "electro_thermal_pathological_joule_heating_scale",
                    "electro_thermal_pathological_conductivity_spread_ratio",
                    "electro_thermal_pathological_transient_severity_peak",
                    "electro_thermal_pathological_temporal_variation",
                    "electro_thermal_pathological_time_scale_mean",
                    "electro_thermal_pathological_potential_residual_norm",
                    "electro_thermal_pathological_current_balance_residual",
                    "electro_thermal_pathological_potential_span_v",
                    "electro_thermal_pathological_ohms_law_residual_ratio",
                    "electro_thermal_pathological_joule_heat_balance_ratio",
                    "electro_thermal_pathological_potential_monotonic_edge_fraction",
                    "electro_thermal_pathological_resistor_known_answer_coverage_ratio",
                    "electro_thermal_pathological_joule_heat_realization_ratio",
                    "electro_thermal_pathological_joule_source_coverage_ratio",
                    "electro_thermal_pathological_thermal_temperature_source_alignment",
                    "electro_thermal_pathological_thermal_source_residual_ratio",
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                },
            )
            | {
                "electro_thermal_coupling_enabled": True,
                "electro_thermal_coupling_fingerprint": 1,
                "electro_joule_heating_scale": 1.0,
                "electro_conductivity_spread_ratio": 1.0,
                "electro_transient_severity": 0.1,
                "electro_nonlinear_severity": 0.1,
            },
            _error_record(
                "electro_thermal_invalid_voltage",
                "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
            ),
            _error_record(
                "electro_thermal_invalid_conductivity_scale",
                "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
            ),
            _error_record(
                "electro_thermal_unmapped_region",
                "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
            ),
            _error_record(
                "cht_coupled_invalid_interface_mapping",
                "RM.FEA.RUN_CHT.INVALID_INTERFACE_MAPPING",
            ),
            _error_record(
                "fsi_coupled_invalid_interface_mapping",
                "RM.FEA.RUN_FSI.INVALID_INTERFACE_MAPPING",
            ),
            _record(
                "electromagnetic_reference_homogeneous_gpu_provider",
                {
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
                    "em_homogeneous_flux_phasor_coherence_ratio",
                    "em_homogeneous_flux_divergence_ratio",
                    "em_homogeneous_energy_imbalance_ratio",
                    "em_homogeneous_boundary_energy_ratio",
                    "em_homogeneous_source_realization_ratio",
                    "em_homogeneous_source_material_alignment_ratio",
                    "em_homogeneous_source_region_coverage_ratio",
                    "em_homogeneous_boundary_anchor_ratio",
                },
            )
            | {
                "electromagnetic_applied_current_a": 120.0,
                "electromagnetic_source_region_energy_consistency_ratio": 1.0,
                "electromagnetic_source_localization_ratio": 1.0,
                "electromagnetic_boundary_condition_localization_ratio": 1.0,
                "electromagnetic_ground_anchor_effectiveness_ratio": 1.0,
                "electromagnetic_source_interference_index": 0.001,
                "electromagnetic_source_realization_ratio": 1.0,
                "electromagnetic_source_region_coverage_ratio": 1.0,
                "electromagnetic_source_material_alignment_ratio": 1.0,
                "electromagnetic_assignment_coverage_ratio": 1.0,
                "electromagnetic_fallback_coefficient_ratio": 0.0,
                "electromagnetic_boundary_anchor_ratio": 1.0,
                "electromagnetic_conductivity_spread_ratio": 1.0,
                "electromagnetic_relative_permittivity_spread_ratio": 1.0,
                "electromagnetic_relative_permeability_spread_ratio": 1.0,
                "electromagnetic_material_heterogeneity_index": 0.0,
                "electromagnetic_region_coefficient_contrast_index": 0.0,
                "electromagnetic_boundary_energy_ratio": 0.463,
                "electromagnetic_boundary_penalty_conditioning_contribution": 0.414,
                "electromagnetic_source_overlap_ratio": 0.0,
                "electromagnetic_insulation_leakage_ratio": 0.0,
                "electromagnetic_solve_quality": 0.99996,
                "electromagnetic_enabled": True,
                "electromagnetic_energy_imbalance_ratio": 4.2e-5,
                "electromagnetic_flux_divergence_ratio": 0.23,
                "electromagnetic_real_residual_norm": 1.0e-10,
                "electromagnetic_imag_residual_norm": 2.0e-8,
                "electromagnetic_condition_number_estimate": 1.83,
                "electromagnetic_reference_frequency_hz": 60.0,
                "electromagnetic_sweep_count": 5.0,
                "electromagnetic_resonance_peak_frequency_hz": 60.0,
                "electromagnetic_resonance_peak_flux_density": 1.0,
                "electromagnetic_resonance_bandwidth_hz": 20.0,
                "electromagnetic_resonance_quality_factor": 3.0,
                "electromagnetic_resonance_flux_gain": 1.2,
            },
            _record(
                "electromagnetic_reference_heterogeneous_gpu_provider",
                {
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
                    "em_heterogeneous_flux_phasor_coherence_ratio",
                    "em_heterogeneous_flux_divergence_ratio",
                    "em_heterogeneous_energy_imbalance_ratio",
                    "em_heterogeneous_boundary_anchor_ratio",
                },
            )
            | {
                "electromagnetic_applied_current_a": 250.0,
                "electromagnetic_source_region_energy_consistency_ratio": 0.52,
                "electromagnetic_source_localization_ratio": 1.0,
                "electromagnetic_boundary_condition_localization_ratio": 1.0,
                "electromagnetic_ground_anchor_effectiveness_ratio": 1.0,
                "electromagnetic_source_interference_index": 0.0,
                "electromagnetic_source_realization_ratio": 0.667,
                "electromagnetic_source_region_coverage_ratio": 1.0,
                "electromagnetic_source_material_alignment_ratio": 1.0,
                "electromagnetic_assignment_coverage_ratio": 1.0,
                "electromagnetic_fallback_coefficient_ratio": 0.0,
                "electromagnetic_boundary_anchor_ratio": 0.75,
                "electromagnetic_conductivity_spread_ratio": 2.56e8,
                "electromagnetic_relative_permittivity_spread_ratio": 6.58,
                "electromagnetic_relative_permeability_spread_ratio": 78.4,
                "electromagnetic_material_heterogeneity_index": 1.1,
                "electromagnetic_region_coefficient_contrast_index": 8.54,
                "electromagnetic_boundary_energy_ratio": 0.839,
                "electromagnetic_boundary_penalty_conditioning_contribution": 0.443,
                "electromagnetic_source_overlap_ratio": 0.0,
                "electromagnetic_insulation_leakage_ratio": 0.515,
                "electromagnetic_solve_quality": 0.772,
                "electromagnetic_enabled": True,
                "electromagnetic_energy_imbalance_ratio": 0.253,
                "electromagnetic_flux_divergence_ratio": 0.143,
                "electromagnetic_real_residual_norm": 1.0e-18,
                "electromagnetic_imag_residual_norm": 1.0e-15,
                "electromagnetic_condition_number_estimate": 3.36,
                "electromagnetic_reference_frequency_hz": 75.0,
                "electromagnetic_sweep_count": 5.0,
                "electromagnetic_resonance_peak_frequency_hz": 75.0,
                "electromagnetic_resonance_peak_flux_density": 1.3,
                "electromagnetic_resonance_bandwidth_hz": 24.0,
                "electromagnetic_resonance_quality_factor": 3.1,
                "electromagnetic_resonance_flux_gain": 1.25,
            },
            _record(
                "electromagnetic_reference_boundary_penalty_stress_gpu_provider",
                {
                    "em_boundary_penalty_anchor_ratio",
                    "em_boundary_penalty_conditioning_contribution",
                    "em_boundary_penalty_real_residual_norm",
                    "em_boundary_penalty_imag_residual_norm",
                },
            ),
            _record(
                "electromagnetic_reference_multi_region_phased_source_gpu_provider",
                {
                    "em_phased_source_energy_consistency_ratio",
                    "em_phased_source_interference_index",
                    "em_phased_source_overlap_ratio",
                    "em_phased_source_region_coverage_ratio",
                },
            ),
            _record(
                "electromagnetic_reference_sparse_assignments_gpu_provider",
                {
                    "em_sparse_assignment_coverage_ratio",
                    "em_sparse_fallback_coefficient_ratio",
                    "em_sparse_source_realization_ratio",
                    "em_sparse_source_region_coverage_ratio",
                    "em_sparse_source_material_alignment_ratio",
                    "em_sparse_boundary_anchor_ratio",
                    "em_sparse_energy_imbalance_ratio",
                },
            ),
            _record(
                "electromagnetic_reference_fallback_heavy_gpu_provider",
                {
                    "em_fallback_heavy_assignment_coverage_ratio",
                    "em_fallback_heavy_fallback_coefficient_ratio",
                    "em_fallback_heavy_source_realization_ratio",
                    "em_fallback_heavy_source_region_coverage_ratio",
                    "em_fallback_heavy_source_material_alignment_ratio",
                    "em_fallback_heavy_boundary_anchor_ratio",
                    "em_fallback_heavy_energy_imbalance_ratio",
                },
            ),
            _record(
                "electromagnetic_reference_overlap_interference_gpu_provider",
                {
                    "em_overlap_source_interference_index",
                    "em_overlap_source_region_coverage_ratio",
                    "em_overlap_source_overlap_ratio",
                    "em_overlap_source_material_alignment_ratio",
                },
            ),
            _record(
                "electromagnetic_reference_boundary_kernel_gpu_provider",
                {
                    "em_boundary_kernel_boundary_localization_ratio",
                    "em_boundary_kernel_ground_anchor_effectiveness_ratio",
                    "em_boundary_kernel_insulation_leakage_ratio",
                },
            ),
            _record(
                "acoustic_harmonic_gpu_provider",
                {
                    "acoustic_normalized_residual_norm",
                    "acoustic_drive_frequency_hz",
                    "acoustic_peak_pressure_pa",
                    "acoustic_domain_node_count",
                    "acoustic_domain_edge_count",
                    "acoustic_domain_active_dimension_count",
                    "acoustic_boundary_node_count",
                    "acoustic_material_coverage_ratio",
                    "acoustic_boundary_coverage_ratio",
                    "acoustic_radiation_boundary_count",
                    "acoustic_impedance_boundary_count",
                    "acoustic_frequency_response_sweep_count",
                    "acoustic_frequency_response_coverage_ratio",
                    "acoustic_sweep_bandwidth_hz",
                    "acoustic_sweep_peak_pressure_pa",
                    "acoustic_sweep_max_residual_norm",
                    "acoustic_tube_mode_alignment_error_ratio",
                    "acoustic_tube_pressure_variation_ratio",
                    "acoustic_cavity_mode_spacing_ratio",
                    "acoustic_cavity_reference_mode_count",
                    "acoustic_known_answer_coverage_ratio",
                },
            ),
            _record(
                "acoustic_harmonic_cpu",
                {
                    "acoustic_normalized_residual_norm",
                    "acoustic_drive_frequency_hz",
                    "acoustic_peak_pressure_pa",
                    "acoustic_domain_node_count",
                    "acoustic_domain_edge_count",
                    "acoustic_domain_active_dimension_count",
                    "acoustic_boundary_node_count",
                    "acoustic_material_coverage_ratio",
                    "acoustic_boundary_coverage_ratio",
                    "acoustic_radiation_boundary_count",
                    "acoustic_impedance_boundary_count",
                    "acoustic_frequency_response_sweep_count",
                    "acoustic_frequency_response_coverage_ratio",
                    "acoustic_sweep_bandwidth_hz",
                    "acoustic_sweep_peak_pressure_pa",
                    "acoustic_sweep_max_residual_norm",
                    "acoustic_tube_mode_alignment_error_ratio",
                    "acoustic_tube_pressure_variation_ratio",
                    "acoustic_cavity_mode_spacing_ratio",
                    "acoustic_cavity_reference_mode_count",
                    "acoustic_known_answer_coverage_ratio",
                },
            ),
            _record(
                "acoustic_harmonic_gpu_fallback",
                {
                    "acoustic_normalized_residual_norm",
                    "acoustic_drive_frequency_hz",
                    "acoustic_peak_pressure_pa",
                    "acoustic_domain_node_count",
                    "acoustic_domain_edge_count",
                    "acoustic_domain_active_dimension_count",
                    "acoustic_boundary_node_count",
                    "acoustic_material_coverage_ratio",
                    "acoustic_boundary_coverage_ratio",
                    "acoustic_radiation_boundary_count",
                    "acoustic_impedance_boundary_count",
                    "acoustic_frequency_response_sweep_count",
                    "acoustic_frequency_response_coverage_ratio",
                    "acoustic_sweep_bandwidth_hz",
                    "acoustic_sweep_peak_pressure_pa",
                    "acoustic_sweep_max_residual_norm",
                    "acoustic_tube_mode_alignment_error_ratio",
                    "acoustic_tube_pressure_variation_ratio",
                    "acoustic_cavity_mode_spacing_ratio",
                    "acoustic_cavity_reference_mode_count",
                    "acoustic_known_answer_coverage_ratio",
                },
            ),
            _error_record(
                "acoustic_harmonic_missing_source",
                "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_SOURCE",
            ),
            _error_record(
                "acoustic_harmonic_missing_boundary",
                "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_BOUNDARY",
            ),
            _record(
                "modal_large_cpu",
                {
                    "modal_max_m_orthogonality_offdiag",
                    "modal_min_relative_frequency_separation",
                },
            ),
            _record(
                "modal_large_cpu_stress16",
                {
                    "modal_max_m_orthogonality_offdiag",
                    "modal_min_relative_frequency_separation",
                },
            ),
            _record(
                "modal_large_gpu_fallback",
                {
                    "modal_max_m_orthogonality_offdiag",
                    "modal_min_relative_frequency_separation",
                },
            ),
            _record(
                "modal_large_gpu_provider",
                {
                    "modal_max_m_orthogonality_offdiag",
                    "modal_min_relative_frequency_separation",
                },
            ),
            _record(
                "modal_large_gpu_provider_stress16",
                {
                    "modal_max_m_orthogonality_offdiag",
                    "modal_min_relative_frequency_separation",
                },
            ),
            _record(
                "transient_long_cpu",
                {
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                }
                | TRANSIENT_ENERGY_BALANCE_ASSERTIONS,
            ),
            _record(
                "transient_long_gpu_fallback",
                {
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                }
                | TRANSIENT_ENERGY_BALANCE_ASSERTIONS,
            ),
            _record(
                "transient_long_gpu_provider",
                {
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                    "transient_adapt_scale_min",
                    "transient_adapt_scale_max",
                    "transient_adapt_scale_mean",
                    "transient_adapt_decrease_steps",
                    "transient_physics_jump_ratio",
                    "transient_physics_nonfinite_count",
                }
                | TRANSIENT_ENERGY_BALANCE_ASSERTIONS,
            ),
            _record(
                "transient_shock_cpu",
                {
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                }
                | TRANSIENT_ENERGY_BALANCE_ASSERTIONS,
            ),
            _record(
                "transient_shock_gpu_provider",
                {
                    "transient_max_residual_norm",
                    "transient_max_energy_growth_ratio",
                    "transient_prepared_cache_hit_ratio",
                    "transient_prepared_cache_misses",
                    "transient_shock_physics_jump_ratio",
                    "transient_shock_physics_nonfinite_count",
                }
                | TRANSIENT_ENERGY_BALANCE_ASSERTIONS,
            ),
            _record(
                "cfd_steady_gpu_provider",
                {
                    "cfd_reference_density_kg_per_m3",
                    "cfd_dynamic_viscosity_pa_s",
                    "cfd_inlet_velocity_m_per_s",
                    "cfd_turbulence_intensity",
                    "cfd_reynolds_number",
                    "cfd_profile_point_count",
                    "cfd_max_momentum_residual",
                    "cfd_max_continuity_residual",
                    "cfd_mass_balance_residual",
                    "cfd_pressure_drop_pa",
                    "cfd_pressure_drop_balance_ratio",
                    "cfd_mass_flux_uniformity_ratio",
                    "cfd_pressure_monotonic_cell_fraction",
                    "cfd_known_answer_coverage_ratio",
                    "cfd_control_volume_count",
                    "cfd_inlet_boundary_count",
                    "cfd_outlet_boundary_count",
                    "cfd_wall_boundary_count",
                    "cfd_boundary_coverage_ratio",
                    "cfd_wall_boundary_coverage_ratio",
                },
            ),
            _record(
                "cfd_steady_cpu",
                {
                    "cfd_reference_density_kg_per_m3",
                    "cfd_dynamic_viscosity_pa_s",
                    "cfd_inlet_velocity_m_per_s",
                    "cfd_turbulence_intensity",
                    "cfd_reynolds_number",
                    "cfd_profile_point_count",
                    "cfd_max_momentum_residual",
                    "cfd_max_continuity_residual",
                    "cfd_mass_balance_residual",
                    "cfd_pressure_drop_pa",
                    "cfd_pressure_drop_balance_ratio",
                    "cfd_mass_flux_uniformity_ratio",
                    "cfd_pressure_monotonic_cell_fraction",
                    "cfd_known_answer_coverage_ratio",
                    "cfd_control_volume_count",
                    "cfd_inlet_boundary_count",
                    "cfd_outlet_boundary_count",
                    "cfd_wall_boundary_count",
                    "cfd_boundary_coverage_ratio",
                    "cfd_wall_boundary_coverage_ratio",
                },
            ),
            _record(
                "cfd_steady_gpu_fallback",
                {
                    "cfd_reference_density_kg_per_m3",
                    "cfd_dynamic_viscosity_pa_s",
                    "cfd_inlet_velocity_m_per_s",
                    "cfd_turbulence_intensity",
                    "cfd_reynolds_number",
                    "cfd_profile_point_count",
                    "cfd_max_momentum_residual",
                    "cfd_max_continuity_residual",
                    "cfd_mass_balance_residual",
                    "cfd_pressure_drop_pa",
                    "cfd_pressure_drop_balance_ratio",
                    "cfd_mass_flux_uniformity_ratio",
                    "cfd_pressure_monotonic_cell_fraction",
                    "cfd_known_answer_coverage_ratio",
                    "cfd_control_volume_count",
                    "cfd_inlet_boundary_count",
                    "cfd_outlet_boundary_count",
                    "cfd_wall_boundary_count",
                    "cfd_boundary_coverage_ratio",
                    "cfd_wall_boundary_coverage_ratio",
                },
            ),
            *[
                _record(
                    fixture_id,
                    {
                        "cfd_reference_density_kg_per_m3",
                        "cfd_dynamic_viscosity_pa_s",
                        "cfd_inlet_velocity_m_per_s",
                        "cfd_turbulence_intensity",
                        "cfd_reynolds_number",
                        "cfd_profile_point_count",
                        "cfd_max_momentum_residual",
                        "cfd_max_continuity_residual",
                        "cfd_mass_balance_residual",
                        "cfd_pressure_drop_pa",
                        "cfd_pressure_drop_balance_ratio",
                        "cfd_mass_flux_uniformity_ratio",
                        "cfd_pressure_monotonic_cell_fraction",
                        "cfd_known_answer_coverage_ratio",
                        "cfd_control_volume_count",
                        "cfd_inlet_boundary_count",
                        "cfd_outlet_boundary_count",
                        "cfd_wall_boundary_count",
                        "cfd_boundary_coverage_ratio",
                        "cfd_wall_boundary_coverage_ratio",
                        "cfd_transient_step_count",
                        "cfd_transient_scale_min",
                        "cfd_transient_scale_max",
                        "cfd_transient_scale_variation",
                    },
                )
                for fixture_id in (
                    "cfd_transient_cpu",
                    "cfd_transient_gpu_provider",
                    "cfd_transient_gpu_fallback",
                )
            ],
            _record(
                "cht_coupled_gpu_provider",
                {
                    "cht_reference_density_kg_per_m3",
                    "cht_dynamic_viscosity_pa_s",
                    "cht_inlet_velocity_m_per_s",
                    "cht_turbulence_intensity",
                    "cht_reynolds_number",
                    "cht_profile_point_count",
                    "cht_applied_temperature_delta_k",
                    "cht_step_count",
                    "cht_time_step_s",
                    "cht_max_momentum_residual",
                    "cht_max_continuity_residual",
                    "cht_max_thermal_residual",
                    "cht_interface_face_count",
                    "cht_max_temperature_jump_k",
                    "cht_max_energy_residual",
                    "cht_heat_flux_balance_ratio",
                    "cht_thermal_transport_residual_ratio",
                    "cht_interface_temperature_continuity_ratio",
                },
            ),
            _record(
                "cht_coupled_cpu",
                {
                    "cht_reference_density_kg_per_m3",
                    "cht_dynamic_viscosity_pa_s",
                    "cht_inlet_velocity_m_per_s",
                    "cht_turbulence_intensity",
                    "cht_reynolds_number",
                    "cht_profile_point_count",
                    "cht_applied_temperature_delta_k",
                    "cht_step_count",
                    "cht_time_step_s",
                    "cht_max_momentum_residual",
                    "cht_max_continuity_residual",
                    "cht_max_thermal_residual",
                    "cht_interface_face_count",
                    "cht_max_temperature_jump_k",
                    "cht_max_energy_residual",
                    "cht_heat_flux_balance_ratio",
                    "cht_thermal_transport_residual_ratio",
                    "cht_interface_temperature_continuity_ratio",
                },
            ),
            _record(
                "cht_coupled_gpu_fallback",
                {
                    "cht_reference_density_kg_per_m3",
                    "cht_dynamic_viscosity_pa_s",
                    "cht_inlet_velocity_m_per_s",
                    "cht_turbulence_intensity",
                    "cht_reynolds_number",
                    "cht_profile_point_count",
                    "cht_applied_temperature_delta_k",
                    "cht_step_count",
                    "cht_time_step_s",
                    "cht_max_momentum_residual",
                    "cht_max_continuity_residual",
                    "cht_max_thermal_residual",
                    "cht_interface_face_count",
                    "cht_max_temperature_jump_k",
                    "cht_max_energy_residual",
                    "cht_heat_flux_balance_ratio",
                    "cht_thermal_transport_residual_ratio",
                    "cht_interface_temperature_continuity_ratio",
                },
            ),
            _record(
                "fsi_coupled_gpu_provider",
                {
                    "fsi_reference_density_kg_per_m3",
                    "fsi_dynamic_viscosity_pa_s",
                    "fsi_inlet_velocity_m_per_s",
                    "fsi_turbulence_intensity",
                    "fsi_reynolds_number",
                    "fsi_profile_point_count",
                    "fsi_step_count",
                    "fsi_time_step_s",
                    "fsi_structural_step_count",
                    "fsi_cfd_profile_point_count",
                    "fsi_max_momentum_residual",
                    "fsi_max_continuity_residual",
                    "fsi_max_interface_residual",
                    "fsi_interface_node_count",
                    "fsi_force_balance_ratio",
                    "fsi_max_displacement_transfer_residual_m",
                },
            ),
            _record(
                "fsi_coupled_cpu",
                {
                    "fsi_reference_density_kg_per_m3",
                    "fsi_dynamic_viscosity_pa_s",
                    "fsi_inlet_velocity_m_per_s",
                    "fsi_turbulence_intensity",
                    "fsi_reynolds_number",
                    "fsi_profile_point_count",
                    "fsi_step_count",
                    "fsi_time_step_s",
                    "fsi_structural_step_count",
                    "fsi_cfd_profile_point_count",
                    "fsi_max_momentum_residual",
                    "fsi_max_continuity_residual",
                    "fsi_max_interface_residual",
                    "fsi_interface_node_count",
                    "fsi_force_balance_ratio",
                    "fsi_max_displacement_transfer_residual_m",
                },
            ),
            _record(
                "fsi_coupled_gpu_fallback",
                {
                    "fsi_reference_density_kg_per_m3",
                    "fsi_dynamic_viscosity_pa_s",
                    "fsi_inlet_velocity_m_per_s",
                    "fsi_turbulence_intensity",
                    "fsi_reynolds_number",
                    "fsi_profile_point_count",
                    "fsi_step_count",
                    "fsi_time_step_s",
                    "fsi_structural_step_count",
                    "fsi_cfd_profile_point_count",
                    "fsi_max_momentum_residual",
                    "fsi_max_continuity_residual",
                    "fsi_max_interface_residual",
                    "fsi_interface_node_count",
                    "fsi_force_balance_ratio",
                    "fsi_max_displacement_transfer_residual_m",
                },
            ),
        ]

    def _run_main_with_report(self, report_path: Path) -> int:
        import sys

        previous_argv = sys.argv[:]
        sys.argv = ["validate_analysis_report_nonlinear.py", str(report_path)]
        try:
            return main()
        finally:
            sys.argv = previous_argv

    def test_passes_with_cfd_cht_fsi_required_assertions_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": self._base_records()}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 0)

    def test_fails_when_fsi_required_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "fsi_coupled_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "fsi_structural_step_count"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_softening_failed_increments_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_softening_benchmark_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "nonlinear_softening_failed_increments"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_nonlinear_converged_increments_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_assembly_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "nonlinear_converged_increments"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_nonlinear_stress_converged_increments_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_assembly_stress_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "nonlinear_stress_converged_increments"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_nonlinear_path_mix_spike_count_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_load_path_mix_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "nonlinear_path_mix_spike_count"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_thermo_gradient_benign_temporal_variation_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "thermo_gradient_benign_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "thermo_gradient_benign_temporal_variation"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_thermo_ramp_field_clamp_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "thermo_ramp_smooth_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "thermo_ramp_smooth_field_clamp_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_thermo_ramp_constitutive_factor_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "thermo_ramp_smooth_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"]
                        != "thermo_ramp_smooth_constitutive_temperature_factor"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_thermo_shock_constitutive_factor_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "thermo_shock_oscillatory_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "thermo_shock_constitutive_temperature_factor"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_contact_complex_load_amplification_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "nonlinear_contact_frictionless_reference_complex_gpu_provider"
                ):
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"]
                        != "contact_frictionless_complex_load_amplification_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_contact_complex_load_realization_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "nonlinear_contact_frictionless_reference_complex_gpu_provider"
                ):
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "contact_frictionless_complex_load_realization_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_plastic_complex_load_realization_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "nonlinear_plastic_hardening_reference_complex_gpu_provider"
                ):
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"]
                        != "plasticity_hardening_reference_complex_load_realization_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_plastic_complex_load_amplification_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "nonlinear_plastic_hardening_reference_complex_gpu_provider"
                ):
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"]
                        != "plasticity_hardening_reference_complex_load_amplification_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_plastic_benchmark_load_realization_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_plasticity_benchmark_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "plasticity_nonlinear_load_realization_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_contact_benchmark_load_realization_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_contact_benchmark_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "contact_nonlinear_load_realization_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_contact_frictionless_reference_load_realization_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_contact_frictionless_reference_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "contact_frictionless_load_realization_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_plastic_reference_load_realization_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_plastic_hardening_reference_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "plasticity_hardening_reference_load_realization_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_plastic_known_answer_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_plastic_hardening_reference_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"]
                        != "plasticity_hardening_reference_known_monotonic_equivalent_plastic_strain_fraction"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_contact_known_answer_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "nonlinear_contact_frictionless_reference_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"]
                        != "contact_frictionless_known_pressure_gap_consistency_residual"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_cfd_cpu_momentum_residual_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "cfd_steady_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "cfd_max_momentum_residual"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_cht_cpu_step_count_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "cht_coupled_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "cht_step_count"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_fsi_cpu_cfd_profile_count_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "fsi_coupled_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "fsi_cfd_profile_point_count"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_acoustic_cpu_peak_pressure_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "acoustic_harmonic_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "acoustic_peak_pressure_pa"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_acoustic_invalid_source_error_code_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "acoustic_harmonic_missing_source":
                    record["run_error_code"] = None
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_thermal_invalid_source_error_code_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "thermal_standalone_ramp_invalid_source":
                    record["run_error_code"] = None
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_electro_invalid_voltage_error_code_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electro_thermal_invalid_voltage":
                    record["run_error_code"] = None
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_modal_cpu_orthogonality_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "modal_large_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "modal_max_m_orthogonality_offdiag"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_transient_long_cpu_residual_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "transient_long_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "transient_max_residual_norm"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_transient_energy_balance_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "transient_shock_cpu":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "transient_energy_balance_growth_ratio"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_transient_long_provider_adapt_mean_assertion_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "transient_long_gpu_provider":
                    record["threshold_assertions"] = [
                        item
                        for item in record["threshold_assertions"]
                        if item["name"] != "transient_adapt_scale_mean"
                    ]
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_required_performance_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "cfd_steady_gpu_provider":
                    record.pop("gpu_solver_solve_ms", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_expanded_performance_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electro_thermal_joule_benign_gpu_provider":
                    record.pop("gpu_solver_solve_ms", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_modal_performance_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "modal_large_gpu_provider":
                    record.pop("gpu_solver_solve_ms", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_fallback_apply_count_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_homogeneous_gpu_provider"
                ):
                    record.pop("gpu_solver_fallback_apply_count", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_applied_current_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_applied_current_a", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_source_energy_consistency_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_source_region_energy_consistency_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_source_localization_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_source_localization_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_boundary_condition_localization_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_boundary_condition_localization_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_ground_anchor_effectiveness_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_ground_anchor_effectiveness_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_source_interference_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_source_interference_index", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_source_fidelity_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_source_realization_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_core_assignment_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_assignment_coverage_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_constitutive_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_conductivity_spread_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_boundary_source_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_boundary_energy_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_solve_quality_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_solve_quality", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_enabled_flag_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record["electromagnetic_enabled"] = False
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_solve_quality_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_solve_quality", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_enabled_flag_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record["electromagnetic_enabled"] = False
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_balance_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_energy_imbalance_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_residual_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_real_residual_norm", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_source_fidelity_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_source_realization_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_core_assignment_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_assignment_coverage_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_constitutive_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_conductivity_spread_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_boundary_source_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_boundary_energy_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_source_localization_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_source_localization_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_boundary_condition_localization_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_boundary_condition_localization_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_non_core_em_applied_current_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if (
                    record["fixture_id"]
                    == "electromagnetic_reference_sparse_assignments_gpu_provider"
                ):
                    record.pop("electromagnetic_applied_current_a", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_residual_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_real_residual_norm", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)

    def test_fails_when_em_balance_field_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._base_records()
            for record in records:
                if record["fixture_id"] == "electromagnetic_reference_homogeneous_gpu_provider":
                    record.pop("electromagnetic_energy_imbalance_ratio", None)
                    break
            path = Path(tmp) / "analysis_benchmark_report.json"
            path.write_text(json.dumps({"records": records}))
            rc = self._run_main_with_report(path)
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
