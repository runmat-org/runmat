#!/usr/bin/env python3
import json
import math
import os
import sys
from pathlib import Path


TRANSIENT_ENERGY_BALANCE_REQUIRED_FIELDS = {
    "transient_initial_total_energy",
    "transient_final_total_energy",
    "transient_max_total_energy",
    "transient_energy_balance_growth_ratio",
    "transient_max_step_energy_jump_ratio",
}

PLASTIC_HARDENING_KNOWN_ANSWER_REQUIRED_FIELDS = {
    "plasticity_hardening_reference_known_monotonic_equivalent_plastic_strain_fraction",
    "plasticity_hardening_reference_known_active_element_coverage_ratio",
    "plasticity_hardening_reference_known_final_to_peak_equivalent_plastic_strain_ratio",
    "plasticity_hardening_reference_known_known_answer_coverage_ratio",
}

PLASTIC_HARDENING_COMPLEX_KNOWN_ANSWER_REQUIRED_FIELDS = {
    "plasticity_hardening_reference_complex_known_monotonic_equivalent_plastic_strain_fraction",
    "plasticity_hardening_reference_complex_known_active_element_coverage_ratio",
    "plasticity_hardening_reference_complex_known_final_to_peak_equivalent_plastic_strain_ratio",
    "plasticity_hardening_reference_complex_known_known_answer_coverage_ratio",
}

CONTACT_FRICTIONLESS_KNOWN_ANSWER_REQUIRED_FIELDS = {
    "contact_frictionless_known_pressure_gap_consistency_residual",
    "contact_frictionless_known_active_entity_coverage_ratio",
    "contact_frictionless_known_nonpenetration_gap_min",
    "contact_frictionless_known_friction_coefficient",
    "contact_frictionless_known_known_answer_coverage_ratio",
}

CONTACT_FRICTIONLESS_COMPLEX_KNOWN_ANSWER_REQUIRED_FIELDS = {
    "contact_frictionless_complex_known_pressure_gap_consistency_residual",
    "contact_frictionless_complex_known_active_entity_coverage_ratio",
    "contact_frictionless_complex_known_nonpenetration_gap_min",
    "contact_frictionless_complex_known_friction_coefficient",
    "contact_frictionless_complex_known_known_answer_coverage_ratio",
}


REQUIRED_FIXTURES = {
    "cantilever_gpu_provider": {
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
    "cantilever_gpu_fallback": {
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
    "cantilever_load_sweep_gpu_provider": {
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
    "cantilever_large_load_sweep_gpu_provider": {
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
    "nonlinear_softening_benchmark_gpu_provider": {
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
    "nonlinear_plasticity_benchmark_gpu_provider": {
        "plasticity_nonlinear_severity_peak",
        "plasticity_nonlinear_severity_mean",
        "plasticity_nonlinear_load_amplification_ratio",
        "plasticity_nonlinear_load_realization_ratio",
        "plasticity_nonlinear_state_topology_element_count",
        "plasticity_nonlinear_state_topology_active_recovery_edge_count",
        "plasticity_nonlinear_state_active_element_count",
        "plasticity_nonlinear_state_max_equivalent_plastic_strain",
    },
    "nonlinear_contact_benchmark_gpu_provider": {
        "contact_nonlinear_severity_peak",
        "contact_nonlinear_severity_mean",
        "contact_nonlinear_load_amplification_ratio",
        "contact_nonlinear_load_realization_ratio",
        "contact_nonlinear_state_topology_element_count",
        "contact_nonlinear_state_topology_active_recovery_edge_count",
        "contact_nonlinear_state_active_entity_count",
        "contact_nonlinear_state_max_contact_pressure",
        "contact_nonlinear_state_min_contact_gap",
    },
    "nonlinear_contact_frictionless_reference_gpu_provider": {
        "contact_frictionless_severity_peak",
        "contact_frictionless_severity_mean",
        "contact_frictionless_load_amplification_ratio",
        "contact_frictionless_load_realization_ratio",
        "contact_frictionless_state_topology_element_count",
        "contact_frictionless_state_topology_active_recovery_edge_count",
        "contact_frictionless_state_active_entity_count",
        "contact_frictionless_state_max_contact_pressure",
        "contact_frictionless_state_min_contact_gap",
    }
    | CONTACT_FRICTIONLESS_KNOWN_ANSWER_REQUIRED_FIELDS,
    "nonlinear_contact_frictionless_reference_complex_gpu_provider": {
        "contact_frictionless_complex_severity_peak",
        "contact_frictionless_complex_severity_mean",
        "contact_frictionless_complex_load_amplification_ratio",
        "contact_frictionless_complex_load_realization_ratio",
        "contact_frictionless_complex_state_topology_element_count",
        "contact_frictionless_complex_state_topology_active_recovery_edge_count",
        "contact_frictionless_complex_state_active_entity_count",
        "contact_frictionless_complex_state_max_contact_pressure",
        "contact_frictionless_complex_state_min_contact_gap",
    }
    | CONTACT_FRICTIONLESS_COMPLEX_KNOWN_ANSWER_REQUIRED_FIELDS,
    "nonlinear_plastic_hardening_reference_gpu_provider": {
        "plasticity_hardening_reference_severity_peak",
        "plasticity_hardening_reference_severity_mean",
        "plasticity_hardening_reference_load_amplification_ratio",
        "plasticity_hardening_reference_load_realization_ratio",
        "plasticity_hardening_reference_state_topology_element_count",
        "plasticity_hardening_reference_state_topology_active_recovery_edge_count",
        "plasticity_hardening_reference_state_active_element_count",
        "plasticity_hardening_reference_state_max_equivalent_plastic_strain",
    }
    | PLASTIC_HARDENING_KNOWN_ANSWER_REQUIRED_FIELDS,
    "nonlinear_plastic_hardening_reference_complex_gpu_provider": {
        "plasticity_hardening_reference_complex_severity_peak",
        "plasticity_hardening_reference_complex_severity_mean",
        "plasticity_hardening_reference_complex_load_realization_ratio",
        "plasticity_hardening_reference_complex_load_amplification_ratio",
        "plasticity_hardening_reference_complex_state_topology_element_count",
        "plasticity_hardening_reference_complex_state_topology_active_recovery_edge_count",
        "plasticity_hardening_reference_complex_state_active_element_count",
        "plasticity_hardening_reference_complex_state_max_equivalent_plastic_strain",
    }
    | PLASTIC_HARDENING_COMPLEX_KNOWN_ANSWER_REQUIRED_FIELDS,
    "thermo_mech_kickoff_gpu_provider": {
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
        "thermo_mech_temperature_field_node_count",
        "thermo_mech_strain_temperature_residual_ratio",
        "thermo_mech_strain_temperature_coverage_ratio",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "thermo_gradient_benign_gpu_provider": {
        "thermo_gradient_benign_spread_ratio",
        "thermo_gradient_benign_heterogeneity",
        "thermo_gradient_benign_temporal_variation",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "thermo_gradient_pathological_gpu_provider": {
        "thermo_gradient_pathological_spread_ratio",
        "thermo_gradient_pathological_heterogeneity",
        "thermo_gradient_pathological_temporal_variation",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "thermo_ramp_smooth_gpu_provider": {
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
    "thermo_ramp_smooth_field_artifact_gpu_provider": {
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
    "thermo_shock_oscillatory_gpu_provider": {
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
    "thermo_shock_oscillatory_field_artifact_gpu_provider": {
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
    "thermal_standalone_ramp_cpu": {
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
    "thermal_standalone_ramp_gpu_provider": {
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
    "thermal_standalone_ramp_gpu_fallback": {
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
    "electro_thermal_joule_benign_gpu_provider": {
        "electro_thermal_benign_joule_heating_scale",
        "electro_thermal_benign_conductivity_spread_ratio",
        "electro_thermal_benign_transient_severity_peak",
        "electro_thermal_benign_temporal_variation",
        "electro_thermal_benign_time_scale_mean",
        "electro_thermal_benign_conductive_node_count",
        "electro_thermal_benign_mapped_voltage_boundary_count",
        "electro_thermal_benign_mapped_current_source_count",
        "electro_thermal_benign_topology_component_count",
        "electro_thermal_benign_source_boundary_alignment_ratio",
        "electro_thermal_benign_domain_conductance_coverage_ratio",
        "electro_thermal_benign_material_region_coverage_ratio",
        "electro_thermal_benign_potential_residual_norm",
        "electro_thermal_benign_current_balance_residual",
        "electro_thermal_benign_potential_span_v",
        "electro_thermal_benign_conduction_edge_count",
        "electro_thermal_benign_topology_coverage_ratio",
        "electro_thermal_benign_ohms_law_residual_ratio",
        "electro_thermal_benign_joule_heat_balance_ratio",
        "electro_thermal_benign_potential_monotonic_edge_fraction",
        "electro_thermal_benign_conduction_graph_coverage_ratio",
        "electro_thermal_benign_joule_heat_realization_ratio",
        "electro_thermal_benign_joule_source_coverage_ratio",
        "electro_thermal_benign_thermal_temperature_source_alignment",
        "electro_thermal_benign_thermal_source_residual_ratio",
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
        "electro_thermal_pathological_conductive_node_count",
        "electro_thermal_pathological_mapped_voltage_boundary_count",
        "electro_thermal_pathological_mapped_current_source_count",
        "electro_thermal_pathological_topology_component_count",
        "electro_thermal_pathological_source_boundary_alignment_ratio",
        "electro_thermal_pathological_domain_conductance_coverage_ratio",
        "electro_thermal_pathological_material_region_coverage_ratio",
        "electro_thermal_pathological_potential_residual_norm",
        "electro_thermal_pathological_current_balance_residual",
        "electro_thermal_pathological_potential_span_v",
        "electro_thermal_pathological_conduction_edge_count",
        "electro_thermal_pathological_topology_coverage_ratio",
        "electro_thermal_pathological_ohms_law_residual_ratio",
        "electro_thermal_pathological_joule_heat_balance_ratio",
        "electro_thermal_pathological_potential_monotonic_edge_fraction",
        "electro_thermal_pathological_conduction_graph_coverage_ratio",
        "electro_thermal_pathological_joule_heat_realization_ratio",
        "electro_thermal_pathological_joule_source_coverage_ratio",
        "electro_thermal_pathological_thermal_temperature_source_alignment",
        "electro_thermal_pathological_thermal_source_residual_ratio",
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
    },
    "electromagnetic_reference_homogeneous_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
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
        "em_homogeneous_flux_phasor_coherence_ratio",
        "em_homogeneous_flux_divergence_ratio",
        "em_homogeneous_energy_imbalance_ratio",
        "em_homogeneous_boundary_energy_ratio",
        "em_homogeneous_edge_dof_count",
        "em_homogeneous_element_count",
        "em_homogeneous_oriented_edge_count",
        "em_homogeneous_gauge_anchor_count",
        "em_homogeneous_gauge_anchor_residual_ratio",
        "em_homogeneous_source_material_alignment_ratio",
        "em_homogeneous_source_region_coverage_ratio",
        "em_homogeneous_boundary_anchor_ratio",
        "em_homogeneous_known_material_residual_ratio",
        "em_homogeneous_known_source_energy_consistency_residual_ratio",
        "em_homogeneous_known_gauge_anchor_residual_ratio",
        "em_homogeneous_known_flux_divergence_ratio",
        "em_homogeneous_known_answer_coverage_ratio",
    },
    "electromagnetic_reference_heterogeneous_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
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
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
        "em_boundary_penalty_known_answer_coverage_ratio",
        "em_boundary_penalty_anchor_ratio",
        "em_boundary_penalty_conditioning_contribution",
        "em_boundary_penalty_real_residual_norm",
        "em_boundary_penalty_imag_residual_norm",
    },
    "electromagnetic_reference_multi_region_phased_source_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
        "em_phased_source_energy_consistency_ratio",
        "em_phased_source_interference_index",
        "em_phased_source_overlap_ratio",
        "em_phased_source_region_coverage_ratio",
    },
    "electromagnetic_reference_sparse_assignments_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
        "em_sparse_assignment_coverage_ratio",
        "em_sparse_fallback_coefficient_ratio",
        "em_sparse_source_realization_ratio",
        "em_sparse_source_region_coverage_ratio",
        "em_sparse_source_material_alignment_ratio",
        "em_sparse_boundary_anchor_ratio",
        "em_sparse_energy_imbalance_ratio",
    },
    "electromagnetic_reference_fallback_heavy_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
        "em_fallback_heavy_assignment_coverage_ratio",
        "em_fallback_heavy_fallback_coefficient_ratio",
        "em_fallback_heavy_source_realization_ratio",
        "em_fallback_heavy_source_region_coverage_ratio",
        "em_fallback_heavy_source_material_alignment_ratio",
        "em_fallback_heavy_boundary_anchor_ratio",
        "em_fallback_heavy_energy_imbalance_ratio",
    },
    "electromagnetic_reference_overlap_interference_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
        "em_overlap_source_interference_index",
        "em_overlap_source_region_coverage_ratio",
        "em_overlap_source_overlap_ratio",
        "em_overlap_source_material_alignment_ratio",
    },
    "electromagnetic_reference_boundary_kernel_gpu_provider": {
        "electromagnetic_source_energy_diagnostic_coverage_ratio",
        "electromagnetic_source_energy_consistency_ratio",
        "electromagnetic_source_energy_imbalance_ratio",
        "electromagnetic_sweep_known_reference_coverage_ratio",
        "electromagnetic_sweep_known_peak_frequency_error_ratio",
        "electromagnetic_sweep_known_quality_factor",
        "electromagnetic_sweep_known_answer_coverage_ratio",
        "em_boundary_kernel_known_answer_coverage_ratio",
        "em_boundary_kernel_boundary_localization_ratio",
        "em_boundary_kernel_ground_anchor_effectiveness_ratio",
        "em_boundary_kernel_insulation_leakage_ratio",
    },
    "acoustic_harmonic_gpu_provider": {
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
    "acoustic_harmonic_cpu": {
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
    "acoustic_harmonic_gpu_fallback": {
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
    "modal_large_cpu": {
        "modal_max_m_orthogonality_offdiag",
        "modal_min_relative_frequency_separation",
    },
    "modal_large_cpu_stress16": {
        "modal_max_m_orthogonality_offdiag",
        "modal_min_relative_frequency_separation",
    },
    "modal_large_gpu_fallback": {
        "modal_max_m_orthogonality_offdiag",
        "modal_min_relative_frequency_separation",
    },
    "modal_large_gpu_provider": {
        "modal_max_m_orthogonality_offdiag",
        "modal_min_relative_frequency_separation",
    },
    "modal_large_gpu_provider_stress16": {
        "modal_max_m_orthogonality_offdiag",
        "modal_min_relative_frequency_separation",
    },
    "transient_long_cpu": {
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
    }
    | TRANSIENT_ENERGY_BALANCE_REQUIRED_FIELDS,
    "transient_long_gpu_fallback": {
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
    }
    | TRANSIENT_ENERGY_BALANCE_REQUIRED_FIELDS,
    "transient_long_gpu_provider": {
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
    | TRANSIENT_ENERGY_BALANCE_REQUIRED_FIELDS,
    "transient_shock_cpu": {
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
    }
    | TRANSIENT_ENERGY_BALANCE_REQUIRED_FIELDS,
    "transient_shock_gpu_provider": {
        "transient_max_residual_norm",
        "transient_max_energy_growth_ratio",
        "transient_prepared_cache_hit_ratio",
        "transient_prepared_cache_misses",
        "transient_shock_physics_jump_ratio",
        "transient_shock_physics_nonfinite_count",
    }
    | TRANSIENT_ENERGY_BALANCE_REQUIRED_FIELDS,
    "cfd_steady_gpu_provider": {
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
    "cfd_steady_cpu": {
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
    "cfd_steady_gpu_fallback": {
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
    "cfd_transient_gpu_provider": {
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
    "cfd_transient_cpu": {
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
    "cfd_transient_gpu_fallback": {
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
    "cht_coupled_gpu_provider": {
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
        "cht_advection_temperature_shift_k",
        "cht_interface_conductance_w_per_m2k",
        "cht_flux_temperature_law_residual_ratio",
        "cht_heated_channel_energy_residual_ratio",
        "cht_conjugate_slab_flux_law_residual_ratio",
        "cht_known_answer_interface_temperature_continuity_ratio",
        "cht_advection_shift_coverage_ratio",
        "cht_known_answer_coverage_ratio",
    },
    "cht_coupled_cpu": {
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
        "cht_advection_temperature_shift_k",
        "cht_interface_conductance_w_per_m2k",
        "cht_flux_temperature_law_residual_ratio",
        "cht_heated_channel_energy_residual_ratio",
        "cht_conjugate_slab_flux_law_residual_ratio",
        "cht_known_answer_interface_temperature_continuity_ratio",
        "cht_advection_shift_coverage_ratio",
        "cht_known_answer_coverage_ratio",
    },
    "cht_coupled_gpu_fallback": {
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
        "cht_advection_temperature_shift_k",
        "cht_interface_conductance_w_per_m2k",
        "cht_flux_temperature_law_residual_ratio",
        "cht_heated_channel_energy_residual_ratio",
        "cht_conjugate_slab_flux_law_residual_ratio",
        "cht_known_answer_interface_temperature_continuity_ratio",
        "cht_advection_shift_coverage_ratio",
        "cht_known_answer_coverage_ratio",
    },
    "fsi_coupled_gpu_provider": {
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
        "fsi_max_coupling_iteration_count",
        "fsi_pressure_feedback_residual_ratio",
        "fsi_pressure_displacement_law_residual_ratio",
        "fsi_interface_stiffness_pa_per_m",
        "fsi_pressure_loaded_wall_displacement_law_residual_ratio",
        "fsi_interface_traction_balance_residual_ratio",
        "fsi_known_answer_displacement_transfer_residual_m",
        "fsi_partitioned_pressure_feedback_residual_ratio",
        "fsi_known_answer_coverage_ratio",
    },
    "fsi_coupled_cpu": {
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
        "fsi_max_coupling_iteration_count",
        "fsi_pressure_feedback_residual_ratio",
        "fsi_pressure_displacement_law_residual_ratio",
        "fsi_interface_stiffness_pa_per_m",
        "fsi_pressure_loaded_wall_displacement_law_residual_ratio",
        "fsi_interface_traction_balance_residual_ratio",
        "fsi_known_answer_displacement_transfer_residual_m",
        "fsi_partitioned_pressure_feedback_residual_ratio",
        "fsi_known_answer_coverage_ratio",
    },
    "fsi_coupled_gpu_fallback": {
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
        "fsi_max_coupling_iteration_count",
        "fsi_pressure_feedback_residual_ratio",
        "fsi_pressure_displacement_law_residual_ratio",
        "fsi_interface_stiffness_pa_per_m",
        "fsi_pressure_loaded_wall_displacement_law_residual_ratio",
        "fsi_interface_traction_balance_residual_ratio",
        "fsi_known_answer_displacement_transfer_residual_m",
        "fsi_partitioned_pressure_feedback_residual_ratio",
        "fsi_known_answer_coverage_ratio",
    },
}

REQUIRED_ERROR_FIXTURES = {
    "acoustic_harmonic_missing_source": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_SOURCE",
    },
    "acoustic_harmonic_missing_boundary": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_ACOUSTIC.MISSING_ACOUSTIC_BOUNDARY",
    },
    "electromagnetic_missing_material": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_MATERIAL",
    },
    "electromagnetic_missing_source": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_SOURCE",
    },
    "electromagnetic_missing_boundary": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_ELECTROMAGNETIC.MISSING_ELECTROMAGNETIC_BOUNDARY",
    },
    "thermal_standalone_ramp_invalid_material": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_THERMAL.INVALID_THERMAL_MATERIAL",
    },
    "thermal_standalone_ramp_invalid_source": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_THERMAL.INVALID_THERMAL_SOURCE",
    },
    "thermal_standalone_ramp_invalid_boundary": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_THERMAL.INVALID_THERMAL_BOUNDARY",
    },
    "electro_thermal_invalid_voltage": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
    },
    "electro_thermal_invalid_conductivity_scale": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
    },
    "electro_thermal_unmapped_region": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_TRANSIENT.INVALID_ELECTRO_THERMAL_OPTIONS",
    },
    "cht_coupled_invalid_interface_mapping": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_CHT.INVALID_INTERFACE_MAPPING",
    },
    "fsi_coupled_invalid_interface_mapping": {
        "validate_ok": True,
        "run_ok": False,
        "run_error_code": "RM.FEA.RUN_FSI.INVALID_INTERFACE_MAPPING",
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
    "electromagnetic_resonance_quality_factor",
    "electromagnetic_resonance_flux_gain",
}

EM_FREQUENCY_REQUIRED_FIELDS = {
    "electromagnetic_reference_frequency_hz",
}

EM_CONDITIONING_REQUIRED_FIELDS = {
    "electromagnetic_condition_number_estimate",
}

EM_APPLIED_CURRENT_REQUIRED_FIELDS = {
    "electromagnetic_applied_current_a",
}

EM_SOURCE_ENERGY_CONSISTENCY_REQUIRED_FIELDS = {
    "electromagnetic_source_region_energy_consistency_ratio",
}

EM_SOURCE_ENERGY_ASSERTION_REQUIRED_FIELDS = {
    "electromagnetic_source_energy_diagnostic_coverage_ratio",
    "electromagnetic_source_energy_consistency_ratio",
    "electromagnetic_source_energy_imbalance_ratio",
}

EM_SWEEP_KNOWN_ANSWER_REQUIRED_FIELDS = {
    "electromagnetic_sweep_known_reference_coverage_ratio",
    "electromagnetic_sweep_known_peak_frequency_error_ratio",
    "electromagnetic_sweep_known_quality_factor",
    "electromagnetic_sweep_known_answer_coverage_ratio",
}

EM_BOUNDARY_KNOWN_ANSWER_REQUIRED_FIELDS = {
    "em_boundary_penalty_known_answer_coverage_ratio",
    "em_boundary_kernel_known_answer_coverage_ratio",
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
    "electromagnetic_insulation_leakage_ratio",
}

EM_SOLVE_QUALITY_REQUIRED_FIELDS = {
    "electromagnetic_solve_quality",
}

EM_RESIDUAL_REQUIRED_FIELDS = {
    "electromagnetic_real_residual_norm",
    "electromagnetic_imag_residual_norm",
}

EM_BALANCE_REQUIRED_FIELDS = {
    "electromagnetic_energy_imbalance_ratio",
    "electromagnetic_flux_divergence_ratio",
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
    "nonlinear_softening_benchmark_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_load_path_mix_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_plasticity_benchmark_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "nonlinear_contact_benchmark_gpu_provider": {
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
    "thermal_standalone_ramp_gpu_fallback": {
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
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_heterogeneous_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_boundary_penalty_stress_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_multi_region_phased_source_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_sparse_assignments_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_fallback_heavy_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_overlap_interference_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "electromagnetic_reference_boundary_kernel_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
        "gpu_solver_fallback_apply_count",
    },
    "acoustic_harmonic_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "acoustic_harmonic_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cantilever_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cantilever_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cantilever_load_sweep_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cantilever_large_load_sweep_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "multi_material_assembly_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "modal_large_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "modal_large_gpu_provider_stress16": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "modal_large_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "transient_long_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "transient_long_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "transient_shock_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cfd_steady_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cfd_steady_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cfd_transient_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cfd_transient_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cht_coupled_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "cht_coupled_gpu_fallback": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "fsi_coupled_gpu_provider": {
        "gpu_speedup_ratio",
        "gpu_solver_solve_ms",
    },
    "fsi_coupled_gpu_fallback": {
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

    for fixture_id, expected in REQUIRED_ERROR_FIXTURES.items():
        record = records.get(fixture_id)
        if record is None:
            errors.append(f"missing fixture record: {fixture_id}")
            continue
        for field, expected_value in expected.items():
            observed = record.get(field)
            if observed != expected_value:
                errors.append(
                    f"fixture {fixture_id} expected {field}={expected_value!r}, got {observed!r}"
                )

    for fixture_id, required in REQUIRED_FIXTURES.items():
        record = records.get(fixture_id)
        if record is None:
            continue

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
            "nonlinear_plasticity_benchmark_gpu_provider",
        }:
            missing_fields = sorted(
                field for field in ELECTRO_REQUIRED_FIELDS if field not in record
            )
            if missing_fields:
                errors.append(
                    f"fixture {fixture_id} missing electro summary fields: {', '.join(missing_fields)}"
                )

        if fixture_id in {
            "nonlinear_plasticity_benchmark_gpu_provider",
        }:
            missing_fields = sorted(
                field for field in PLASTIC_REQUIRED_FIELDS if field not in record
            )
            if missing_fields:
                errors.append(
                    f"fixture {fixture_id} missing plastic summary fields: {', '.join(missing_fields)}"
                )

        if fixture_id in {
            "nonlinear_contact_benchmark_gpu_provider",
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
            for field in sorted(EM_SOLVE_QUALITY_REQUIRED_FIELDS):
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
