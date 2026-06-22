use runmat_analysis_core::{AnalysisModel, EvidenceConfidence, MaterialAssignment};

use crate::{
    assembly,
    contracts::{FeaPrepCalibrationProfile, FeaPrepContext},
    diagnostics::{FeaDiagnostic, FeaDiagnosticSeverity},
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct CommonRunDiagnosticInputs<'a> {
    pub(crate) model: &'a AnalysisModel,
    pub(crate) summary: &'a assembly::AssemblySummary,
    pub(crate) prep_context: Option<FeaPrepContext>,
    pub(crate) iteration_metric: f64,
    pub(crate) residual_metric: f64,
    pub(crate) requested_preconditioner: &'a str,
    pub(crate) effective_preconditioner: &'a str,
}

pub(crate) fn extend_common_run_diagnostics(
    diagnostics: &mut Vec<FeaDiagnostic>,
    inputs: CommonRunDiagnosticInputs<'_>,
) {
    diagnostics.extend(material_assignment_diagnostics(
        &inputs.model.material_assignments,
    ));
    if let Some(prep) = inputs.prep_context {
        diagnostics.push(prep_diagnostic(prep));
    }
    if let Some(prep_summary) = inputs.summary.prep_assembly.as_ref() {
        diagnostics.push(prep_assembly_diagnostic(prep_summary));
        if let Some(prep) = inputs.prep_context {
            diagnostics.push(prep_topology_diagnostic(prep, inputs.summary.dof_count));
        }
    }
    if let Some(operator_topology) = inputs.summary.prep_operator_topology.as_ref() {
        diagnostics.push(prep_operator_topology_diagnostic(operator_topology));
    }
    if let Some(region_topology) = inputs.summary.prep_region_topology.as_ref() {
        diagnostics.push(prep_region_topology_diagnostic(region_topology));
    }
    if let Some(element_assembly) = inputs.summary.prep_element_assembly.as_ref() {
        diagnostics.push(prep_element_assembly_diagnostic(element_assembly));
    }
    if let Some(element_connectivity) = inputs.summary.prep_element_connectivity.as_ref() {
        diagnostics.push(prep_element_connectivity_diagnostic(element_connectivity));
    }
    if let Some(graph_assembly) = inputs.summary.prep_graph_assembly.as_ref() {
        diagnostics.push(prep_graph_assembly_diagnostic(graph_assembly));
        diagnostics.push(prep_graph_solver_diagnostic(
            graph_assembly,
            inputs.iteration_metric,
            inputs.residual_metric,
            inputs.requested_preconditioner,
            inputs.effective_preconditioner,
        ));
    }
    if let Some(calibration) = inputs.summary.prep_calibration.as_ref() {
        diagnostics.push(prep_calibration_diagnostic(calibration));
    }
    if let Some(acceptance) = inputs.summary.prep_acceptance.as_ref() {
        diagnostics.push(prep_acceptance_diagnostic(acceptance));
    }
    if let Some(thermo_mechanical) = inputs.summary.thermo_mechanical.as_ref() {
        diagnostics.push(thermo_mechanical_diagnostic(thermo_mechanical));
    }
    if let Some(electro_thermal) = inputs.summary.electro_thermal.as_ref() {
        diagnostics.push(electro_thermal_diagnostic(electro_thermal));
    }
}

pub(crate) fn material_assignment_diagnostics(
    assignments: &[MaterialAssignment],
) -> Vec<FeaDiagnostic> {
    let mut out = Vec::new();
    for assignment in assignments {
        if assignment.expected_material_id == assignment.assigned_material_id {
            continue;
        }

        let (code, severity) = match assignment.confidence {
            EvidenceConfidence::Verified => (
                "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_VERIFIED",
                FeaDiagnosticSeverity::Error,
            ),
            EvidenceConfidence::Probable => (
                "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_PROBABLE",
                FeaDiagnosticSeverity::Warning,
            ),
            EvidenceConfidence::Inferred => (
                "ANALYSIS_MATERIAL_ASSIGNMENT_CONFLICT_INFERRED",
                FeaDiagnosticSeverity::Warning,
            ),
        };

        out.push(FeaDiagnostic {
            code: code.to_string(),
            severity,
            message: format!(
                "region={} expected_material={} assigned_material={} confidence={:?}",
                assignment.region_id,
                assignment.expected_material_id,
                assignment.assigned_material_id,
                assignment.confidence
            ),
        });
    }
    out
}

pub(crate) fn prep_diagnostic(prep: FeaPrepContext) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_CONTEXT".to_string(),
        severity: if prep.inverted_element_count == 0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "prepared_mesh_count={} prepared_node_count={} prepared_element_count={} mapped_region_count={} mapped_load_count={} mapped_bc_count={} min_scaled_jacobian={} mean_aspect_ratio={} inverted_element_count={} topology_dof_multiplier={} topology_bandwidth_estimate={} mapped_region_participation_ratio={} topology_surface_patch_ratio={} topology_volume_core_ratio={} topology_mixed_family_ratio={} topology_region_span_mean={} topology_region_block_count={} topology_region_mesh_mean={} topology_region_mesh_variance={} topology_triangle_family_ratio={} topology_quad_family_ratio={} topology_tet_family_ratio={} topology_hex_family_ratio={} element_geometry_node_count={} element_geometry_edge_count={} mean_element_edge_length_m={} mean_element_area_m2={} element_geometry_coverage_ratio={} reference_element_area_m2={} calibration_profile_override={}",
            prep.prepared_mesh_count,
            prep.prepared_node_count,
            prep.prepared_element_count,
            prep.mapped_region_count,
            prep.mapped_load_count,
            prep.mapped_bc_count,
            prep.min_scaled_jacobian,
            prep.mean_aspect_ratio,
            prep.inverted_element_count,
            prep.topology_dof_multiplier,
            prep.topology_bandwidth_estimate,
            prep.mapped_region_participation_ratio,
            prep.topology_surface_patch_ratio,
            prep.topology_volume_core_ratio,
            prep.topology_mixed_family_ratio,
            prep.topology_region_span_mean,
            prep.topology_region_block_count,
            prep.topology_region_mesh_mean,
            prep.topology_region_mesh_variance,
            prep.topology_triangle_family_ratio,
            prep.topology_quad_family_ratio,
            prep.topology_tet_family_ratio,
            prep.topology_hex_family_ratio,
            prep.element_geometry_node_count,
            prep.element_geometry_edge_count,
            prep.mean_element_edge_length_m,
            prep.mean_element_area_m2,
            prep.element_geometry_coverage_ratio,
            prep.reference_element_area_m2,
            prep.calibration_profile_override
                .map(|profile| match profile {
                    FeaPrepCalibrationProfile::Fast => "fast",
                    FeaPrepCalibrationProfile::Balanced => "balanced",
                    FeaPrepCalibrationProfile::Conservative => "conservative",
                })
                .unwrap_or("auto"),
        ),
    }
}

pub(crate) fn prep_assembly_diagnostic(summary: &assembly::PrepAssemblySummary) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ASSEMBLY".to_string(),
        severity: if summary.mapped_load_ratio > 0.0 || summary.constrained_prep_ratio > 0.0 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "active_region_count={} mapped_load_count={} mapped_bc_count={} mapped_load_ratio={} constrained_prep_ratio={} layout_seed={}",
            summary.active_region_count,
            summary.mapped_load_count,
            summary.mapped_bc_count,
            summary.mapped_load_ratio,
            summary.constrained_prep_ratio,
            summary.layout_seed
        ),
    }
}

pub(crate) fn prep_topology_diagnostic(prep: FeaPrepContext, dof_count: usize) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_TOPOLOGY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "effective_dof_multiplier={} effective_dof_count={} coupling_bandwidth_estimate={} mapped_region_participation_ratio={} surface_patch_ratio={} volume_core_ratio={} mixed_family_ratio={} mean_region_span={}",
            prep.topology_dof_multiplier,
            dof_count,
            prep.topology_bandwidth_estimate,
            prep.mapped_region_participation_ratio,
            prep.topology_surface_patch_ratio,
            prep.topology_volume_core_ratio,
            prep.topology_mixed_family_ratio,
            prep.topology_region_span_mean,
        ),
    }
}

pub(crate) fn prep_operator_topology_diagnostic(
    summary: &assembly::PrepOperatorTopologySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_OPERATOR_TOPOLOGY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "stiffness_scale={} mass_scale={} damping_scale={} rhs_scale={} coupling_nonzero_ratio={} stiffness_spread_ratio={} topology_fingerprint={}",
            summary.stiffness_scale,
            summary.mass_scale,
            summary.damping_scale,
            summary.rhs_scale,
            summary.coupling_nonzero_ratio,
            summary.stiffness_spread_ratio,
            summary.topology_fingerprint,
        ),
    }
}

pub(crate) fn prep_region_topology_diagnostic(
    summary: &assembly::PrepRegionTopologySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_REGION_TOPOLOGY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "region_block_count={} inter_block_edge_count={} coupling_nonzero_ratio={} block_size_min={} block_size_max={} block_size_mean={} region_topology_fingerprint={}",
            summary.region_block_count,
            summary.inter_block_edge_count,
            summary.coupling_nonzero_ratio,
            summary.block_size_min,
            summary.block_size_max,
            summary.block_size_mean,
            summary.region_topology_fingerprint,
        ),
    }
}

pub(crate) fn prep_element_assembly_diagnostic(
    summary: &assembly::PrepElementAssemblySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ELEMENT_ASSEMBLY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "assembled_element_count={} triangle_element_count={} quad_element_count={} tet_element_count={} hex_element_count={} mixed_element_count={} scatter_nnz_count={} assembly_fingerprint={}",
            summary.assembled_element_count,
            summary.triangle_element_count,
            summary.quad_element_count,
            summary.tet_element_count,
            summary.hex_element_count,
            summary.mixed_element_count,
            summary.scatter_nnz_count,
            summary.assembly_fingerprint,
        ),
    }
}

pub(crate) fn prep_element_connectivity_diagnostic(
    summary: &assembly::PrepElementConnectivitySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ELEMENT_CONNECTIVITY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "assembled_element_count={} stiffness_offdiag_nnz_count={} mass_offdiag_nnz_count={} damping_offdiag_nnz_count={} triangle_contrib_share={} quad_contrib_share={} tet_contrib_share={} hex_contrib_share={} mixed_contrib_share={} mean_connectivity_hop={} connectivity_fingerprint={}",
            summary.assembled_element_count,
            summary.stiffness_offdiag_nnz_count,
            summary.mass_offdiag_nnz_count,
            summary.damping_offdiag_nnz_count,
            summary.triangle_contrib_share,
            summary.quad_contrib_share,
            summary.tet_contrib_share,
            summary.hex_contrib_share,
            summary.mixed_contrib_share,
            summary.mean_connectivity_hop,
            summary.connectivity_fingerprint,
        ),
    }
}

pub(crate) fn prep_graph_assembly_diagnostic(
    summary: &assembly::PrepGraphAssemblySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_GRAPH_ASSEMBLY".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "node_count={} edge_count={} degree_min={} degree_max={} degree_mean={} degree_p95={} fill_ratio={} connected_component_count={} ordering_bandwidth_before={} ordering_bandwidth_after={} ordering_reduction_ratio={} ordering_fingerprint={} recommend_ilu0={} graph_fingerprint={}",
            summary.node_count,
            summary.edge_count,
            summary.degree_min,
            summary.degree_max,
            summary.degree_mean,
            summary.degree_p95,
            summary.fill_ratio,
            summary.connected_component_count,
            summary.ordering_bandwidth_before,
            summary.ordering_bandwidth_after,
            summary.ordering_reduction_ratio,
            summary.ordering_fingerprint,
            summary.recommend_ilu0,
            summary.graph_fingerprint,
        ),
    }
}

pub(crate) fn prep_graph_solver_diagnostic(
    summary: &assembly::PrepGraphAssemblySummary,
    iteration_metric: f64,
    residual_metric: f64,
    requested_preconditioner: &str,
    effective_preconditioner: &str,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_GRAPH_SOLVER".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "ordering_bandwidth_before={} ordering_bandwidth_after={} ordering_reduction_ratio={} ordering_fingerprint={} recommend_ilu0={} requested_preconditioner={} effective_preconditioner={} iteration_metric={} residual_metric={} graph_fingerprint={}",
            summary.ordering_bandwidth_before,
            summary.ordering_bandwidth_after,
            summary.ordering_reduction_ratio,
            summary.ordering_fingerprint,
            summary.recommend_ilu0,
            requested_preconditioner,
            effective_preconditioner,
            iteration_metric,
            residual_metric,
            summary.graph_fingerprint,
        ),
    }
}

pub(crate) fn prep_calibration_diagnostic(
    summary: &assembly::PrepCalibrationSummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_CALIBRATION".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "profile={} triangle_weight={} quad_weight={} tet_weight={} hex_weight={} mixed_weight={} stiffness_calibration_scale={} mass_calibration_scale={} damping_calibration_scale={} calibration_fingerprint={}",
            summary.profile,
            summary.triangle_weight,
            summary.quad_weight,
            summary.tet_weight,
            summary.hex_weight,
            summary.mixed_weight,
            summary.stiffness_calibration_scale,
            summary.mass_calibration_scale,
            summary.damping_calibration_scale,
            summary.calibration_fingerprint,
        ),
    }
}

pub(crate) fn prep_acceptance_diagnostic(
    summary: &assembly::PrepAcceptanceSummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_PREP_ACCEPTANCE".to_string(),
        severity: if summary.accepted {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "profile={} accepted={} bounded_displacement_scale={} bounded_stress_scale={} bounded_connectivity_fill={} acceptance_score={} acceptance_fingerprint={}",
            summary.profile,
            summary.accepted,
            summary.bounded_displacement_scale,
            summary.bounded_stress_scale,
            summary.bounded_connectivity_fill,
            summary.acceptance_score,
            summary.acceptance_fingerprint,
        ),
    }
}

pub(crate) fn thermo_mechanical_diagnostic(
    summary: &assembly::ThermoMechanicalAssemblySummary,
) -> FeaDiagnostic {
    FeaDiagnostic {
        code: "FEA_TM_COUPLING".to_string(),
        severity: FeaDiagnosticSeverity::Info,
        message: format!(
            "enabled={} reference_temperature_k={} applied_temperature_delta_k={} thermal_expansion_coefficient={} thermal_strain_scale={} thermal_load_scale={} constitutive_temperature_factor={} constitutive_poisson_coupling={} effective_modulus_scale={} constitutive_material_spread_ratio={} assignment_heterogeneity_index={} spatial_gradient_index={} spatial_coverage_ratio={} temporal_profile_variation={} region_delta_count={} coupling_fingerprint={}",
            summary.enabled,
            summary.reference_temperature_k,
            summary.applied_temperature_delta_k,
            summary.thermal_expansion_coefficient,
            summary.thermal_strain_scale,
            summary.thermal_load_scale,
            summary.constitutive_temperature_factor,
            summary.constitutive_poisson_coupling,
            summary.effective_modulus_scale,
            summary.constitutive_material_spread_ratio,
            summary.assignment_heterogeneity_index,
            summary.spatial_gradient_index,
            summary.spatial_coverage_ratio,
            summary.temporal_profile_variation,
            summary.region_delta_count,
            summary.coupling_fingerprint,
        ),
    }
}

pub(crate) fn electro_thermal_diagnostic(
    summary: &assembly::ElectroThermalAssemblySummary,
) -> FeaDiagnostic {
    let electrical_power_in_w = summary.applied_voltage_v.powi(2)
        * summary.base_electrical_conductivity_s_per_m.max(1.0e-9)
        * summary.resistive_heating_coefficient.max(0.0)
        / 1.0e6;
    let integrated_joule_heat_w = summary.joule_heating_scale;
    let power_balance_ratio = if electrical_power_in_w > 1.0e-12 {
        integrated_joule_heat_w / electrical_power_in_w
    } else {
        1.0
    };
    let conservation_residual = (1.0 - power_balance_ratio).abs();
    FeaDiagnostic {
        code: "FEA_ET_COUPLING".to_string(),
        severity: if conservation_residual <= 1.0e-6 {
            FeaDiagnosticSeverity::Info
        } else {
            FeaDiagnosticSeverity::Warning
        },
        message: format!(
            "enabled={} reference_temperature_k={} applied_voltage_v={} base_electrical_conductivity_s_per_m={} resistive_heating_coefficient={} joule_heating_scale={} conductivity_spread_ratio={} temporal_profile_variation={} region_scale_count={} coupling_fingerprint={} electrical_power_in_w={} integrated_joule_heat_w={} power_balance_ratio={} conservation_residual={}",
            summary.enabled,
            summary.reference_temperature_k,
            summary.applied_voltage_v,
            summary.base_electrical_conductivity_s_per_m,
            summary.resistive_heating_coefficient,
            summary.joule_heating_scale,
            summary.conductivity_spread_ratio,
            summary.temporal_profile_variation,
            summary.region_scale_count,
            summary.coupling_fingerprint,
            electrical_power_in_w,
            integrated_joule_heat_w,
            power_balance_ratio,
            conservation_residual,
        ),
    }
}
