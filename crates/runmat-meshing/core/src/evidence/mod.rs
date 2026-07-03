use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    adaptive::{AdaptiveConvergenceStatus, RefinementIndicatorStatus},
    artifact::{AnalysisMeshArtifact, MeshBackendSummary},
    quality::QualityThresholds,
    validation::{
        analysis_mesh_validation_error_code, mesh_contains_point,
        validate_analysis_mesh_with_options, volume_component_count,
        volume_component_element_counts, AnalysisMeshValidationOptions,
    },
};

pub const MESH_EVIDENCE_SCHEMA_VERSION: &str = "mesh-evidence/v1";

#[cfg(feature = "dev-evidence")]
pub use dev_traces::{build_mesh_evidence_artifact_with_debug, MeshDebugEvent, MeshDebugEvidence};
use summaries::quality_evidence;
use summaries::region_evidence;
pub use summaries::{MeshQualityEvidence, MeshRegionEvidence};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshEvidenceArtifact {
    pub schema_version: String,
    pub mesh_id: String,
    pub backend: MeshBackendSummary,
    #[cfg(feature = "dev-evidence")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug: Option<MeshDebugEvidence>,
    #[serde(default)]
    pub cad: MeshCadEvidence,
    pub topology: MeshTopologyEvidence,
    #[serde(default)]
    pub adaptive: MeshAdaptiveEvidence,
    pub sizing: MeshSizingEvidence,
    pub quality: MeshQualityEvidence,
    #[serde(default)]
    pub tetrahedron_recovery: MeshTetrahedronRecoveryEvidence,
    pub regions: MeshRegionEvidence,
    pub validation: MeshValidationEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshCadEvidence {
    pub topology_source: String,
    pub evaluation_source: String,
    pub vertex_count: usize,
    pub edge_count: usize,
    pub face_count: usize,
    pub shell_count: usize,
    pub volume_count: usize,
    pub imported_face_count: usize,
    pub evaluator_face_count: usize,
    #[serde(default)]
    pub live_query_face_count: usize,
    pub exact_query_face_count: usize,
    #[serde(default)]
    pub missing_exact_query_face_count: usize,
    #[serde(default)]
    pub missing_derivative_query_face_count: usize,
    #[serde(default)]
    pub missing_curvature_query_face_count: usize,
    #[serde(default)]
    pub point_evaluation_supported_face_count: usize,
    #[serde(default)]
    pub projection_supported_face_count: usize,
    #[serde(default)]
    pub normal_supported_face_count: usize,
    #[serde(default)]
    pub derivative_supported_face_count: usize,
    #[serde(default)]
    pub curvature_supported_face_count: usize,
    pub evaluator_sample_count: usize,
    #[serde(default)]
    pub evaluator_rejected_sample_count: usize,
    pub projection_query_count: usize,
    #[serde(default)]
    pub derivative_query_count: usize,
    #[serde(default)]
    pub curvature_query_count: usize,
    #[serde(default)]
    pub uv_domain_face_count: usize,
    #[serde(default)]
    pub uv_projection_out_of_bounds_count: usize,
    pub max_projection_error_m: f64,
    pub max_normal_deviation: f64,
    #[serde(default)]
    pub max_curvature_estimate_1_per_m: f64,
    pub surface_cad_face_count: usize,
    #[serde(default)]
    pub surface_source_edge_loop_count: usize,
    #[serde(default)]
    pub surface_closed_edge_loop_count: usize,
    #[serde(default)]
    pub surface_conforming_source_edge_count: usize,
    #[serde(default)]
    pub surface_missing_source_edge_count: usize,
    #[serde(default)]
    pub surface_exact_cad_sample_node_count: usize,
    #[serde(default)]
    pub surface_rejected_exact_cad_sample_count: usize,
    pub surface_max_projection_error_m: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshTopologyEvidence {
    pub node_count: usize,
    pub volume_element_count: usize,
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub adaptive_iteration_count: usize,
    pub bounds_min_m: Option<[f64; 3]>,
    pub bounds_max_m: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshAdaptiveEvidence {
    pub iteration_count: usize,
    #[serde(default)]
    pub latest_iteration_index: Option<usize>,
    #[serde(default)]
    pub latest_convergence_status: Option<String>,
    #[serde(default)]
    pub latest_indicator_count: usize,
    #[serde(default)]
    pub latest_used_indicator_count: usize,
    #[serde(default)]
    pub latest_marker_count: usize,
    #[serde(default)]
    pub latest_sizing_update_sample_count: usize,
    #[serde(default)]
    pub marker_count: usize,
    #[serde(default)]
    pub sizing_update_sample_count: usize,
    #[serde(default)]
    pub latest_indicator_status_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub latest_marker_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub latest_sizing_update_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub marker_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub sizing_update_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshSizingEvidence {
    pub global_target_size_m: Option<f64>,
    pub min_size_m: Option<f64>,
    pub max_size_m: Option<f64>,
    #[serde(default)]
    pub growth_rate: Option<f64>,
    pub sample_count: usize,
    #[serde(default)]
    pub generated_cad_sample_count: usize,
    #[serde(default)]
    pub anisotropic_sample_count: usize,
    #[serde(default)]
    pub valid_anisotropic_sample_count: usize,
    #[serde(default)]
    pub invalid_anisotropic_sample_count: usize,
    pub applied_sample_count: usize,
    pub rejected_sample_count: usize,
    pub inserted_breakpoint_count: usize,
    #[serde(default)]
    pub inserted_breakpoint_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub uninserted_sample_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_location_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_surrogate_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tetrahedron_refinement_exact_point_count: usize,
    #[serde(default)]
    pub rejected_requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub requested_tetrahedron_refinement_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub dropped_requested_tetrahedron_refinement_point_count: usize,
    #[serde(default)]
    pub requested_tetrahedron_refinement_dropped_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_acceptance_ratio: Option<f64>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_rejection_ratio: Option<f64>,
    #[serde(default)]
    pub requested_tetrahedron_refinement_surrogate_ratio: Option<f64>,
    #[serde(default)]
    pub generated_cad_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub anisotropic_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub invalid_anisotropic_by_reason: BTreeMap<String, usize>,
    pub applied_by_reason: BTreeMap<String, usize>,
    pub rejected_by_status: BTreeMap<String, usize>,
    pub rejected_by_reason: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshTetrahedronRecoveryEvidence {
    pub element_count: usize,
    pub recovered_component_ratio: f64,
    pub fan_fallback_component_count: usize,
    pub volume_coverage_ratio: f64,
    pub refinement_pass_count: usize,
    pub refinement_point_count: usize,
    pub optimization_pass_count: usize,
    pub smoothed_point_count: usize,
    pub sliver_count: usize,
    #[serde(default)]
    pub sliver_removed_count: usize,
    #[serde(default)]
    pub optimization_target_seed_count: usize,
    #[serde(default)]
    pub optimization_skipped_target_seed_count: usize,
    #[serde(default)]
    pub optimization_rejected_edit_count: usize,
    #[serde(default)]
    pub optimization_initial_max_aspect_ratio: f64,
    #[serde(default)]
    pub optimization_final_max_aspect_ratio: f64,
    #[serde(default)]
    pub optimization_initial_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub optimization_final_min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub untangling_pass_count: usize,
    #[serde(default)]
    pub untangling_initial_near_singular_count: usize,
    #[serde(default)]
    pub untangling_final_near_singular_count: usize,
    #[serde(default)]
    pub untangling_relocated_seed_count: usize,
    #[serde(default)]
    pub untangling_reconnected_edge_star_count: usize,
    #[serde(default)]
    pub untangling_reconnected_boundary_adjacent_cavity_count: usize,
    #[serde(default)]
    pub untangling_reconnected_node_adjacent_cavity_count: usize,
    pub exact_quality_repair_pass_count: usize,
    pub exact_quality_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_reconnection_quality_gain_count: usize,
    #[serde(default)]
    pub exact_quality_face_neighbor_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_connected_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_node_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_boundary_adjacent_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_expanded_connected_reconnected_cavity_count: usize,
    pub exact_quality_split_cavity_count: usize,
    pub exact_quality_seed_star_collapse_count: usize,
    #[serde(default)]
    pub exact_quality_seed_star_relocation_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_total_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_general_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_boundary_adjacent_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_node_adjacent_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_interior_seed_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_edge_star_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshValidationEvidence {
    #[serde(default = "default_solve_ready")]
    pub solve_ready: bool,
    #[serde(default)]
    pub validation_error_code: Option<String>,
    #[serde(default)]
    pub validation_error_message: Option<String>,
    pub quality: QualityThresholds,
    #[serde(default)]
    pub volume_element_count: usize,
    #[serde(default)]
    pub max_volume_element_count: Option<usize>,
    #[serde(default)]
    pub volume_component_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub volume_component_element_counts: Vec<usize>,
    #[serde(default)]
    pub max_volume_component_count: Option<usize>,
    #[serde(default)]
    pub coverage_sample_count: usize,
    #[serde(default)]
    pub covered_coverage_sample_count: usize,
    #[serde(default)]
    pub coverage_sample_ratio: Option<f64>,
    #[serde(default = "default_min_coverage_sample_ratio")]
    pub min_coverage_sample_ratio: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coverage_sample_points_m: Vec<[f64; 3]>,
    pub expected_bounds_m: Option<[[f64; 3]; 2]>,
    pub min_bounds_coverage_ratio: f64,
    pub expected_volume_m3: Option<f64>,
    pub min_volume_coverage_ratio: f64,
    pub expected_boundary_area_m2: Option<f64>,
    pub min_boundary_area_ratio: f64,
    pub min_boundary_face_recovery_ratio: f64,
    pub min_boundary_edge_recovery_ratio: f64,
    #[serde(default)]
    pub require_no_fan_fallback: bool,
    #[serde(default)]
    pub require_no_unrepaired_exact_quality: bool,
    #[serde(default)]
    pub fan_fallback_component_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_total_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_general_cavity_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_boundary_adjacent_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_node_adjacent_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_interior_seed_count: usize,
    #[serde(default)]
    pub unrepaired_exact_quality_edge_star_count: usize,
    pub required_boundary_region_ids: Vec<String>,
    pub required_material_region_ids: Vec<String>,
    pub boundary_recovery: MeshBoundaryRecoveryEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBoundaryRecoveryEvidence {
    pub boundary_face_recovery_ratio: f64,
    pub boundary_edge_recovery_ratio: f64,
    pub recovered_boundary_face_count: usize,
    pub recovered_boundary_edge_count: usize,
}

pub fn build_mesh_evidence_artifact(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshEvidenceArtifact {
    build_mesh_evidence_artifact_with_validation_evidence(
        mesh,
        validation_evidence(mesh, validation),
    )
}

pub fn build_mesh_evidence_artifact_with_validation_evidence(
    mesh: &AnalysisMeshArtifact,
    validation: MeshValidationEvidence,
) -> MeshEvidenceArtifact {
    let validation = validation_evidence(mesh, &validation_options_from_evidence(&validation));
    MeshEvidenceArtifact {
        schema_version: MESH_EVIDENCE_SCHEMA_VERSION.to_string(),
        mesh_id: mesh.mesh_id.clone(),
        backend: mesh.backend.clone(),
        #[cfg(feature = "dev-evidence")]
        debug: None,
        cad: cad_evidence(mesh),
        topology: topology_evidence(mesh),
        adaptive: adaptive_evidence(mesh),
        sizing: sizing_evidence(mesh),
        quality: quality_evidence(mesh),
        tetrahedron_recovery: tetrahedron_recovery_evidence(mesh),
        regions: region_evidence(mesh),
        validation,
    }
}

fn validation_options_from_evidence(
    validation: &MeshValidationEvidence,
) -> AnalysisMeshValidationOptions {
    AnalysisMeshValidationOptions {
        quality: validation.quality,
        max_volume_element_count: validation.max_volume_element_count,
        max_volume_component_count: validation.max_volume_component_count,
        coverage_sample_points_m: validation.coverage_sample_points_m.clone(),
        min_coverage_sample_ratio: validation.min_coverage_sample_ratio,
        expected_bounds_m: validation.expected_bounds_m,
        min_bounds_coverage_ratio: validation.min_bounds_coverage_ratio,
        expected_volume_m3: validation.expected_volume_m3,
        min_volume_coverage_ratio: validation.min_volume_coverage_ratio,
        expected_boundary_area_m2: validation.expected_boundary_area_m2,
        min_boundary_area_ratio: validation.min_boundary_area_ratio,
        min_boundary_face_recovery_ratio: validation.min_boundary_face_recovery_ratio,
        min_boundary_edge_recovery_ratio: validation.min_boundary_edge_recovery_ratio,
        require_no_fan_fallback: validation.require_no_fan_fallback,
        require_no_unrepaired_exact_quality: validation.require_no_unrepaired_exact_quality,
        required_boundary_region_ids: validation.required_boundary_region_ids.clone(),
        required_material_region_ids: validation.required_material_region_ids.clone(),
    }
}

fn cad_evidence(mesh: &AnalysisMeshArtifact) -> MeshCadEvidence {
    MeshCadEvidence {
        topology_source: mesh.backend.cad_topology_source.clone(),
        evaluation_source: mesh.backend.cad_evaluation_source.clone(),
        vertex_count: mesh.backend.cad_vertex_count,
        edge_count: mesh.backend.cad_edge_count,
        face_count: mesh.backend.cad_face_count,
        shell_count: mesh.backend.cad_shell_count,
        volume_count: mesh.backend.cad_volume_count,
        imported_face_count: mesh.backend.cad_imported_face_count,
        evaluator_face_count: mesh.backend.cad_evaluation_evaluator_face_count,
        live_query_face_count: mesh.backend.cad_evaluation_live_query_face_count,
        exact_query_face_count: mesh.backend.cad_evaluation_exact_query_face_count,
        missing_exact_query_face_count: mesh.backend.cad_evaluation_missing_exact_query_face_count,
        missing_derivative_query_face_count: mesh
            .backend
            .cad_evaluation_missing_derivative_query_face_count,
        missing_curvature_query_face_count: mesh
            .backend
            .cad_evaluation_missing_curvature_query_face_count,
        point_evaluation_supported_face_count: mesh
            .backend
            .cad_evaluation_point_supported_face_count,
        projection_supported_face_count: mesh
            .backend
            .cad_evaluation_projection_supported_face_count,
        normal_supported_face_count: mesh.backend.cad_evaluation_normal_supported_face_count,
        derivative_supported_face_count: mesh
            .backend
            .cad_evaluation_derivative_supported_face_count,
        curvature_supported_face_count: mesh.backend.cad_evaluation_curvature_supported_face_count,
        evaluator_sample_count: mesh.backend.cad_evaluation_sample_count,
        evaluator_rejected_sample_count: mesh.backend.cad_evaluation_rejected_sample_count,
        projection_query_count: mesh.backend.cad_projection_query_count,
        derivative_query_count: mesh.backend.cad_derivative_query_count,
        curvature_query_count: mesh.backend.cad_curvature_query_count,
        uv_domain_face_count: mesh.backend.cad_uv_domain_face_count,
        uv_projection_out_of_bounds_count: mesh.backend.cad_uv_projection_out_of_bounds_count,
        max_projection_error_m: mesh.backend.cad_max_projection_error_m,
        max_normal_deviation: mesh.backend.cad_max_normal_deviation,
        max_curvature_estimate_1_per_m: mesh.backend.cad_max_curvature_estimate_1_per_m,
        surface_cad_face_count: mesh.backend.surface_cad_face_count,
        surface_source_edge_loop_count: mesh.backend.surface_source_edge_loop_count,
        surface_closed_edge_loop_count: mesh.backend.surface_closed_edge_loop_count,
        surface_conforming_source_edge_count: mesh.backend.surface_conforming_source_edge_count,
        surface_missing_source_edge_count: mesh.backend.surface_missing_source_edge_count,
        surface_exact_cad_sample_node_count: mesh.backend.surface_exact_cad_sample_node_count,
        surface_rejected_exact_cad_sample_count: mesh
            .backend
            .surface_rejected_exact_cad_sample_count,
        surface_max_projection_error_m: mesh.backend.surface_max_cad_projection_error_m,
    }
}

fn topology_evidence(mesh: &AnalysisMeshArtifact) -> MeshTopologyEvidence {
    let bounds = mesh_bounds_m(mesh);
    MeshTopologyEvidence {
        node_count: mesh.nodes.len(),
        volume_element_count: mesh.volume_elements.len(),
        boundary_face_count: mesh.boundary_faces.len(),
        boundary_edge_count: mesh.boundary_edges.len(),
        adaptive_iteration_count: mesh.adaptive_iterations.len(),
        bounds_min_m: bounds.map(|bounds| bounds[0]),
        bounds_max_m: bounds.map(|bounds| bounds[1]),
    }
}

fn adaptive_evidence(mesh: &AnalysisMeshArtifact) -> MeshAdaptiveEvidence {
    let mut marker_count = 0_usize;
    let mut sizing_update_sample_count = 0_usize;
    let mut marker_by_reason = BTreeMap::<String, usize>::new();
    let mut sizing_update_by_reason = BTreeMap::<String, usize>::new();
    for iteration in &mesh.adaptive_iterations {
        marker_count += iteration.markers.len();
        sizing_update_sample_count += iteration.sizing_update.samples.len();
        for marker in &iteration.markers {
            *marker_by_reason.entry(marker.reason.clone()).or_default() += 1;
        }
        for sample in &iteration.sizing_update.samples {
            let reason = sample
                .reason
                .clone()
                .unwrap_or_else(|| "unspecified".to_string());
            *sizing_update_by_reason.entry(reason).or_default() += 1;
        }
    }

    let Some(latest) = mesh.adaptive_iterations.last() else {
        return MeshAdaptiveEvidence {
            iteration_count: 0,
            ..MeshAdaptiveEvidence::default()
        };
    };

    let mut latest_indicator_status_counts = BTreeMap::<String, usize>::new();
    for indicator in &latest.indicators {
        *latest_indicator_status_counts
            .entry(indicator_status_label(indicator.status))
            .or_default() += 1;
    }
    let mut latest_marker_by_reason = BTreeMap::<String, usize>::new();
    for marker in &latest.markers {
        *latest_marker_by_reason
            .entry(marker.reason.clone())
            .or_default() += 1;
    }
    let mut latest_sizing_update_by_reason = BTreeMap::<String, usize>::new();
    for sample in &latest.sizing_update.samples {
        let reason = sample
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *latest_sizing_update_by_reason.entry(reason).or_default() += 1;
    }

    MeshAdaptiveEvidence {
        iteration_count: mesh.adaptive_iterations.len(),
        latest_iteration_index: Some(latest.iteration_index),
        latest_convergence_status: Some(convergence_status_label(latest.convergence_status)),
        latest_indicator_count: latest.indicators.len(),
        latest_used_indicator_count: latest
            .indicators
            .iter()
            .filter(|indicator| indicator.status == RefinementIndicatorStatus::Used)
            .count(),
        latest_marker_count: latest.markers.len(),
        latest_sizing_update_sample_count: latest.sizing_update.samples.len(),
        marker_count,
        sizing_update_sample_count,
        latest_indicator_status_counts,
        latest_marker_by_reason,
        latest_sizing_update_by_reason,
        marker_by_reason,
        sizing_update_by_reason,
    }
}

fn convergence_status_label(status: AdaptiveConvergenceStatus) -> String {
    match status {
        AdaptiveConvergenceStatus::NotStarted => "not_started",
        AdaptiveConvergenceStatus::Disabled => "disabled",
        AdaptiveConvergenceStatus::Pending => "pending",
        AdaptiveConvergenceStatus::Converged => "converged",
        AdaptiveConvergenceStatus::MaxIterationsReached => "max_iterations_reached",
        AdaptiveConvergenceStatus::ElementBudgetReached => "element_budget_reached",
    }
    .to_string()
}

fn indicator_status_label(status: RefinementIndicatorStatus) -> String {
    match status {
        RefinementIndicatorStatus::Used => "used",
        RefinementIndicatorStatus::SkippedMissingField => "skipped_missing_field",
        RefinementIndicatorStatus::SkippedNotApplicable => "skipped_not_applicable",
        RefinementIndicatorStatus::SkippedBudget => "skipped_budget",
        RefinementIndicatorStatus::SkippedQuality => "skipped_quality",
    }
    .to_string()
}

fn sizing_evidence(mesh: &AnalysisMeshArtifact) -> MeshSizingEvidence {
    let mut generated_cad_by_reason = BTreeMap::<String, usize>::new();
    for sample in &mesh.sizing.samples {
        if let Some(reason) = sample
            .reason
            .as_deref()
            .filter(|reason| reason.starts_with("cad."))
        {
            *generated_cad_by_reason
                .entry(reason.to_string())
                .or_default() += 1;
        }
    }

    let mut anisotropic_by_reason = BTreeMap::<String, usize>::new();
    let mut invalid_anisotropic_by_reason = BTreeMap::<String, usize>::new();
    for sample in &mesh.sizing.anisotropic_samples {
        let reason = sample
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *anisotropic_by_reason.entry(reason.clone()).or_default() += 1;
        if !sample.is_valid_metric() {
            *invalid_anisotropic_by_reason.entry(reason).or_default() += 1;
        }
    }
    let invalid_anisotropic_sample_count = invalid_anisotropic_by_reason.values().sum::<usize>();

    let mut applied_by_reason = BTreeMap::<String, usize>::new();
    let mut inserted_breakpoint_by_reason = BTreeMap::<String, usize>::new();
    let mut uninserted_sample_by_reason = BTreeMap::<String, usize>::new();
    let mut inserted_breakpoint_count = 0_usize;
    for application in &mesh.sizing.applied_samples {
        let reason = application
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *applied_by_reason.entry(reason.clone()).or_default() += 1;
        if application.inserted_breakpoint_count > 0 {
            *inserted_breakpoint_by_reason.entry(reason).or_default() +=
                application.inserted_breakpoint_count;
        } else {
            *uninserted_sample_by_reason.entry(reason).or_default() += 1;
        }
        inserted_breakpoint_count += application.inserted_breakpoint_count;
    }

    let mut rejected_by_status = BTreeMap::<String, usize>::new();
    let mut rejected_by_reason = BTreeMap::<String, usize>::new();
    for rejection in &mesh.sizing.rejected_samples {
        *rejected_by_status
            .entry(rejection.status.clone())
            .or_default() += 1;
        let reason = rejection
            .reason
            .clone()
            .unwrap_or_else(|| "unspecified".to_string());
        *rejected_by_reason.entry(reason).or_default() += 1;
    }

    let accepted_requested_tetrahedron_refinement_point_count = mesh
        .backend
        .tetrahedron_accepted_requested_refinement_point_count;
    let accepted_requested_tetrahedron_refinement_location_count = mesh
        .backend
        .tetrahedron_accepted_requested_refinement_location_count;
    let accepted_requested_tetrahedron_refinement_surrogate_point_count = mesh
        .backend
        .tetrahedron_accepted_requested_refinement_surrogate_point_count;

    MeshSizingEvidence {
        global_target_size_m: mesh.sizing.global_target_size_m,
        min_size_m: mesh.sizing.min_size_m,
        max_size_m: mesh.sizing.max_size_m,
        growth_rate: mesh.sizing.growth_rate,
        sample_count: mesh.sizing.samples.len(),
        generated_cad_sample_count: generated_cad_by_reason.values().sum(),
        anisotropic_sample_count: mesh.sizing.anisotropic_samples.len(),
        valid_anisotropic_sample_count: mesh
            .sizing
            .anisotropic_samples
            .len()
            .saturating_sub(invalid_anisotropic_sample_count),
        invalid_anisotropic_sample_count,
        applied_sample_count: mesh.sizing.applied_samples.len(),
        rejected_sample_count: mesh.sizing.rejected_samples.len(),
        inserted_breakpoint_count,
        inserted_breakpoint_by_reason,
        uninserted_sample_by_reason,
        requested_tetrahedron_refinement_point_count: mesh
            .backend
            .tetrahedron_requested_refinement_point_count,
        accepted_requested_tetrahedron_refinement_location_count,
        accepted_requested_tetrahedron_refinement_point_count,
        accepted_requested_tetrahedron_refinement_surrogate_point_count,
        accepted_requested_tetrahedron_refinement_exact_point_count:
            accepted_requested_tetrahedron_refinement_point_count
                .saturating_sub(accepted_requested_tetrahedron_refinement_surrogate_point_count),
        rejected_requested_tetrahedron_refinement_point_count: mesh
            .backend
            .tetrahedron_rejected_requested_refinement_point_count,
        requested_tetrahedron_refinement_rejected_by_reason: mesh
            .backend
            .tetrahedron_requested_refinement_rejected_by_reason
            .clone(),
        dropped_requested_tetrahedron_refinement_point_count: mesh
            .backend
            .tetrahedron_dropped_requested_refinement_point_count,
        requested_tetrahedron_refinement_dropped_by_reason: mesh
            .backend
            .tetrahedron_requested_refinement_dropped_by_reason
            .clone(),
        requested_tetrahedron_refinement_acceptance_ratio: if mesh
            .backend
            .tetrahedron_requested_refinement_point_count
            > 0
        {
            Some(
                mesh.backend
                    .tetrahedron_accepted_requested_refinement_point_count as f64
                    / mesh.backend.tetrahedron_requested_refinement_point_count as f64,
            )
        } else {
            None
        },
        requested_tetrahedron_refinement_rejection_ratio: if mesh
            .backend
            .tetrahedron_requested_refinement_point_count
            > 0
        {
            Some(
                mesh.backend
                    .tetrahedron_rejected_requested_refinement_point_count as f64
                    / mesh.backend.tetrahedron_requested_refinement_point_count as f64,
            )
        } else {
            None
        },
        requested_tetrahedron_refinement_surrogate_ratio:
            if accepted_requested_tetrahedron_refinement_point_count > 0 {
                Some(
                    accepted_requested_tetrahedron_refinement_surrogate_point_count as f64
                        / accepted_requested_tetrahedron_refinement_point_count as f64,
                )
            } else {
                None
            },
        generated_cad_by_reason,
        anisotropic_by_reason,
        invalid_anisotropic_by_reason,
        applied_by_reason,
        rejected_by_status,
        rejected_by_reason,
    }
}

fn tetrahedron_recovery_evidence(mesh: &AnalysisMeshArtifact) -> MeshTetrahedronRecoveryEvidence {
    MeshTetrahedronRecoveryEvidence {
        element_count: mesh.backend.tetrahedron_element_count,
        recovered_component_ratio: mesh.backend.tetrahedron_recovered_component_ratio,
        fan_fallback_component_count: mesh.backend.tetrahedron_fan_fallback_component_count,
        volume_coverage_ratio: mesh.backend.tetrahedron_volume_coverage_ratio,
        refinement_pass_count: mesh.backend.tetrahedron_refinement_pass_count,
        refinement_point_count: mesh.backend.tetrahedron_refinement_point_count,
        optimization_pass_count: mesh.backend.tetrahedron_optimization_pass_count,
        smoothed_point_count: mesh.backend.tetrahedron_smoothed_point_count,
        sliver_count: mesh.backend.tetrahedron_sliver_count,
        sliver_removed_count: mesh.backend.tetrahedron_sliver_removed_count,
        optimization_target_seed_count: mesh.backend.tetrahedron_optimization_target_seed_count,
        optimization_skipped_target_seed_count: mesh
            .backend
            .tetrahedron_optimization_skipped_target_seed_count,
        optimization_rejected_edit_count: mesh.backend.tetrahedron_optimization_rejected_edit_count,
        optimization_initial_max_aspect_ratio: mesh
            .backend
            .tetrahedron_optimization_initial_max_aspect_ratio,
        optimization_final_max_aspect_ratio: mesh
            .backend
            .tetrahedron_optimization_final_max_aspect_ratio,
        optimization_initial_min_exact_scaled_jacobian: mesh
            .backend
            .tetrahedron_optimization_initial_min_exact_scaled_jacobian,
        optimization_final_min_exact_scaled_jacobian: mesh
            .backend
            .tetrahedron_optimization_final_min_exact_scaled_jacobian,
        untangling_pass_count: mesh.backend.tetrahedron_untangling_pass_count,
        untangling_initial_near_singular_count: mesh
            .backend
            .tetrahedron_untangling_initial_near_singular_count,
        untangling_final_near_singular_count: mesh
            .backend
            .tetrahedron_untangling_final_near_singular_count,
        untangling_relocated_seed_count: mesh.backend.tetrahedron_untangling_relocated_seed_count,
        untangling_reconnected_edge_star_count: mesh
            .backend
            .tetrahedron_untangling_reconnected_edge_star_count,
        untangling_reconnected_boundary_adjacent_cavity_count: mesh
            .backend
            .tetrahedron_untangling_reconnected_boundary_adjacent_cavity_count,
        untangling_reconnected_node_adjacent_cavity_count: mesh
            .backend
            .tetrahedron_untangling_reconnected_node_adjacent_cavity_count,
        exact_quality_repair_pass_count: mesh.backend.tetrahedron_exact_quality_repair_pass_count,
        exact_quality_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_reconnected_cavity_count,
        exact_quality_reconnection_quality_gain_count: mesh
            .backend
            .tetrahedron_exact_quality_reconnection_quality_gain_count,
        exact_quality_face_neighbor_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_face_neighbor_reconnected_cavity_count,
        exact_quality_connected_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_connected_reconnected_cavity_count,
        exact_quality_node_adjacent_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_node_adjacent_reconnected_cavity_count,
        exact_quality_boundary_adjacent_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_boundary_adjacent_reconnected_cavity_count,
        exact_quality_expanded_connected_reconnected_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_expanded_connected_reconnected_cavity_count,
        exact_quality_split_cavity_count: mesh.backend.tetrahedron_exact_quality_split_cavity_count,
        exact_quality_seed_star_collapse_count: mesh
            .backend
            .tetrahedron_exact_quality_seed_star_collapse_count,
        exact_quality_seed_star_relocation_count: mesh
            .backend
            .tetrahedron_exact_quality_seed_star_relocation_count,
        exact_quality_unrepaired_total_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_total_count,
        exact_quality_unrepaired_general_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_general_cavity_count,
        exact_quality_unrepaired_boundary_adjacent_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_boundary_adjacent_count,
        exact_quality_unrepaired_node_adjacent_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_node_adjacent_count,
        exact_quality_unrepaired_interior_seed_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_interior_seed_count,
        exact_quality_unrepaired_edge_star_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_edge_star_count,
    }
}

fn validation_evidence(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
) -> MeshValidationEvidence {
    let validation_result = validate_analysis_mesh_with_options(mesh, validation.clone());
    let (solve_ready, validation_error_code, validation_error_message) = match validation_result {
        Ok(()) => (true, None, None),
        Err(err) => (
            false,
            Some(analysis_mesh_validation_error_code(&err).to_string()),
            Some(format!("{err:?}")),
        ),
    };

    MeshValidationEvidence {
        solve_ready,
        validation_error_code,
        validation_error_message,
        quality: validation.quality,
        volume_element_count: mesh.volume_elements.len(),
        max_volume_element_count: validation.max_volume_element_count,
        volume_component_count: volume_component_count(mesh),
        volume_component_element_counts: volume_component_element_counts(mesh),
        max_volume_component_count: validation.max_volume_component_count,
        coverage_sample_count: finite_coverage_sample_count(&validation.coverage_sample_points_m),
        covered_coverage_sample_count: covered_coverage_sample_count(
            mesh,
            &validation.coverage_sample_points_m,
        ),
        coverage_sample_ratio: coverage_sample_ratio(mesh, &validation.coverage_sample_points_m),
        min_coverage_sample_ratio: validation.min_coverage_sample_ratio,
        coverage_sample_points_m: validation.coverage_sample_points_m.clone(),
        expected_bounds_m: validation.expected_bounds_m,
        min_bounds_coverage_ratio: validation.min_bounds_coverage_ratio,
        expected_volume_m3: validation.expected_volume_m3,
        min_volume_coverage_ratio: validation.min_volume_coverage_ratio,
        expected_boundary_area_m2: validation.expected_boundary_area_m2,
        min_boundary_area_ratio: validation.min_boundary_area_ratio,
        min_boundary_face_recovery_ratio: validation.min_boundary_face_recovery_ratio,
        min_boundary_edge_recovery_ratio: validation.min_boundary_edge_recovery_ratio,
        require_no_fan_fallback: validation.require_no_fan_fallback,
        require_no_unrepaired_exact_quality: validation.require_no_unrepaired_exact_quality,
        fan_fallback_component_count: mesh.backend.tetrahedron_fan_fallback_component_count,
        unrepaired_exact_quality_total_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_total_count,
        unrepaired_exact_quality_general_cavity_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_general_cavity_count,
        unrepaired_exact_quality_boundary_adjacent_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_boundary_adjacent_count,
        unrepaired_exact_quality_node_adjacent_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_node_adjacent_count,
        unrepaired_exact_quality_interior_seed_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_interior_seed_count,
        unrepaired_exact_quality_edge_star_count: mesh
            .backend
            .tetrahedron_exact_quality_unrepaired_edge_star_count,
        required_boundary_region_ids: validation.required_boundary_region_ids.clone(),
        required_material_region_ids: validation.required_material_region_ids.clone(),
        boundary_recovery: boundary_recovery_evidence(mesh),
    }
}

fn default_solve_ready() -> bool {
    true
}

fn default_min_coverage_sample_ratio() -> f64 {
    1.0
}

fn finite_coverage_sample_count(points: &[[f64; 3]]) -> usize {
    points
        .iter()
        .filter(|point| point.iter().all(|value| value.is_finite()))
        .count()
}

fn covered_coverage_sample_count(mesh: &AnalysisMeshArtifact, points: &[[f64; 3]]) -> usize {
    points
        .iter()
        .filter(|point| point.iter().all(|value| value.is_finite()))
        .filter(|point| mesh_contains_point(mesh, **point))
        .count()
}

fn coverage_sample_ratio(mesh: &AnalysisMeshArtifact, points: &[[f64; 3]]) -> Option<f64> {
    let finite_count = finite_coverage_sample_count(points);
    if finite_count == 0 {
        return None;
    }
    Some(covered_coverage_sample_count(mesh, points) as f64 / finite_count as f64)
}

fn boundary_recovery_evidence(mesh: &AnalysisMeshArtifact) -> MeshBoundaryRecoveryEvidence {
    MeshBoundaryRecoveryEvidence {
        boundary_face_recovery_ratio: boundary_face_recovery_ratio(mesh),
        boundary_edge_recovery_ratio: boundary_edge_recovery_ratio(mesh),
        recovered_boundary_face_count: mesh
            .boundary_faces
            .iter()
            .filter(|face| !face.adjacent_volume_element_ids.is_empty())
            .count(),
        recovered_boundary_edge_count: recovered_boundary_edge_count(mesh),
    }
}

fn mesh_bounds_m(mesh: &AnalysisMeshArtifact) -> Option<[[f64; 3]; 2]> {
    let mut iter = mesh.nodes.iter();
    let first = iter.next()?.coordinates_m;
    let mut min = first;
    let mut max = first;
    for node in iter {
        for axis in 0..3 {
            min[axis] = min[axis].min(node.coordinates_m[axis]);
            max[axis] = max[axis].max(node.coordinates_m[axis]);
        }
    }
    Some([min, max])
}

fn boundary_face_recovery_ratio(mesh: &AnalysisMeshArtifact) -> f64 {
    if mesh.boundary_faces.is_empty() {
        return 1.0;
    }
    mesh.boundary_faces
        .iter()
        .filter(|face| !face.adjacent_volume_element_ids.is_empty())
        .count() as f64
        / mesh.boundary_faces.len() as f64
}

fn boundary_edge_recovery_ratio(mesh: &AnalysisMeshArtifact) -> f64 {
    let expected_edges = boundary_face_edges(mesh);
    if expected_edges.is_empty() {
        return 1.0;
    }
    recovered_boundary_edge_count(mesh) as f64 / expected_edges.len() as f64
}

fn recovered_boundary_edge_count(mesh: &AnalysisMeshArtifact) -> usize {
    let expected_edges = boundary_face_edges(mesh);
    mesh.boundary_edges
        .iter()
        .filter(|edge| expected_edges.contains(&ordered_edge(edge.node_ids[0], edge.node_ids[1])))
        .count()
}

fn boundary_face_edges(mesh: &AnalysisMeshArtifact) -> BTreeSet<[u32; 2]> {
    let mut edges = BTreeSet::<[u32; 2]>::new();
    for face in &mesh.boundary_faces {
        if face.node_ids.len() != 3 {
            continue;
        }
        edges.insert(ordered_edge(face.node_ids[0], face.node_ids[1]));
        edges.insert(ordered_edge(face.node_ids[1], face.node_ids[2]));
        edges.insert(ordered_edge(face.node_ids[2], face.node_ids[0]));
    }
    edges
}

fn ordered_edge(left: u32, right: u32) -> [u32; 2] {
    if left <= right {
        [left, right]
    } else {
        [right, left]
    }
}

#[cfg(feature = "dev-evidence")]
pub mod dev_traces;
pub mod summaries;
#[cfg(test)]
mod tests;
