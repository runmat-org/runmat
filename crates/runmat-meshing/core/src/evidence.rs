use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    artifact::{AnalysisMeshArtifact, AnalysisVolumeElement, MeshBackendSummary},
    quality::QualityThresholds,
    topology::VolumeElementKind,
    validation::{
        analysis_mesh_validation_error_code, mesh_contains_point,
        validate_analysis_mesh_with_options, volume_component_count,
        volume_component_element_counts, AnalysisMeshValidationOptions,
    },
};

pub const MESH_EVIDENCE_SCHEMA_VERSION: &str = "mesh-evidence/v1";

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
    pub sizing: MeshSizingEvidence,
    pub quality: MeshQualityEvidence,
    #[serde(default)]
    pub tet_recovery: MeshTetRecoveryEvidence,
    pub regions: MeshRegionEvidence,
    pub validation: MeshValidationEvidence,
}

#[cfg(feature = "dev-evidence")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshDebugEvidence {
    pub event_cap: usize,
    pub event_count: usize,
    pub emitted_event_count: usize,
    pub truncated_event_count: usize,
    pub events: Vec<MeshDebugEvent>,
}

#[cfg(feature = "dev-evidence")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshDebugEvent {
    pub stage: String,
    pub severity: String,
    pub message: String,
}

#[cfg(feature = "dev-evidence")]
impl MeshDebugEvent {
    pub fn new(
        stage: impl Into<String>,
        severity: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage: stage.into(),
            severity: severity.into(),
            message: message.into(),
        }
    }
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
    pub requested_tet_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tet_refinement_candidate_count: usize,
    #[serde(default)]
    pub accepted_requested_tet_refinement_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tet_refinement_surrogate_point_count: usize,
    #[serde(default)]
    pub accepted_requested_tet_refinement_exact_point_count: usize,
    #[serde(default)]
    pub rejected_requested_tet_refinement_point_count: usize,
    #[serde(default)]
    pub requested_tet_refinement_rejected_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub dropped_requested_tet_refinement_point_count: usize,
    #[serde(default)]
    pub requested_tet_refinement_dropped_by_reason: BTreeMap<String, usize>,
    #[serde(default)]
    pub requested_tet_refinement_acceptance_ratio: Option<f64>,
    #[serde(default)]
    pub requested_tet_refinement_rejection_ratio: Option<f64>,
    #[serde(default)]
    pub requested_tet_refinement_surrogate_ratio: Option<f64>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshQualityEvidence {
    pub min_scaled_jacobian: f64,
    #[serde(default)]
    pub min_exact_scaled_jacobian: f64,
    #[serde(default)]
    pub scaled_jacobian_p05: Option<f64>,
    #[serde(default)]
    pub scaled_jacobian_p50: Option<f64>,
    #[serde(default)]
    pub scaled_jacobian_p95: Option<f64>,
    #[serde(default)]
    pub exact_scaled_jacobian_p05: Option<f64>,
    #[serde(default)]
    pub exact_scaled_jacobian_p50: Option<f64>,
    #[serde(default)]
    pub exact_scaled_jacobian_p95: Option<f64>,
    pub mean_aspect_ratio: f64,
    pub max_aspect_ratio: f64,
    #[serde(default)]
    pub aspect_ratio_p50: Option<f64>,
    #[serde(default)]
    pub aspect_ratio_p95: Option<f64>,
    pub inverted_element_count: usize,
    pub mean_boundary_projection_error_m: f64,
    pub max_boundary_projection_error_m: f64,
    pub element_quality_sample_count: usize,
    pub scaled_jacobian_bins: BTreeMap<String, usize>,
    #[serde(default)]
    pub exact_scaled_jacobian_bins: BTreeMap<String, usize>,
    pub aspect_ratio_bins: BTreeMap<String, usize>,
    pub volume_bins: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MeshTetRecoveryEvidence {
    pub candidate_count: usize,
    pub recovered_component_ratio: f64,
    pub fan_fallback_component_count: usize,
    pub candidate_volume_ratio: f64,
    pub refinement_pass_count: usize,
    pub refinement_point_count: usize,
    pub optimization_pass_count: usize,
    pub smoothed_point_count: usize,
    pub sliver_candidate_count: usize,
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
    pub exact_quality_repair_pass_count: usize,
    pub exact_quality_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_reconnection_quality_gain_count: usize,
    #[serde(default)]
    pub exact_quality_face_neighbor_reconnected_cavity_count: usize,
    #[serde(default)]
    pub exact_quality_connected_reconnected_cavity_count: usize,
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
    pub exact_quality_unrepaired_interior_seed_count: usize,
    #[serde(default)]
    pub exact_quality_unrepaired_edge_star_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshRegionEvidence {
    pub material_region_element_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub material_region_volume_m3: BTreeMap<String, f64>,
    pub boundary_region_face_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub boundary_region_recovered_face_counts: BTreeMap<String, usize>,
    pub boundary_region_edge_counts: BTreeMap<String, usize>,
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

#[cfg(feature = "dev-evidence")]
pub fn build_mesh_evidence_artifact_with_debug(
    mesh: &AnalysisMeshArtifact,
    validation: &AnalysisMeshValidationOptions,
    debug_events: Vec<MeshDebugEvent>,
    event_cap: usize,
) -> MeshEvidenceArtifact {
    let mut artifact = build_mesh_evidence_artifact(mesh, validation);
    artifact.debug = Some(debug_evidence(debug_events, event_cap));
    artifact
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
        sizing: sizing_evidence(mesh),
        quality: quality_evidence(mesh),
        tet_recovery: tet_recovery_evidence(mesh),
        regions: region_evidence(mesh),
        validation,
    }
}

#[cfg(feature = "dev-evidence")]
fn debug_evidence(mut events: Vec<MeshDebugEvent>, event_cap: usize) -> MeshDebugEvidence {
    let event_count = events.len();
    let cap = event_cap.min(event_count);
    events.truncate(cap);
    MeshDebugEvidence {
        event_cap,
        event_count,
        emitted_event_count: events.len(),
        truncated_event_count: event_count.saturating_sub(events.len()),
        events,
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

    let accepted_requested_tet_refinement_point_count =
        mesh.backend.tet_accepted_requested_refinement_point_count;
    let accepted_requested_tet_refinement_candidate_count = mesh
        .backend
        .tet_accepted_requested_refinement_candidate_count;
    let accepted_requested_tet_refinement_surrogate_point_count = mesh
        .backend
        .tet_accepted_requested_refinement_surrogate_point_count;

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
        requested_tet_refinement_point_count: mesh.backend.tet_requested_refinement_point_count,
        accepted_requested_tet_refinement_candidate_count,
        accepted_requested_tet_refinement_point_count,
        accepted_requested_tet_refinement_surrogate_point_count,
        accepted_requested_tet_refinement_exact_point_count:
            accepted_requested_tet_refinement_point_count
                .saturating_sub(accepted_requested_tet_refinement_surrogate_point_count),
        rejected_requested_tet_refinement_point_count: mesh
            .backend
            .tet_rejected_requested_refinement_point_count,
        requested_tet_refinement_rejected_by_reason: mesh
            .backend
            .tet_requested_refinement_rejected_by_reason
            .clone(),
        dropped_requested_tet_refinement_point_count: mesh
            .backend
            .tet_dropped_requested_refinement_point_count,
        requested_tet_refinement_dropped_by_reason: mesh
            .backend
            .tet_requested_refinement_dropped_by_reason
            .clone(),
        requested_tet_refinement_acceptance_ratio: if mesh
            .backend
            .tet_requested_refinement_point_count
            > 0
        {
            Some(
                mesh.backend.tet_accepted_requested_refinement_point_count as f64
                    / mesh.backend.tet_requested_refinement_point_count as f64,
            )
        } else {
            None
        },
        requested_tet_refinement_rejection_ratio: if mesh
            .backend
            .tet_requested_refinement_point_count
            > 0
        {
            Some(
                mesh.backend.tet_rejected_requested_refinement_point_count as f64
                    / mesh.backend.tet_requested_refinement_point_count as f64,
            )
        } else {
            None
        },
        requested_tet_refinement_surrogate_ratio: if accepted_requested_tet_refinement_point_count
            > 0
        {
            Some(
                accepted_requested_tet_refinement_surrogate_point_count as f64
                    / accepted_requested_tet_refinement_point_count as f64,
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

fn quality_evidence(mesh: &AnalysisMeshArtifact) -> MeshQualityEvidence {
    let mut scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut exact_scaled_jacobian_bins = BTreeMap::<String, usize>::new();
    let mut aspect_ratio_bins = BTreeMap::<String, usize>::new();
    let mut volume_bins = BTreeMap::<String, usize>::new();
    let mut scaled_jacobians = Vec::<f64>::new();
    let mut exact_scaled_jacobians = Vec::<f64>::new();
    let mut aspect_ratios = Vec::<f64>::new();
    for element in &mesh.quality.elements {
        *scaled_jacobian_bins
            .entry(scaled_jacobian_bin(element.scaled_jacobian))
            .or_default() += 1;
        *exact_scaled_jacobian_bins
            .entry(scaled_jacobian_bin(element.exact_scaled_jacobian))
            .or_default() += 1;
        *aspect_ratio_bins
            .entry(aspect_ratio_bin(element.aspect_ratio))
            .or_default() += 1;
        *volume_bins
            .entry(volume_bin(element.volume_m3))
            .or_default() += 1;
        if element.scaled_jacobian.is_finite() {
            scaled_jacobians.push(element.scaled_jacobian);
        }
        if element.exact_scaled_jacobian.is_finite() {
            exact_scaled_jacobians.push(element.exact_scaled_jacobian);
        }
        if element.aspect_ratio.is_finite() {
            aspect_ratios.push(element.aspect_ratio);
        }
    }
    scaled_jacobians.sort_by(f64::total_cmp);
    exact_scaled_jacobians.sort_by(f64::total_cmp);
    aspect_ratios.sort_by(f64::total_cmp);

    MeshQualityEvidence {
        min_scaled_jacobian: mesh.quality.min_scaled_jacobian,
        min_exact_scaled_jacobian: mesh.quality.min_exact_scaled_jacobian,
        scaled_jacobian_p05: percentile(&scaled_jacobians, 0.05),
        scaled_jacobian_p50: percentile(&scaled_jacobians, 0.50),
        scaled_jacobian_p95: percentile(&scaled_jacobians, 0.95),
        exact_scaled_jacobian_p05: percentile(&exact_scaled_jacobians, 0.05),
        exact_scaled_jacobian_p50: percentile(&exact_scaled_jacobians, 0.50),
        exact_scaled_jacobian_p95: percentile(&exact_scaled_jacobians, 0.95),
        mean_aspect_ratio: mesh.quality.mean_aspect_ratio,
        max_aspect_ratio: mesh.quality.max_aspect_ratio,
        aspect_ratio_p50: percentile(&aspect_ratios, 0.50),
        aspect_ratio_p95: percentile(&aspect_ratios, 0.95),
        inverted_element_count: mesh.quality.inverted_element_count,
        mean_boundary_projection_error_m: mesh.quality.mean_boundary_projection_error_m,
        max_boundary_projection_error_m: mesh.quality.max_boundary_projection_error_m,
        element_quality_sample_count: mesh.quality.elements.len(),
        scaled_jacobian_bins,
        exact_scaled_jacobian_bins,
        aspect_ratio_bins,
        volume_bins,
    }
}

fn tet_recovery_evidence(mesh: &AnalysisMeshArtifact) -> MeshTetRecoveryEvidence {
    MeshTetRecoveryEvidence {
        candidate_count: mesh.backend.tet_candidate_count,
        recovered_component_ratio: mesh.backend.tet_recovered_component_ratio,
        fan_fallback_component_count: mesh.backend.tet_fan_fallback_component_count,
        candidate_volume_ratio: mesh.backend.tet_candidate_volume_ratio,
        refinement_pass_count: mesh.backend.tet_refinement_pass_count,
        refinement_point_count: mesh.backend.tet_refinement_point_count,
        optimization_pass_count: mesh.backend.tet_optimization_pass_count,
        smoothed_point_count: mesh.backend.tet_smoothed_point_count,
        sliver_candidate_count: mesh.backend.tet_sliver_candidate_count,
        sliver_removed_count: mesh.backend.tet_sliver_removed_count,
        optimization_target_seed_count: mesh.backend.tet_optimization_target_seed_count,
        optimization_skipped_target_seed_count: mesh
            .backend
            .tet_optimization_skipped_target_seed_count,
        optimization_rejected_edit_count: mesh.backend.tet_optimization_rejected_edit_count,
        optimization_initial_max_aspect_ratio: mesh
            .backend
            .tet_optimization_initial_max_aspect_ratio,
        optimization_final_max_aspect_ratio: mesh.backend.tet_optimization_final_max_aspect_ratio,
        optimization_initial_min_exact_scaled_jacobian: mesh
            .backend
            .tet_optimization_initial_min_exact_scaled_jacobian,
        optimization_final_min_exact_scaled_jacobian: mesh
            .backend
            .tet_optimization_final_min_exact_scaled_jacobian,
        untangling_pass_count: mesh.backend.tet_untangling_pass_count,
        untangling_initial_near_singular_count: mesh
            .backend
            .tet_untangling_initial_near_singular_count,
        untangling_final_near_singular_count: mesh.backend.tet_untangling_final_near_singular_count,
        untangling_relocated_seed_count: mesh.backend.tet_untangling_relocated_seed_count,
        untangling_reconnected_edge_star_count: mesh
            .backend
            .tet_untangling_reconnected_edge_star_count,
        untangling_reconnected_boundary_adjacent_cavity_count: mesh
            .backend
            .tet_untangling_reconnected_boundary_adjacent_cavity_count,
        exact_quality_repair_pass_count: mesh.backend.tet_exact_quality_repair_pass_count,
        exact_quality_reconnected_cavity_count: mesh
            .backend
            .tet_exact_quality_reconnected_cavity_count,
        exact_quality_reconnection_quality_gain_count: mesh
            .backend
            .tet_exact_quality_reconnection_quality_gain_count,
        exact_quality_face_neighbor_reconnected_cavity_count: mesh
            .backend
            .tet_exact_quality_face_neighbor_reconnected_cavity_count,
        exact_quality_connected_reconnected_cavity_count: mesh
            .backend
            .tet_exact_quality_connected_reconnected_cavity_count,
        exact_quality_boundary_adjacent_reconnected_cavity_count: mesh
            .backend
            .tet_exact_quality_boundary_adjacent_reconnected_cavity_count,
        exact_quality_expanded_connected_reconnected_cavity_count: mesh
            .backend
            .tet_exact_quality_expanded_connected_reconnected_cavity_count,
        exact_quality_split_cavity_count: mesh.backend.tet_exact_quality_split_cavity_count,
        exact_quality_seed_star_collapse_count: mesh
            .backend
            .tet_exact_quality_seed_star_collapse_count,
        exact_quality_seed_star_relocation_count: mesh
            .backend
            .tet_exact_quality_seed_star_relocation_count,
        exact_quality_unrepaired_total_count: mesh.backend.tet_exact_quality_unrepaired_total_count,
        exact_quality_unrepaired_general_cavity_count: mesh
            .backend
            .tet_exact_quality_unrepaired_general_cavity_count,
        exact_quality_unrepaired_boundary_adjacent_count: mesh
            .backend
            .tet_exact_quality_unrepaired_boundary_adjacent_count,
        exact_quality_unrepaired_interior_seed_count: mesh
            .backend
            .tet_exact_quality_unrepaired_interior_seed_count,
        exact_quality_unrepaired_edge_star_count: mesh
            .backend
            .tet_exact_quality_unrepaired_edge_star_count,
    }
}

fn percentile(sorted_values: &[f64], ratio: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let ratio = ratio.clamp(0.0, 1.0);
    let index = ((sorted_values.len() - 1) as f64 * ratio).round() as usize;
    sorted_values.get(index).copied()
}

fn region_evidence(mesh: &AnalysisMeshArtifact) -> MeshRegionEvidence {
    let mut material_region_element_counts = BTreeMap::<String, usize>::new();
    let mut material_region_volume_m3 = BTreeMap::<String, f64>::new();
    for element in &mesh.volume_elements {
        *material_region_element_counts
            .entry(element.material_region_id.clone())
            .or_default() += 1;
        let volume_m3 = element_volume_m3(mesh, element);
        if volume_m3.is_finite() && volume_m3 > 0.0 {
            *material_region_volume_m3
                .entry(element.material_region_id.clone())
                .or_default() += volume_m3;
        }
    }

    let mut boundary_region_face_counts = BTreeMap::<String, usize>::new();
    let mut boundary_region_recovered_face_counts = BTreeMap::<String, usize>::new();
    for face in &mesh.boundary_faces {
        for region_id in &face.region_ids {
            *boundary_region_face_counts
                .entry(region_id.clone())
                .or_default() += 1;
            if !face.adjacent_volume_element_ids.is_empty() {
                *boundary_region_recovered_face_counts
                    .entry(region_id.clone())
                    .or_default() += 1;
            }
        }
    }

    let mut boundary_region_edge_counts = BTreeMap::<String, usize>::new();
    for edge in &mesh.boundary_edges {
        for region_id in &edge.region_ids {
            *boundary_region_edge_counts
                .entry(region_id.clone())
                .or_default() += 1;
        }
    }

    MeshRegionEvidence {
        material_region_element_counts,
        material_region_volume_m3,
        boundary_region_face_counts,
        boundary_region_recovered_face_counts,
        boundary_region_edge_counts,
    }
}

fn element_volume_m3(mesh: &AnalysisMeshArtifact, element: &AnalysisVolumeElement) -> f64 {
    if element.kind != VolumeElementKind::Tet4 || element.node_ids.len() != 4 {
        return 0.0;
    }
    let Some(points) = element_tet_points(mesh, element.node_ids.as_slice()) else {
        return 0.0;
    };
    tet_volume_m3(points)
}

fn element_tet_points(mesh: &AnalysisMeshArtifact, node_ids: &[u32]) -> Option<[[f64; 3]; 4]> {
    Some([
        mesh_node(mesh, node_ids[0])?,
        mesh_node(mesh, node_ids[1])?,
        mesh_node(mesh, node_ids[2])?,
        mesh_node(mesh, node_ids[3])?,
    ])
}

fn mesh_node(mesh: &AnalysisMeshArtifact, node_id: u32) -> Option<[f64; 3]> {
    mesh.nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
}

fn tet_volume_m3(points: [[f64; 3]; 4]) -> f64 {
    let ab = [
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    ];
    let ac = [
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
        points[2][2] - points[0][2],
    ];
    let ad = [
        points[3][0] - points[0][0],
        points[3][1] - points[0][1],
        points[3][2] - points[0][2],
    ];
    let cross = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    ((ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0).abs()
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
        fan_fallback_component_count: mesh.backend.tet_fan_fallback_component_count,
        unrepaired_exact_quality_total_count: mesh.backend.tet_exact_quality_unrepaired_total_count,
        unrepaired_exact_quality_general_cavity_count: mesh
            .backend
            .tet_exact_quality_unrepaired_general_cavity_count,
        unrepaired_exact_quality_boundary_adjacent_count: mesh
            .backend
            .tet_exact_quality_unrepaired_boundary_adjacent_count,
        unrepaired_exact_quality_interior_seed_count: mesh
            .backend
            .tet_exact_quality_unrepaired_interior_seed_count,
        unrepaired_exact_quality_edge_star_count: mesh
            .backend
            .tet_exact_quality_unrepaired_edge_star_count,
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

fn scaled_jacobian_bin(value: f64) -> String {
    if value < 0.0 {
        "lt_0".to_string()
    } else if value < 0.15 {
        "0_to_0_15".to_string()
    } else if value < 0.35 {
        "0_15_to_0_35".to_string()
    } else if value < 0.65 {
        "0_35_to_0_65".to_string()
    } else {
        "gte_0_65".to_string()
    }
}

fn aspect_ratio_bin(value: f64) -> String {
    if value < 2.0 {
        "lt_2".to_string()
    } else if value < 5.0 {
        "2_to_5".to_string()
    } else if value < 10.0 {
        "5_to_10".to_string()
    } else if value < 20.0 {
        "10_to_20".to_string()
    } else {
        "gte_20".to_string()
    }
}

fn volume_bin(value: f64) -> String {
    if value <= 0.0 {
        "lte_0".to_string()
    } else if value < 1.0e-12 {
        "lt_1e-12".to_string()
    } else if value < 1.0e-9 {
        "1e-12_to_1e-9".to_string()
    } else if value < 1.0e-6 {
        "1e-9_to_1e-6".to_string()
    } else {
        "gte_1e-6".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::{
            AnalysisBoundaryEdge, AnalysisBoundaryFace, AnalysisMeshNode, AnalysisVolumeElement,
            ANALYSIS_MESH_SCHEMA_VERSION,
        },
        provenance::AnalysisMeshProvenance,
        quality::{AnalysisMeshQualityReport, ElementQuality},
        sizing::{
            AnisotropicSizingSample, MeshSizingField, SizingSample, SizingSampleApplication,
            SizingSampleRejection,
        },
        topology::{BoundaryElementKind, VolumeElementKind},
    };

    #[test]
    fn evidence_summarizes_mesh_without_raw_sizing_samples() {
        let mesh = AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "mesh_1".to_string(),
            nodes: vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
            ],
            volume_elements: vec![AnalysisVolumeElement {
                element_id: "tet_1".to_string(),
                kind: VolumeElementKind::Tet4,
                node_ids: vec![1, 2, 3, 4],
                material_region_id: "solid".to_string(),
                provenance: Vec::new(),
            }],
            boundary_faces: vec![AnalysisBoundaryFace {
                face_id: "face_1".to_string(),
                kind: BoundaryElementKind::Tri3,
                node_ids: vec![1, 2, 3],
                adjacent_volume_element_ids: vec!["tet_1".to_string()],
                region_ids: vec!["fixed".to_string()],
                provenance: Vec::new(),
            }],
            boundary_edges: vec![
                boundary_edge("edge_1", [1, 2]),
                boundary_edge("edge_2", [2, 3]),
                boundary_edge("edge_3", [1, 3]),
            ],
            quality: AnalysisMeshQualityReport {
                min_scaled_jacobian: 0.5,
                min_exact_scaled_jacobian: 0.45,
                mean_aspect_ratio: 2.0,
                max_aspect_ratio: 2.0,
                inverted_element_count: 0,
                mean_boundary_projection_error_m: 0.0,
                max_boundary_projection_error_m: 0.0,
                elements: vec![ElementQuality {
                    element_id: "tet_1".to_string(),
                    scaled_jacobian: 0.5,
                    exact_scaled_jacobian: 0.45,
                    aspect_ratio: 2.0,
                    volume_m3: 1.0 / 6.0,
                }],
            },
            sizing: MeshSizingField {
                growth_rate: Some(1.4),
                samples: vec![
                    SizingSample {
                        position_m: [0.0, 0.0, 0.0],
                        target_size_m: 0.25,
                        reason: Some("load_region".to_string()),
                    },
                    SizingSample {
                        position_m: [0.5, 0.0, 0.0],
                        target_size_m: 0.2,
                        reason: Some("cad.curvature".to_string()),
                    },
                    SizingSample {
                        position_m: [0.0, 0.5, 0.0],
                        target_size_m: 0.15,
                        reason: Some("cad.interface".to_string()),
                    },
                ],
                anisotropic_samples: vec![
                    AnisotropicSizingSample {
                        position_m: [0.2, 0.2, 0.2],
                        target_sizes_m: [0.02, 0.04, 0.08],
                        directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        reason: Some("boundary_layer".to_string()),
                    },
                    AnisotropicSizingSample {
                        position_m: [0.3, 0.2, 0.2],
                        target_sizes_m: [0.02, -0.04, 0.08],
                        directions: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        reason: Some("cad.proximity".to_string()),
                    },
                ],
                applied_samples: vec![
                    SizingSampleApplication {
                        position_m: [0.0, 0.0, 0.0],
                        target_size_m: 0.25,
                        inserted_breakpoint_count: 2,
                        reason: Some("load_region".to_string()),
                        detail: Some("sample detail should not be copied".to_string()),
                    },
                    SizingSampleApplication {
                        position_m: [0.5, 0.0, 0.0],
                        target_size_m: 0.2,
                        inserted_breakpoint_count: 0,
                        reason: Some("cad.curvature".to_string()),
                        detail: Some("sample detail should not be copied".to_string()),
                    },
                ],
                rejected_samples: vec![SizingSampleRejection {
                    position_m: [0.1, 0.0, 0.0],
                    target_size_m: 0.1,
                    status: "outside_bounds".to_string(),
                    reason: Some("adaptive".to_string()),
                    detail: Some("rejection detail should not be copied".to_string()),
                }],
                ..MeshSizingField::default()
            },
            backend: MeshBackendSummary {
                backend: "production".to_string(),
                cad_topology_source: "semantic_cad".to_string(),
                cad_evaluation_source: "imported_evaluator_samples".to_string(),
                cad_vertex_count: 4,
                cad_edge_count: 6,
                cad_face_count: 4,
                cad_shell_count: 1,
                cad_volume_count: 1,
                cad_imported_face_count: 3,
                cad_evaluation_evaluator_face_count: 2,
                cad_evaluation_live_query_face_count: 0,
                cad_evaluation_exact_query_face_count: 1,
                cad_evaluation_missing_exact_query_face_count: 1,
                cad_evaluation_point_supported_face_count: 2,
                cad_evaluation_projection_supported_face_count: 2,
                cad_evaluation_normal_supported_face_count: 2,
                cad_evaluation_derivative_supported_face_count: 2,
                cad_evaluation_curvature_supported_face_count: 1,
                cad_evaluation_sample_count: 8,
                cad_evaluation_rejected_sample_count: 9,
                cad_projection_query_count: 12,
                cad_derivative_query_count: 6,
                cad_curvature_query_count: 5,
                cad_uv_domain_face_count: 10,
                cad_uv_projection_out_of_bounds_count: 2,
                cad_max_projection_error_m: 2.0e-6,
                cad_max_normal_deviation: 1.0e-5,
                cad_max_curvature_estimate_1_per_m: 0.125,
                surface_cad_face_count: 3,
                surface_exact_cad_sample_node_count: 4,
                surface_rejected_exact_cad_sample_count: 5,
                surface_max_cad_projection_error_m: 3.0e-6,
                tet_candidate_count: 12,
                tet_recovered_component_ratio: 1.0,
                tet_fan_fallback_component_count: 0,
                tet_candidate_volume_ratio: 0.99,
                tet_refinement_pass_count: 2,
                tet_refinement_point_count: 5,
                tet_requested_refinement_point_count: 5,
                tet_accepted_requested_refinement_candidate_count: 5,
                tet_accepted_requested_refinement_point_count: 3,
                tet_accepted_requested_refinement_surrogate_point_count: 2,
                tet_rejected_requested_refinement_point_count: 1,
                tet_requested_refinement_rejected_by_reason: BTreeMap::from([(
                    "quality_or_recovery".to_string(),
                    1,
                )]),
                tet_dropped_requested_refinement_point_count: 2,
                tet_requested_refinement_dropped_by_reason: BTreeMap::from([(
                    "not_retained_after_repair".to_string(),
                    2,
                )]),
                tet_optimization_pass_count: 1,
                tet_smoothed_point_count: 2,
                tet_sliver_candidate_count: 1,
                tet_sliver_removed_count: 2,
                tet_optimization_target_seed_count: 7,
                tet_optimization_skipped_target_seed_count: 4,
                tet_optimization_rejected_edit_count: 3,
                tet_optimization_initial_max_aspect_ratio: 6.0,
                tet_optimization_final_max_aspect_ratio: 4.0,
                tet_optimization_initial_min_exact_scaled_jacobian: 0.32,
                tet_optimization_final_min_exact_scaled_jacobian: 0.40,
                tet_untangling_pass_count: 2,
                tet_untangling_initial_near_singular_count: 6,
                tet_untangling_final_near_singular_count: 1,
                tet_untangling_relocated_seed_count: 3,
                tet_untangling_reconnected_edge_star_count: 4,
                tet_untangling_reconnected_boundary_adjacent_cavity_count: 5,
                tet_exact_quality_repair_pass_count: 1,
                tet_exact_quality_reconnected_cavity_count: 2,
                tet_exact_quality_reconnection_quality_gain_count: 1,
                tet_exact_quality_face_neighbor_reconnected_cavity_count: 6,
                tet_exact_quality_connected_reconnected_cavity_count: 7,
                tet_exact_quality_boundary_adjacent_reconnected_cavity_count: 8,
                tet_exact_quality_expanded_connected_reconnected_cavity_count: 9,
                tet_exact_quality_split_cavity_count: 3,
                tet_exact_quality_seed_star_collapse_count: 4,
                tet_exact_quality_seed_star_relocation_count: 5,
                tet_exact_quality_unrepaired_total_count: 9,
                tet_exact_quality_unrepaired_general_cavity_count: 1,
                tet_exact_quality_unrepaired_boundary_adjacent_count: 6,
                tet_exact_quality_unrepaired_interior_seed_count: 7,
                tet_exact_quality_unrepaired_edge_star_count: 8,
                ..MeshBackendSummary::default()
            },
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        };

        let validation = AnalysisMeshValidationOptions {
            max_volume_element_count: Some(7),
            max_volume_component_count: Some(1),
            coverage_sample_points_m: vec![[0.1, 0.1, 0.1]],
            min_coverage_sample_ratio: 1.0,
            require_no_fan_fallback: true,
            ..AnalysisMeshValidationOptions::default()
        };
        let evidence = build_mesh_evidence_artifact(&mesh, &validation);

        assert_eq!(evidence.schema_version, MESH_EVIDENCE_SCHEMA_VERSION);
        assert!(evidence.validation.solve_ready);
        assert_eq!(evidence.validation.validation_error_code, None);
        assert_eq!(evidence.validation.validation_error_message, None);
        assert_eq!(evidence.cad.topology_source, "semantic_cad");
        assert_eq!(evidence.cad.evaluation_source, "imported_evaluator_samples");
        assert_eq!(evidence.cad.imported_face_count, 3);
        assert_eq!(evidence.cad.exact_query_face_count, 1);
        assert_eq!(evidence.cad.missing_exact_query_face_count, 1);
        assert_eq!(evidence.cad.point_evaluation_supported_face_count, 2);
        assert_eq!(evidence.cad.projection_supported_face_count, 2);
        assert_eq!(evidence.cad.normal_supported_face_count, 2);
        assert_eq!(evidence.cad.derivative_supported_face_count, 2);
        assert_eq!(evidence.cad.curvature_supported_face_count, 1);
        assert_eq!(evidence.cad.evaluator_sample_count, 8);
        assert_eq!(evidence.cad.evaluator_rejected_sample_count, 9);
        assert_eq!(evidence.cad.projection_query_count, 12);
        assert_eq!(evidence.cad.derivative_query_count, 6);
        assert_eq!(evidence.cad.curvature_query_count, 5);
        assert_eq!(evidence.cad.uv_domain_face_count, 10);
        assert_eq!(evidence.cad.uv_projection_out_of_bounds_count, 2);
        assert_eq!(evidence.cad.max_projection_error_m, 2.0e-6);
        assert_eq!(evidence.cad.max_normal_deviation, 1.0e-5);
        assert_eq!(evidence.cad.max_curvature_estimate_1_per_m, 0.125);
        assert_eq!(evidence.cad.surface_max_projection_error_m, 3.0e-6);
        assert_eq!(evidence.cad.surface_exact_cad_sample_node_count, 4);
        assert_eq!(evidence.cad.surface_rejected_exact_cad_sample_count, 5);
        assert_eq!(evidence.topology.node_count, 4);
        assert_eq!(evidence.validation.volume_element_count, 1);
        assert_eq!(evidence.validation.max_volume_element_count, Some(7));
        assert_eq!(evidence.validation.volume_component_count, 1);
        assert_eq!(evidence.validation.volume_component_element_counts, vec![1]);
        assert_eq!(evidence.validation.max_volume_component_count, Some(1));
        assert_eq!(evidence.validation.coverage_sample_count, 1);
        assert_eq!(evidence.validation.covered_coverage_sample_count, 1);
        assert_eq!(evidence.validation.coverage_sample_ratio, Some(1.0));
        assert_eq!(evidence.validation.min_coverage_sample_ratio, 1.0);
        assert_eq!(
            evidence.validation.coverage_sample_points_m,
            vec![[0.1, 0.1, 0.1]]
        );
        assert!(evidence.validation.require_no_fan_fallback);
        assert!(!evidence.validation.require_no_unrepaired_exact_quality);
        assert_eq!(evidence.validation.fan_fallback_component_count, 0);
        assert_eq!(evidence.validation.unrepaired_exact_quality_total_count, 9);
        assert_eq!(
            evidence
                .validation
                .unrepaired_exact_quality_general_cavity_count,
            1
        );
        assert_eq!(
            evidence
                .validation
                .unrepaired_exact_quality_boundary_adjacent_count,
            6
        );
        assert_eq!(
            evidence
                .validation
                .unrepaired_exact_quality_interior_seed_count,
            7
        );
        assert_eq!(
            evidence.validation.unrepaired_exact_quality_edge_star_count,
            8
        );
        assert_eq!(evidence.sizing.inserted_breakpoint_count, 2);
        assert_eq!(evidence.sizing.requested_tet_refinement_point_count, 5);
        assert_eq!(
            evidence
                .sizing
                .accepted_requested_tet_refinement_candidate_count,
            5
        );
        assert_eq!(
            evidence
                .sizing
                .accepted_requested_tet_refinement_point_count,
            3
        );
        assert_eq!(
            evidence
                .sizing
                .rejected_requested_tet_refinement_point_count,
            1
        );
        assert_eq!(
            evidence
                .sizing
                .requested_tet_refinement_rejected_by_reason
                .get("quality_or_recovery"),
            Some(&1)
        );
        assert_eq!(
            evidence.sizing.dropped_requested_tet_refinement_point_count,
            2
        );
        assert_eq!(
            evidence
                .sizing
                .requested_tet_refinement_dropped_by_reason
                .get("not_retained_after_repair"),
            Some(&2)
        );
        assert_eq!(
            evidence
                .sizing
                .accepted_requested_tet_refinement_surrogate_point_count,
            2
        );
        assert_eq!(
            evidence
                .sizing
                .accepted_requested_tet_refinement_exact_point_count,
            1
        );
        assert_eq!(
            evidence.sizing.requested_tet_refinement_acceptance_ratio,
            Some(0.6)
        );
        assert_eq!(
            evidence.sizing.requested_tet_refinement_rejection_ratio,
            Some(0.2)
        );
        assert_eq!(
            evidence.sizing.requested_tet_refinement_surrogate_ratio,
            Some(2.0 / 3.0)
        );
        assert_eq!(evidence.sizing.sample_count, 3);
        assert_eq!(evidence.sizing.generated_cad_sample_count, 2);
        assert_eq!(evidence.sizing.anisotropic_sample_count, 2);
        assert_eq!(evidence.sizing.valid_anisotropic_sample_count, 1);
        assert_eq!(evidence.sizing.invalid_anisotropic_sample_count, 1);
        assert_eq!(
            evidence.sizing.anisotropic_by_reason.get("boundary_layer"),
            Some(&1)
        );
        assert_eq!(
            evidence
                .sizing
                .invalid_anisotropic_by_reason
                .get("cad.proximity"),
            Some(&1)
        );
        assert_eq!(
            evidence.sizing.generated_cad_by_reason.get("cad.curvature"),
            Some(&1)
        );
        assert_eq!(
            evidence.sizing.generated_cad_by_reason.get("cad.interface"),
            Some(&1)
        );
        assert_eq!(
            evidence.sizing.applied_by_reason.get("load_region"),
            Some(&1)
        );
        assert_eq!(
            evidence
                .sizing
                .inserted_breakpoint_by_reason
                .get("load_region"),
            Some(&2)
        );
        assert_eq!(
            evidence
                .sizing
                .uninserted_sample_by_reason
                .get("cad.curvature"),
            Some(&1)
        );
        assert_eq!(evidence.tet_recovery.candidate_count, 12);
        assert_eq!(evidence.tet_recovery.recovered_component_ratio, 1.0);
        assert_eq!(evidence.tet_recovery.candidate_volume_ratio, 0.99);
        assert_eq!(evidence.tet_recovery.refinement_pass_count, 2);
        assert_eq!(evidence.tet_recovery.refinement_point_count, 5);
        assert_eq!(evidence.tet_recovery.optimization_pass_count, 1);
        assert_eq!(evidence.tet_recovery.smoothed_point_count, 2);
        assert_eq!(evidence.tet_recovery.sliver_candidate_count, 1);
        assert_eq!(evidence.tet_recovery.sliver_removed_count, 2);
        assert_eq!(evidence.tet_recovery.optimization_target_seed_count, 7);
        assert_eq!(
            evidence.tet_recovery.optimization_skipped_target_seed_count,
            4
        );
        assert_eq!(evidence.tet_recovery.optimization_rejected_edit_count, 3);
        assert_eq!(
            evidence.tet_recovery.optimization_initial_max_aspect_ratio,
            6.0
        );
        assert_eq!(
            evidence.tet_recovery.optimization_final_max_aspect_ratio,
            4.0
        );
        assert_eq!(
            evidence
                .tet_recovery
                .optimization_initial_min_exact_scaled_jacobian,
            0.32
        );
        assert_eq!(
            evidence
                .tet_recovery
                .optimization_final_min_exact_scaled_jacobian,
            0.40
        );
        assert_eq!(evidence.tet_recovery.untangling_pass_count, 2);
        assert_eq!(
            evidence.tet_recovery.untangling_initial_near_singular_count,
            6
        );
        assert_eq!(
            evidence.tet_recovery.untangling_final_near_singular_count,
            1
        );
        assert_eq!(evidence.tet_recovery.untangling_relocated_seed_count, 3);
        assert_eq!(
            evidence.tet_recovery.untangling_reconnected_edge_star_count,
            4
        );
        assert_eq!(
            evidence
                .tet_recovery
                .untangling_reconnected_boundary_adjacent_cavity_count,
            5
        );
        assert_eq!(evidence.tet_recovery.exact_quality_repair_pass_count, 1);
        assert_eq!(
            evidence.tet_recovery.exact_quality_reconnected_cavity_count,
            2
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_reconnection_quality_gain_count,
            1
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_face_neighbor_reconnected_cavity_count,
            6
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_connected_reconnected_cavity_count,
            7
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_boundary_adjacent_reconnected_cavity_count,
            8
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_expanded_connected_reconnected_cavity_count,
            9
        );
        assert_eq!(evidence.tet_recovery.exact_quality_split_cavity_count, 3);
        assert_eq!(
            evidence.tet_recovery.exact_quality_seed_star_collapse_count,
            4
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_seed_star_relocation_count,
            5
        );
        assert_eq!(
            evidence.tet_recovery.exact_quality_unrepaired_total_count,
            9
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_unrepaired_general_cavity_count,
            1
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_unrepaired_boundary_adjacent_count,
            6
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_unrepaired_interior_seed_count,
            7
        );
        assert_eq!(
            evidence
                .tet_recovery
                .exact_quality_unrepaired_edge_star_count,
            8
        );
        assert_eq!(evidence.sizing.growth_rate, Some(1.4));
        assert_eq!(
            evidence.sizing.rejected_by_status.get("outside_bounds"),
            Some(&1)
        );
        assert_eq!(
            evidence.regions.boundary_region_face_counts.get("fixed"),
            Some(&1)
        );
        assert_eq!(
            evidence.regions.material_region_volume_m3.get("solid"),
            Some(&(1.0 / 6.0))
        );
        assert_eq!(
            evidence
                .regions
                .boundary_region_recovered_face_counts
                .get("fixed"),
            Some(&1)
        );
        assert_eq!(evidence.quality.min_exact_scaled_jacobian, 0.45);
        assert_eq!(evidence.quality.scaled_jacobian_p05, Some(0.5));
        assert_eq!(evidence.quality.scaled_jacobian_p50, Some(0.5));
        assert_eq!(evidence.quality.scaled_jacobian_p95, Some(0.5));
        assert_eq!(evidence.quality.exact_scaled_jacobian_p05, Some(0.45));
        assert_eq!(evidence.quality.exact_scaled_jacobian_p50, Some(0.45));
        assert_eq!(evidence.quality.exact_scaled_jacobian_p95, Some(0.45));
        assert_eq!(evidence.quality.aspect_ratio_p50, Some(2.0));
        assert_eq!(evidence.quality.aspect_ratio_p95, Some(2.0));
        assert_eq!(
            evidence
                .quality
                .exact_scaled_jacobian_bins
                .get("0_35_to_0_65"),
            Some(&1)
        );
        assert_eq!(
            evidence
                .validation
                .boundary_recovery
                .boundary_edge_recovery_ratio,
            1.0
        );

        let encoded = serde_json::to_value(&evidence).expect("serialize evidence");
        assert!(encoded.get("sizing").is_some());
        assert!(encoded.get("debug").is_none());
        assert_eq!(
            encoded["cad"]["evaluation_source"],
            serde_json::Value::String("imported_evaluator_samples".to_string())
        );
        assert!(
            encoded
                .to_string()
                .contains("sample detail should not be copied")
                == false
        );

        let failed_validation = AnalysisMeshValidationOptions {
            max_volume_element_count: Some(0),
            ..AnalysisMeshValidationOptions::default()
        };
        let failed_evidence = build_mesh_evidence_artifact(&mesh, &failed_validation);
        assert!(!failed_evidence.validation.solve_ready);
        assert_eq!(
            failed_evidence.validation.validation_error_code.as_deref(),
            Some("element_budget_exceeded")
        );
        assert!(failed_evidence
            .validation
            .validation_error_message
            .as_deref()
            .is_some_and(|message| message.contains("ElementBudgetExceeded")));

        let mut stale_validation = evidence.validation.clone();
        stale_validation.solve_ready = false;
        stale_validation.validation_error_code = Some("stale".to_string());
        stale_validation.validation_error_message = Some("stale".to_string());
        stale_validation.volume_element_count = 999;
        stale_validation.volume_component_count = 999;
        stale_validation
            .boundary_recovery
            .boundary_edge_recovery_ratio = 0.0;
        let refreshed_evidence =
            build_mesh_evidence_artifact_with_validation_evidence(&mesh, stale_validation);
        assert!(refreshed_evidence.validation.solve_ready);
        assert_eq!(refreshed_evidence.validation.validation_error_code, None);
        assert_eq!(refreshed_evidence.validation.volume_element_count, 1);
        assert_eq!(refreshed_evidence.validation.volume_component_count, 1);
        assert_eq!(
            refreshed_evidence
                .validation
                .boundary_recovery
                .boundary_edge_recovery_ratio,
            1.0
        );
    }

    #[cfg(feature = "dev-evidence")]
    #[test]
    fn dev_mesh_evidence_caps_debug_events() {
        let mesh = AnalysisMeshArtifact {
            schema_version: ANALYSIS_MESH_SCHEMA_VERSION.to_string(),
            mesh_id: "debug_mesh".to_string(),
            nodes: vec![
                node(1, [0.0, 0.0, 0.0]),
                node(2, [1.0, 0.0, 0.0]),
                node(3, [0.0, 1.0, 0.0]),
                node(4, [0.0, 0.0, 1.0]),
            ],
            volume_elements: vec![volume_element("tet_1", [1, 2, 3, 4])],
            boundary_faces: vec![boundary_face("face_1", [1, 2, 3])],
            boundary_edges: vec![
                boundary_edge("edge_1", [1, 2]),
                boundary_edge("edge_2", [2, 3]),
                boundary_edge("edge_3", [1, 3]),
            ],
            sizing: MeshSizingField::default(),
            quality: quality_report(),
            backend: MeshBackendSummary::default(),
            adaptive_iterations: Vec::new(),
            provenance: AnalysisMeshProvenance {
                algorithm: "test".to_string(),
                source_geometry_id: "geo".to_string(),
                source_geometry_revision: 1,
                source_geometry_sha256: None,
            },
        };

        let evidence = build_mesh_evidence_artifact_with_debug(
            &mesh,
            &AnalysisMeshValidationOptions::default(),
            vec![
                MeshDebugEvent::new("surface", "info", "surface recovery accepted"),
                MeshDebugEvent::new("volume", "warning", "candidate quality improved"),
                MeshDebugEvent::new("validation", "info", "solve readiness checked"),
            ],
            2,
        );

        let debug = evidence.debug.expect("dev evidence should include debug");
        assert_eq!(debug.event_cap, 2);
        assert_eq!(debug.event_count, 3);
        assert_eq!(debug.emitted_event_count, 2);
        assert_eq!(debug.truncated_event_count, 1);
        assert_eq!(debug.events[0].stage, "surface");
        assert_eq!(debug.events[1].stage, "volume");

        let encoded = serde_json::to_value(&debug).expect("serialize debug evidence");
        assert_eq!(encoded["events"].as_array().map(Vec::len), Some(2));
    }

    fn node(node_id: u32, coordinates_m: [f64; 3]) -> AnalysisMeshNode {
        AnalysisMeshNode {
            node_id,
            coordinates_m,
            provenance: Vec::new(),
        }
    }

    fn boundary_edge(edge_id: &str, node_ids: [u32; 2]) -> AnalysisBoundaryEdge {
        AnalysisBoundaryEdge {
            edge_id: edge_id.to_string(),
            node_ids,
            adjacent_boundary_face_ids: vec!["face_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }
    }

    #[cfg(feature = "dev-evidence")]
    fn boundary_face(face_id: &str, node_ids: [u32; 3]) -> AnalysisBoundaryFace {
        AnalysisBoundaryFace {
            face_id: face_id.to_string(),
            kind: BoundaryElementKind::Tri3,
            node_ids: node_ids.into(),
            adjacent_volume_element_ids: vec!["tet_1".to_string()],
            region_ids: vec!["fixed".to_string()],
            provenance: Vec::new(),
        }
    }

    #[cfg(feature = "dev-evidence")]
    fn volume_element(element_id: &str, node_ids: [u32; 4]) -> AnalysisVolumeElement {
        AnalysisVolumeElement {
            element_id: element_id.to_string(),
            kind: VolumeElementKind::Tet4,
            node_ids: node_ids.into(),
            material_region_id: "solid".to_string(),
            provenance: Vec::new(),
        }
    }

    #[cfg(feature = "dev-evidence")]
    fn quality_report() -> AnalysisMeshQualityReport {
        AnalysisMeshQualityReport {
            min_scaled_jacobian: 0.5,
            min_exact_scaled_jacobian: 0.45,
            mean_aspect_ratio: 2.0,
            max_aspect_ratio: 2.0,
            inverted_element_count: 0,
            mean_boundary_projection_error_m: 0.0,
            max_boundary_projection_error_m: 0.0,
            elements: vec![ElementQuality {
                element_id: "tet_1".to_string(),
                scaled_jacobian: 0.5,
                exact_scaled_jacobian: 0.45,
                aspect_ratio: 2.0,
                volume_m3: 1.0 / 6.0,
            }],
        }
    }
}
