use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use runmat_meshing_core::{
    contracts::AnalysisMeshArtifact,
    quality::QualityThresholds,
    validation::{
        analysis_mesh_validation_error_code, mesh_contains_point,
        validate_analysis_mesh_with_options, volume_component_count,
        volume_component_element_counts, AnalysisMeshValidationOptions,
    },
};

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
    pub require_no_unrecovered_tetrahedron_components: bool,
    #[serde(default)]
    pub require_no_unrepaired_exact_quality: bool,
    #[serde(default)]
    pub unrecovered_tetrahedron_component_count: usize,
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

pub(super) fn validation_options_from_evidence(
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
        require_no_unrecovered_tetrahedron_components: validation
            .require_no_unrecovered_tetrahedron_components,
        require_no_unrepaired_exact_quality: validation.require_no_unrepaired_exact_quality,
        required_boundary_region_ids: validation.required_boundary_region_ids.clone(),
        required_material_region_ids: validation.required_material_region_ids.clone(),
    }
}

pub(super) fn validation_evidence(
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
        require_no_unrecovered_tetrahedron_components: validation
            .require_no_unrecovered_tetrahedron_components,
        require_no_unrepaired_exact_quality: validation.require_no_unrepaired_exact_quality,
        unrecovered_tetrahedron_component_count: mesh
            .backend
            .tetrahedron_unrecovered_component_count,
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
