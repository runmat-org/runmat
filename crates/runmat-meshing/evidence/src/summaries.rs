use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use runmat_meshing_core::contracts::{
    AnalysisMeshArtifact, AnalysisVolumeElement, VolumeElementKind,
};
use runmat_meshing_size::adaptive::{AdaptiveConvergenceStatus, RefinementIndicatorStatus};

pub const MODULE_PURPOSE: &str = "compact solid evidence summaries";

pub use runmat_meshing_core::contracts::StageEvidence;

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

pub(super) fn adaptive_evidence(mesh: &AnalysisMeshArtifact) -> MeshAdaptiveEvidence {
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

pub(super) fn sizing_evidence(mesh: &AnalysisMeshArtifact) -> MeshSizingEvidence {
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

pub(super) fn topology_evidence(mesh: &AnalysisMeshArtifact) -> MeshTopologyEvidence {
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

pub(super) fn quality_evidence(mesh: &AnalysisMeshArtifact) -> MeshQualityEvidence {
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

pub(super) fn region_evidence(mesh: &AnalysisMeshArtifact) -> MeshRegionEvidence {
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
    if element.kind != VolumeElementKind::Tetrahedron4 || element.node_ids.len() != 4 {
        return 0.0;
    }
    let Some(points) = element_tetrahedron_points(mesh, element.node_ids.as_slice()) else {
        return 0.0;
    };
    tetrahedron_volume_m3(points)
}

fn element_tetrahedron_points(
    mesh: &AnalysisMeshArtifact,
    node_ids: &[u32],
) -> Option<[[f64; 3]; 4]> {
    Some([
        mesh_node(mesh, node_ids[0])?,
        mesh_node(mesh, node_ids[1])?,
        mesh_node(mesh, node_ids[2])?,
        mesh_node(mesh, node_ids[3])?,
    ])
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

fn mesh_node(mesh: &AnalysisMeshArtifact, node_id: u32) -> Option<[f64; 3]> {
    mesh.nodes
        .iter()
        .find(|node| node.node_id == node_id)
        .map(|node| node.coordinates_m)
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

fn tetrahedron_volume_m3(points: [[f64; 3]; 4]) -> f64 {
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

fn percentile(sorted_values: &[f64], ratio: f64) -> Option<f64> {
    if sorted_values.is_empty() {
        return None;
    }
    let ratio = ratio.clamp(0.0, 1.0);
    let index = ((sorted_values.len() - 1) as f64 * ratio).round() as usize;
    sorted_values.get(index).copied()
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
