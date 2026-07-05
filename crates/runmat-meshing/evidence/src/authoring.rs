use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::MeshEvidenceArtifact;

pub const MESH_AUTHORING_SUMMARY_SCHEMA_VERSION: &str = "mesh-authoring-summary/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringSummary {
    pub schema_version: String,
    pub mesh_id: String,
    pub solve_ready: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_error_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_error_message: Option<String>,
    pub backend: String,
    pub tetrahedron_generation_family: String,
    pub topology: MeshAuthoringTopologySummary,
    pub quality: MeshAuthoringQualitySummary,
    pub recovery: MeshAuthoringRecoverySummary,
    pub regions: MeshAuthoringRegionSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringTopologySummary {
    pub node_count: usize,
    pub volume_element_count: usize,
    pub boundary_face_count: usize,
    pub boundary_edge_count: usize,
    pub adaptive_iteration_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds_min_m: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds_max_m: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringQualitySummary {
    pub meets_quality_thresholds: bool,
    pub min_scaled_jacobian: f64,
    pub min_exact_scaled_jacobian: f64,
    pub max_aspect_ratio: f64,
    pub max_boundary_projection_error_m: f64,
    pub inverted_element_count: usize,
    pub sliver_count: usize,
    pub sliver_removed_count: usize,
    pub unrepaired_exact_quality_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringRecoverySummary {
    pub boundary_face_recovery_ratio: f64,
    pub boundary_edge_recovery_ratio: f64,
    pub recovery_item_count: usize,
    pub recovered_item_count: usize,
    pub missing_recovery_item_count: usize,
    pub unrecovered_tetrahedron_component_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringRegionSummary {
    pub material_regions: Vec<MeshAuthoringMaterialRegion>,
    pub boundary_regions: Vec<MeshAuthoringBoundaryRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_material_region_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_required_material_region_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_boundary_region_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub missing_required_boundary_region_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringMaterialRegion {
    pub region_id: String,
    pub element_count: usize,
    #[serde(default)]
    pub volume_m3: f64,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAuthoringBoundaryRegion {
    pub region_id: String,
    pub face_count: usize,
    pub recovered_face_count: usize,
    pub edge_count: usize,
    pub fully_recovered: bool,
    pub required: bool,
}

pub fn build_mesh_authoring_summary(evidence: &MeshEvidenceArtifact) -> MeshAuthoringSummary {
    MeshAuthoringSummary {
        schema_version: MESH_AUTHORING_SUMMARY_SCHEMA_VERSION.to_string(),
        mesh_id: evidence.mesh_id.clone(),
        solve_ready: evidence.validation.solve_ready,
        validation_error_code: evidence.validation.validation_error_code.clone(),
        validation_error_message: evidence.validation.validation_error_message.clone(),
        backend: evidence.backend.backend.clone(),
        tetrahedron_generation_family: evidence.backend.tetrahedron_generation_family.clone(),
        topology: MeshAuthoringTopologySummary {
            node_count: evidence.topology.node_count,
            volume_element_count: evidence.topology.volume_element_count,
            boundary_face_count: evidence.topology.boundary_face_count,
            boundary_edge_count: evidence.topology.boundary_edge_count,
            adaptive_iteration_count: evidence.topology.adaptive_iteration_count,
            bounds_min_m: evidence.topology.bounds_min_m,
            bounds_max_m: evidence.topology.bounds_max_m,
        },
        quality: MeshAuthoringQualitySummary {
            meets_quality_thresholds: meets_quality_thresholds(evidence),
            min_scaled_jacobian: evidence.quality.min_scaled_jacobian,
            min_exact_scaled_jacobian: evidence.quality.min_exact_scaled_jacobian,
            max_aspect_ratio: evidence.quality.max_aspect_ratio,
            max_boundary_projection_error_m: evidence.quality.max_boundary_projection_error_m,
            inverted_element_count: evidence.quality.inverted_element_count,
            sliver_count: evidence.backend.tetrahedron_sliver_count,
            sliver_removed_count: evidence.backend.tetrahedron_sliver_removed_count,
            unrepaired_exact_quality_count: evidence
                .backend
                .tetrahedron_exact_quality_unrepaired_total_count,
        },
        recovery: MeshAuthoringRecoverySummary {
            boundary_face_recovery_ratio: evidence
                .validation
                .boundary_recovery
                .boundary_face_recovery_ratio,
            boundary_edge_recovery_ratio: evidence
                .validation
                .boundary_recovery
                .boundary_edge_recovery_ratio,
            recovery_item_count: evidence.backend.tetrahedron_recovery_item_count,
            recovered_item_count: evidence.backend.tetrahedron_recovered_item_count,
            missing_recovery_item_count: evidence.backend.tetrahedron_missing_recovery_item_count,
            unrecovered_tetrahedron_component_count: evidence
                .backend
                .tetrahedron_unrecovered_component_count,
        },
        regions: region_summary(evidence),
    }
}

fn meets_quality_thresholds(evidence: &MeshEvidenceArtifact) -> bool {
    let thresholds = &evidence.validation.quality;
    evidence.quality.min_scaled_jacobian >= thresholds.min_scaled_jacobian
        && evidence.quality.max_aspect_ratio <= thresholds.max_aspect_ratio
        && evidence.quality.max_boundary_projection_error_m
            <= thresholds.max_boundary_projection_error_m
        && (thresholds.allow_inverted_elements || evidence.quality.inverted_element_count == 0)
}

fn region_summary(evidence: &MeshEvidenceArtifact) -> MeshAuthoringRegionSummary {
    let required_materials: BTreeSet<_> = evidence
        .validation
        .required_material_region_ids
        .iter()
        .cloned()
        .collect();
    let required_boundaries: BTreeSet<_> = evidence
        .validation
        .required_boundary_region_ids
        .iter()
        .cloned()
        .collect();

    let material_region_ids: BTreeSet<_> = evidence
        .regions
        .material_region_element_counts
        .keys()
        .chain(evidence.regions.material_region_volume_m3.keys())
        .cloned()
        .collect();
    let boundary_region_ids: BTreeSet<_> = evidence
        .regions
        .boundary_region_face_counts
        .keys()
        .chain(
            evidence
                .regions
                .boundary_region_recovered_face_counts
                .keys(),
        )
        .chain(evidence.regions.boundary_region_edge_counts.keys())
        .cloned()
        .collect();

    MeshAuthoringRegionSummary {
        material_regions: material_region_ids
            .iter()
            .map(|region_id| MeshAuthoringMaterialRegion {
                region_id: region_id.clone(),
                element_count: evidence
                    .regions
                    .material_region_element_counts
                    .get(region_id)
                    .copied()
                    .unwrap_or_default(),
                volume_m3: evidence
                    .regions
                    .material_region_volume_m3
                    .get(region_id)
                    .copied()
                    .unwrap_or_default(),
                required: required_materials.contains(region_id),
            })
            .collect(),
        boundary_regions: boundary_region_ids
            .iter()
            .map(|region_id| {
                let face_count = evidence
                    .regions
                    .boundary_region_face_counts
                    .get(region_id)
                    .copied()
                    .unwrap_or_default();
                let recovered_face_count = evidence
                    .regions
                    .boundary_region_recovered_face_counts
                    .get(region_id)
                    .copied()
                    .unwrap_or_default();
                MeshAuthoringBoundaryRegion {
                    region_id: region_id.clone(),
                    face_count,
                    recovered_face_count,
                    edge_count: evidence
                        .regions
                        .boundary_region_edge_counts
                        .get(region_id)
                        .copied()
                        .unwrap_or_default(),
                    fully_recovered: recovered_face_count >= face_count,
                    required: required_boundaries.contains(region_id),
                }
            })
            .collect(),
        required_material_region_ids: required_materials.iter().cloned().collect(),
        missing_required_material_region_ids: required_materials
            .difference(&material_region_ids)
            .cloned()
            .collect(),
        required_boundary_region_ids: required_boundaries.iter().cloned().collect(),
        missing_required_boundary_region_ids: required_boundaries
            .difference(&boundary_region_ids)
            .cloned()
            .collect(),
    }
}
