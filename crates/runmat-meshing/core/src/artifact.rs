use serde::{Deserialize, Serialize};

use crate::{
    adaptive::AdaptiveIterationSummary,
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance},
    quality::AnalysisMeshQualityReport,
    sizing::MeshSizingField,
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub const ANALYSIS_MESH_SCHEMA_VERSION: &str = "analysis-mesh/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMeshNode {
    pub node_id: u32,
    pub coordinates_m: [f64; 3],
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisVolumeElement {
    pub element_id: String,
    pub kind: VolumeElementKind,
    pub node_ids: Vec<u32>,
    pub material_region_id: String,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisBoundaryFace {
    pub face_id: String,
    pub kind: BoundaryElementKind,
    pub node_ids: Vec<u32>,
    #[serde(default)]
    pub adjacent_volume_element_ids: Vec<String>,
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisBoundaryEdge {
    pub edge_id: String,
    pub node_ids: [u32; 2],
    #[serde(default)]
    pub adjacent_boundary_face_ids: Vec<String>,
    #[serde(default)]
    pub region_ids: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<MeshEntityProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshBackendSummary {
    pub backend: String,
    pub algorithm: String,
    #[serde(default)]
    pub source_topology_vertex_count: usize,
    #[serde(default)]
    pub source_topology_edge_count: usize,
    #[serde(default)]
    pub source_topology_face_count: usize,
    #[serde(default)]
    pub curve_element_count: usize,
    #[serde(default)]
    pub surface_element_count: usize,
    #[serde(default)]
    pub volume_candidate_count: usize,
    #[serde(default)]
    pub interior_seed_point_count: usize,
    #[serde(default)]
    pub tet_candidate_count: usize,
    #[serde(default)]
    pub boundary_face_recovery_ratio: f64,
    #[serde(default)]
    pub boundary_edge_recovery_ratio: f64,
}

impl Default for MeshBackendSummary {
    fn default() -> Self {
        Self {
            backend: "unknown".to_string(),
            algorithm: "unknown".to_string(),
            source_topology_vertex_count: 0,
            source_topology_edge_count: 0,
            source_topology_face_count: 0,
            curve_element_count: 0,
            surface_element_count: 0,
            volume_candidate_count: 0,
            interior_seed_point_count: 0,
            tet_candidate_count: 0,
            boundary_face_recovery_ratio: 0.0,
            boundary_edge_recovery_ratio: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMeshArtifact {
    pub schema_version: String,
    pub mesh_id: String,
    pub nodes: Vec<AnalysisMeshNode>,
    pub volume_elements: Vec<AnalysisVolumeElement>,
    #[serde(default)]
    pub boundary_faces: Vec<AnalysisBoundaryFace>,
    #[serde(default)]
    pub boundary_edges: Vec<AnalysisBoundaryEdge>,
    pub quality: AnalysisMeshQualityReport,
    pub sizing: MeshSizingField,
    #[serde(default)]
    pub backend: MeshBackendSummary,
    #[serde(default)]
    pub adaptive_iterations: Vec<AdaptiveIterationSummary>,
    pub provenance: AnalysisMeshProvenance,
}
