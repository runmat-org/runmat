use serde::{Deserialize, Serialize};

use crate::{quality::AnalysisMeshQualityReport, size::field::MeshSizingField};
use runmat_meshing_size::adaptive::AdaptiveIterationSummary;

use super::MeshBackendSummary;
use crate::contracts::{
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance},
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
