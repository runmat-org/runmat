use serde::{Deserialize, Serialize};

use crate::{quality::AnalysisMeshQualityReport, size::field::MeshSizingField};
use runmat_meshing_size::adaptive::AdaptiveIterationSummary;

use super::MeshBackendSummary;
use crate::contracts::{
    provenance::{AnalysisMeshProvenance, MeshEntityProvenance},
    topology::{BoundaryElementKind, VolumeElementKind},
};

pub const ANALYSIS_MESH_SCHEMA_VERSION: &str = "analysis-mesh/v1";

pub const ANALYSIS_MESH_FIELD_TOPOLOGY_ID: &str = "analysis_mesh";
pub const ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID: &str = "analysis_mesh_boundary_faces";
pub const ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID: &str = "analysis_mesh_boundary_edges";
pub const TETRAHEDRON4_FIELD_ELEMENT_KIND: &str = "tetrahedron4";
pub const TRI3_FIELD_ELEMENT_KIND: &str = "tri3";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisFieldTopologyLocation {
    Node,
    VolumeElement,
    BoundaryFace,
    BoundaryEdge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisFieldTopologyDescriptor {
    pub topology_id: String,
    pub location: AnalysisFieldTopologyLocation,
    pub entity_count: usize,
    #[serde(default)]
    pub element_kind: Option<String>,
}

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
    pub field_topology: Vec<AnalysisFieldTopologyDescriptor>,
    #[serde(default)]
    pub backend: MeshBackendSummary,
    #[serde(default)]
    pub adaptive_iterations: Vec<AdaptiveIterationSummary>,
    pub provenance: AnalysisMeshProvenance,
}

impl AnalysisMeshArtifact {
    pub fn refresh_field_topology(&mut self) {
        self.field_topology = analysis_mesh_field_topology(
            &self.nodes,
            &self.volume_elements,
            &self.boundary_faces,
            &self.boundary_edges,
        );
    }
}

pub fn analysis_mesh_field_topology(
    nodes: &[AnalysisMeshNode],
    volume_elements: &[AnalysisVolumeElement],
    boundary_faces: &[AnalysisBoundaryFace],
    boundary_edges: &[AnalysisBoundaryEdge],
) -> Vec<AnalysisFieldTopologyDescriptor> {
    let mut descriptors = vec![AnalysisFieldTopologyDescriptor {
        topology_id: ANALYSIS_MESH_FIELD_TOPOLOGY_ID.to_string(),
        location: AnalysisFieldTopologyLocation::Node,
        entity_count: nodes.len(),
        element_kind: None,
    }];

    descriptors.push(AnalysisFieldTopologyDescriptor {
        topology_id: ANALYSIS_MESH_FIELD_TOPOLOGY_ID.to_string(),
        location: AnalysisFieldTopologyLocation::VolumeElement,
        entity_count: volume_elements.len(),
        element_kind: uniform_volume_element_kind(volume_elements),
    });

    descriptors.push(AnalysisFieldTopologyDescriptor {
        topology_id: ANALYSIS_MESH_BOUNDARY_FACE_TOPOLOGY_ID.to_string(),
        location: AnalysisFieldTopologyLocation::BoundaryFace,
        entity_count: boundary_faces.len(),
        element_kind: uniform_boundary_face_element_kind(boundary_faces),
    });

    descriptors.push(AnalysisFieldTopologyDescriptor {
        topology_id: ANALYSIS_MESH_BOUNDARY_EDGE_TOPOLOGY_ID.to_string(),
        location: AnalysisFieldTopologyLocation::BoundaryEdge,
        entity_count: boundary_edges.len(),
        element_kind: None,
    });

    descriptors
}

fn uniform_volume_element_kind(elements: &[AnalysisVolumeElement]) -> Option<String> {
    let first = elements.first()?.kind;
    if elements.iter().all(|element| element.kind == first) {
        match first {
            VolumeElementKind::Tetrahedron4 => Some(TETRAHEDRON4_FIELD_ELEMENT_KIND.to_string()),
            _ => None,
        }
    } else {
        None
    }
}

fn uniform_boundary_face_element_kind(faces: &[AnalysisBoundaryFace]) -> Option<String> {
    let first = faces.first()?.kind;
    if faces.iter().all(|face| face.kind == first) {
        match first {
            BoundaryElementKind::Tri3 => Some(TRI3_FIELD_ELEMENT_KIND.to_string()),
            _ => None,
        }
    } else {
        None
    }
}
