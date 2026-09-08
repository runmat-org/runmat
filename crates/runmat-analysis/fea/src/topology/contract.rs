use std::collections::BTreeMap;

use runmat_execution::Digest;
use runmat_meshing_core::{
    BoundaryFaceRole, BoundaryTriangleOrder, ElementOrder, PersistentEntityId, StableDigest,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct AnalysisProjection {
    pub schema_version: u16,
    pub mesh_digest: StableDigest,
    pub projection_digest: Digest,
    pub policy: AnalysisProjectionPolicy,
    pub nodes: Vec<ProjectedNode>,
    pub elements: Vec<ProjectedElement>,
    pub boundary_faces: Vec<ProjectedBoundaryFace>,
    pub node_neighbors: Vec<Vec<usize>>,
    pub bounds_m: [[f64; 2]; 3],
    pub characteristic_length_m: f64,
    pub(super) node_by_id: BTreeMap<u64, usize>,
    pub(super) element_by_id: BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisProjectionPolicy {
    pub topology_geometry_revision: u16,
}

impl Default for AnalysisProjectionPolicy {
    fn default() -> Self {
        Self {
            topology_geometry_revision: 1,
        }
    }
}

impl AnalysisProjection {
    pub fn node_index(&self, node_id: u64) -> Option<usize> {
        self.node_by_id.get(&node_id).copied()
    }

    pub fn element_index(&self, element_id: u64) -> Option<usize> {
        self.element_by_id.get(&element_id).copied()
    }

    pub fn boundary_faces_matching<'a>(
        &'a self,
        selector: &'a str,
    ) -> impl Iterator<Item = &'a ProjectedBoundaryFace> + 'a {
        self.boundary_faces.iter().filter(move |face| {
            face.provenance
                .iter()
                .any(|entity| entity.matches_analysis_selection(selector))
        })
    }
}

#[derive(Debug, Clone)]
pub struct ProjectedNode {
    pub node_id: u64,
    pub stable_identity: StableDigest,
    pub coordinates_m: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct ProjectedElement {
    pub element_id: u64,
    pub stable_identity: StableDigest,
    pub order: ElementOrder,
    pub node_indices: Vec<usize>,
    pub region_id: PersistentEntityId,
    pub signed_volume_m3: f64,
    pub centroid_m: [f64; 3],
    pub characteristic_length_m: f64,
}

#[derive(Debug, Clone)]
pub struct ProjectedBoundaryFace {
    pub face_id: u64,
    pub stable_identity: StableDigest,
    pub order: BoundaryTriangleOrder,
    pub role: BoundaryFaceRole,
    pub node_indices: Vec<usize>,
    pub adjacent_element_indices: Vec<usize>,
    pub provenance: Vec<PersistentEntityId>,
    pub centroid_m: [f64; 3],
    pub oriented_area_vector_m2: [f64; 3],
    pub normal_orientation: BoundaryNormalOrientation,
    pub area_m2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryNormalOrientation {
    ExteriorOutward,
    FirstAdjacentElementOutward,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AnalysisProjectionError {
    #[error("canonical solver mesh is invalid: {0}")]
    InvalidMesh(String),
    #[error("analysis projection identity encoding failed: {0}")]
    IdentityEncoding(String),
    #[error("{entity_kind} {entity_id} references unknown node {node_id}")]
    UnknownNode {
        entity_kind: &'static str,
        entity_id: u64,
        node_id: u64,
    },
    #[error("boundary face {face_id} references unknown adjacent element {element_id}")]
    UnknownAdjacentElement { face_id: u64, element_id: u64 },
    #[error("volume element {element_id} has unsupported or degenerate geometry")]
    InvalidElementGeometry { element_id: u64 },
    #[error("boundary face {face_id} has unsupported, degenerate, or unoriented geometry")]
    InvalidBoundaryGeometry { face_id: u64 },
}
