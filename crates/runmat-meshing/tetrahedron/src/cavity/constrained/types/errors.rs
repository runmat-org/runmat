use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityValidationError {
    EmptyRemovedTetrahedronSet,
    InvalidTargetVolume {
        target_volume_m3: f64,
    },
    TooFewBoundaryFaces {
        boundary_face_count: usize,
    },
    DegenerateBoundaryFace {
        face_index: usize,
        node_ids: [u32; 3],
    },
    DuplicateBoundaryFace {
        node_ids: [u32; 3],
    },
    NonManifoldBoundaryEdge {
        node_ids: [u32; 2],
        face_count: usize,
    },
    ProtectedNodeOutsideBoundary {
        node_id: u32,
    },
    InvalidRefillVolume {
        target_volume_m3: f64,
        candidate_volume_m3: f64,
        tolerance_m3: f64,
    },
    BoundaryFaceCountMismatch {
        expected_count: usize,
        candidate_count: usize,
    },
    MissingBoundaryFace {
        node_ids: [u32; 3],
    },
    UnexpectedBoundaryFace {
        node_ids: [u32; 3],
    },
    BoundarySourceFaceMismatch {
        node_ids: [u32; 3],
        expected_source_face_id: Option<u32>,
        candidate_source_face_id: Option<u32>,
    },
    BoundarySourceEdgeMismatch {
        node_ids: [u32; 2],
        expected_source_edge_id: Option<u32>,
        candidate_source_edge_id: Option<u32>,
    },
    BoundaryRegionMismatch {
        node_ids: [u32; 3],
        expected_region_ids: Vec<String>,
        candidate_region_ids: Vec<String>,
    },
    BoundaryOutsideTetrahedronMismatch {
        node_ids: [u32; 3],
        expected_outside_tetrahedron_ids: Vec<u32>,
        candidate_outside_tetrahedron_ids: Vec<u32>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundarySplitError {
    SplitNodeReusesFaceNode { node_id: u32 },
    MissingBoundaryFace { node_ids: [u32; 3] },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryEdgeSplitError {
    MissingBoundaryEdge {
        node_ids: [u32; 2],
    },
    MissingBoundaryNode {
        node_id: u32,
    },
    #[cfg(test)]
    InvalidPatchWeights {
        weights: [f64; 4],
    },
    Split(ConstrainedCavityBoundarySplitError),
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryFaceSplitError {
    MissingBoundaryFace { node_ids: [u32; 3] },
    MissingBoundaryNode { node_id: u32 },
    DuplicateBoundaryFace { node_ids: [u32; 3] },
    InvalidBarycentricCoordinates { barycentric: [f64; 3] },
    Split(ConstrainedCavityBoundarySplitError),
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityBoundaryPatchSplitError {
    Edge(ConstrainedCavityBoundaryEdgeSplitError),
    Face(ConstrainedCavityBoundaryFaceSplitError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavitySolidEmptyBoundaryRecoveryError {
    Refill(ConstrainedCavityRefillError),
    Split(ConstrainedCavityBoundaryPatchSplitError),
    Expansion(ConstrainedCavityExpansionError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavitySourceEdgeSplitError {
    MissingBoundaryEdge { node_ids: [u32; 2] },
    MissingBoundaryNode { node_id: u32 },
    MissingSourceNode { node_id: u32 },
    MissingRemovedSourceTetrahedron { tetrahedron_id: u32 },
    NoIncidentSourceTetrahedron { node_ids: [u32; 2] },
    DegenerateSplitTetrahedron { tetrahedron_id: u32 },
    Validation(ConstrainedCavityValidationError),
    Refill(ConstrainedCavityRefillError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityExpansionError {
    BoundaryFaceNotFound { node_ids: [u32; 3] },
    BoundaryFaceHasNoOutsideTetrahedron { node_ids: [u32; 3] },
    BoundaryEdgeHasNoOutsideTetrahedron { node_ids: [u32; 2] },
    SourceTetrahedronIdNotFound { tetrahedron_id: u32 },
    ExpansionDidNotConverge { step_count: usize },
    Extraction(ConstrainedCavityExtractionError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityExtractionError {
    EmptySelection,
    SelectedTetrahedronIndexOutOfBounds {
        tetrahedron_index: usize,
        tetrahedron_count: usize,
    },
    DuplicateSelectedTetrahedronIndex {
        tetrahedron_index: usize,
    },
    Validation(ConstrainedCavityValidationError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillError {
    InvalidOptions,
    Validation(ConstrainedCavityValidationError),
    MissingBoundaryNode {
        node_id: u32,
    },
    DuplicateInteriorNode {
        node_id: u32,
    },
    InteriorNodeReusesBoundaryNode {
        node_id: u32,
    },
    InteriorPointOutsideCavity {
        node_id: u32,
    },
    NoValidCandidate {
        rejected_by_reason: BTreeMap<String, usize>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillTetrahedronSplitError {
    InvalidBarycentricCoordinates {
        barycentric: [f64; 3],
    },
    MissingNode {
        node_id: u32,
    },
    FaceIncidenceNotTwo {
        node_ids: [u32; 3],
        incident_tetrahedron_count: usize,
    },
    RejectedChildTetrahedron {
        node_ids: [u32; 4],
        reason: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedCavityRefillTetrahedronFlipError {
    MissingNode {
        node_id: u32,
    },
    FaceIncidenceNotTwo {
        node_ids: [u32; 3],
        incident_tetrahedron_count: usize,
    },
    EdgeIncidenceNotThree {
        node_ids: [u32; 2],
        incident_tetrahedron_count: usize,
    },
    InvalidFlipTopology {
        reason: &'static str,
    },
    RejectedCreatedTetrahedron {
        node_ids: [u32; 4],
        reason: &'static str,
    },
}
