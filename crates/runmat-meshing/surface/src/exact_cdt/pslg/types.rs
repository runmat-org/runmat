use runmat_geometry_core::{PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::StableDigest;

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFacePslg {
    pub source_face_id: PersistentEntityId,
    /// Canonically ordered boundary vertices and face-owned interior Steiner vertices.
    /// Only boundary vertices are referenced by `segments`.
    pub vertices: Vec<ExactFacePslgVertex>,
    pub segments: Vec<ExactFacePslgSegment>,
    pub loops: Vec<ExactFacePslgLoop>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFacePslgVertex {
    /// Shared-curve identity on the boundary, or deterministic face-owned identity in the interior.
    pub node_id: StableDigest,
    pub seam_image: Option<u8>,
    pub uv: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFacePslgSegment {
    pub source_coedge_id: PersistentEntityId,
    pub source_edge_id: PersistentEntityId,
    pub vertex_indices: [u32; 2],
    /// Exact curve parameters oriented with this face-local coedge segment.
    pub edge_parameters: [f64; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFacePslgLoop {
    pub source_wire_id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
    pub first_segment: u32,
    pub segment_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFacePslgError {
    pub kind: ExactFacePslgErrorKind,
    pub source_face_id: PersistentEntityId,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFacePslgErrorKind {
    InvalidBoundary,
    InvalidTopology,
    ResourceLimit,
}

impl ExactFacePslgError {
    pub(super) fn new(
        kind: ExactFacePslgErrorKind,
        source_face_id: &PersistentEntityId,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source_face_id: source_face_id.clone(),
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for ExactFacePslgError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face PSLG {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFacePslgError {}
