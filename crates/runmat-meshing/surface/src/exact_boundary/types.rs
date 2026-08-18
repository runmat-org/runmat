use runmat_geometry_core::{PersistentEntityId, TopologicalOrientation};
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};

pub const EXACT_SURFACE_BOUNDARY_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactSurfaceBoundary {
    pub schema_version: u16,
    pub faces: Vec<ExactFaceBoundary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceBoundary {
    pub source_face_id: PersistentEntityId,
    pub outer_loop: ExactFaceBoundaryLoop,
    pub inner_loops: Vec<ExactFaceBoundaryLoop>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceBoundaryLoop {
    pub source_wire_id: PersistentEntityId,
    pub orientation: TopologicalOrientation,
    /// Cyclic face-local traversal. Segment order is incidence, not a sortable collection.
    pub segments: Vec<ExactFaceBoundarySegment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactFaceBoundarySegment {
    pub source_coedge_id: PersistentEntityId,
    pub source_edge_id: PersistentEntityId,
    pub node_ids: [StableDigest; 2],
    pub node_uv: [[f64; 2]; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactSurfaceBoundaryError {
    pub kind: ExactSurfaceBoundaryErrorKind,
    pub entity_id: Option<PersistentEntityId>,
    pub conflict: Option<Box<ExactSurfaceBoundaryConflict>>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactSurfaceBoundaryConflict {
    pub source_edge_ids: [PersistentEntityId; 2],
    pub segment_uv: [[[f64; 2]; 2]; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactSurfaceBoundaryErrorKind {
    InvalidCurveInput,
    MissingTopology,
    InvalidContract,
    InvalidPslg,
    ResourceLimit,
}

impl ExactSurfaceBoundaryError {
    pub(super) fn new(
        kind: ExactSurfaceBoundaryErrorKind,
        entity_id: Option<PersistentEntityId>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            entity_id,
            conflict: None,
            reason: reason.into(),
        }
    }

    pub(super) fn with_conflict(mut self, conflict: ExactSurfaceBoundaryConflict) -> Self {
        self.conflict = Some(Box::new(conflict));
        self
    }
}

impl std::fmt::Display for ExactSurfaceBoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "exact surface boundary {:?}", self.kind)?;
        if let Some(entity_id) = &self.entity_id {
            write!(formatter, " for {entity_id:?}")?;
        }
        write!(formatter, ": {}", self.reason)
    }
}

impl std::error::Error for ExactSurfaceBoundaryError {}
