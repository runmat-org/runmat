use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use crate::{ExactFaceDelaunayTriangle, ExactFaceMetricErrorKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExactFaceRefinementReason {
    ChordalDeviation,
    NormalDeviation,
    MetricEdgeLength,
    MetricAngle,
    PhysicalAspectRatio,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceRefinementCandidate {
    pub source_face_id: PersistentEntityId,
    pub triangle_index: u32,
    pub triangle: ExactFaceDelaunayTriangle,
    pub reason: ExactFaceRefinementReason,
    pub uv: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExactFaceCandidateDisposition {
    Insert,
    SplitProtectedSegment(Box<ExactProtectedSegmentSplit>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactProtectedSegmentSplit {
    pub source_face_id: PersistentEntityId,
    pub pslg_segment_index: u32,
    pub source_coedge_id: PersistentEntityId,
    pub source_edge_id: PersistentEntityId,
    /// Ordered by increasing exact source-edge parameter, independent of coedge orientation.
    pub endpoint_node_ids: [StableDigest; 2],
    /// Strictly increasing exact source-edge parameter interval.
    pub edge_parameters: [f64; 2],
    pub split_parameter: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceRefinementErrorKind {
    InvalidQuality,
    InvalidGeometry,
    Metric(ExactFaceMetricErrorKind),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceRefinementError {
    pub kind: ExactFaceRefinementErrorKind,
    pub source_face_id: PersistentEntityId,
    pub reason: String,
}

impl ExactFaceRefinementError {
    pub(super) fn new(
        kind: ExactFaceRefinementErrorKind,
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

impl std::fmt::Display for ExactFaceRefinementError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face refinement {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceRefinementError {}
