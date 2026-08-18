use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_curve::SharedCurveSegmentSplit;

use crate::{
    ExactFaceConstrainedDelaunay, ExactFaceDelaunay, ExactFaceDelaunayErrorKind,
    ExactFaceDelaunayTriangle, ExactFaceMetricErrorKind, ExactFacePslg, ExactFaceTrimmedDelaunay,
};

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
    pub curve_split: SharedCurveSegmentSplit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceRefinedTopology {
    pub pslg: ExactFacePslg,
    pub delaunay: ExactFaceDelaunay,
    pub constrained: ExactFaceConstrainedDelaunay,
    pub trimmed: ExactFaceTrimmedDelaunay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceRefinementErrorKind {
    InvalidQuality,
    InvalidGeometry,
    Metric(ExactFaceMetricErrorKind),
    Delaunay(ExactFaceDelaunayErrorKind),
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
