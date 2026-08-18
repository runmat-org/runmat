use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{MeshingCancellationSignal, MetricFieldRequest, SurfaceQualityTargets};
use runmat_meshing_curve::SharedCurveSegmentSplit;

use crate::{
    ExactFaceConstrainedDelaunay, ExactFaceDelaunayErrorKind, ExactFaceDelaunayTriangle,
    ExactFaceGeometry, ExactFaceGeometryErrorKind, ExactFaceMetricErrorKind, ExactFacePslg,
    ExactFaceTrimmedDelaunay,
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
    pub constrained: ExactFaceConstrainedDelaunay,
    pub trimmed: ExactFaceTrimmedDelaunay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactFaceRefinementOptions {
    pub maximum_interior_insertions: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceRefinementPolicy {
    pub quality: SurfaceQualityTargets,
    pub delaunay: crate::ExactFaceDelaunayOptions,
    pub refinement: ExactFaceRefinementOptions,
}

pub struct ExactFaceRefinementContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub metric_request: &'a MetricFieldRequest,
    pub evaluator: &'a dyn ExactSurfaceEvaluator,
    pub geometry_control: &'a dyn GeometryEvaluationControl,
    pub cancellation: &'a dyn MeshingCancellationSignal,
}

impl<'a> ExactFaceRefinementContext<'a> {
    pub fn new(
        topology: &'a ExactBRepTopology,
        metric_request: &'a MetricFieldRequest,
        evaluator: &'a dyn ExactSurfaceEvaluator,
        geometry_control: &'a dyn GeometryEvaluationControl,
        cancellation: &'a dyn MeshingCancellationSignal,
    ) -> Self {
        Self {
            topology,
            metric_request,
            evaluator,
            geometry_control,
            cancellation,
        }
    }
}

impl Default for ExactFaceRefinementOptions {
    fn default() -> Self {
        Self {
            maximum_interior_insertions: 1_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceRefinedMesh {
    pub topology: ExactFaceRefinedTopology,
    pub geometry: ExactFaceGeometry,
    pub interior_insertion_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExactFaceRefinementOutcome {
    Converged(Box<ExactFaceRefinedMesh>),
    /// The global shared curve must be rebuilt before this face is restarted. Any provisional
    /// face-owned insertions are intentionally not authoritative across that boundary change.
    RequiresCurveSplit {
        split: Box<ExactProtectedSegmentSplit>,
        completed_interior_insertions: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceRefinementErrorKind {
    InvalidOptions,
    InvalidQuality,
    InvalidGeometry,
    Metric(ExactFaceMetricErrorKind),
    Geometry(ExactFaceGeometryErrorKind),
    Delaunay(ExactFaceDelaunayErrorKind),
    ResourceLimit,
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
