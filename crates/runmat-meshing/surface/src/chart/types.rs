use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use crate::{
    ExactFaceBoundary, ExactFaceConstrainedDelaunay, ExactFaceDelaunay, ExactFaceDelaunayErrorKind,
    ExactFacePslg, ExactFaceTrimmedDelaunay,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceChart {
    pub chart_id: StableDigest,
    pub source_face_id: PersistentEntityId,
    pub periodicity: [Option<f64>; 2],
    pub boundary: ExactFaceBoundary,
    pub pslg: ExactFacePslg,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceCharts {
    pub source_face_id: PersistentEntityId,
    pub periodicity: [Option<f64>; 2],
    pub charts: Vec<ExactFaceChart>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceChartDelaunay {
    pub chart_id: StableDigest,
    pub triangulation: ExactFaceDelaunay,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceChartConstrainedDomain {
    pub chart_id: StableDigest,
    pub delaunay: ExactFaceDelaunay,
    pub constrained: ExactFaceConstrainedDelaunay,
    pub trimmed: ExactFaceTrimmedDelaunay,
}

#[derive(Clone, Copy)]
pub struct ExactFaceChartDelaunayContext<'a> {
    pub topology: &'a ExactBRepTopology,
    pub evaluator: &'a dyn ExactSurfaceEvaluator,
    pub geometry_control: &'a dyn GeometryEvaluationControl,
    pub cancellation: &'a dyn MeshingCancellationSignal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceChartOptions {
    pub maximum_periodic_residual: f64,
    pub maximum_period_shifts: i32,
    pub maximum_charts_per_face: u16,
}

impl Default for ExactFaceChartOptions {
    fn default() -> Self {
        Self {
            maximum_periodic_residual: 1.0e-12,
            maximum_period_shifts: 1_000_000,
            maximum_charts_per_face: 64,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceChartErrorKind {
    InvalidOptions,
    InvalidInput,
    GeometryEvaluation,
    RequiresMultipleCharts,
    Delaunay(ExactFaceDelaunayErrorKind),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceChartError {
    pub kind: ExactFaceChartErrorKind,
    pub source_face_id: Box<PersistentEntityId>,
    pub source_wire_id: Option<Box<PersistentEntityId>>,
    pub axis: Option<u8>,
    pub residual: Option<f64>,
    pub reason: String,
}

impl ExactFaceChartError {
    pub(super) fn new(
        kind: ExactFaceChartErrorKind,
        face_id: &PersistentEntityId,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source_face_id: Box::new(face_id.clone()),
            source_wire_id: None,
            axis: None,
            residual: None,
            reason: reason.into(),
        }
    }

    pub(super) fn with_witness(
        mut self,
        wire_id: &PersistentEntityId,
        axis: usize,
        residual: f64,
    ) -> Self {
        self.source_wire_id = Some(Box::new(wire_id.clone()));
        self.axis = Some(axis as u8);
        self.residual = Some(residual);
        self
    }
}

impl std::fmt::Display for ExactFaceChartError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face chart {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceChartError {}
