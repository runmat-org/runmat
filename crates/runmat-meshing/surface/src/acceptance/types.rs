use runmat_geometry_core::PersistentEntityId;

use crate::{ExactFaceDelaunayTriangle, ExactFaceGeometryErrorKind, ExactFaceMetricErrorKind};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceAcceptanceOptions {
    pub minimum_subdivision_depth: u8,
    pub maximum_subdivision_depth: u8,
    pub refinement_margin_ratio: f64,
    pub maximum_samples: u64,
}

impl Default for ExactFaceAcceptanceOptions {
    fn default() -> Self {
        Self {
            minimum_subdivision_depth: 2,
            maximum_subdivision_depth: 8,
            refinement_margin_ratio: 0.5,
            maximum_samples: 10_000_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceAcceptanceReport {
    pub source_face_id: PersistentEntityId,
    pub triangles: Vec<ExactFaceTriangleAcceptance>,
    pub sample_count: u64,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_chordal_deviation_uv: [f64; 2],
    pub maximum_normal_deviation_rad: f64,
    pub maximum_normal_deviation_uv: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceTriangleAcceptance {
    pub triangle: ExactFaceDelaunayTriangle,
    pub sample_count: u64,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_normal_deviation_rad: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceAcceptanceErrorKind {
    InvalidOptions,
    InvalidInput,
    Geometry(ExactFaceGeometryErrorKind),
    Metric(ExactFaceMetricErrorKind),
    ResourceLimit,
    UnsatisfiedQuality,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceAcceptanceError {
    pub kind: ExactFaceAcceptanceErrorKind,
    pub source_face_id: Box<PersistentEntityId>,
    pub triangle: Option<ExactFaceDelaunayTriangle>,
    pub witness_uv: Option<[f64; 2]>,
    pub reason: String,
}

impl ExactFaceAcceptanceError {
    pub(super) fn new(
        kind: ExactFaceAcceptanceErrorKind,
        source_face_id: &PersistentEntityId,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            source_face_id: Box::new(source_face_id.clone()),
            triangle: None,
            witness_uv: None,
            reason: reason.into(),
        }
    }

    pub(super) fn with_witness(
        mut self,
        triangle: ExactFaceDelaunayTriangle,
        uv: [f64; 2],
    ) -> Self {
        self.triangle = Some(triangle);
        self.witness_uv = Some(uv);
        self
    }
}

impl std::fmt::Display for ExactFaceAcceptanceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face acceptance {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceAcceptanceError {}
