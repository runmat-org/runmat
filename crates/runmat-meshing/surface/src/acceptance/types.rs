use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use crate::{ExactFaceDelaunayTriangle, ExactFaceGeometryErrorKind, ExactFaceMetricErrorKind};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceAcceptanceOptions {
    pub minimum_subdivision_depth: u8,
    pub maximum_subdivision_depth: u8,
    pub refinement_margin_ratio: f64,
    pub maximum_samples: u64,
}

impl ExactFaceAcceptanceOptions {
    /// Minimum adaptive sample demand, excluding additional boundary checks and
    /// deeper subdivision near the quality envelope. This is a lower bound,
    /// never authority to skip sampling or bypass the shared work control.
    pub fn minimum_sample_count(&self, triangles: u64) -> Option<u64> {
        let mut cells = 1u64;
        let mut samples = 0u64;
        for depth in 0..=self.minimum_subdivision_depth {
            samples = samples.checked_add(cells.checked_mul(4)?)?;
            if depth < self.minimum_subdivision_depth {
                cells = cells.checked_mul(4)?;
            }
        }
        samples.checked_mul(triangles)
    }
}

impl Default for ExactFaceAcceptanceOptions {
    fn default() -> Self {
        Self {
            minimum_subdivision_depth: 2,
            maximum_subdivision_depth: 4,
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
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_chordal_deviation_uv: [f64; 2],
    pub maximum_normal_deviation_rad: f64,
    pub maximum_normal_deviation_uv: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceChartAcceptanceReport {
    pub chart_id: StableDigest,
    pub acceptance: ExactFaceAcceptanceReport,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExactFaceTriangleAcceptance {
    pub triangle: ExactFaceDelaunayTriangle,
    pub sample_count: u64,
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_chordal_deviation_uv: [f64; 2],
    pub maximum_normal_deviation_rad: f64,
    pub maximum_normal_deviation_uv: [f64; 2],
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceAcceptanceViolation {
    ChordalDeviation,
    NormalDeviation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceAcceptanceError {
    pub kind: ExactFaceAcceptanceErrorKind,
    pub source_face_id: Box<PersistentEntityId>,
    pub triangle: Option<ExactFaceDelaunayTriangle>,
    pub witness_uv: Option<[f64; 2]>,
    pub violation: Option<ExactFaceAcceptanceViolation>,
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
            violation: None,
            reason: reason.into(),
        }
    }

    pub(super) fn with_witness(
        mut self,
        triangle: ExactFaceDelaunayTriangle,
        uv: [f64; 2],
        violation: ExactFaceAcceptanceViolation,
    ) -> Self {
        self.triangle = Some(triangle);
        self.witness_uv = Some(uv);
        self.violation = Some(violation);
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
