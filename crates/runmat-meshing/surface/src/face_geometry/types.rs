use runmat_geometry_core::PersistentEntityId;

use crate::{ExactFaceDelaunayTriangle, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation};

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceGeometry {
    pub source_face_id: PersistentEntityId,
    pub vertices: Vec<ExactFaceGeometryVertex>,
    pub triangles: Vec<ExactFaceTriangleGeometry>,
    pub maximum_metric_edge_length: f64,
    pub minimum_metric_angle_rad: f64,
    pub maximum_physical_aspect_ratio: f64,
    pub maximum_chordal_deviation_m: f64,
    pub maximum_normal_deviation_rad: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceGeometryVertex {
    pub pslg_vertex_index: u32,
    pub evaluation: ExactFaceMetricEvaluation,
    pub unit_normal: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceTriangleGeometry {
    pub triangle: ExactFaceDelaunayTriangle,
    pub centroid: ExactFaceMetricEvaluation,
    pub unit_normal: [f64; 3],
    pub physical_area_m2: f64,
    pub metric_edge_lengths: [f64; 3],
    pub minimum_metric_angle_rad: f64,
    pub physical_aspect_ratio: f64,
    pub chordal_deviation_m: f64,
    pub normal_deviation_rad: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactFaceGeometryErrorKind {
    InvalidInput,
    Metric(ExactFaceMetricErrorKind),
    InvalidEvaluation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactFaceGeometryError {
    pub kind: ExactFaceGeometryErrorKind,
    pub source_face_id: PersistentEntityId,
    pub reason: String,
}

impl ExactFaceGeometryError {
    pub(super) fn new(
        kind: ExactFaceGeometryErrorKind,
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

impl std::fmt::Display for ExactFaceGeometryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "exact face geometry {:?} for {:?}: {}",
            self.kind, self.source_face_id, self.reason
        )
    }
}

impl std::error::Error for ExactFaceGeometryError {}
