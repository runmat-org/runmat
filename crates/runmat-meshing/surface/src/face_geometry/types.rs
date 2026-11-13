use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::MetricFieldRequest;

use crate::{ExactFaceDelaunayTriangle, ExactFaceMetricErrorKind, ExactFaceMetricEvaluation};

#[derive(Clone, Copy)]
pub struct ExactFaceGeometryContext<'a> {
    pub(crate) metric_field: Option<&'a crate::ResolvedFaceMetricField>,
    pub topology: &'a ExactBRepTopology,
    pub metric_request: &'a MetricFieldRequest,
    pub evaluator: &'a dyn ExactSurfaceEvaluator,
    pub control: &'a dyn GeometryEvaluationControl,
    pub boundary_coordinates: Option<&'a super::ExactBoundaryNodeCoordinates<'a>>,
}

impl<'a> ExactFaceGeometryContext<'a> {
    pub fn new(
        topology: &'a ExactBRepTopology,
        metric_request: &'a MetricFieldRequest,
        evaluator: &'a dyn ExactSurfaceEvaluator,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Self {
        Self {
            metric_field: None,
            topology,
            metric_request,
            evaluator,
            control,
            boundary_coordinates: None,
        }
    }
    pub fn with_boundary_coordinates(
        mut self,
        coordinates: &'a super::ExactBoundaryNodeCoordinates<'a>,
    ) -> Self {
        self.boundary_coordinates = Some(coordinates);
        self
    }
    pub(crate) fn resolve_metric(
        &self,
    ) -> Result<std::borrow::Cow<'a, crate::ResolvedFaceMetricField>, crate::ExactFaceMetricError>
    {
        match self.metric_field {
            Some(field) => Ok(std::borrow::Cow::Borrowed(field)),
            None => crate::ResolvedFaceMetricField::new_with_control(
                self.topology,
                self.metric_request,
                self.control,
            )
            .map(std::borrow::Cow::Owned),
        }
    }
    pub(crate) fn realize(
        &self,
        face: &PersistentEntityId,
        node_id: runmat_meshing_core::StableDigest,
        surface_point: [f64; 3],
    ) -> Result<[f64; 3], ExactFaceGeometryError> {
        match self.boundary_coordinates {
            None => Ok(surface_point),
            Some(coordinates) => coordinates
                .point(node_id, self.control)
                .map(|point| point.unwrap_or(surface_point))
                .map_err(|failure| {
                    ExactFaceGeometryError::new(
                        ExactFaceGeometryErrorKind::Metric(
                            ExactFaceMetricErrorKind::GeometryEvaluation(failure.kind),
                        ),
                        face,
                        failure.reason,
                    )
                }),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceGeometry {
    pub source_face_id: PersistentEntityId,
    pub vertices: Vec<ExactFaceGeometryVertex>,
    pub triangles: Vec<ExactFaceTriangleGeometry>,
    pub maximum_metric_edge_length: f64,
    pub minimum_metric_angle_rad: f64,
    pub maximum_physical_aspect_ratio: f64,
    pub maximum_chordal_deviation_m: runmat_types::physical::Length,
    pub maximum_normal_deviation_rad: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceGeometryVertex {
    pub pslg_vertex_index: u32,
    /// Realized discrete position: original shared curve coordinate at a boundary,
    /// otherwise the evaluated surface point. All physical triangle checks use it.
    pub point_m: [f64; 3],
    /// Original exact-surface evaluation at the face parameter; never overwritten.
    pub evaluation: ExactFaceMetricEvaluation,
    pub unit_normal: [f64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExactFaceTriangleGeometry {
    pub triangle: ExactFaceDelaunayTriangle,
    pub deviation_samples: [ExactFaceMetricEvaluation; 4],
    pub unit_normal: [f64; 3],
    pub physical_area_m2: f64,
    pub metric_edge_lengths: [f64; 3],
    pub minimum_metric_angle_rad: f64,
    pub physical_aspect_ratio: f64,
    pub chordal_deviation_m: f64,
    pub chordal_deviation_uv: [f64; 2],
    pub normal_deviation_rad: f64,
    pub normal_deviation_uv: [f64; 2],
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
