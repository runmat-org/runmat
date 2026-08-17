use super::{
    CurveEvaluatorId, MassPropertiesEvaluatorId, PcurveEvaluatorId, SurfaceEvaluatorId,
    TrimClassifierId,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterRange {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveDerivatives {
    pub point_m: [f64; 3],
    pub first_m: [f64; 3],
    pub second_m: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcurveDerivatives {
    pub point_uv: [f64; 2],
    pub first_uv: [f64; 2],
    pub second_uv: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveProjection {
    pub parameter: f64,
    pub point_m: [f64; 3],
    pub distance_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceDerivatives {
    pub point_m: [f64; 3],
    pub du_m: [f64; 3],
    pub dv_m: [f64; 3],
    pub duu_m: [f64; 3],
    pub duv_m: [f64; 3],
    pub dvv_m: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCurvature {
    pub minimum_1_per_m: f64,
    pub maximum_1_per_m: f64,
    pub minimum_direction_uv: [f64; 2],
    pub maximum_direction_uv: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceProjection {
    pub uv: [f64; 2],
    pub point_m: [f64; 3],
    pub distance_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimDomainLocation {
    Inside,
    OnBoundary,
    Outside,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyMassProperties {
    pub volume_m3: f64,
    pub surface_area_m2: f64,
    pub centroid_m: [f64; 3],
    /// `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` about the centroid, before density.
    pub inertia_about_centroid_m5: [f64; 6],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryEvaluationErrorKind {
    Cancelled,
    BudgetExceeded,
    UnknownEvaluator,
    ParameterOutsideDomain,
    ProjectionDidNotConverge,
    KernelUnavailable,
    KernelFailure,
    InvalidResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryEvaluationError {
    pub kind: GeometryEvaluationErrorKind,
    pub reason: String,
}

impl GeometryEvaluationError {
    pub fn new(kind: GeometryEvaluationErrorKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }
}

impl std::fmt::Display for GeometryEvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "geometry evaluation {:?}: {}",
            self.kind, self.reason
        )
    }
}

impl std::error::Error for GeometryEvaluationError {}

/// Execution supplies this narrow authority so iterative, allocating, or
/// kernel-backed evaluation remains cancellable and bounded without geometry
/// owning jobs.
pub trait GeometryEvaluationControl: Send + Sync {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError>;
    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError>;
    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError>;
    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError>;
}

pub trait ExactCurveEvaluator: Send + Sync {
    fn parameter_range(
        &self,
        id: &CurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError>;
    fn point(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn unit_tangent(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn derivatives(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveDerivatives, GeometryEvaluationError>;
    fn curvature_1_per_m(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError>;
    fn arc_length_m(
        &self,
        id: &CurveEvaluatorId,
        range: ParameterRange,
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError>;
    fn inverse_project(
        &self,
        id: &CurveEvaluatorId,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveProjection, GeometryEvaluationError>;
}

pub trait ExactPcurveEvaluator: Send + Sync {
    fn parameter_range(
        &self,
        id: &PcurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError>;
    fn point(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 2], GeometryEvaluationError>;
    fn derivatives(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError>;
}

pub trait ExactSurfaceEvaluator: Send + Sync {
    fn parameter_bounds(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<[ParameterRange; 2], GeometryEvaluationError>;
    fn periodicity(
        &self,
        id: &SurfaceEvaluatorId,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError>;
    fn point(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn derivatives(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError>;
    fn unit_normal(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn principal_curvature(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvature, GeometryEvaluationError>;
    fn closest_point(
        &self,
        id: &SurfaceEvaluatorId,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjection, GeometryEvaluationError>;
}

pub trait ExactTrimClassifier: Send + Sync {
    fn classify(
        &self,
        id: &TrimClassifierId,
        uv: [f64; 2],
        boundary_tolerance_uv: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<TrimDomainLocation, GeometryEvaluationError>;
}

pub trait ExactMassPropertiesEvaluator: Send + Sync {
    fn mass_properties(
        &self,
        id: &MassPropertiesEvaluatorId,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<BodyMassProperties, GeometryEvaluationError>;
}
