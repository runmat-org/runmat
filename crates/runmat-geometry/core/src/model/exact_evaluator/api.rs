use super::{
    CurveEvaluatorIdV2, MassPropertiesEvaluatorIdV2, PcurveEvaluatorIdV2, SurfaceEvaluatorIdV2,
    TrimClassifierIdV2,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParameterRangeV2 {
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveDerivativesV2 {
    pub point_m: [f64; 3],
    pub first_m: [f64; 3],
    pub second_m: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcurveDerivativesV2 {
    pub point_uv: [f64; 2],
    pub first_uv: [f64; 2],
    pub second_uv: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveProjectionV2 {
    pub parameter: f64,
    pub point_m: [f64; 3],
    pub distance_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceDerivativesV2 {
    pub point_m: [f64; 3],
    pub du_m: [f64; 3],
    pub dv_m: [f64; 3],
    pub duu_m: [f64; 3],
    pub duv_m: [f64; 3],
    pub dvv_m: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCurvatureV2 {
    pub minimum_1_per_m: f64,
    pub maximum_1_per_m: f64,
    pub minimum_direction_uv: [f64; 2],
    pub maximum_direction_uv: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceProjectionV2 {
    pub uv: [f64; 2],
    pub point_m: [f64; 3],
    pub distance_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimDomainLocationV2 {
    Inside,
    OnBoundary,
    Outside,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyMassPropertiesV2 {
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

/// Execution supplies this narrow authority so iterative or kernel-backed
/// evaluation remains cancellable and bounded without geometry owning jobs.
pub trait GeometryEvaluationControl: Send + Sync {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError>;
    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError>;
    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError>;
}

pub trait ExactCurveEvaluatorV2: Send + Sync {
    fn parameter_range(
        &self,
        id: &CurveEvaluatorIdV2,
    ) -> Result<ParameterRangeV2, GeometryEvaluationError>;
    fn point(
        &self,
        id: &CurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn unit_tangent(
        &self,
        id: &CurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn derivatives(
        &self,
        id: &CurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveDerivativesV2, GeometryEvaluationError>;
    fn curvature_1_per_m(
        &self,
        id: &CurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError>;
    fn arc_length_m(
        &self,
        id: &CurveEvaluatorIdV2,
        range: ParameterRangeV2,
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError>;
    fn inverse_project(
        &self,
        id: &CurveEvaluatorIdV2,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveProjectionV2, GeometryEvaluationError>;
}

pub trait ExactPcurveEvaluatorV2: Send + Sync {
    fn parameter_range(
        &self,
        id: &PcurveEvaluatorIdV2,
    ) -> Result<ParameterRangeV2, GeometryEvaluationError>;
    fn point(
        &self,
        id: &PcurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 2], GeometryEvaluationError>;
    fn derivatives(
        &self,
        id: &PcurveEvaluatorIdV2,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivativesV2, GeometryEvaluationError>;
}

pub trait ExactSurfaceEvaluatorV2: Send + Sync {
    fn parameter_bounds(
        &self,
        id: &SurfaceEvaluatorIdV2,
    ) -> Result<[ParameterRangeV2; 2], GeometryEvaluationError>;
    fn periodicity(
        &self,
        id: &SurfaceEvaluatorIdV2,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError>;
    fn point(
        &self,
        id: &SurfaceEvaluatorIdV2,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn derivatives(
        &self,
        id: &SurfaceEvaluatorIdV2,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivativesV2, GeometryEvaluationError>;
    fn unit_normal(
        &self,
        id: &SurfaceEvaluatorIdV2,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError>;
    fn principal_curvature(
        &self,
        id: &SurfaceEvaluatorIdV2,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvatureV2, GeometryEvaluationError>;
    fn closest_point(
        &self,
        id: &SurfaceEvaluatorIdV2,
        point_m: [f64; 3],
        absolute_error_m: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjectionV2, GeometryEvaluationError>;
}

pub trait ExactTrimClassifierV2: Send + Sync {
    fn classify(
        &self,
        id: &TrimClassifierIdV2,
        uv: [f64; 2],
        boundary_tolerance_uv: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<TrimDomainLocationV2, GeometryEvaluationError>;
}

pub trait ExactMassPropertiesEvaluatorV2: Send + Sync {
    fn mass_properties(
        &self,
        id: &MassPropertiesEvaluatorIdV2,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<BodyMassPropertiesV2, GeometryEvaluationError>;
}
