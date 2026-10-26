use super::super::GeometryContractError;
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
pub(crate) struct CurveDerivativeCoefficients {
    pub point_m: [f64; 3],
    pub first_m: [f64; 3],
    pub second_m: [f64; 3],
}

pub type GeometryDerivative<const D: usize = 1> = crate::physical::ParametricDerivative<
    crate::physical::kind::Length,
    [f64; 3],
    super::GeometryParameterFrame,
    D,
>;

#[derive(Debug, Clone, PartialEq)]
pub struct CurveDerivatives {
    pub point_m: crate::physical::Position,
    pub first_m: GeometryDerivative,
    pub second_m: GeometryDerivative,
}

impl CurveDerivatives {
    pub fn from_parameter_coefficients(
        point: [f64; 3],
        first: [f64; 3],
        second: [f64; 3],
        frame: super::GeometryParameterFrame,
    ) -> Result<Self, GeometryEvaluationError> {
        Ok(Self {
            point_m: crate::physical::Position::try_new(point).map_err(derivative_result_error)?,
            first_m: geometry_derivative(
                first,
                [crate::physical::DerivativeAxis::new(frame.clone(), 1)],
            )?,
            second_m: geometry_derivative(
                second,
                [crate::physical::DerivativeAxis::new(frame, 2)],
            )?,
        })
    }
    pub fn transformed(
        self,
        transform: &super::super::GeometryTransform,
    ) -> Result<Self, GeometryEvaluationError> {
        Ok(Self {
            point_m: crate::physical::Position::try_new(
                transform.transform_point(*self.point_m.canonical_values()),
            )
            .map_err(derivative_result_error)?,
            first_m: self
                .first_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
            second_m: self
                .second_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
        })
    }
}
impl CurveDerivativeCoefficients {
    pub(crate) fn into_parameter_frame(
        self,
        frame: super::GeometryParameterFrame,
    ) -> Result<CurveDerivatives, GeometryEvaluationError> {
        CurveDerivatives::from_parameter_coefficients(
            self.point_m,
            self.first_m,
            self.second_m,
            frame,
        )
    }
}

fn derivative_result_error(error: impl std::fmt::Display) -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::InvalidResult,
        error.to_string(),
    )
}
fn geometry_derivative<const D: usize>(
    values: [f64; 3],
    axes: [crate::physical::DerivativeAxis<super::GeometryParameterFrame>; D],
) -> Result<GeometryDerivative<D>, GeometryEvaluationError> {
    crate::physical::ParametricDerivative::try_new(
        crate::physical::Quantity::try_new(values).map_err(derivative_result_error)?,
        axes,
    )
    .map_err(derivative_result_error)
}

pub type GeometryParameterDerivative =
    crate::physical::ParameterCoordinateDerivative<super::GeometryParameterFrame>;
#[derive(Debug, Clone, PartialEq)]
pub struct PcurveDerivatives {
    pub point_uv: [GeometryParameterCoordinate; 2],
    pub first_uv: [GeometryParameterDerivative; 2],
    pub second_uv: [GeometryParameterDerivative; 2],
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PcurveDerivativeCoefficients {
    pub point_uv: [f64; 2],
    pub first_uv: [f64; 2],
    pub second_uv: [f64; 2],
}
impl PcurveDerivatives {
    pub fn from_parameter_coefficients(
        point: [f64; 2],
        first: [f64; 2],
        second: [f64; 2],
        numerator: [super::GeometryParameterFrame; 2],
        denominator: super::GeometryParameterFrame,
    ) -> Result<Self, GeometryEvaluationError> {
        let [u, v] = numerator;
        let first_uv = [
            GeometryParameterDerivative::try_new(
                first[0],
                u.clone(),
                [crate::physical::DerivativeAxis::new(denominator.clone(), 1)],
            )
            .map_err(derivative_result_error)?,
            GeometryParameterDerivative::try_new(
                first[1],
                v.clone(),
                [crate::physical::DerivativeAxis::new(denominator.clone(), 1)],
            )
            .map_err(derivative_result_error)?,
        ];
        let second_uv = [
            GeometryParameterDerivative::try_new(
                second[0],
                u.clone(),
                [crate::physical::DerivativeAxis::new(denominator.clone(), 2)],
            )
            .map_err(derivative_result_error)?,
            GeometryParameterDerivative::try_new(
                second[1],
                v.clone(),
                [crate::physical::DerivativeAxis::new(denominator, 2)],
            )
            .map_err(derivative_result_error)?,
        ];
        Ok(Self {
            point_uv: [
                GeometryParameterCoordinate::try_new(point[0], u)
                    .map_err(derivative_result_error)?,
                GeometryParameterCoordinate::try_new(point[1], v)
                    .map_err(derivative_result_error)?,
            ],
            first_uv,
            second_uv,
        })
    }
}
impl PcurveDerivativeCoefficients {
    pub(crate) fn into_parameter_frames(
        self,
        numerator: [super::GeometryParameterFrame; 2],
        denominator: super::GeometryParameterFrame,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError> {
        PcurveDerivatives::from_parameter_coefficients(
            self.point_uv,
            self.first_uv,
            self.second_uv,
            numerator,
            denominator,
        )
    }
}

pub type GeometryParameterCoordinate =
    crate::physical::ParameterCoordinate<super::GeometryParameterFrame>;

#[derive(Debug, Clone, PartialEq)]
pub struct CurveProjection {
    pub parameter: GeometryParameterCoordinate,
    pub point_m: crate::physical::Position,
    pub distance_m: crate::physical::Length,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CurveProjectionCoefficients {
    pub parameter: f64,
    pub point_m: [f64; 3],
    pub distance_m: f64,
}
impl CurveProjectionCoefficients {
    pub(crate) fn into_parameter_frame(
        self,
        frame: super::GeometryParameterFrame,
    ) -> Result<CurveProjection, GeometryEvaluationError> {
        CurveProjection::from_parameter_coefficients(
            self.parameter,
            self.point_m,
            self.distance_m,
            frame,
        )
    }
}
impl CurveProjection {
    pub fn from_parameter_coefficients(
        parameter: f64,
        point: [f64; 3],
        distance: f64,
        frame: super::GeometryParameterFrame,
    ) -> Result<Self, GeometryEvaluationError> {
        if distance < 0.0 {
            return Err(derivative_result_error(
                "negative curve projection distance",
            ));
        }
        Ok(Self {
            parameter: GeometryParameterCoordinate::try_new(parameter, frame)
                .map_err(derivative_result_error)?,
            point_m: crate::physical::Position::try_new(point).map_err(derivative_result_error)?,
            distance_m: crate::physical::Length::try_new(distance)
                .map_err(derivative_result_error)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurfaceDerivativeCoefficients {
    pub point_m: [f64; 3],
    pub du_m: [f64; 3],
    pub dv_m: [f64; 3],
    pub duu_m: [f64; 3],
    pub duv_m: [f64; 3],
    pub dvv_m: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceDerivatives {
    pub point_m: crate::physical::Position,
    pub du_m: GeometryDerivative,
    pub dv_m: GeometryDerivative,
    pub duu_m: GeometryDerivative,
    pub duv_m: GeometryDerivative<2>,
    pub dvv_m: GeometryDerivative,
}
impl SurfaceDerivatives {
    pub fn from_parameter_coefficients(
        point: [f64; 3],
        du: [f64; 3],
        dv: [f64; 3],
        duu: [f64; 3],
        duv: [f64; 3],
        dvv: [f64; 3],
        frames: [super::GeometryParameterFrame; 2],
    ) -> Result<Self, GeometryEvaluationError> {
        let [u, v] = frames;
        Ok(Self {
            point_m: crate::physical::Position::try_new(point).map_err(derivative_result_error)?,
            du_m: geometry_derivative(du, [crate::physical::DerivativeAxis::new(u.clone(), 1)])?,
            dv_m: geometry_derivative(dv, [crate::physical::DerivativeAxis::new(v.clone(), 1)])?,
            duu_m: geometry_derivative(duu, [crate::physical::DerivativeAxis::new(u.clone(), 2)])?,
            duv_m: geometry_derivative(
                duv,
                [
                    crate::physical::DerivativeAxis::new(u, 1),
                    crate::physical::DerivativeAxis::new(v.clone(), 1),
                ],
            )?,
            dvv_m: geometry_derivative(dvv, [crate::physical::DerivativeAxis::new(v, 2)])?,
        })
    }
    pub fn transformed(
        self,
        transform: &super::super::GeometryTransform,
    ) -> Result<Self, GeometryEvaluationError> {
        Ok(Self {
            point_m: crate::physical::Position::try_new(
                transform.transform_point(*self.point_m.canonical_values()),
            )
            .map_err(derivative_result_error)?,
            du_m: self
                .du_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
            dv_m: self
                .dv_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
            duu_m: self
                .duu_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
            duv_m: self
                .duv_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
            dvv_m: self
                .dvv_m
                .try_map_canonical_storage(|value| transform.transform_vector(value))
                .map_err(derivative_result_error)?,
        })
    }
}
impl SurfaceDerivativeCoefficients {
    pub(crate) fn into_parameter_frames(
        self,
        frames: [super::GeometryParameterFrame; 2],
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        SurfaceDerivatives::from_parameter_coefficients(
            self.point_m,
            self.du_m,
            self.dv_m,
            self.duu_m,
            self.duv_m,
            self.dvv_m,
            frames,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCurvature {
    pub minimum_1_per_m: crate::physical::Curvature,
    pub maximum_1_per_m: crate::physical::Curvature,
    pub minimum_direction_uv: [f64; 2],
    pub maximum_direction_uv: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceProjection {
    pub uv: [GeometryParameterCoordinate; 2],
    pub point_m: crate::physical::Position,
    pub distance_m: crate::physical::Length,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurfaceProjectionCoefficients {
    pub uv: [f64; 2],
    pub point_m: [f64; 3],
    pub distance_m: f64,
}
impl SurfaceProjectionCoefficients {
    pub(crate) fn into_parameter_frames(
        self,
        frames: [super::GeometryParameterFrame; 2],
    ) -> Result<SurfaceProjection, GeometryEvaluationError> {
        SurfaceProjection::from_parameter_coefficients(
            self.uv,
            self.point_m,
            self.distance_m,
            frames,
        )
    }
}
impl SurfaceProjection {
    pub fn from_parameter_coefficients(
        uv: [f64; 2],
        point: [f64; 3],
        distance: f64,
        frames: [super::GeometryParameterFrame; 2],
    ) -> Result<Self, GeometryEvaluationError> {
        if distance < 0.0 {
            return Err(derivative_result_error(
                "negative surface projection distance",
            ));
        }
        let [u, v] = frames;
        Ok(Self {
            uv: [
                GeometryParameterCoordinate::try_new(uv[0], u).map_err(derivative_result_error)?,
                GeometryParameterCoordinate::try_new(uv[1], v).map_err(derivative_result_error)?,
            ],
            point_m: crate::physical::Position::try_new(point).map_err(derivative_result_error)?,
            distance_m: crate::physical::Length::try_new(distance)
                .map_err(derivative_result_error)?,
        })
    }
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
    #[serde(with = "crate::physical::canonical_scalar")]
    pub volume_m3: crate::physical::Volume,
    #[serde(with = "crate::physical::canonical_scalar")]
    pub surface_area_m2: crate::physical::Area,
    #[serde(with = "crate::physical::canonical_storage")]
    pub centroid_m: crate::physical::Position,
    /// `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` about the centroid, before density.
    #[serde(with = "crate::physical::canonical_storage")]
    pub inertia_about_centroid_m5:
        crate::physical::Quantity<crate::physical::kind::GeometricVolumeSecondMoment, [f64; 6]>,
}

impl BodyMassProperties {
    pub fn validate(&self) -> Result<(), GeometryContractError> {
        if !self.volume_m3.canonical_value().is_finite()
            || self.volume_m3.canonical_value() < 0.0
            || !self.surface_area_m2.canonical_value().is_finite()
            || self.surface_area_m2.canonical_value() <= 0.0
            || self
                .centroid_m
                .canonical_values()
                .iter()
                .any(|value| !value.is_finite())
            || self
                .inertia_about_centroid_m5
                .canonical_values()
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(GeometryContractError::invalid(
                "body mass properties",
                "volume, area, centroid, and inertia must be finite with non-negative volume and positive area",
            ));
        }
        let [ixx, iyy, izz, ..] = *self.inertia_about_centroid_m5.canonical_values();
        if ixx < 0.0
            || iyy < 0.0
            || izz < 0.0
            || ixx + iyy < izz
            || ixx + izz < iyy
            || iyy + izz < ixx
        {
            return Err(GeometryContractError::invalid(
                "body inertia",
                "principal diagonal moments must be non-negative and satisfy triangle inequalities",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryEvaluationErrorKind {
    Cancelled,
    BudgetExceeded,
    TimeBudgetExceeded,
    AllocationBudgetExceeded,
    SearchWorkBudgetExceeded,
    IterationBudgetExceeded,
    UnknownEvaluator,
    ParameterOutsideDomain,
    ProjectionDidNotConverge,
    ClassificationDidNotConverge,
    InconsistentGeometry,
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
    // Preserve the actual probe site for cheap execution diagnostics.
    #[track_caller]
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError>;
    fn consume_iterations(&self, count: u64) -> Result<(), GeometryEvaluationError>;
    fn consume_search_work(&self, count: u64) -> Result<(), GeometryEvaluationError>;
    fn consume_allocation_bytes(&self, count: u64) -> Result<(), GeometryEvaluationError>;
    /// Release a successful allocation charge after its storage has been dropped.
    /// Legacy cumulative controls may keep the charge conservatively. Callers must
    /// release only their own successful charge, exactly once.
    fn release_allocation_bytes(&self, _count: u64) {}
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
    ) -> Result<crate::physical::Position, GeometryEvaluationError>;
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
    ) -> Result<crate::physical::Curvature, GeometryEvaluationError>;
    fn arc_length_m(
        &self,
        id: &CurveEvaluatorId,
        range: ParameterRange,
        absolute_error_m: crate::physical::Length,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<crate::physical::Length, GeometryEvaluationError>;
    fn inverse_project(
        &self,
        id: &CurveEvaluatorId,
        point_m: crate::physical::Position,
        absolute_error_m: crate::physical::Length,
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
    ) -> Result<[crate::GeometryParameterCoordinate; 2], GeometryEvaluationError>;
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
    ) -> Result<crate::physical::Position, GeometryEvaluationError>;
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
        point_m: crate::physical::Position,
        absolute_error_m: crate::physical::Length,
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
