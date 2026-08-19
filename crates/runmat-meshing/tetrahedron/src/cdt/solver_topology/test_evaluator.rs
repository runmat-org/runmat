use runmat_geometry_core::{
    CurveDerivatives, CurveEvaluatorId, CurveProjection, ExactCurveEvaluator, ExactPcurveEvaluator,
    ExactSurfaceEvaluator, ExactTrimClassifier, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange, PcurveDerivatives, PcurveEvaluatorId,
    SurfaceCurvature, SurfaceDerivatives, SurfaceEvaluatorId, SurfaceProjection, TrimClassifierId,
    TrimDomainLocation,
};

const POINTS: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
];
const EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const FACETS: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];

pub(super) static EVALUATOR: LinearTetrahedronEvaluator = LinearTetrahedronEvaluator {
    warped_curve_parameter: false,
};
pub(super) static WARPED_EVALUATOR: LinearTetrahedronEvaluator = LinearTetrahedronEvaluator {
    warped_curve_parameter: true,
};
pub(super) static CONTROL: UnlimitedControl = UnlimitedControl;

pub(super) struct LinearTetrahedronEvaluator {
    warped_curve_parameter: bool,
}

impl ExactCurveEvaluator for LinearTetrahedronEvaluator {
    fn parameter_range(
        &self,
        _id: &CurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        Ok(ParameterRange {
            start: 0.0,
            end: 1.0,
        })
    }

    fn point(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        curve_point(
            index(id.as_str(), "curve:")?,
            parameter,
            self.warped_curve_parameter,
        )
    }

    fn unit_tangent(
        &self,
        id: &CurveEvaluatorId,
        _parameter: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        let [left, right] = EDGES[index(id.as_str(), "curve:")?];
        normalize(subtract(POINTS[right], POINTS[left]))
    }

    fn derivatives(
        &self,
        id: &CurveEvaluatorId,
        parameter: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveDerivatives, GeometryEvaluationError> {
        let edge = index(id.as_str(), "curve:")?;
        let [left, right] = EDGES[edge];
        let direction = subtract(POINTS[right], POINTS[left]);
        let first_scale = if self.warped_curve_parameter {
            1.5 - parameter
        } else {
            1.0
        };
        let second_scale = if self.warped_curve_parameter {
            -1.0
        } else {
            0.0
        };
        Ok(CurveDerivatives {
            point_m: curve_point(edge, parameter, self.warped_curve_parameter)?,
            first_m: direction.map(|value| value * first_scale),
            second_m: direction.map(|value| value * second_scale),
        })
    }

    fn curvature_1_per_m(
        &self,
        _id: &CurveEvaluatorId,
        _parameter: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        Ok(0.0)
    }

    fn arc_length_m(
        &self,
        id: &CurveEvaluatorId,
        range: ParameterRange,
        _absolute_error_m: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<f64, GeometryEvaluationError> {
        let [left, right] = EDGES[index(id.as_str(), "curve:")?];
        let span = if self.warped_curve_parameter {
            warped_parameter(range.end) - warped_parameter(range.start)
        } else {
            range.end - range.start
        };
        Ok(length(subtract(POINTS[right], POINTS[left])) * span.abs())
    }

    fn inverse_project(
        &self,
        _id: &CurveEvaluatorId,
        _point_m: [f64; 3],
        _absolute_error_m: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<CurveProjection, GeometryEvaluationError> {
        Err(unused())
    }
}

impl ExactPcurveEvaluator for LinearTetrahedronEvaluator {
    fn parameter_range(
        &self,
        _id: &PcurveEvaluatorId,
    ) -> Result<ParameterRange, GeometryEvaluationError> {
        Ok(ParameterRange {
            start: 0.0,
            end: 1.0,
        })
    }

    fn point(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 2], GeometryEvaluationError> {
        let (face, local_edge) = pcurve_indices(id.as_str())?;
        let from = FACETS[face][local_edge];
        let to = FACETS[face][(local_edge + 1) % 3];
        let edge = edge_index(from, to)?;
        Ok(surface_uv(
            face,
            curve_point(edge, parameter, self.warped_curve_parameter)?,
        ))
    }

    fn derivatives(
        &self,
        id: &PcurveEvaluatorId,
        parameter: f64,
        control: &dyn GeometryEvaluationControl,
    ) -> Result<PcurveDerivatives, GeometryEvaluationError> {
        let epsilon = 1.0e-6;
        let left = ExactPcurveEvaluator::point(self, id, parameter - epsilon, control)?;
        let right = ExactPcurveEvaluator::point(self, id, parameter + epsilon, control)?;
        Ok(PcurveDerivatives {
            point_uv: ExactPcurveEvaluator::point(self, id, parameter, control)?,
            first_uv: [
                (right[0] - left[0]) / (2.0 * epsilon),
                (right[1] - left[1]) / (2.0 * epsilon),
            ],
            second_uv: [0.0; 2],
        })
    }
}

impl ExactSurfaceEvaluator for LinearTetrahedronEvaluator {
    fn parameter_bounds(
        &self,
        _id: &SurfaceEvaluatorId,
    ) -> Result<[ParameterRange; 2], GeometryEvaluationError> {
        Ok([ParameterRange {
            start: 0.0,
            end: 1.0,
        }; 2])
    }

    fn periodicity(
        &self,
        _id: &SurfaceEvaluatorId,
    ) -> Result<[Option<f64>; 2], GeometryEvaluationError> {
        Ok([None, None])
    }

    fn point(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        surface_point(index(id.as_str(), "surface:")?, uv)
    }

    fn derivatives(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceDerivatives, GeometryEvaluationError> {
        let face = index(id.as_str(), "surface:")?;
        let (du_m, dv_m) = match face {
            0 => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            1 => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            2 => ([-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]),
            3 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            _ => return Err(unknown()),
        };
        Ok(SurfaceDerivatives {
            point_m: ExactSurfaceEvaluator::point(self, id, uv, control)?,
            du_m,
            dv_m,
            duu_m: [0.0; 3],
            duv_m: [0.0; 3],
            dvv_m: [0.0; 3],
        })
    }

    fn unit_normal(
        &self,
        id: &SurfaceEvaluatorId,
        uv: [f64; 2],
        control: &dyn GeometryEvaluationControl,
    ) -> Result<[f64; 3], GeometryEvaluationError> {
        let derivatives = ExactSurfaceEvaluator::derivatives(self, id, uv, control)?;
        normalize(cross(derivatives.du_m, derivatives.dv_m))
    }

    fn principal_curvature(
        &self,
        _id: &SurfaceEvaluatorId,
        _uv: [f64; 2],
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceCurvature, GeometryEvaluationError> {
        Ok(SurfaceCurvature {
            minimum_1_per_m: 0.0,
            maximum_1_per_m: 0.0,
            minimum_direction_uv: [1.0, 0.0],
            maximum_direction_uv: [0.0, 1.0],
        })
    }

    fn closest_point(
        &self,
        _id: &SurfaceEvaluatorId,
        _point_m: [f64; 3],
        _absolute_error_m: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<SurfaceProjection, GeometryEvaluationError> {
        Err(unused())
    }
}

impl ExactTrimClassifier for LinearTetrahedronEvaluator {
    fn classify(
        &self,
        _id: &TrimClassifierId,
        _uv: [f64; 2],
        _boundary_tolerance_uv: f64,
        _control: &dyn GeometryEvaluationControl,
    ) -> Result<TrimDomainLocation, GeometryEvaluationError> {
        Ok(TrimDomainLocation::Inside)
    }
}

pub(super) struct UnlimitedControl;

impl GeometryEvaluationControl for UnlimitedControl {
    fn checkpoint(&self) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_iterations(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_search_work(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }

    fn consume_allocation_bytes(&self, _count: u64) -> Result<(), GeometryEvaluationError> {
        Ok(())
    }
}

fn curve_point(
    edge: usize,
    parameter: f64,
    warped: bool,
) -> Result<[f64; 3], GeometryEvaluationError> {
    let [left, right] = *EDGES.get(edge).ok_or_else(unknown)?;
    let parameter = if warped {
        warped_parameter(parameter)
    } else {
        parameter
    };
    Ok(std::array::from_fn(|axis| {
        POINTS[left][axis] * (1.0 - parameter) + POINTS[right][axis] * parameter
    }))
}

fn warped_parameter(parameter: f64) -> f64 {
    parameter + 0.5 * parameter * (1.0 - parameter)
}

fn surface_point(face: usize, uv: [f64; 2]) -> Result<[f64; 3], GeometryEvaluationError> {
    match face {
        0 => Ok([uv[0], uv[1], 0.0]),
        1 => Ok([uv[0], 0.0, uv[1]]),
        2 => Ok([1.0 - uv[0] - uv[1], uv[0], uv[1]]),
        3 => Ok([0.0, uv[0], uv[1]]),
        _ => Err(unknown()),
    }
}

fn surface_uv(face: usize, point: [f64; 3]) -> [f64; 2] {
    match face {
        0 => [point[0], point[1]],
        1 => [point[0], point[2]],
        2 | 3 => [point[1], point[2]],
        _ => [f64::NAN; 2],
    }
}

fn pcurve_indices(value: &str) -> Result<(usize, usize), GeometryEvaluationError> {
    let mut pieces = value
        .strip_prefix("pcurve:")
        .ok_or_else(unknown)?
        .split(':');
    let face = pieces
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(unknown)?;
    let edge = pieces
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(unknown)?;
    if pieces.next().is_some() {
        return Err(unknown());
    }
    Ok((face, edge))
}

fn index(value: &str, prefix: &str) -> Result<usize, GeometryEvaluationError> {
    value
        .strip_prefix(prefix)
        .and_then(|value| value.parse().ok())
        .ok_or_else(unknown)
}

fn edge_index(left: usize, right: usize) -> Result<usize, GeometryEvaluationError> {
    let mut endpoints = [left, right];
    endpoints.sort_unstable();
    EDGES
        .iter()
        .position(|edge| *edge == endpoints)
        .ok_or_else(unknown)
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(vector: [f64; 3]) -> Result<[f64; 3], GeometryEvaluationError> {
    let length = length(vector);
    if !length.is_finite() || length <= 0.0 {
        return Err(unknown());
    }
    Ok(vector.map(|value| value / length))
}

fn length(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn unknown() -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::UnknownEvaluator,
        "unknown linear tetrahedron test evaluator",
    )
}

fn unused() -> GeometryEvaluationError {
    GeometryEvaluationError::new(
        GeometryEvaluationErrorKind::InvalidResult,
        "unused test evaluator operation",
    )
}
