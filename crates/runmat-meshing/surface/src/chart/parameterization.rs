use runmat_geometry_core::{
    ExactSurfaceEvaluator, GeometryEvaluationControl, SurfaceDerivatives, SurfaceEvaluatorId,
};

use crate::ExactFaceBoundary;

use super::types::{ExactFaceChartError, ExactFaceChartErrorKind, ExactFaceChartOptions};

/// Relationship between chart-local topology coordinates and exact evaluator parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum ExactFaceChartParameterization {
    /// Chart coordinates are evaluator parameters, possibly lifted by integral periods.
    EvaluatorParameters,
    /// Chart coordinates are orthogonal coordinates in a deterministic local secant plane.
    /// Exact points and evaluator parameters are recovered through closest-point evaluation.
    ClosestPointProjection {
        origin_m: [f64; 3],
        axes: [[f64; 3]; 2],
        differential_step_m: f64,
        projection_tolerance_m: f64,
    },
}

impl ExactFaceChartParameterization {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::EvaluatorParameters => Ok(()),
            Self::ClosestPointProjection {
                origin_m,
                axes,
                differential_step_m,
                projection_tolerance_m,
            } => {
                if origin_m
                    .iter()
                    .chain(axes.iter().flatten())
                    .any(|value| !value.is_finite())
                    || !differential_step_m.is_finite()
                    || *differential_step_m <= 0.0
                    || !projection_tolerance_m.is_finite()
                    || *projection_tolerance_m <= 0.0
                    || (norm(axes[0]) - 1.0).abs() > 1.0e-12
                    || (norm(axes[1]) - 1.0).abs() > 1.0e-12
                    || dot(axes[0], axes[1]).abs() > 1.0e-12
                {
                    Err("projected chart frame and numerical policy must be finite and orthonormal")
                } else {
                    Ok(())
                }
            }
        }
    }

    pub(crate) fn project_point(&self, point_m: [f64; 3]) -> Result<[f64; 2], &'static str> {
        if point_m.iter().any(|value| !value.is_finite()) {
            return Err("projected chart point must be finite");
        }
        match self {
            Self::EvaluatorParameters => Err("evaluator parameter charts do not project 3D points"),
            Self::ClosestPointProjection { origin_m, axes, .. } => {
                let offset = subtract(point_m, *origin_m);
                Ok([dot(offset, axes[0]), dot(offset, axes[1])])
            }
        }
    }

    pub(crate) fn chart_plane_point(&self, coordinates: [f64; 2]) -> Option<[f64; 3]> {
        match self {
            Self::EvaluatorParameters => None,
            Self::ClosestPointProjection { origin_m, axes, .. } => {
                Some(std::array::from_fn(|axis| {
                    origin_m[axis] + coordinates[0] * axes[0][axis] + coordinates[1] * axes[1][axis]
                }))
            }
        }
    }
}

pub(super) fn build_projected_parameterization(
    boundary: &mut ExactFaceBoundary,
    evaluator_id: &SurfaceEvaluatorId,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    let mut points = Vec::new();
    let mut singular = None::<SurfaceDerivatives>;
    for segment in boundary_segments(boundary) {
        for uv in segment.node_uv {
            control
                .checkpoint()
                .map_err(|error| geometry_error(boundary, error))?;
            let derivatives = evaluator
                .derivatives(evaluator_id, uv, control)
                .map_err(|error| geometry_error(boundary, error))?;
            let point = derivatives.point_m;
            if point.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    boundary,
                    "singular chart boundary point is not finite",
                ));
            }
            if points.iter().all(|existing| {
                squared_norm(subtract(point, *existing))
                    > options.projection_tolerance_m * options.projection_tolerance_m
            }) {
                points.push(point);
            }
            if is_singular(&derivatives) {
                if singular.as_ref().is_some_and(|existing| {
                    squared_norm(subtract(existing.point_m, point))
                        > options.projection_tolerance_m * options.projection_tolerance_m
                }) {
                    return Err(ExactFaceChartError::new(
                        ExactFaceChartErrorKind::RequiresMultipleCharts,
                        &boundary.source_face_id,
                        "one projected chart cannot contain multiple surface singularities",
                    ));
                }
                singular.get_or_insert(derivatives);
            }
        }
    }
    let parameterization = frame(boundary, &points, singular.as_ref(), options)?;
    let source_face_id = boundary.source_face_id.clone();
    for segment in boundary_segments_mut(boundary) {
        for endpoint in 0..2 {
            let point = evaluator
                .point(evaluator_id, segment.node_uv[endpoint], control)
                .map_err(|error| {
                    ExactFaceChartError::new(
                        ExactFaceChartErrorKind::GeometryEvaluation(error.kind),
                        &source_face_id,
                        error.reason,
                    )
                })?;
            segment.node_uv[endpoint] =
                parameterization.project_point(point).map_err(|reason| {
                    ExactFaceChartError::new(
                        ExactFaceChartErrorKind::InvalidInput,
                        &source_face_id,
                        reason,
                    )
                })?;
        }
    }
    Ok(parameterization)
}

fn frame(
    boundary: &ExactFaceBoundary,
    points: &[[f64; 3]],
    singular: Option<&SurfaceDerivatives>,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    if let Some(singular) = singular {
        return differential_frame(boundary, points, singular, options);
    }
    let Some(origin) = points.first().copied() else {
        return Err(invalid(boundary, "singular chart has no boundary points"));
    };
    let (first_index, first_distance) = points
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, point)| (index, squared_norm(subtract(*point, origin))))
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .ok_or_else(|| {
            invalid(
                boundary,
                "singular chart has fewer than two distinct points",
            )
        })?;
    let first_axis = normalize(subtract(points[first_index], origin)).ok_or_else(|| {
        invalid(
            boundary,
            "singular chart first projection axis is unresolved",
        )
    })?;
    let (normal, area) = points
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != first_index)
        .map(|(index, point)| {
            let normal = cross(first_axis, subtract(*point, origin));
            (index, normal, squared_norm(normal))
        })
        .max_by(|left, right| {
            left.2
                .total_cmp(&right.2)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(_, normal, area)| (normal, area))
        .ok_or_else(|| invalid(boundary, "singular chart has no third projection point"))?;
    let tolerance_squared = options.projection_tolerance_m * options.projection_tolerance_m;
    if first_distance <= tolerance_squared || area <= tolerance_squared {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "singular face boundary cannot define one regular projected chart",
        ));
    }
    let normal = normalize(normal)
        .ok_or_else(|| invalid(boundary, "singular chart projection normal is unresolved"))?;
    let second_axis = normalize(cross(normal, first_axis)).ok_or_else(|| {
        invalid(
            boundary,
            "singular chart second projection axis is unresolved",
        )
    })?;
    let extent_m = first_distance.sqrt().max(area.sqrt());
    projected_parameterization(
        boundary,
        origin,
        [first_axis, second_axis],
        extent_m,
        options,
    )
}

fn differential_frame(
    boundary: &ExactFaceBoundary,
    points: &[[f64; 3]],
    singular: &SurfaceDerivatives,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    let candidates = [
        singular.du_m,
        singular.dv_m,
        singular.duu_m,
        singular.duv_m,
        singular.dvv_m,
    ];
    let (first_index, first) = candidates
        .iter()
        .enumerate()
        .filter(|(_, vector)| vector.iter().all(|value| value.is_finite()))
        .max_by(|left, right| {
            squared_norm(*left.1)
                .total_cmp(&squared_norm(*right.1))
                .then_with(|| right.0.cmp(&left.0))
        })
        .ok_or_else(|| {
            invalid(
                boundary,
                "singular chart has no finite differential direction",
            )
        })?;
    let first_axis = normalize(*first).ok_or_else(|| {
        invalid(
            boundary,
            "singular chart has no nonzero differential direction",
        )
    })?;
    let second = candidates
        .iter()
        .enumerate()
        .filter(|(index, vector)| {
            *index != first_index && vector.iter().all(|value| value.is_finite())
        })
        .map(|(index, vector)| {
            let orthogonal = subtract(*vector, scale(first_axis, dot(*vector, first_axis)));
            (index, orthogonal, squared_norm(orthogonal))
        })
        .max_by(|left, right| {
            left.2
                .total_cmp(&right.2)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(_, vector, _)| vector)
        .ok_or_else(|| {
            invalid(
                boundary,
                "singular chart has no second differential direction",
            )
        })?;
    let second_axis = normalize(second).ok_or_else(|| {
        ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "surface differential cannot define one regular singular chart",
        )
    })?;
    let extent_m = points
        .iter()
        .map(|point| norm(subtract(*point, singular.point_m)))
        .fold(0.0, f64::max);
    if !extent_m.is_finite() || extent_m <= options.projection_tolerance_m {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "singular chart has no finite physical extent",
        ));
    }
    projected_parameterization(
        boundary,
        singular.point_m,
        [first_axis, second_axis],
        extent_m,
        options,
    )
}

fn projected_parameterization(
    boundary: &ExactFaceBoundary,
    origin_m: [f64; 3],
    axes: [[f64; 3]; 2],
    extent_m: f64,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    let differential_step_m = (extent_m * f64::EPSILON.cbrt())
        .max(options.projection_tolerance_m * 8.0)
        .min(extent_m * 0.125);
    let result = ExactFaceChartParameterization::ClosestPointProjection {
        origin_m,
        axes,
        differential_step_m,
        projection_tolerance_m: options.projection_tolerance_m,
    };
    result
        .validate()
        .map_err(|reason| invalid(boundary, reason))?;
    Ok(result)
}

fn is_singular(derivatives: &SurfaceDerivatives) -> bool {
    let area = norm(cross(derivatives.du_m, derivatives.dv_m));
    let scale = norm(derivatives.du_m) * norm(derivatives.dv_m);
    area.is_finite() && area <= f64::EPSILON.sqrt() * scale.max(1.0)
}

fn boundary_segments(
    boundary: &ExactFaceBoundary,
) -> impl Iterator<Item = &crate::ExactFaceBoundarySegment> {
    boundary
        .outer_loop
        .segments
        .iter()
        .chain(boundary.inner_loops.iter().flat_map(|wire| &wire.segments))
}

fn boundary_segments_mut(
    boundary: &mut ExactFaceBoundary,
) -> impl Iterator<Item = &mut crate::ExactFaceBoundarySegment> {
    boundary.outer_loop.segments.iter_mut().chain(
        boundary
            .inner_loops
            .iter_mut()
            .flat_map(|wire| &mut wire.segments),
    )
}

fn geometry_error(
    boundary: &ExactFaceBoundary,
    error: runmat_geometry_core::GeometryEvaluationError,
) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::GeometryEvaluation(error.kind),
        &boundary.source_face_id,
        error.reason,
    )
}

fn invalid(boundary: &ExactFaceBoundary, reason: impl Into<String>) -> ExactFaceChartError {
    ExactFaceChartError::new(
        ExactFaceChartErrorKind::InvalidInput,
        &boundary.source_face_id,
        reason,
    )
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn squared_norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector)
}

fn norm(vector: [f64; 3]) -> f64 {
    squared_norm(vector).sqrt()
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(vector);
    (length.is_finite() && length > 0.0).then(|| vector.map(|value| value / length))
}

fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|value| value * factor)
}
