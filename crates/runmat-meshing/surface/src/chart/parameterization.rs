use std::collections::BTreeMap;

use runmat_geometry_core::{
    ExactBRepTopology, ExactSurfaceEvaluator, GeometryEvaluationControl, SurfaceDerivatives,
    SurfaceEvaluatorId,
};

use crate::ExactFaceBoundary;

use super::types::{ExactFaceChartError, ExactFaceChartErrorKind, ExactFaceChartOptions};

const MAXIMUM_POLAR_ANGLE_RAD: f64 = std::f64::consts::FRAC_PI_2;
const PARAMETER_TOLERANCE: f64 = 1.0e-10;

/// Relationship between chart-local topology coordinates and exact evaluator parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum ExactFaceChartParameterization {
    /// Chart coordinates are evaluator parameters, possibly lifted by integral periods.
    EvaluatorParameters,
    /// An axis-aligned affine map from evaluator parameters into locally
    /// physical chart coordinates. One chart unit is approximately one metre
    /// at the deterministic reference point used to construct the chart.
    AffineEvaluatorParameters {
        origins: [f64; 2],
        evaluator_units_per_chart_unit: [f64; 2],
    },
    /// A bounded quarter-turn around one collapsed evaluator-parameter boundary. This is an
    /// exact, invertible parameter transform; it never recovers geometry through projection.
    PolarEvaluatorParameters {
        angular_axis: u8,
        radial_axis: u8,
        angular_origin: f64,
        angular_span: f64,
        singular_coordinate: f64,
        radial_direction: f64,
        differential_step: f64,
    },
}

impl ExactFaceChartParameterization {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::EvaluatorParameters => Ok(()),
            Self::AffineEvaluatorParameters {
                origins,
                evaluator_units_per_chart_unit,
            } => {
                if origins.iter().any(|value| !value.is_finite())
                    || evaluator_units_per_chart_unit
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    Err("affine chart origins and parameter scales must be finite and scales positive")
                } else {
                    Ok(())
                }
            }
            Self::PolarEvaluatorParameters {
                angular_axis,
                radial_axis,
                angular_origin,
                angular_span,
                singular_coordinate,
                radial_direction,
                differential_step,
            } => {
                if *angular_axis > 1
                    || *radial_axis > 1
                    || angular_axis == radial_axis
                    || [
                        *angular_origin,
                        *angular_span,
                        *singular_coordinate,
                        *radial_direction,
                        *differential_step,
                    ]
                    .iter()
                    .any(|value| !value.is_finite())
                    || *angular_span <= 0.0
                    || *angular_span > MAXIMUM_POLAR_ANGLE_RAD + PARAMETER_TOLERANCE
                    || radial_direction.abs() != 1.0
                    || *differential_step <= 0.0
                {
                    Err("polar chart axes, bounds, direction, and numerical policy are invalid")
                } else {
                    Ok(())
                }
            }
        }
    }

    pub(crate) fn evaluator_parameters(
        &self,
        coordinates: [f64; 2],
    ) -> Result<[f64; 2], &'static str> {
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err("chart coordinates must be finite");
        }
        match self {
            Self::EvaluatorParameters => Ok(coordinates),
            Self::AffineEvaluatorParameters {
                origins,
                evaluator_units_per_chart_unit,
            } => Ok([
                origins[0] + coordinates[0] * evaluator_units_per_chart_unit[0],
                origins[1] + coordinates[1] * evaluator_units_per_chart_unit[1],
            ]),
            Self::PolarEvaluatorParameters {
                angular_axis,
                radial_axis,
                angular_origin,
                angular_span,
                singular_coordinate,
                radial_direction,
                ..
            } => {
                let radius = coordinates[0].hypot(coordinates[1]);
                let mut angle = if radius == 0.0 {
                    0.0
                } else {
                    coordinates[1].atan2(coordinates[0])
                };
                if angle < 0.0 && angle.abs() <= PARAMETER_TOLERANCE {
                    angle = 0.0;
                }
                if angle < 0.0 || angle > *angular_span + PARAMETER_TOLERANCE {
                    return Err("chart point lies outside the bounded polar sector");
                }
                let mut uv = [0.0; 2];
                uv[*angular_axis as usize] = angular_origin + angle.min(*angular_span);
                uv[*radial_axis as usize] = singular_coordinate + radial_direction * radius;
                Ok(uv)
            }
        }
    }

    pub(crate) fn reverses_evaluator_orientation(&self) -> bool {
        match self {
            Self::EvaluatorParameters => false,
            Self::AffineEvaluatorParameters { .. } => false,
            Self::PolarEvaluatorParameters {
                angular_axis,
                radial_direction,
                ..
            } => {
                let determinant_sign = if *angular_axis == 0 {
                    -*radial_direction
                } else {
                    *radial_direction
                };
                determinant_sign < 0.0
            }
        }
    }

    fn chart_coordinates(&self, uv: [f64; 2]) -> Result<[f64; 2], &'static str> {
        match self {
            Self::EvaluatorParameters => Ok(uv),
            Self::AffineEvaluatorParameters {
                origins,
                evaluator_units_per_chart_unit,
            } => Ok([
                (uv[0] - origins[0]) / evaluator_units_per_chart_unit[0],
                (uv[1] - origins[1]) / evaluator_units_per_chart_unit[1],
            ]),
            Self::PolarEvaluatorParameters {
                angular_axis,
                radial_axis,
                angular_origin,
                angular_span,
                singular_coordinate,
                radial_direction,
                ..
            } => {
                let angle = uv[*angular_axis as usize] - angular_origin;
                let radius = radial_direction * (uv[*radial_axis as usize] - singular_coordinate);
                if angle < -PARAMETER_TOLERANCE
                    || angle > *angular_span + PARAMETER_TOLERANCE
                    || radius < -PARAMETER_TOLERANCE
                {
                    return Err("evaluator parameters lie outside the bounded polar sector");
                }
                let angle = angle.clamp(0.0, *angular_span);
                let radius = radius.max(0.0);
                Ok([radius * angle.cos(), radius * angle.sin()])
            }
        }
    }
}

pub(super) fn build_singular_parameterization(
    boundary: &mut ExactFaceBoundary,
    topology: &ExactBRepTopology,
    evaluator_id: &SurfaceEvaluatorId,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    control: &dyn GeometryEvaluationControl,
    options: ExactFaceChartOptions,
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    let mut samples = Vec::new();
    let mut singular_samples = Vec::new();
    let mut singular_point = None::<[f64; 3]>;
    for segment in boundary_segments(boundary) {
        for uv in segment.node_uv {
            control
                .checkpoint()
                .map_err(|error| geometry_error(boundary, error))?;
            let derivatives = evaluator
                .derivatives(evaluator_id, uv, control)
                .map_err(|error| geometry_error(boundary, error))?;
            if derivatives
                .point_m
                .canonical_values()
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(invalid(
                    boundary,
                    "singular chart boundary point is not finite",
                ));
            }
            if is_singular(&derivatives) {
                if singular_point.is_some_and(|point| {
                    squared_distance(point, *derivatives.point_m.canonical_values())
                        > options.projection_tolerance_m.canonical_value()
                            * options.projection_tolerance_m.canonical_value()
                }) {
                    return Err(ExactFaceChartError::new(
                        ExactFaceChartErrorKind::RequiresMultipleCharts,
                        &boundary.source_face_id,
                        "one chart cannot contain multiple surface singularities",
                    ));
                }
                singular_point.get_or_insert(*derivatives.point_m.canonical_values());
                singular_samples.push(uv);
            }
            samples.push(uv);
        }
    }
    let parameterization = polar_parameterization(
        boundary,
        evaluator,
        evaluator_id,
        &samples,
        &singular_samples,
    )?;
    let source_face_id = boundary.source_face_id.clone();
    let mut canonical_nodes =
        BTreeMap::<runmat_meshing_core::StableDigest, ([f64; 3], [f64; 2])>::new();
    for segment in boundary_segments_mut(boundary) {
        for endpoint in 0..2 {
            let evaluator_uv = segment.node_uv[endpoint];
            let point = evaluator
                .point(evaluator_id, evaluator_uv, control)
                .map_err(|error| {
                    ExactFaceChartError::new(
                        ExactFaceChartErrorKind::GeometryEvaluation(error.kind),
                        &source_face_id,
                        error.reason,
                    )
                })?;
            let coordinates =
                parameterization
                    .chart_coordinates(evaluator_uv)
                    .map_err(|reason| {
                        ExactFaceChartError::new(
                            ExactFaceChartErrorKind::InvalidInput,
                            &source_face_id,
                            reason,
                        )
                    })?;
            let node_id = segment.node_ids[endpoint];
            segment.node_uv[endpoint] = if let Some((canonical_point, canonical_coordinates)) =
                canonical_nodes.get(&node_id)
            {
                if squared_distance(point.canonical_value(), *canonical_point)
                    > options.projection_tolerance_m.canonical_value()
                        * options.projection_tolerance_m.canonical_value()
                {
                    return Err(ExactFaceChartError::new(
                        ExactFaceChartErrorKind::InvalidInput,
                        &source_face_id,
                        "one topological boundary node evaluates to distinct physical points",
                    ));
                }
                *canonical_coordinates
            } else {
                canonical_nodes.insert(node_id, (point.canonical_value(), coordinates));
                coordinates
            };
        }
    }
    remove_collapsed_degenerate_segments(boundary, topology)?;
    Ok(parameterization)
}

fn polar_parameterization(
    boundary: &ExactFaceBoundary,
    evaluator: &(impl ExactSurfaceEvaluator + ?Sized),
    evaluator_id: &SurfaceEvaluatorId,
    samples: &[[f64; 2]],
    singular_samples: &[[f64; 2]],
) -> Result<ExactFaceChartParameterization, ExactFaceChartError> {
    if singular_samples.is_empty() || samples.len() < 3 {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "singular face has no bounded singular boundary",
        ));
    }
    let periodicity = evaluator
        .periodicity(evaluator_id)
        .map_err(|error| geometry_error(boundary, error))?;
    let angular_axis = periodicity
        .iter()
        .position(Option::is_some)
        .ok_or_else(|| {
            ExactFaceChartError::new(
                ExactFaceChartErrorKind::RequiresMultipleCharts,
                &boundary.source_face_id,
                "singular face has no periodic angular parameter",
            )
        })?;
    let radial_axis = 1 - angular_axis;
    let angular_origin = samples
        .iter()
        .map(|uv| uv[angular_axis])
        .min_by(f64::total_cmp)
        .unwrap();
    let angular_end = samples
        .iter()
        .map(|uv| uv[angular_axis])
        .max_by(f64::total_cmp)
        .unwrap();
    let angular_span = angular_end - angular_origin;
    let singular_coordinate = singular_samples[0][radial_axis];
    let parameter_scale = samples
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let tolerance = PARAMETER_TOLERANCE * parameter_scale;
    if singular_samples
        .iter()
        .any(|uv| (uv[radial_axis] - singular_coordinate).abs() > tolerance)
        || !angular_span.is_finite()
        || angular_span <= tolerance
        || angular_span > MAXIMUM_POLAR_ANGLE_RAD + tolerance
    {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "singular face exceeds one bounded polar sector",
        ));
    }
    let farthest = samples
        .iter()
        .map(|uv| uv[radial_axis] - singular_coordinate)
        .max_by(|left, right| left.abs().total_cmp(&right.abs()))
        .unwrap();
    if !farthest.is_finite() || farthest.abs() <= tolerance {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "singular face has no radial parameter extent",
        ));
    }
    let radial_direction = farthest.signum();
    if samples
        .iter()
        .any(|uv| radial_direction * (uv[radial_axis] - singular_coordinate) < -tolerance)
    {
        return Err(ExactFaceChartError::new(
            ExactFaceChartErrorKind::RequiresMultipleCharts,
            &boundary.source_face_id,
            "singular face crosses its collapsed radial boundary",
        ));
    }
    let radial_extent = farthest.abs();
    let result = ExactFaceChartParameterization::PolarEvaluatorParameters {
        angular_axis: angular_axis as u8,
        radial_axis: radial_axis as u8,
        angular_origin,
        angular_span,
        singular_coordinate,
        radial_direction,
        differential_step: (radial_extent * f64::EPSILON.cbrt())
            .max(radial_extent * 1.0e-9)
            .min(radial_extent * 0.125),
    };
    result
        .validate()
        .map_err(|reason| invalid(boundary, reason))?;
    Ok(result)
}

fn remove_collapsed_degenerate_segments(
    boundary: &mut ExactFaceBoundary,
    topology: &ExactBRepTopology,
) -> Result<(), ExactFaceChartError> {
    let source_face_id = boundary.source_face_id.clone();
    for loop_boundary in std::iter::once(&mut boundary.outer_loop).chain(&mut boundary.inner_loops)
    {
        for segment in &loop_boundary.segments {
            let collapsed = segment.node_ids[0] == segment.node_ids[1]
                && segment.node_uv[0] == segment.node_uv[1];
            if collapsed
                && topology
                    .edges
                    .iter()
                    .find(|edge| edge.id == segment.source_edge_id)
                    .is_none_or(|edge| !edge.is_degenerate)
            {
                return Err(ExactFaceChartError::new(
                    ExactFaceChartErrorKind::InvalidInput,
                    &source_face_id,
                    "a nondegenerate exact trim collapses in the singular chart",
                ));
            }
        }
        loop_boundary.segments.retain(|segment| {
            segment.node_ids[0] != segment.node_ids[1] || segment.node_uv[0] != segment.node_uv[1]
        });
        if loop_boundary.segments.is_empty() {
            return Err(ExactFaceChartError::new(
                ExactFaceChartErrorKind::RequiresMultipleCharts,
                &source_face_id,
                "singular chart boundary collapses to one point",
            ));
        }
    }
    Ok(())
}

pub(super) fn is_singular(derivatives: &SurfaceDerivatives) -> bool {
    let area = norm(cross(
        derivatives
            .du_m
            .with_canonical_storage(|values, _axes| *values),
        derivatives
            .dv_m
            .with_canonical_storage(|values, _axes| *values),
    ));
    let scale = norm(
        derivatives
            .du_m
            .with_canonical_storage(|values, _axes| *values),
    ) * norm(
        derivatives
            .dv_m
            .with_canonical_storage(|values, _axes| *values),
    );
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

fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum()
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    vector
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}
