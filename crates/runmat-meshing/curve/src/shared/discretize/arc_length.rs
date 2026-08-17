use crate::shared::{SharedCurveError, SharedCurveErrorKind};
use runmat_geometry_core::{
    ExactCurveEvaluator, ExactEdge, GeometryEvaluationControl, GeometryTransform, ParameterRange,
};

use super::{
    error::{edge_error, geometry_error},
    math::{dot, norm},
};

pub(crate) fn world_arc_length(
    edge: &ExactEdge,
    curves: &dyn ExactCurveEvaluator,
    control: &dyn GeometryEvaluationControl,
    transform: GeometryTransform,
    range: ParameterRange,
    absolute_error_m: f64,
    maximum_depth: u16,
) -> Result<f64, SharedCurveError> {
    if let Some(scale) = similarity_scale(transform) {
        return curves
            .arc_length_m(
                &edge.curve_evaluator_id,
                range,
                absolute_error_m / scale,
                control,
            )
            .map(|length| length * scale)
            .map_err(|error| geometry_error(edge, error));
    }
    let speed = |parameter| {
        control
            .checkpoint()
            .map_err(|error| geometry_error(edge, error))?;
        let derivatives = curves
            .derivatives(&edge.curve_evaluator_id, parameter, control)
            .map_err(|error| geometry_error(edge, error))?;
        let value = norm(transform.transform_vector(derivatives.first_m));
        if !value.is_finite() || value <= 0.0 {
            return Err(edge_error(
                edge,
                SharedCurveErrorKind::GeometricMismatch,
                "curve arc-length derivative",
                "transformed curve speed must be finite and positive",
            ));
        }
        Ok(value)
    };
    let start = speed(range.start)?;
    let midpoint_parameter = (range.start + range.end) * 0.5;
    let midpoint = speed(midpoint_parameter)?;
    let end = speed(range.end)?;
    adaptive_simpson(
        edge,
        &speed,
        range.start,
        range.end,
        [start, midpoint, end],
        absolute_error_m,
        0,
        maximum_depth,
    )
}

#[allow(clippy::too_many_arguments)]
fn adaptive_simpson(
    edge: &ExactEdge,
    speed: &dyn Fn(f64) -> Result<f64, SharedCurveError>,
    start: f64,
    end: f64,
    values: [f64; 3],
    tolerance: f64,
    depth: u16,
    maximum_depth: u16,
) -> Result<f64, SharedCurveError> {
    let midpoint = (start + end) * 0.5;
    let left_midpoint = (start + midpoint) * 0.5;
    let right_midpoint = (midpoint + end) * 0.5;
    let left_value = speed(left_midpoint)?;
    let right_value = speed(right_midpoint)?;
    let whole = (end - start) * (values[0] + 4.0 * values[1] + values[2]) / 6.0;
    let left = (midpoint - start) * (values[0] + 4.0 * left_value + values[1]) / 6.0;
    let right = (end - midpoint) * (values[1] + 4.0 * right_value + values[2]) / 6.0;
    let difference = (left + right - whole).abs();
    if difference <= 15.0 * tolerance {
        return Ok(left + right + (left + right - whole) / 15.0);
    }
    if depth >= maximum_depth {
        return Err(edge_error(
            edge,
            SharedCurveErrorKind::ResourceLimit,
            "curve arc-length integration",
            "requested absolute error exceeds the integration depth limit",
        ));
    }
    Ok(adaptive_simpson(
        edge,
        speed,
        start,
        midpoint,
        [values[0], left_value, values[1]],
        tolerance * 0.5,
        depth + 1,
        maximum_depth,
    )? + adaptive_simpson(
        edge,
        speed,
        midpoint,
        end,
        [values[1], right_value, values[2]],
        tolerance * 0.5,
        depth + 1,
        maximum_depth,
    )?)
}

fn similarity_scale(transform: GeometryTransform) -> Option<f64> {
    let columns = [
        [transform.0[0], transform.0[4], transform.0[8]],
        [transform.0[1], transform.0[5], transform.0[9]],
        [transform.0[2], transform.0[6], transform.0[10]],
    ];
    let squared = columns.map(|column| dot(column, column));
    let scale_squared = squared[0];
    let tolerance = scale_squared.max(1.0) * 1.0e-12;
    (scale_squared.is_finite()
        && scale_squared > 0.0
        && (squared[1] - scale_squared).abs() <= tolerance
        && (squared[2] - scale_squared).abs() <= tolerance
        && dot(columns[0], columns[1]).abs() <= tolerance
        && dot(columns[0], columns[2]).abs() <= tolerance
        && dot(columns[1], columns[2]).abs() <= tolerance)
        .then(|| scale_squared.sqrt())
}
