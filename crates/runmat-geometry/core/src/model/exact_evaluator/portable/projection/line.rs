//! Allocation-free analytic projection shared by spatial and parametric lines.
use super::{
    invalid_result, GeometryEvaluationControl, GeometryEvaluationError, ParameterRange,
    ParametricProjection,
};
use crate::model::exact_evaluator::portable::vector::{add_scaled, distance, dot, subtract};

pub(in crate::model::exact_evaluator::portable) fn project_line<const N: usize>(
    range: ParameterRange,
    origin: &[f64; N],
    direction: &[f64; N],
    point: [f64; N],
    control: &dyn GeometryEvaluationControl,
) -> Result<ParametricProjection<N>, GeometryEvaluationError> {
    control.checkpoint()?;
    control.consume_search_work(1)?;
    let offset = subtract(&point, origin);
    let parameter = dot(&offset, direction) / dot(direction, direction);
    if !parameter.is_finite() {
        return Err(invalid_result(
            "line projection produced an invalid parameter",
        ));
    }
    let parameter = parameter.clamp(range.start, range.end);
    let projected = add_scaled(origin, direction, parameter, &[0.0; N], 0.0);
    let distance = distance(&projected, &point);
    if !distance.is_finite() {
        return Err(invalid_result(
            "line projection produced an invalid distance",
        ));
    }
    Ok(ParametricProjection {
        parameter,
        point: projected,
        distance,
    })
}
