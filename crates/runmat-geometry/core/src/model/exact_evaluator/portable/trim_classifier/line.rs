//! Exact straight-segment formulas with fallback at floating decision boundaries.
use super::{
    evaluate_pcurve, ExactPcurveDefinition, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange,
};
use crate::model::exact_evaluator::portable::projection::project_line;

pub(super) enum LineLocation {
    Boundary,
    Winding(f64),
}

pub(super) fn classify_line(
    definition: &ExactPcurveDefinition,
    range: ParameterRange,
    query: [f64; 2],
    tolerance: f64,
    angular_error: f64,
    control: &dyn GeometryEvaluationControl,
) -> Result<Option<LineLocation>, GeometryEvaluationError> {
    let ExactPcurveDefinition::Line {
        origin_uv,
        direction_uv_per_parameter,
        ..
    } = definition
    else {
        return Ok(None);
    };
    let projected = match project_line(range, origin_uv, direction_uv_per_parameter, query, control)
    {
        Ok(value) => value,
        Err(failure) if failure.kind == GeometryEvaluationErrorKind::InvalidResult => {
            return Ok(None)
        }
        Err(failure) => return Err(failure),
    };
    // Equality is the same represented line evaluation used by the general
    // projection path; no tolerance enlargement or near-boundary shortcut.
    if projected.point == query {
        return Ok(Some(LineLocation::Boundary));
    }
    let first = evaluate_pcurve(definition, range.start, control)?.point_uv;
    let last = evaluate_pcurve(definition, range.end, control)?.point_uv;
    let scale = first
        .into_iter()
        .chain(last)
        .chain(query)
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let roundoff = 64.0 * f64::EPSILON * scale;
    if (projected.distance - tolerance).abs() <= roundoff {
        return Ok(None);
    }
    if projected.distance < tolerance {
        return Ok(Some(LineLocation::Boundary));
    }
    if angular_error < 64.0 * f64::EPSILON {
        return Ok(None);
    }
    let first = [first[0] - query[0], first[1] - query[1]];
    let last = [last[0] - query[0], last[1] - query[1]];
    let first_length = first[0].hypot(first[1]);
    let last_length = last[0].hypot(last[1]);
    if !first_length.is_finite()
        || !last_length.is_finite()
        || first_length == 0.0
        || last_length == 0.0
    {
        return Ok(None);
    }
    let first = first.map(|value| value / first_length);
    let last = last.map(|value| value / last_length);
    let cross = first[0] * last[1] - first[1] * last[0];
    let dot = first[0] * last[0] + first[1] * last[1];
    // The principal-angle branch must not choose the wrong sign at +/-pi.
    if dot < 0.0 && cross.abs() <= 64.0 * f64::EPSILON {
        return Ok(None);
    }
    let angle = cross.atan2(dot);
    Ok(angle.is_finite().then_some(LineLocation::Winding(angle)))
}

#[cfg(test)]
mod tests;
