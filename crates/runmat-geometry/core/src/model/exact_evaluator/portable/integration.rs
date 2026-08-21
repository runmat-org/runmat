use super::super::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind, ParameterRange,
};

const MAX_INTEGRATION_DEPTH: u8 = 32;
const MAX_INTEGRATION_INTERVALS: usize = 1_000_000;

#[derive(Debug, Clone, Copy)]
struct Interval {
    start: f64,
    midpoint: f64,
    end: f64,
    start_value: f64,
    midpoint_value: f64,
    end_value: f64,
    estimate: f64,
    tolerance: f64,
    depth: u8,
}

pub(super) fn adaptive_arc_length(
    range: ParameterRange,
    absolute_error_m: f64,
    control: &dyn GeometryEvaluationControl,
    speed: impl FnMut(f64) -> Result<f64, GeometryEvaluationError>,
) -> Result<f64, GeometryEvaluationError> {
    let value = adaptive_scalar_integral(
        range,
        absolute_error_m,
        control,
        "arc-length integration",
        speed,
    )?;
    if value < 0.0 {
        return Err(invalid("arc-length integration produced a negative result"));
    }
    Ok(value)
}

pub(super) fn adaptive_scalar_integral(
    range: ParameterRange,
    absolute_error: f64,
    control: &dyn GeometryEvaluationControl,
    operation: &str,
    mut value_at: impl FnMut(f64) -> Result<f64, GeometryEvaluationError>,
) -> Result<f64, GeometryEvaluationError> {
    if !absolute_error.is_finite() || absolute_error <= 0.0 {
        return Err(invalid(
            "integration error bound must be finite and positive",
        ));
    }
    let initial_midpoint = midpoint(range.start, range.end);
    let start_value = finite_sample(value_at(range.start)?, operation)?;
    let midpoint_value = finite_sample(value_at(initial_midpoint)?, operation)?;
    let end_value = finite_sample(value_at(range.end)?, operation)?;
    let initial = simpson(
        range.start,
        range.end,
        start_value,
        midpoint_value,
        end_value,
    );
    let mut pending = vec![Interval {
        start: range.start,
        midpoint: initial_midpoint,
        end: range.end,
        start_value,
        midpoint_value,
        end_value,
        estimate: initial,
        tolerance: absolute_error,
        depth: 0,
    }];
    let mut accepted = 0.0;
    let mut visited = 0usize;
    while let Some(interval) = pending.pop() {
        control.checkpoint()?;
        control.consume_iterations(1)?;
        visited = visited.saturating_add(1);
        if visited > MAX_INTEGRATION_INTERVALS {
            return Err(budget(&format!(
                "{operation} exceeded its hard interval bound"
            )));
        }
        let left_midpoint = midpoint(interval.start, interval.midpoint);
        let right_midpoint = midpoint(interval.midpoint, interval.end);
        let left_midpoint_value = finite_sample(value_at(left_midpoint)?, operation)?;
        let right_midpoint_value = finite_sample(value_at(right_midpoint)?, operation)?;
        let left = simpson(
            interval.start,
            interval.midpoint,
            interval.start_value,
            left_midpoint_value,
            interval.midpoint_value,
        );
        let right = simpson(
            interval.midpoint,
            interval.end,
            interval.midpoint_value,
            right_midpoint_value,
            interval.end_value,
        );
        let refined = left + right;
        let error = (refined - interval.estimate).abs() / 15.0;
        if error <= interval.tolerance {
            accepted += refined + (refined - interval.estimate) / 15.0;
            continue;
        }
        if interval.depth >= MAX_INTEGRATION_DEPTH
            || left_midpoint == interval.start
            || right_midpoint == interval.midpoint
        {
            return Err(budget(&format!(
                "{operation} could not meet the requested error within its hard depth bound"
            )));
        }
        let child_tolerance = interval.tolerance * 0.5;
        let depth = interval.depth + 1;
        pending.push(Interval {
            start: interval.midpoint,
            midpoint: right_midpoint,
            end: interval.end,
            start_value: interval.midpoint_value,
            midpoint_value: right_midpoint_value,
            end_value: interval.end_value,
            estimate: right,
            tolerance: child_tolerance,
            depth,
        });
        pending.push(Interval {
            start: interval.start,
            midpoint: left_midpoint,
            end: interval.midpoint,
            start_value: interval.start_value,
            midpoint_value: left_midpoint_value,
            end_value: interval.midpoint_value,
            estimate: left,
            tolerance: child_tolerance,
            depth,
        });
    }
    if !accepted.is_finite() {
        return Err(invalid("integration produced a non-finite result"));
    }
    Ok(accepted)
}

fn finite_sample(value: f64, operation: &str) -> Result<f64, GeometryEvaluationError> {
    if !value.is_finite() {
        return Err(invalid(&format!(
            "{operation} produced a non-finite sample"
        )));
    }
    Ok(value)
}

fn simpson(start: f64, end: f64, start_value: f64, midpoint_value: f64, end_value: f64) -> f64 {
    (end - start) * (start_value + 4.0 * midpoint_value + end_value) / 6.0
}

fn midpoint(start: f64, end: f64) -> f64 {
    start + (end - start) * 0.5
}

fn budget(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::BudgetExceeded, reason)
}

fn invalid(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
