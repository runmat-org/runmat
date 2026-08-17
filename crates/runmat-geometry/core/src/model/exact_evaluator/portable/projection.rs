use super::super::{
    CurveDerivativesV2, CurveProjectionV2, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRangeV2,
};
use super::vector::{distance, dot, norm, subtract};

const MAX_NEWTON_ITERATIONS: usize = 64;
const MAX_PROJECTION_SEEDS: usize = 1_000_000;

pub(super) fn project_curve(
    range: ParameterRangeV2,
    point_m: [f64; 3],
    absolute_error_m: f64,
    seeds: impl IntoIterator<Item = f64>,
    control: &dyn GeometryEvaluationControl,
    mut evaluate: impl FnMut(f64) -> Result<CurveDerivativesV2, GeometryEvaluationError>,
) -> Result<CurveProjectionV2, GeometryEvaluationError> {
    // Callers seed every analytic interval or nonzero NURBS knot span in
    // parameter order. We refine each seed independently and resolve equal
    // distances by the lowest parameter, so search order cannot change output.
    if point_m.iter().any(|value| !value.is_finite())
        || !absolute_error_m.is_finite()
        || absolute_error_m <= 0.0
    {
        return Err(invalid(
            "projection input and error bound must be finite and valid",
        ));
    }
    let mut candidates = Vec::new();
    for seed in seeds {
        if seed.is_finite() && seed >= range.start && seed <= range.end {
            if candidates.len() >= MAX_PROJECTION_SEEDS {
                return Err(GeometryEvaluationError::new(
                    GeometryEvaluationErrorKind::BudgetExceeded,
                    "curve projection exceeds its hard seed bound",
                ));
            }
            candidates.push(seed);
        }
    }
    candidates.sort_by(f64::total_cmp);
    candidates.dedup_by(|left, right| left.to_bits() == right.to_bits());

    let mut best = None;
    let endpoint_optimal =
        endpoint_is_optimal(range.start, true, &point_m, absolute_error_m, &mut evaluate)?
            || endpoint_is_optimal(range.end, false, &point_m, absolute_error_m, &mut evaluate)?;
    consider(range.start, &point_m, &mut best, &mut evaluate)?;
    consider(range.end, &point_m, &mut best, &mut evaluate)?;
    let mut converged_stationary_point = false;
    for seed in candidates {
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let Some(parameter) = refine_stationary_point(
            seed,
            range,
            &point_m,
            absolute_error_m,
            control,
            &mut evaluate,
        )?
        else {
            continue;
        };
        converged_stationary_point = true;
        consider(parameter, &point_m, &mut best, &mut evaluate)?;
    }
    if !converged_stationary_point && !endpoint_optimal {
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::ProjectionDidNotConverge,
            "curve projection did not isolate a stationary point within its hard iteration bound",
        ));
    }
    best.ok_or_else(|| invalid("projection produced no candidate"))
}

fn endpoint_is_optimal(
    parameter: f64,
    is_start: bool,
    point_m: &[f64; 3],
    absolute_error_m: f64,
    evaluate: &mut impl FnMut(f64) -> Result<CurveDerivativesV2, GeometryEvaluationError>,
) -> Result<bool, GeometryEvaluationError> {
    let value = evaluate(parameter)?;
    let gradient = dot(&subtract(&value.point_m, point_m), &value.first_m);
    let threshold = absolute_error_m * norm(&value.first_m).max(1.0);
    Ok(if is_start {
        gradient >= -threshold
    } else {
        gradient <= threshold
    })
}

fn consider(
    parameter: f64,
    point_m: &[f64; 3],
    best: &mut Option<CurveProjectionV2>,
    evaluate: &mut impl FnMut(f64) -> Result<CurveDerivativesV2, GeometryEvaluationError>,
) -> Result<(), GeometryEvaluationError> {
    let evaluation = evaluate(parameter)?;
    let candidate = CurveProjectionV2 {
        parameter,
        point_m: evaluation.point_m,
        distance_m: distance(&evaluation.point_m, point_m),
    };
    if !candidate.distance_m.is_finite() {
        return Err(invalid("projection produced a non-finite distance"));
    }
    if best.as_ref().is_none_or(|current| {
        candidate.distance_m < current.distance_m
            || (candidate.distance_m == current.distance_m
                && candidate.parameter < current.parameter)
    }) {
        *best = Some(candidate);
    }
    Ok(())
}

fn refine_stationary_point(
    seed: f64,
    range: ParameterRangeV2,
    point_m: &[f64; 3],
    absolute_error_m: f64,
    control: &dyn GeometryEvaluationControl,
    evaluate: &mut impl FnMut(f64) -> Result<CurveDerivativesV2, GeometryEvaluationError>,
) -> Result<Option<f64>, GeometryEvaluationError> {
    let mut parameter = seed;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        control.checkpoint()?;
        control.consume_iterations(1)?;
        let value = evaluate(parameter)?;
        let offset = subtract(&value.point_m, point_m);
        let gradient = dot(&offset, &value.first_m);
        let hessian = dot(&value.first_m, &value.first_m) + dot(&offset, &value.second_m);
        let speed = norm(&value.first_m);
        if gradient.abs() <= absolute_error_m * speed.max(1.0) {
            return Ok(Some(parameter));
        }
        if !hessian.is_finite() || hessian.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let next = (parameter - gradient / hessian).clamp(range.start, range.end);
        if next == parameter || (next - parameter).abs() * speed.max(1.0) <= absolute_error_m {
            return Ok(Some(next));
        }
        parameter = next;
    }
    Ok(None)
}

pub(super) fn uniform_seeds(range: ParameterRangeV2, interval_count: usize) -> Vec<f64> {
    let interval_count = interval_count.clamp(1, MAX_PROJECTION_SEEDS - 1);
    (0..=interval_count)
        .map(|index| range.start + (range.end - range.start) * index as f64 / interval_count as f64)
        .collect()
}

fn invalid(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
