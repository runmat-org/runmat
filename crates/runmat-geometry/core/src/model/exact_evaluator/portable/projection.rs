use super::super::{
    CurveDerivatives, CurveProjection, GeometryEvaluationControl, GeometryEvaluationError,
    GeometryEvaluationErrorKind, ParameterRange,
};
use super::invalid_result;
use super::vector::{distance, dot, norm, subtract};

const MAX_NEWTON_ITERATIONS: usize = 64;
const MAX_PROJECTION_SEEDS: usize = 1_000_000;

#[derive(Clone, Copy)]
pub(super) struct ParametricDerivatives<const N: usize> {
    pub point: [f64; N],
    pub first: [f64; N],
    pub second: [f64; N],
}

pub(super) struct ParametricProjection<const N: usize> {
    pub parameter: f64,
    pub point: [f64; N],
    pub distance: f64,
}

pub(super) fn project_curve(
    range: ParameterRange,
    point_m: [f64; 3],
    absolute_error_m: f64,
    seeds: impl IntoIterator<Item = f64>,
    control: &dyn GeometryEvaluationControl,
    mut evaluate: impl FnMut(f64) -> Result<CurveDerivatives, GeometryEvaluationError>,
) -> Result<CurveProjection, GeometryEvaluationError> {
    let projection = project_parametric(
        range,
        point_m,
        absolute_error_m,
        seeds,
        control,
        |parameter| {
            let value = evaluate(parameter)?;
            Ok(ParametricDerivatives {
                point: value.point_m,
                first: value.first_m,
                second: value.second_m,
            })
        },
    )?;
    Ok(CurveProjection {
        parameter: projection.parameter,
        point_m: projection.point,
        distance_m: projection.distance,
    })
}

pub(super) fn project_parametric<const N: usize>(
    range: ParameterRange,
    point: [f64; N],
    absolute_error: f64,
    seeds: impl IntoIterator<Item = f64>,
    control: &dyn GeometryEvaluationControl,
    mut evaluate: impl FnMut(f64) -> Result<ParametricDerivatives<N>, GeometryEvaluationError>,
) -> Result<ParametricProjection<N>, GeometryEvaluationError> {
    // Callers seed every analytic interval or nonzero NURBS knot span in
    // parameter order. We refine each seed independently and resolve equal
    // distances by the lowest parameter, so search order cannot change output.
    if point.iter().any(|value| !value.is_finite())
        || !absolute_error.is_finite()
        || absolute_error <= 0.0
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
            charge_seed_allocation(1, std::mem::size_of::<f64>(), control)?;
            candidates.push(seed);
        }
    }
    candidates.sort_by(f64::total_cmp);
    candidates.dedup_by(|left, right| left.to_bits() == right.to_bits());

    let mut best = None;
    let endpoint_optimal =
        endpoint_is_optimal(range.start, true, &point, absolute_error, &mut evaluate)?
            || endpoint_is_optimal(range.end, false, &point, absolute_error, &mut evaluate)?;
    consider(range.start, &point, &mut best, &mut evaluate)?;
    consider(range.end, &point, &mut best, &mut evaluate)?;
    let mut converged_stationary_point = false;
    for seed in candidates {
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let Some(parameter) =
            refine_stationary_point(seed, range, &point, absolute_error, control, &mut evaluate)?
        else {
            continue;
        };
        converged_stationary_point = true;
        consider(parameter, &point, &mut best, &mut evaluate)?;
    }
    if !converged_stationary_point && !endpoint_optimal {
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::ProjectionDidNotConverge,
            "curve projection did not isolate a stationary point within its hard iteration bound",
        ));
    }
    best.ok_or_else(|| invalid("projection produced no candidate"))
}

fn endpoint_is_optimal<const N: usize>(
    parameter: f64,
    is_start: bool,
    point: &[f64; N],
    absolute_error: f64,
    evaluate: &mut impl FnMut(f64) -> Result<ParametricDerivatives<N>, GeometryEvaluationError>,
) -> Result<bool, GeometryEvaluationError> {
    let value = evaluate(parameter)?;
    let gradient = dot(&subtract(&value.point, point), &value.first);
    let threshold = absolute_error * norm(&value.first).max(1.0);
    Ok(if is_start {
        gradient >= -threshold
    } else {
        gradient <= threshold
    })
}

fn consider<const N: usize>(
    parameter: f64,
    point: &[f64; N],
    best: &mut Option<ParametricProjection<N>>,
    evaluate: &mut impl FnMut(f64) -> Result<ParametricDerivatives<N>, GeometryEvaluationError>,
) -> Result<(), GeometryEvaluationError> {
    let evaluation = evaluate(parameter)?;
    let candidate = ParametricProjection {
        parameter,
        point: evaluation.point,
        distance: distance(&evaluation.point, point),
    };
    if !candidate.distance.is_finite() {
        return Err(invalid("projection produced a non-finite distance"));
    }
    if best.as_ref().is_none_or(|current| {
        candidate.distance < current.distance
            || (candidate.distance == current.distance && candidate.parameter < current.parameter)
    }) {
        *best = Some(candidate);
    }
    Ok(())
}

fn refine_stationary_point<const N: usize>(
    seed: f64,
    range: ParameterRange,
    point: &[f64; N],
    absolute_error: f64,
    control: &dyn GeometryEvaluationControl,
    evaluate: &mut impl FnMut(f64) -> Result<ParametricDerivatives<N>, GeometryEvaluationError>,
) -> Result<Option<f64>, GeometryEvaluationError> {
    let mut parameter = seed;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        control.checkpoint()?;
        control.consume_iterations(1)?;
        let value = evaluate(parameter)?;
        let offset = subtract(&value.point, point);
        let gradient = dot(&offset, &value.first);
        let hessian = dot(&value.first, &value.first) + dot(&offset, &value.second);
        let speed = norm(&value.first);
        if gradient.abs() <= absolute_error * speed.max(1.0) {
            return Ok(Some(parameter));
        }
        if !hessian.is_finite() || hessian.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let next = (parameter - gradient / hessian).clamp(range.start, range.end);
        if next == parameter || (next - parameter).abs() * speed.max(1.0) <= absolute_error {
            return Ok(Some(next));
        }
        parameter = next;
    }
    Ok(None)
}

pub(super) fn uniform_seeds(range: ParameterRange, interval_count: usize) -> Vec<f64> {
    let interval_count = interval_count.clamp(1, MAX_PROJECTION_SEEDS - 1);
    (0..=interval_count)
        .map(|index| range.start + (range.end - range.start) * index as f64 / interval_count as f64)
        .collect()
}

pub(super) fn charge_seed_allocation(
    count: usize,
    bytes_per_seed: usize,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), GeometryEvaluationError> {
    let bytes = count
        .checked_mul(bytes_per_seed)
        .ok_or_else(|| invalid_result("projection seed allocation-byte count overflow"))?;
    control.consume_allocation_bytes(
        u64::try_from(bytes).map_err(|_| {
            invalid_result("projection seed allocation-byte count does not fit u64")
        })?,
    )
}

fn invalid(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::InvalidResult, reason)
}
