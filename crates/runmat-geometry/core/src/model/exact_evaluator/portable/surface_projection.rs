use super::super::{
    GeometryEvaluationControl, GeometryEvaluationError, GeometryEvaluationErrorKind,
    ParameterRange, SurfaceDerivatives, SurfaceProjection,
};
use super::invalid_result;
use super::vector::{distance, dot, norm, subtract};

const MAX_NEWTON_ITERATIONS: usize = 64;
const MAX_SURFACE_SEEDS: usize = 1_000_000;

pub(super) fn project_surface(
    bounds: [ParameterRange; 2],
    point_m: [f64; 3],
    absolute_error_m: f64,
    seeds: impl IntoIterator<Item = [f64; 2]>,
    control: &dyn GeometryEvaluationControl,
    mut evaluate: impl FnMut([f64; 2]) -> Result<SurfaceDerivatives, GeometryEvaluationError>,
) -> Result<SurfaceProjection, GeometryEvaluationError> {
    if point_m.iter().any(|value| !value.is_finite())
        || !absolute_error_m.is_finite()
        || absolute_error_m <= 0.0
    {
        return Err(invalid_result(
            "surface projection input and error bound must be finite and valid",
        ));
    }
    let mut best = None;
    let mut converged = false;
    let mut seed_count = 0usize;
    for seed in seeds {
        if !in_bounds(seed, bounds) {
            continue;
        }
        seed_count = seed_count.saturating_add(1);
        if seed_count > MAX_SURFACE_SEEDS {
            return Err(budget("surface projection exceeds its hard seed bound"));
        }
        control.checkpoint()?;
        control.consume_search_work(1)?;
        let Some(uv) = refine(
            seed,
            bounds,
            &point_m,
            absolute_error_m,
            control,
            &mut evaluate,
        )?
        else {
            continue;
        };
        converged = true;
        consider(uv, &point_m, &mut best, &mut evaluate)?;
    }
    if !converged {
        return Err(GeometryEvaluationError::new(
            GeometryEvaluationErrorKind::ProjectionDidNotConverge,
            "surface projection did not isolate a constrained stationary point within its hard iteration bound",
        ));
    }
    best.ok_or_else(|| invalid_result("surface projection produced no candidate"))
}

fn refine(
    seed: [f64; 2],
    bounds: [ParameterRange; 2],
    point_m: &[f64; 3],
    absolute_error_m: f64,
    control: &dyn GeometryEvaluationControl,
    evaluate: &mut impl FnMut([f64; 2]) -> Result<SurfaceDerivatives, GeometryEvaluationError>,
) -> Result<Option<[f64; 2]>, GeometryEvaluationError> {
    let mut uv = seed;
    for _ in 0..MAX_NEWTON_ITERATIONS {
        control.checkpoint()?;
        control.consume_iterations(1)?;
        let value = evaluate(uv)?;
        let offset = subtract(&value.point_m, point_m);
        let gradient = [dot(&offset, &value.du_m), dot(&offset, &value.dv_m)];
        let scale = [norm(&value.du_m).max(1.0), norm(&value.dv_m).max(1.0)];
        if constrained_stationary(uv, bounds, gradient, scale, absolute_error_m) {
            return Ok(Some(uv));
        }
        let hessian = [
            [
                dot(&value.du_m, &value.du_m) + dot(&offset, &value.duu_m),
                dot(&value.du_m, &value.dv_m) + dot(&offset, &value.duv_m),
            ],
            [
                dot(&value.du_m, &value.dv_m) + dot(&offset, &value.duv_m),
                dot(&value.dv_m, &value.dv_m) + dot(&offset, &value.dvv_m),
            ],
        ];
        let determinant = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
        if !determinant.is_finite() || determinant.abs() <= f64::MIN_POSITIVE {
            break;
        }
        let step = [
            (hessian[1][1] * gradient[0] - hessian[0][1] * gradient[1]) / determinant,
            (-hessian[1][0] * gradient[0] + hessian[0][0] * gradient[1]) / determinant,
        ];
        let next = [
            (uv[0] - step[0]).clamp(bounds[0].start, bounds[0].end),
            (uv[1] - step[1]).clamp(bounds[1].start, bounds[1].end),
        ];
        if next.iter().any(|value| !value.is_finite()) || next == uv {
            break;
        }
        uv = next;
    }
    Ok(None)
}

fn constrained_stationary(
    uv: [f64; 2],
    bounds: [ParameterRange; 2],
    gradient: [f64; 2],
    scale: [f64; 2],
    tolerance: f64,
) -> bool {
    (0..2).all(|axis| {
        let threshold = tolerance * scale[axis];
        gradient[axis].abs() <= threshold
            || (uv[axis] == bounds[axis].start && gradient[axis] >= -threshold)
            || (uv[axis] == bounds[axis].end && gradient[axis] <= threshold)
    })
}

fn consider(
    uv: [f64; 2],
    point_m: &[f64; 3],
    best: &mut Option<SurfaceProjection>,
    evaluate: &mut impl FnMut([f64; 2]) -> Result<SurfaceDerivatives, GeometryEvaluationError>,
) -> Result<(), GeometryEvaluationError> {
    let value = evaluate(uv)?;
    let candidate = SurfaceProjection {
        uv,
        point_m: value.point_m,
        distance_m: distance(&value.point_m, point_m),
    };
    if !candidate.distance_m.is_finite() {
        return Err(invalid_result(
            "surface projection produced a non-finite distance",
        ));
    }
    if best.as_ref().is_none_or(|current| {
        candidate.distance_m < current.distance_m
            || (candidate.distance_m == current.distance_m
                && (candidate.uv[0] < current.uv[0]
                    || (candidate.uv[0] == current.uv[0] && candidate.uv[1] < current.uv[1])))
    }) {
        *best = Some(candidate);
    }
    Ok(())
}

fn in_bounds(uv: [f64; 2], bounds: [ParameterRange; 2]) -> bool {
    uv.iter().all(|value| value.is_finite())
        && uv[0] >= bounds[0].start
        && uv[0] <= bounds[0].end
        && uv[1] >= bounds[1].start
        && uv[1] <= bounds[1].end
}

pub(super) fn uniform_surface_seeds(
    bounds: [ParameterRange; 2],
    intervals_per_axis: usize,
) -> Vec<[f64; 2]> {
    let intervals = intervals_per_axis.clamp(1, 999);
    let mut seeds = Vec::with_capacity((intervals + 1).saturating_mul(intervals + 1));
    for u_index in 0..=intervals {
        let u =
            bounds[0].start + (bounds[0].end - bounds[0].start) * u_index as f64 / intervals as f64;
        for v_index in 0..=intervals {
            let v = bounds[1].start
                + (bounds[1].end - bounds[1].start) * v_index as f64 / intervals as f64;
            seeds.push([u, v]);
        }
    }
    seeds
}

fn budget(reason: &str) -> GeometryEvaluationError {
    GeometryEvaluationError::new(GeometryEvaluationErrorKind::BudgetExceeded, reason)
}
