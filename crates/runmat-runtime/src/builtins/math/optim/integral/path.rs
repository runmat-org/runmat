//! Finite real waypoint geometry. Complex contours and improper domains have
//! separate numerical rules and do not enter this finite Simpson path.

use num_complex::Complex;
use runmat_builtins::INTEGRAL_ERROR_INVALID_ARGUMENT;

use super::{
    integral_error_with_detail, integral_map_error, scalar_bound_host, IntegralEvaluator,
    IntegralOptionName, IntegralOptions,
};
use crate::builtins::math::optim::quadrature::{
    integrate_scaled, QuadratureCooperation, ScaledEstimate,
};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use runmat_builtins::INTEGRAL_ERROR_INVALID_INPUT;
use runmat_value::Value;

fn resource_error(source: std::collections::TryReserveError) -> crate::RuntimeError {
    build_runtime_error("integral: could not reserve waypoint path")
        .with_identifier("RunMat:Internal")
        .with_builtin("integral")
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

pub(super) fn finite_real_segments(
    lower: f64,
    upper: f64,
    waypoints: &[Complex<f64>],
) -> BuiltinResult<Vec<(f64, f64)>> {
    let lo = lower.min(upper);
    let hi = lower.max(upper);
    let mut points = Vec::new();
    points
        .try_reserve_exact(waypoints.len())
        .map_err(resource_error)?;
    let ascending = lower <= upper;
    let mut previous = lower;
    for point in waypoints {
        if point.im != 0.0 || point.re < lo || point.re > hi {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "real Waypoints must lie within the integration interval",
            ));
        }
        if (ascending && point.re < previous) || (!ascending && point.re > previous) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "real Waypoints must be monotone along the integration path",
            ));
        }
        points.push(point.re);
        previous = point.re;
    }
    if !ascending {
        points.reverse();
    }
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(points.len().checked_add(1).ok_or_else(|| {
            integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, "too many Waypoints")
        })?)
        .map_err(resource_error)?;
    let mut start = lo;
    for end in points.into_iter().chain(std::iter::once(hi)) {
        if end > start {
            segments.push((start, end));
        }
        start = end;
    }
    Ok(segments)
}

pub(super) fn preflight_host_path(
    lower: &Value,
    upper: &Value,
    options: &[(IntegralOptionName, &Value)],
    waypoint_receipts: &[Option<Vec<Complex<f64>>>],
) -> BuiltinResult<()> {
    if matches!(lower, Value::GpuTensor(_)) || matches!(upper, Value::GpuTensor(_)) {
        return Ok(());
    }
    let lower = scalar_bound_host("lower bound", lower)?;
    let upper = scalar_bound_host("upper bound", upper)?;
    let last_waypoints = options
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, (name, _))| {
            (*name == IntegralOptionName::Waypoints).then_some(&waypoint_receipts[index])
        });
    if let Some(Some(points)) = last_waypoints {
        finite_real_segments(lower, upper, points)?;
    }
    Ok(())
}

pub(super) async fn integrate_real_segments(
    function: &Value,
    options: &IntegralOptions,
    segments: &[(f64, f64)],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<f64> {
    let mut estimate = ScaledEstimate::from_f64(0.0);
    let mut remaining = options.max_fun_evals;
    let mut segment_options = options.clone();
    segment_options.abs_tol = options.abs_tol / segments.len() as f64;
    for &(start, end) in segments {
        cooperation.completed(0).await?;
        if remaining < 5 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_INPUT,
                "exceeded maximum function evaluations",
            ));
        }
        segment_options.max_fun_evals = remaining;
        let mut evaluator = IntegralEvaluator {
            function,
            options: &segment_options,
            cooperation,
        };
        let (part, evaluations) = integrate_scaled(&mut evaluator, start, end)
            .await
            .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
        remaining -= evaluations;
        estimate = estimate.add(part);
    }
    estimate.to_f64().ok_or_else(|| {
        integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_INPUT,
            "quadrature estimate is not representable as a finite double",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_waypoints_keep_monotone_order_and_skip_duplicate_intervals() {
        assert_eq!(
            finite_real_segments(
                1.0,
                0.0,
                &[
                    Complex::new(0.75, 0.0),
                    Complex::new(0.25, 0.0),
                    Complex::new(0.25, 0.0)
                ]
            )
            .unwrap(),
            vec![(0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]
        );
        assert!(finite_real_segments(0.0, 1.0, &[Complex::new(2.0, 0.0)]).is_err());
        assert!(finite_real_segments(0.0, 1.0, &[Complex::new(0.5, 1.0)]).is_err());
        assert!(finite_real_segments(
            0.0,
            1.0,
            &[Complex::new(0.75, 0.0), Complex::new(0.25, 0.0)]
        )
        .is_err());
    }
}
