//! Finite real waypoint geometry. Complex contours and improper domains have
//! separate numerical rules and do not enter this finite Simpson path.

use num_complex::Complex;
use runmat_builtins::INTEGRAL_ERROR_INVALID_ARGUMENT;

use super::{
    integral_error_with_detail, integral_map_error, scalar_bound_host, IntegralEvaluator,
    IntegralOptionName, IntegralOptions,
};
use crate::builtins::math::optim::quadrature::{integrate, integrate_mesh, QuadratureCooperation};
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

pub(super) async fn finite_real_segments(
    lower: f64,
    upper: f64,
    waypoints: &[Complex<f64>],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<(f64, f64)>> {
    cooperation.completed(0).await?;
    let lo = lower.min(upper);
    let hi = lower.max(upper);
    let ascending = lower <= upper;
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(waypoints.len().checked_add(1).ok_or_else(|| {
            integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, "too many Waypoints")
        })?)
        .map_err(resource_error)?;
    let mut start = lower;
    for (index, point) in waypoints.iter().enumerate() {
        if point.im != 0.0 || point.re < lo || point.re > hi {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "real Waypoints must lie within the integration interval",
            ));
        }
        if (ascending && point.re < start) || (!ascending && point.re > start) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "real Waypoints must be monotone along the integration path",
            ));
        }
        if point.re != start {
            segments.push(if ascending {
                (start, point.re)
            } else {
                (point.re, start)
            });
        }
        start = point.re;
        if (index + 1).is_multiple_of(4096) {
            cooperation.completed(4096).await?;
        }
    }
    cooperation
        .completed((waypoints.len() % 4096) as u64)
        .await?;
    if upper != start {
        segments.push(if ascending {
            (start, upper)
        } else {
            (upper, start)
        });
    }
    if !ascending {
        let count = segments.len();
        let swaps = count / 2;
        for index in 0..swaps {
            segments.swap(index, count - 1 - index);
            if (index + 1).is_multiple_of(4096) {
                cooperation.completed(4096).await?;
            }
        }
        cooperation.completed((swaps % 4096) as u64).await?;
    }
    Ok(segments)
}

pub(super) async fn preflight_host_path(
    lower: &Value,
    upper: &Value,
    options: &[(IntegralOptionName, &Value)],
    waypoint_receipts: &[Option<Vec<Complex<f64>>>],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Option<Vec<(f64, f64)>>> {
    if matches!(lower, Value::GpuTensor(_)) || matches!(upper, Value::GpuTensor(_)) {
        return Ok(None);
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
        return finite_real_segments(lower, upper, points, cooperation)
            .await
            .map(Some);
    }
    if last_waypoints.is_none() {
        return finite_real_segments(lower, upper, &[], cooperation)
            .await
            .map(Some);
    }
    Ok(None)
}

pub(super) async fn integrate_real_segments(
    function: &Value,
    options: &IntegralOptions,
    segments: &[(f64, f64)],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<f64> {
    let mut evaluator = IntegralEvaluator {
        function,
        options,
        cooperation,
    };
    if options.waypoints.is_empty() {
        return integrate(&mut evaluator, segments[0].0, segments[0].1)
            .await
            .map(|result| result.q)
            .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT));
    }
    let estimate = integrate_mesh(&mut evaluator, segments)
        .await
        .map_err(|err| integral_map_error(err, &INTEGRAL_ERROR_INVALID_INPUT))?;
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
        let mut cooperation =
            QuadratureCooperation::new(super::super::QuadratureIdentity::Integral, None);
        assert_eq!(
            futures::executor::block_on(finite_real_segments(
                1.0,
                0.0,
                &[
                    Complex::new(0.75, 0.0),
                    Complex::new(0.25, 0.0),
                    Complex::new(0.25, 0.0)
                ],
                &mut cooperation,
            ))
            .unwrap(),
            vec![(0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]
        );
        assert!(futures::executor::block_on(finite_real_segments(
            0.0,
            1.0,
            &[Complex::new(2.0, 0.0)],
            &mut cooperation
        ))
        .is_err());
        assert!(futures::executor::block_on(finite_real_segments(
            0.0,
            1.0,
            &[Complex::new(0.5, 1.0)],
            &mut cooperation
        ))
        .is_err());
        assert!(futures::executor::block_on(finite_real_segments(
            0.0,
            1.0,
            &[Complex::new(0.75, 0.0), Complex::new(0.25, 0.0)],
            &mut cooperation,
        ))
        .is_err());
    }
}
