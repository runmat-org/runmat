//! Ordered real integration path admission. Finite endpoints and infinite
//! tails become segments on the shared open Gauss–Kronrod mesh.

use num_complex::Complex;
use runmat_builtins::INTEGRAL_ERROR_INVALID_ARGUMENT;
use runmat_value::Value;

use super::{integral_error_with_detail, scalar_bound_host, IntegralOptionName};
use crate::builtins::common::cooperative_sort::{SortProgress, SortResourceError, StableSortWork};
use crate::builtins::math::optim::quadrature::{OpenDomain, OpenSegment, QuadratureCooperation};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

const PATH_CHUNK: usize = 4096;

fn resource_error(source: std::collections::TryReserveError) -> crate::RuntimeError {
    build_runtime_error("integral: could not reserve integration path")
        .with_identifier("RunMat:Internal")
        .with_builtin("integral")
        .with_source(source)
        .with_gpu_gather_retry(GpuGatherRetry::Never)
        .build()
}

fn invalid_path(detail: &str) -> crate::RuntimeError {
    integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, detail)
}

fn push_span(segments: &mut Vec<OpenSegment>, lower: f64, upper: f64) {
    if lower == upper {
        return;
    }
    match (lower.is_infinite(), upper.is_infinite()) {
        (false, false) => segments.push(OpenSegment {
            a: lower,
            b: upper,
            domain: OpenDomain::Finite,
        }),
        (true, false) => segments.push(OpenSegment {
            a: 0.0,
            b: 1.0,
            domain: OpenDomain::LeftInfinite { bound: upper },
        }),
        (false, true) => segments.push(OpenSegment {
            a: 0.0,
            b: 1.0,
            domain: OpenDomain::RightInfinite { bound: lower },
        }),
        (true, true) => {
            push_span(segments, f64::NEG_INFINITY, 0.0);
            push_span(segments, 0.0, f64::INFINITY);
        }
    }
}

pub(super) async fn real_segments(
    lower: f64,
    upper: f64,
    waypoints: &[Complex<f64>],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<OpenSegment>> {
    cooperation.completed(0).await?;
    if lower == upper && lower.is_infinite() {
        return Err(invalid_path(
            "equal infinite bounds do not define an integration path",
        ));
    }
    let lo = lower.min(upper);
    let hi = lower.max(upper);
    let mut sorted = Vec::new();
    sorted
        .try_reserve_exact(waypoints.len())
        .map_err(resource_error)?;
    for (index, point) in waypoints.iter().enumerate() {
        if point.im != 0.0 || !point.re.is_finite() || point.re < lo || point.re > hi {
            return Err(invalid_path(
                "real Waypoints must be finite and lie within the integration interval",
            ));
        }
        sorted.push(point.re);
        if (index + 1).is_multiple_of(PATH_CHUNK) {
            cooperation.completed(PATH_CHUNK as u64).await?;
        }
    }
    cooperation
        .completed((waypoints.len() % PATH_CHUNK) as u64)
        .await?;
    let mut sort =
        StableSortWork::new(&mut sorted, |left: &f64, right: &f64| left.total_cmp(right));
    loop {
        let progress = sort
            .advance()
            .map_err(|SortResourceError(source)| resource_error(source))?;
        let units = match progress {
            SortProgress::Pending(units) | SortProgress::Complete(units) => units,
        };
        cooperation.completed(units).await?;
        if matches!(progress, SortProgress::Complete(_)) {
            break;
        }
    }

    let capacity = waypoints
        .len()
        .checked_add(2)
        .ok_or_else(|| invalid_path("too many Waypoints"))?;
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(capacity)
        .map_err(resource_error)?;
    let mut start = lo;
    for (index, &point) in sorted.iter().enumerate() {
        push_span(&mut segments, start, point);
        start = point;
        if (index + 1).is_multiple_of(PATH_CHUNK) {
            cooperation.completed(PATH_CHUNK as u64).await?;
        }
    }
    cooperation
        .completed((waypoints.len() % PATH_CHUNK) as u64)
        .await?;
    push_span(&mut segments, start, hi);
    Ok(segments)
}

pub(super) async fn preflight_host_path(
    lower: &Value,
    upper: &Value,
    options: &[(IntegralOptionName, &Value)],
    waypoint_receipts: &[Option<Vec<Complex<f64>>>],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Option<Vec<OpenSegment>>> {
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
        return real_segments(lower, upper, points, cooperation)
            .await
            .map(Some);
    }
    if last_waypoints.is_none() {
        return real_segments(lower, upper, &[], cooperation)
            .await
            .map(Some);
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::optim::quadrature::QuadratureIdentity;

    #[test]
    fn real_waypoints_normalize_order_and_skip_duplicate_intervals() {
        let mut cooperation = QuadratureCooperation::new(QuadratureIdentity::Integral, None);
        let segments = futures::executor::block_on(real_segments(
            1.0,
            0.0,
            &[
                Complex::new(0.75, 0.0),
                Complex::new(0.25, 0.0),
                Complex::new(0.25, 0.0),
            ],
            &mut cooperation,
        ))
        .expect("ordered finite path");
        assert_eq!(segments.len(), 3);
        assert!(matches!(segments[0].domain, OpenDomain::Finite));
        assert_eq!((segments[0].a, segments[0].b), (0.0, 0.25));
        assert_eq!((segments[1].a, segments[1].b), (0.25, 0.75));
        assert_eq!((segments[2].a, segments[2].b), (0.75, 1.0));
        assert!(futures::executor::block_on(real_segments(
            0.0,
            1.0,
            &[Complex::new(2.0, 0.0)],
            &mut cooperation,
        ))
        .is_err());
        assert!(futures::executor::block_on(real_segments(
            0.0,
            1.0,
            &[Complex::new(0.5, 1.0)],
            &mut cooperation,
        ))
        .is_err());
        let normalized = futures::executor::block_on(real_segments(
            0.0,
            1.0,
            &[Complex::new(0.75, 0.0), Complex::new(0.25, 0.0)],
            &mut cooperation,
        ))
        .expect("unordered real waypoints form an initial mesh");
        assert_eq!(normalized.len(), 3);
        assert_eq!((normalized[0].a, normalized[0].b), (0.0, 0.25));
        assert_eq!((normalized[1].a, normalized[1].b), (0.25, 0.75));
        assert_eq!((normalized[2].a, normalized[2].b), (0.75, 1.0));
    }

    #[test]
    fn improper_paths_use_open_tails_and_finite_waypoint_spans() {
        let mut cooperation = QuadratureCooperation::new(QuadratureIdentity::Integral, None);
        let segments = futures::executor::block_on(real_segments(
            f64::NEG_INFINITY,
            f64::INFINITY,
            &[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)],
            &mut cooperation,
        ))
        .expect("improper path");
        assert_eq!(segments.len(), 3);
        assert!(matches!(segments[0].domain, OpenDomain::LeftInfinite { bound } if bound == 1.0));
        assert!(matches!(segments[1].domain, OpenDomain::Finite));
        assert_eq!((segments[1].a, segments[1].b), (1.0, 2.0));
        assert!(matches!(segments[2].domain, OpenDomain::RightInfinite { bound } if bound == 2.0));
    }
}
