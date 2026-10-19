//! Finite complex bounds and ordered directed contour segments.

use num_complex::Complex;
use runmat_value::Value;

use super::{invalid, materialize, resource_error, COMPONENT_CHUNK};
use crate::builtins::math::optim::integral::waypoints;
use crate::builtins::math::optim::quadrature::{OpenDomain, OpenSegment, QuadratureCooperation};
use crate::BuiltinResult;

fn checked_complex(real: f64, imaginary: f64) -> BuiltinResult<Complex<f64>> {
    if real.is_finite() && imaginary.is_finite() {
        Ok(Complex::new(real, imaginary))
    } else {
        Err(invalid("complex contour values must be finite"))
    }
}

pub(in super::super) async fn scalar_bound(
    label: &str,
    value: &Value,
) -> BuiltinResult<Complex<f64>> {
    let host = materialize(value).await?;
    scalar_bound_host(label, host.as_ref())
}

pub(in super::super) fn scalar_bound_host(
    label: &str,
    value: &Value,
) -> BuiltinResult<Complex<f64>> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(invalid(format!(
            "{label} must be exactly representable as double"
        )));
    }
    match value {
        Value::Complex(real, imaginary) => checked_complex(*real, *imaginary),
        Value::ComplexTensor(array) if array.len() == 1 => {
            let (real, imaginary) = array
                .numeric_value_at(0)
                .ok_or_else(|| invalid(format!("{label} has invalid complex storage")))?;
            checked_complex(waypoints::value_f64(real), waypoints::value_f64(imaginary))
        }
        other => super::super::scalar_bound_host(label, other)
            .and_then(|real| checked_complex(real, 0.0)),
    }
}

pub(in super::super) async fn ordered_segments(
    lower: Complex<f64>,
    upper: Complex<f64>,
    waypoints: &[Complex<f64>],
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Vec<OpenSegment>> {
    if !lower.re.is_finite()
        || !lower.im.is_finite()
        || !upper.re.is_finite()
        || !upper.im.is_finite()
    {
        return Err(invalid("complex contours require finite bounds"));
    }
    let capacity = waypoints
        .len()
        .checked_add(1)
        .ok_or_else(|| invalid("too many Waypoints"))?;
    let mut segments = Vec::new();
    segments
        .try_reserve_exact(capacity)
        .map_err(resource_error)?;
    let mut start = lower;
    for (index, &point) in waypoints.iter().enumerate() {
        if !point.re.is_finite() || !point.im.is_finite() {
            return Err(invalid("complex Waypoints must be finite"));
        }
        if start != point {
            segments.push(OpenSegment {
                a: 0.0,
                b: 1.0,
                domain: OpenDomain::Contour { start, end: point },
            });
        }
        start = point;
        if (index + 1).is_multiple_of(COMPONENT_CHUNK) {
            cooperation.completed(COMPONENT_CHUNK as u64).await?;
        }
    }
    cooperation
        .completed((waypoints.len() % COMPONENT_CHUNK) as u64)
        .await?;
    if start != upper {
        segments.push(OpenSegment {
            a: 0.0,
            b: 1.0,
            domain: OpenDomain::Contour { start, end: upper },
        });
    }
    Ok(segments)
}
