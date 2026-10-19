use runmat_builtins::SQRT_ERROR_INVALID_INPUT;
use runmat_value::{ComplexStorage, ComplexTensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::BuiltinResult;

use super::errors;

pub(super) fn evaluate_scalar(real: f64, imag: f64) -> Value {
    let (real, imag) = parts_f64(real, imag);
    Value::Complex(zero_f64(real), zero_f64(imag))
}

pub(super) fn evaluate_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let (storage, shape) = tensor.into_complex_storage_and_shape();
    let storage = match storage {
        ComplexStorage::F64(values) => ComplexStorage::F64(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (real, imag) = parts_f64(real, imag);
                    (zero_f64(real), zero_f64(imag))
                })
                .collect(),
        ),
        ComplexStorage::F32(values) => ComplexStorage::F32(
            values
                .into_iter()
                .map(|(real, imag)| {
                    let (real, imag) = parts_f32(real, imag);
                    (zero_f32(real), zero_f32(imag))
                })
                .collect(),
        ),
        ComplexStorage::Integer(_) => {
            return Err(errors::with_detail(
                &SQRT_ERROR_INVALID_INPUT,
                "typed complex integer input is not supported",
            ))
        }
    };
    let output = ComplexTensor::from_complex_storage(storage, shape).map_err(errors::internal)?;
    Ok(complex_tensor_into_value(output))
}

fn parts_f64(real: f64, imag: f64) -> (f64, f64) {
    if imag == 0.0 {
        return if real < 0.0 {
            (0.0, (-real).sqrt())
        } else {
            (real.sqrt(), 0.0)
        };
    }
    if imag.is_infinite() {
        return (f64::INFINITY, imag);
    }
    if real.is_infinite() {
        return if real.is_sign_positive() {
            (real, if imag.is_nan() { imag } else { 0.0 })
        } else {
            (
                if imag.is_nan() { imag } else { 0.0 },
                f64::INFINITY.copysign(imag),
            )
        };
    }
    // Scale before forming the magnitude: both hypot(MAX, MAX) and
    // hypot(x, y) + |x| can overflow even though their square root is finite.
    let x = real.abs();
    let y = imag.abs();
    let major = if x >= y {
        let ratio = y / x;
        x.sqrt() * ((1.0 + 1.0_f64.hypot(ratio)) * 0.5).sqrt()
    } else {
        let ratio = x / y;
        y.sqrt() * ((ratio + 1.0_f64.hypot(ratio)) * 0.5).sqrt()
    };
    // Recover the smaller component from 2*u*v = imag, avoiding subtraction
    // of nearly equal numbers when one component is tiny.
    let minor = y / (2.0 * major);
    if real >= 0.0 {
        (major, minor.copysign(imag))
    } else {
        (minor, major.copysign(imag))
    }
}

pub(super) fn parts_f32(real: f32, imag: f32) -> (f32, f32) {
    let (real, imag) = parts_f64(f64::from(real), f64::from(imag));
    (real as f32, imag as f32)
}

pub(super) fn zero_f64(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

pub(super) fn zero_f32(value: f32) -> f32 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}
