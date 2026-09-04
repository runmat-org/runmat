use runmat_builtins::SQRT_ERROR_INVALID_INPUT;
use runmat_value::{ComplexStorage, ComplexTensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::BuiltinResult;

use super::errors;

const ZERO_EPS: f64 = 1e-12;

pub(super) fn evaluate_scalar(real: f64, imag: f64) -> Value {
    let (real, imag) = parts_f64(real, imag);
    Value::Complex(zero_f64(real), zero_f64(imag))
}

pub(super) fn evaluate_tensor(tensor: ComplexTensor) -> BuiltinResult<Value> {
    let shape = tensor.shape.clone();
    let storage = match tensor.into_complex_storage() {
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
    let magnitude = real.hypot(imag);
    if magnitude == 0.0 {
        return (0.0, 0.0);
    }
    let output_real = ((magnitude + real) / 2.0).sqrt();
    let output_imag = ((magnitude - real) / 2.0).sqrt().copysign(imag);
    (output_real, output_imag)
}

pub(super) fn parts_f32(real: f32, imag: f32) -> (f32, f32) {
    if imag == 0.0 {
        return if real < 0.0 {
            (0.0, (-real).sqrt())
        } else {
            (real.sqrt(), 0.0)
        };
    }
    let magnitude = real.hypot(imag);
    if magnitude == 0.0 {
        return (0.0, 0.0);
    }
    let output_real = ((magnitude + real) / 2.0).sqrt();
    let output_imag = ((magnitude - real) / 2.0).sqrt().copysign(imag);
    (output_real, output_imag)
}

pub(super) fn zero_f64(value: f64) -> f64 {
    if value.abs() < ZERO_EPS {
        0.0
    } else {
        value
    }
}

pub(super) fn zero_f32(value: f32) -> f32 {
    if value.abs() < ZERO_EPS as f32 {
        0.0
    } else {
        value
    }
}
