//! Bounded, typed host coefficient materialization for convolution families.
//! Class/shape/compatibility admission belongs to each public identity.

use num_complex::Complex;
use runmat_value::Value;

use super::super::{SignalCooperation, SIGNAL_WORK_CHUNK};
use crate::{BuiltinResult, RuntimeError};

pub(crate) async fn materialize_host_coefficients(
    value: &Value,
    cooperation: &mut SignalCooperation,
    invalid: impl Fn() -> RuntimeError,
    resource: impl Fn(std::collections::TryReserveError) -> RuntimeError,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let len = match value {
        Value::Tensor(tensor) => tensor.len(),
        Value::ComplexTensor(tensor) => tensor.len(),
        Value::LogicalArray(array) => array.data.len(),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => 1,
        _ => return Err(invalid()),
    };
    let mut coefficients = Vec::new();
    coefficients.try_reserve_exact(len).map_err(resource)?;
    for index in 0..len {
        let coefficient = match value {
            Value::Tensor(tensor) => Complex::new(
                tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid())?
                    .materialize_f64(),
                0.0,
            ),
            Value::ComplexTensor(tensor) => {
                let (real, imaginary) = tensor.numeric_value_at(index).ok_or_else(|| invalid())?;
                Complex::new(real.materialize_f64(), imaginary.materialize_f64())
            }
            Value::LogicalArray(array) => {
                Complex::new(if array.data[index] != 0 { 1.0 } else { 0.0 }, 0.0)
            }
            Value::Num(number) => Complex::new(*number, 0.0),
            Value::Int(integer) => Complex::new(integer.to_f64(), 0.0),
            Value::Bool(flag) => Complex::new(if *flag { 1.0 } else { 0.0 }, 0.0),
            Value::Complex(real, imaginary) => Complex::new(*real, *imaginary),
            _ => return Err(invalid()),
        };
        coefficients.push(coefficient);
        if (index + 1).is_multiple_of(SIGNAL_WORK_CHUNK as usize) {
            cooperation.completed(SIGNAL_WORK_CHUNK).await?;
        }
    }
    cooperation
        .completed((len % SIGNAL_WORK_CHUNK as usize) as u64)
        .await?;
    Ok(coefficients)
}
