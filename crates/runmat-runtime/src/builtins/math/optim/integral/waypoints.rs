//! Typed waypoint admission; path traversal is owned by the integration domain.

use num_complex::Complex;
use runmat_builtins::INTEGRAL_ERROR_INVALID_ARGUMENT;
use runmat_value::{NumericScalar, Value};

use super::integral_error_with_detail;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

fn invalid(detail: &str) -> crate::RuntimeError {
    integral_error_with_detail(&INTEGRAL_ERROR_INVALID_ARGUMENT, detail)
}

fn vector_shape(shape: &[usize]) -> bool {
    let (rows, cols) = match shape {
        [] => return true,
        [len] => (1, *len),
        [rows, cols, ..] => (*rows, *cols),
    };
    (rows <= 1 || cols <= 1 || rows == 0 || cols == 0)
        && shape.iter().skip(2).all(|&axis| axis == 1)
}

fn value_f64(value: NumericScalar) -> f64 {
    match value {
        NumericScalar::F64(value) => value,
        NumericScalar::F32(value) => f64::from(value),
        integer => integer
            .into_int_value()
            .expect("nonfloating scalar is an integer")
            .to_f64(),
    }
}

fn reserve(len: usize) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|source| {
        build_runtime_error("integral: could not reserve waypoint sequence")
            .with_identifier("RunMat:Internal")
            .with_builtin("integral")
            .with_source(source)
            .with_gpu_gather_retry(GpuGatherRetry::Never)
            .build()
    })?;
    Ok(values)
}

pub(super) fn admit_shape(value: &Value) -> BuiltinResult<()> {
    let shape = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => return Ok(()),
        Value::Tensor(tensor) => tensor.shape.as_slice(),
        Value::ComplexTensor(tensor) => tensor.shape.as_slice(),
        Value::LogicalArray(array) => array.shape.as_slice(),
        Value::GpuTensor(handle) => handle.shape(),
        _ => {
            return Err(invalid(
                "Waypoints must be a real or complex numeric vector",
            ))
        }
    };
    if vector_shape(shape) {
        Ok(())
    } else {
        Err(invalid("Waypoints must be a vector"))
    }
}

pub(super) fn decode(value: &Value) -> BuiltinResult<Vec<Complex<f64>>> {
    admit_shape(value)?;
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(invalid(
            "Waypoints integer values must convert to double exactly",
        ));
    }
    let values = match value {
        Value::Num(number) => vec![Complex::new(*number, 0.0)],
        Value::Int(integer) => vec![Complex::new(integer.to_f64(), 0.0)],
        Value::Bool(flag) => vec![Complex::new(if *flag { 1.0 } else { 0.0 }, 0.0)],
        Value::Complex(real, imaginary) => vec![Complex::new(*real, *imaginary)],
        Value::Tensor(tensor) => {
            let mut values = reserve(tensor.len())?;
            for index in 0..tensor.len() {
                let number = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid Waypoints storage"))?;
                values.push(Complex::new(value_f64(number), 0.0));
            }
            values
        }
        Value::ComplexTensor(tensor) => {
            let mut values = reserve(tensor.len())?;
            for index in 0..tensor.len() {
                let (real, imaginary) = tensor
                    .numeric_value_at(index)
                    .ok_or_else(|| invalid("invalid Waypoints storage"))?;
                values.push(Complex::new(value_f64(real), value_f64(imaginary)));
            }
            values
        }
        Value::LogicalArray(array) => {
            let mut values = reserve(array.data.len())?;
            values.extend(
                array
                    .data
                    .iter()
                    .map(|&bit| Complex::new(if bit == 0 { 0.0 } else { 1.0 }, 0.0)),
            );
            values
        }
        Value::GpuTensor(_) => {
            return Err(invalid("Waypoints must be materialized by their owner"))
        }
        _ => {
            return Err(invalid(
                "Waypoints must be a real or complex numeric vector",
            ))
        }
    };
    if values
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(invalid("Waypoints must be finite"));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::{ComplexStorage, ComplexTensor, Tensor};

    #[test]
    fn waypoint_values_keep_complex_order_and_exact_integer_boundary() {
        let complex = ComplexTensor::from_complex_storage(
            ComplexStorage::F64(vec![(1.0, 1.0), (2.0, -1.0)]),
            vec![1, 2],
        )
        .unwrap();
        assert_eq!(
            decode(&Value::ComplexTensor(complex)).unwrap(),
            vec![Complex::new(1.0, 1.0), Complex::new(2.0, -1.0)]
        );
        let wide = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )
        .unwrap();
        assert!(decode(&Value::Tensor(wide)).is_err());
    }
}
