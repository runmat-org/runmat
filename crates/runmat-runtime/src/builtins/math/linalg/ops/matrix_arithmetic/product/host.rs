use runmat_value::Value;

use super::ProductResult as BuiltinResult;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{linalg, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};

use super::{errors, integer, NAME};

#[async_recursion::async_recursion(?Send)]
pub(super) async fn evaluate(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    use Value::*;

    let lhs = crate::dispatcher::gather_if_needed_async(&lhs)
        .await
        .map_err(errors::map_control_flow)?;
    let rhs = crate::dispatcher::gather_if_needed_async(&rhs)
        .await
        .map_err(errors::map_control_flow)?;

    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Mul) {
        return Ok(result);
    }

    if integer::contains(&lhs) || integer::contains(&rhs) {
        if !is_scalar(&lhs) && !is_scalar(&rhs) {
            return Err(errors::invalid_input(
                "mtimes: if one input is an integer class, the other input must be scalar",
            ));
        }
        if let Some(result) = try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Multiply, NAME)
            .map_err(errors::invalid_input)?
        {
            return Ok(result);
        }
    }

    if is_scalar(&lhs) || is_scalar(&rhs) {
        return crate::builtins::math::elementwise::binary_arithmetic::times::times_host(lhs, rhs)
            .map_err(errors::map_control_flow);
    }

    let single = [&lhs, &rhs].iter().any(|value| match value {
        Tensor(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
        ComplexTensor(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
        _ => false,
    });
    let result = match (lhs, rhs) {
        (LogicalArray(lhs), other) => {
            let tensor = tensor::logical_to_tensor(&lhs).map_err(errors::invalid_input)?;
            evaluate(Value::Tensor(tensor), other).await
        }
        (other, LogicalArray(rhs)) => {
            let tensor = tensor::logical_to_tensor(&rhs).map_err(errors::invalid_input)?;
            evaluate(other, Value::Tensor(tensor)).await
        }
        (Bool(value), other) => evaluate(Value::Num(f64::from(value)), other).await,
        (other, Bool(value)) => evaluate(other, Value::Num(f64::from(value))).await,
        (Complex(ar, ai), Complex(br, bi)) => Ok(Complex(ar * br - ai * bi, ar * bi + ai * br)),
        (Complex(ar, ai), Num(scalar)) => Ok(Complex(ar * scalar, ai * scalar)),
        (Num(scalar), Complex(br, bi)) => Ok(Complex(scalar * br, scalar * bi)),
        (Tensor(tensor), Complex(real, imag)) | (Complex(real, imag), Tensor(tensor)) => Ok(
            complex_tensor_into_value(linalg::scalar_mul_complex(&tensor, real, imag)),
        ),
        (ComplexTensor(tensor), Num(scalar)) | (Num(scalar), ComplexTensor(tensor)) => Ok(
            complex_tensor_into_value(linalg::scalar_mul_complex_tensor(&tensor, scalar, 0.0)),
        ),
        (ComplexTensor(tensor), Complex(real, imag))
        | (Complex(real, imag), ComplexTensor(tensor)) => Ok(complex_tensor_into_value(
            linalg::scalar_mul_complex_tensor(&tensor, real, imag),
        )),
        (ComplexTensor(lhs), ComplexTensor(rhs)) => linalg::matmul_complex(&lhs, &rhs)
            .map(complex_tensor_into_value)
            .map_err(errors::invalid_input),
        (ComplexTensor(lhs), Tensor(rhs)) => linalg::matmul_complex_real(&lhs, &rhs)
            .map(complex_tensor_into_value)
            .map_err(errors::invalid_input),
        (Tensor(lhs), ComplexTensor(rhs)) => linalg::matmul_real_complex(&lhs, &rhs)
            .map(complex_tensor_into_value)
            .map_err(errors::invalid_input),
        (Tensor(lhs), Tensor(rhs)) => linalg::matmul_real(&lhs, &rhs)
            .map(tensor::tensor_into_value)
            .map_err(errors::invalid_input),
        (Num(lhs), Num(rhs)) => Ok(Num(lhs * rhs)),
        _ => Err(errors::invalid_input(
            "matrix multiplication: unsupported input types",
        )),
    }?;
    if !single {
        return Ok(result);
    }
    match result {
        Tensor(value) => Ok(Tensor(tensor::coerce_tensor_dtype(
            value,
            runmat_value::NumericDType::F32,
        ))),
        ComplexTensor(value) => runmat_value::ComplexTensor::from_f64_values_with_dtype(
            value.materialize_f64(),
            value.shape,
            runmat_value::NumericDType::F32,
        )
        .map(ComplexTensor)
        .map_err(errors::invalid_input),
        Num(value) => runmat_value::Tensor::from_f32(vec![value as f32], vec![1, 1])
            .map(Tensor)
            .map_err(errors::invalid_input),
        Complex(real, imag) => {
            runmat_value::ComplexTensor::from_f32(vec![(real as f32, imag as f32)], vec![1, 1])
                .map(ComplexTensor)
                .map_err(errors::invalid_input)
        }
        value => Ok(value),
    }
}

fn is_scalar(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::ComplexTensor(tensor) => tensor::is_scalar_complex_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::GpuTensor(handle) => crate::builtins::common::shape::is_scalar_shape(&handle.shape),
        _ => false,
    }
}
