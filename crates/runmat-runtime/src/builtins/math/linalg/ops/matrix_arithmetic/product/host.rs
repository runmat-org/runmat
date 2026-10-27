use runmat_value::Value;

use super::ProductResult as BuiltinResult;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::{linalg, tensor};
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};

use super::{errors, integer, NAME};

#[async_recursion::async_recursion(?Send)]
pub(super) async fn evaluate(
    context: Option<&crate::context::RuntimeContext>,
    lhs: Value,
    rhs: Value,
) -> crate::BuiltinResult<BuiltinResult<Value>> {
    use Value::*;

    let character_runtime = if matches!(&lhs, CharArray(_)) || matches!(&rhs, CharArray(_)) {
        context
            .cloned()
            .or_else(crate::context::legacy::active)
            .map(crate::context::RuntimeContext::with_pinned_cancellation)
    } else {
        None
    };
    let lhs = crate::dispatcher::gather_if_needed_async(&lhs).await?;
    let rhs = crate::dispatcher::gather_if_needed_async(&rhs).await?;

    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Mul) {
        return Ok(Ok(result));
    }

    if integer::contains(&lhs) || integer::contains(&rhs) {
        if !is_scalar(&lhs) && !is_scalar(&rhs) {
            return Ok(Err(errors::invalid_input(
                "mtimes: if one input is an integer class, the other input must be scalar",
            )));
        }
        if let Some(result) =
            product_try!(
                try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Multiply, NAME)
                    .map_err(errors::invalid_input)
            )
        {
            return Ok(Ok(result));
        }
    }

    // Normalize only after exact integer admission; character codes are double,
    // not integer storage, and must reach scalar/dense/CSC selection together.
    let normalize = |value: Value| -> crate::BuiltinResult<BuiltinResult<Value>> {
        let Value::CharArray(chars) = value else {
            return Ok(Ok(value));
        };
        match tensor::character_numeric::to_double(character_runtime.as_ref(), &chars) {
            Ok(value) => Ok(Ok(Value::Tensor(value))),
            Err(tensor::character_numeric::CharacterNumericError::InvalidLayout(message)) => {
                Ok(Err(errors::invalid_input(message)))
            }
            Err(tensor::character_numeric::CharacterNumericError::Runtime(error)) => Err(*error),
        }
    };
    let lhs = product_try!(normalize(lhs)?);
    let rhs = product_try!(normalize(rhs)?);

    if is_scalar(&lhs) || is_scalar(&rhs) {
        return crate::builtins::math::elementwise::binary_arithmetic::times::times_host(lhs, rhs)
            .map(Ok);
    }

    if matches!(lhs, SparseTensor(_)) || matches!(rhs, SparseTensor(_)) {
        return super::sparse::evaluate(context, &lhs, &rhs).await;
    }
    let single = [&lhs, &rhs].iter().any(|value| match value {
        Tensor(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
        ComplexTensor(tensor) => tensor.numeric_dtype() == runmat_value::NumericDType::F32,
        _ => false,
    });
    let result = product_try!(match (lhs, rhs) {
        (LogicalArray(lhs), other) => {
            let tensor =
                product_try!(tensor::logical_to_tensor(&lhs).map_err(errors::invalid_input));
            evaluate(context, Value::Tensor(tensor), other).await?
        }
        (other, LogicalArray(rhs)) => {
            let tensor =
                product_try!(tensor::logical_to_tensor(&rhs).map_err(errors::invalid_input));
            evaluate(context, other, Value::Tensor(tensor)).await?
        }
        (Bool(value), other) => evaluate(context, Value::Num(f64::from(value)), other).await?,
        (other, Bool(value)) => evaluate(context, other, Value::Num(f64::from(value))).await?,
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
    });
    if !single {
        return Ok(Ok(result));
    }
    Ok(match result {
        Tensor(value) => Ok(Tensor(tensor::coerce_tensor_dtype(
            value,
            runmat_value::NumericDType::F32,
        ))),
        ComplexTensor(value) => {
            let (storage, shape) = value.into_complex_storage_and_shape();
            runmat_value::ComplexTensor::from_f64_values_with_dtype(
                storage.materialize_f64(),
                shape,
                runmat_value::NumericDType::F32,
            )
            .map(ComplexTensor)
            .map_err(errors::invalid_input)
        }
        Num(value) => runmat_value::Tensor::from_f32(vec![value as f32], vec![1, 1])
            .map(Tensor)
            .map_err(errors::invalid_input),
        Complex(real, imag) => {
            runmat_value::ComplexTensor::from_f32(vec![(real as f32, imag as f32)], vec![1, 1])
                .map(ComplexTensor)
                .map_err(errors::invalid_input)
        }
        value => Ok(value),
    })
}

fn is_scalar(value: &Value) -> bool {
    match value {
        Value::Int(_) | Value::Num(_) | Value::Bool(_) | Value::Complex(_, _) => true,
        Value::Tensor(tensor) => tensor::is_scalar_tensor(tensor),
        Value::ComplexTensor(tensor) => tensor::is_scalar_complex_tensor(tensor),
        Value::LogicalArray(logical) => logical.data.len() == 1,
        Value::SparseTensor(sparse) => sparse.rows == 1 && sparse.cols == 1,
        Value::GpuTensor(handle) => crate::builtins::common::shape::is_scalar_shape(handle.shape()),
        _ => false,
    }
}
