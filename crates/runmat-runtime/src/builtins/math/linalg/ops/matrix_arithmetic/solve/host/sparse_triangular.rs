//! Host CSC triangular left solve. The coefficient matrix is borrowed in its
//! native precision; a general sparse system remains on the existing path.

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor::tensor_into_value;
use crate::builtins::math::linalg::ops::triangular_operator::{
    Diagonal, Triangle, TriangularLinearOperator, TriangularOperatorError,
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, Tensor, Value};

use super::super::{errors, SolveOrientation};

fn invalid(message: impl Into<String>) -> RuntimeError {
    errors::invalid_input(SolveOrientation::Left, message)
}

fn internal(message: String) -> RuntimeError {
    errors::internal(SolveOrientation::Left, format!("mldivide: {message}"))
}

fn admission(error: HostAllocationError, context: &RuntimeContext) -> RuntimeError {
    crate::builtin::builtin_control_error(Some(&context.cancellation_context()), "mldivide", error)
}

fn map_operator(error: TriangularOperatorError, context: &RuntimeContext) -> RuntimeError {
    match error {
        TriangularOperatorError::Admission(source)
        | TriangularOperatorError::Operator(
            crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                source,
            ),
        ) => admission(source, context),
        other => invalid(format!("mldivide: {other}")),
    }
}

fn candidate<'a>(
    lhs: &'a Value,
    context: &RuntimeContext,
) -> BuiltinResult<Option<TriangularLinearOperator<'a>>> {
    let borrow = |triangle| {
        TriangularLinearOperator::borrow_with_context(lhs, triangle, Diagonal::Explicit, context)
    };
    let lower = match borrow(Triangle::Lower) {
        Ok(operator) => return Ok(Some(operator)),
        Err(error @ TriangularOperatorError::Admission(_))
        | Err(
            error @ TriangularOperatorError::Operator(
                crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                    _,
                ),
            ),
        ) => return Err(map_operator(error, context)),
        Err(error) => error,
    };
    match borrow(Triangle::Upper) {
        Ok(operator) => Ok(Some(operator)),
        Err(error @ TriangularOperatorError::Admission(_))
        | Err(
            error @ TriangularOperatorError::Operator(
                crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                    _,
                ),
            ),
        ) => Err(map_operator(error, context)),
        Err(error) => {
            if matches!(&lower, TriangularOperatorError::NonTriangular { .. }) {
                if matches!(&error, TriangularOperatorError::NonTriangular { .. }) {
                    Ok(None)
                } else {
                    Err(map_operator(error, context))
                }
            } else if matches!(&error, TriangularOperatorError::NonTriangular { .. }) {
                // The lower probe can hit an invalid diagonal before a later
                // upper entry proves the matrix general. Preserve the prior
                // general-sparse path in that ambiguous case.
                Ok(None)
            } else {
                Err(map_operator(lower, context))
            }
        }
    }
}

fn rhs_shape(shape: &[usize], order: usize) -> Option<usize> {
    if shape.first().copied()? != order || shape.iter().skip(2).any(|&extent| extent != 1) {
        return None;
    }
    Some(shape.get(1).copied().unwrap_or(1))
}

macro_rules! solve_columns {
    ($operator:expr, $input:expr, $columns:expr, $zero:expr, $context:expr) => {{
        let order = $operator.order();
        HostAllocationReceipt::policy(Some($context))
            .map_err(|error| admission(error, $context))?;
        let mut output = Vec::with_capacity($input.len());
        while output.len() < $input.len() {
            HostAllocationReceipt::policy(Some($context))
                .map_err(|error| admission(error, $context))?;
            output.resize(output.len().saturating_add(1024).min($input.len()), $zero);
        }
        for column in 0..$columns {
            let start = column * order;
            let end = start + order;
            let input = &$input[start..end];
            let output = &mut output[start..end];
            $operator
                .apply_into(input, output, $context)
                .map_err(|error| map_operator(error, $context))?;
        }
        output
    }};
}

pub(super) fn try_solve(
    lhs: &Value,
    rhs: &Value,
    context: &RuntimeContext,
) -> BuiltinResult<Option<Value>> {
    let Value::SparseTensor(sparse) = lhs else {
        return Ok(None);
    };
    let order = sparse.rows;
    if order == 0 || order != sparse.cols {
        return Ok(None);
    }
    let columns = match rhs {
        Value::Tensor(tensor) => rhs_shape(tensor.shape(), order).filter(|_| {
            (sparse.as_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        Value::ComplexTensor(tensor) => rhs_shape(tensor.shape(), order).filter(|_| {
            (sparse.as_complex_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_complex_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        Value::Num(_) if order == 1 && sparse.as_f64_slice().is_some() => Some(1),
        Value::Complex(_, _) if order == 1 && sparse.as_complex_f64_slice().is_some() => Some(1),
        _ => None,
    };
    let Some(columns) = columns else {
        return Ok(None);
    };
    let policy =
        HostAllocationReceipt::policy(Some(context)).map_err(|error| admission(error, context))?;
    let Some(operator) = candidate(lhs, context)? else {
        return Ok(None);
    };
    if policy == HostAllocationPolicy::Managed {
        return Err(admission(HostAllocationError::MissingManagedOwner, context));
    }
    let result = match (operator, rhs) {
        (TriangularLinearOperator::RealF32(operator), Value::Tensor(tensor)) => {
            let input = tensor
                .as_f32_slice()
                .ok_or_else(|| invalid("RHS class changed"))?;
            let output = solve_columns!(operator, input, columns, 0.0_f32, context);
            tensor_into_value(Tensor::from_f32(output, vec![order, columns]).map_err(internal)?)
        }
        (TriangularLinearOperator::RealF64(operator), Value::Tensor(tensor)) => {
            let input = tensor
                .as_f64_slice()
                .ok_or_else(|| invalid("RHS class changed"))?;
            let output = solve_columns!(operator, input, columns, 0.0_f64, context);
            tensor_into_value(Tensor::new(output, vec![order, columns]).map_err(internal)?)
        }
        (TriangularLinearOperator::ComplexF32(operator), Value::ComplexTensor(tensor)) => {
            let input = tensor
                .as_f32_slice()
                .ok_or_else(|| invalid("RHS class changed"))?;
            let output = solve_columns!(
                operator,
                input,
                columns,
                ComplexElement(0.0_f32, 0.0_f32),
                context
            );
            complex_tensor_into_value(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(output.into()),
                    vec![order, columns],
                )
                .map_err(internal)?,
            )
        }
        (TriangularLinearOperator::ComplexF64(operator), Value::ComplexTensor(tensor)) => {
            let input = tensor
                .as_f64_slice()
                .ok_or_else(|| invalid("RHS class changed"))?;
            let output = solve_columns!(
                operator,
                input,
                columns,
                ComplexElement(0.0_f64, 0.0_f64),
                context
            );
            complex_tensor_into_value(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(output.into()),
                    vec![order, columns],
                )
                .map_err(internal)?,
            )
        }
        (TriangularLinearOperator::RealF64(operator), Value::Num(value)) => {
            let input = [*value];
            let output = solve_columns!(operator, &input, 1, 0.0_f64, context);
            Value::Num(output[0])
        }
        (TriangularLinearOperator::ComplexF64(operator), Value::Complex(real, imaginary)) => {
            let input = [ComplexElement(*real, *imaginary)];
            let output = solve_columns!(
                operator,
                &input,
                1,
                ComplexElement(0.0_f64, 0.0_f64),
                context
            );
            Value::Complex(output[0].0, output[0].1)
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}
