//! Host CSC triangular left solve. The coefficient matrix is borrowed in its
//! native precision; a general sparse system remains on the existing path.

mod managed;
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
    let mut error = crate::builtin::builtin_control_error(
        Some(&context.cancellation_context()),
        "mldivide",
        error,
    );
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}

fn map_operator(error: TriangularOperatorError, context: &RuntimeContext) -> RuntimeError {
    match error {
        TriangularOperatorError::Admission(source)
        | TriangularOperatorError::Operator(
            crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError::Admission(
                source,
            ),
        ) => admission(source, context),
        TriangularOperatorError::Execution(error) => {
            let mut error = error.into_runtime_error_in_context(Some(context));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        }
        other => {
            let mut builder = crate::build_runtime_error(format!("mldivide: {other}"))
                .with_builtin("mldivide")
                .with_source(other)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never);
            if let Some(identifier) = runmat_builtins::MLDIVIDE_ERROR_INVALID_INPUT.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        }
    }
}

enum Candidate<'a> {
    General,
    Operator(TriangularLinearOperator<'a>),
    Singular,
}

async fn candidate<'a>(lhs: &'a Value, context: &RuntimeContext) -> BuiltinResult<Candidate<'a>> {
    match TriangularLinearOperator::borrow_cooperatively(
        lhs,
        Triangle::Lower,
        Diagonal::Explicit,
        context,
    )
    .await
    {
        Ok(operator) => return Ok(Candidate::Operator(operator)),
        Err(TriangularOperatorError::NonTriangular { .. }) => {}
        Err(
            TriangularOperatorError::MissingDiagonal(_)
            | TriangularOperatorError::SingularDiagonal(_),
        ) => return Ok(Candidate::Singular),
        // The canonical validator proves the complete finite triangle before
        // its deferred diagonal result. Missing authority is never singularity.
        Err(error) => return Err(map_operator(error, context)),
    }
    match TriangularLinearOperator::borrow_cooperatively(
        lhs,
        Triangle::Upper,
        Diagonal::Explicit,
        context,
    )
    .await
    {
        Ok(operator) => Ok(Candidate::Operator(operator)),
        Err(TriangularOperatorError::NonTriangular { .. }) => Ok(Candidate::General),
        Err(
            TriangularOperatorError::MissingDiagonal(_)
            | TriangularOperatorError::SingularDiagonal(_),
        ) => Ok(Candidate::Singular),
        Err(error) => Err(map_operator(error, context)),
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

pub(in super::super) async fn try_solve(
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
    let operator = match candidate(lhs, context).await? {
        Candidate::General => return Ok(None),
        Candidate::Singular => {
            return super::general_sparse::solve_singular(sparse, rhs, columns, context)
                .await
                .map(Some)
        }
        Candidate::Operator(operator) => operator,
    };
    if policy == HostAllocationPolicy::Managed {
        return managed::solve(operator, rhs, columns, context)
            .await
            .map(Some);
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
