//! Tall CSC least-squares selection. The coefficient remains borrowed; faer's
//! privately allocated QR factor is used only with an Unbounded host policy.

use faer::{c32, c64};
use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor::tensor_into_value;
use crate::builtins::math::linalg::ops::sparse_factor::{SparseQrError, SparseQrFactor};
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

use super::super::super::{errors, SolveOrientation};
use super::{map_values, rhs_columns};

fn map_qr(error: SparseQrError, context: &RuntimeContext) -> RuntimeError {
    match error {
        SparseQrError::Admission(error)
        | SparseQrError::Operator(SparseOperatorError::Admission(error)) => {
            crate::builtin::builtin_control_error(
                Some(&context.cancellation_context()),
                "mldivide",
                error,
            )
        }
        SparseQrError::NonFiniteInput => errors::invalid_input(
            SolveOrientation::Left,
            "mldivide: sparse QR coefficient values must be finite",
        ),
        SparseQrError::NumericalBreakdown => errors::invalid_input(
            SolveOrientation::Left,
            "mldivide: sparse QR least-squares solution is non-finite",
        ),
        SparseQrError::Library(source) => errors::internal(
            SolveOrientation::Left,
            format!("mldivide: sparse QR factorization failed: {source:?}"),
        ),
        other => errors::internal(SolveOrientation::Left, format!("mldivide: {other:?}")),
    }
}

fn compact<T: Copy>(
    mut solved: Vec<T>,
    rows: usize,
    cols: usize,
    columns: usize,
    context: &RuntimeContext,
) -> Result<Vec<T>, RuntimeError> {
    let len = cols
        .checked_mul(columns)
        .ok_or_else(|| map_qr(SparseQrError::TemporaryAllocationFailed, context))?;
    for index in 1..columns {
        super::require_unbounded(context)?;
        let source = index * rows;
        let destination = index * cols;
        solved.copy_within(source..source + cols, destination);
    }
    solved.truncate(len);
    Ok(solved)
}

pub(in super::super::super) fn try_solve(
    lhs: &Value,
    rhs: &Value,
    context: &RuntimeContext,
) -> BuiltinResult<Option<Value>> {
    let Value::SparseTensor(sparse) = lhs else {
        return Ok(None);
    };
    let (rows, cols) = (sparse.rows, sparse.cols);
    if rows <= cols || cols == 0 {
        return Ok(None);
    }
    let columns = match rhs {
        Value::Tensor(tensor) => rhs_columns(tensor.shape(), rows).filter(|_| {
            (sparse.as_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        Value::ComplexTensor(tensor) => rhs_columns(tensor.shape(), rows).filter(|_| {
            (sparse.as_complex_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_complex_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        _ => None,
    };
    let Some(columns) = columns else {
        return Ok(None);
    };
    let factor = SparseQrFactor::factor_unbounded(sparse, context)
        .map_err(|error| map_qr(error, context))?;
    let result = match (&factor, rhs) {
        (SparseQrFactor::RealF32 { .. }, Value::Tensor(tensor)) => {
            let mut values = map_values(
                tensor.as_f32_slice().expect("class checked"),
                context,
                std::convert::identity,
            )?;
            factor
                .solve_real_f32(&mut values, columns, context)
                .map_err(|error| map_qr(error, context))?;
            let output = compact(values, rows, cols, columns, context)?;
            tensor_into_value(
                Tensor::from_f32(output, vec![cols, columns]).map_err(|error| {
                    errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
                })?,
            )
        }
        (SparseQrFactor::RealF64 { .. }, Value::Tensor(tensor)) => {
            let mut values = map_values(
                tensor.as_f64_slice().expect("class checked"),
                context,
                std::convert::identity,
            )?;
            factor
                .solve_real_f64(&mut values, columns, context)
                .map_err(|error| map_qr(error, context))?;
            let output = compact(values, rows, cols, columns, context)?;
            tensor_into_value(Tensor::new(output, vec![cols, columns]).map_err(|error| {
                errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
            })?)
        }
        (SparseQrFactor::ComplexF32 { .. }, Value::ComplexTensor(tensor)) => {
            let mut values = map_values(
                tensor.as_f32_slice().expect("class checked"),
                context,
                |value| c32::new(value.0, value.1),
            )?;
            factor
                .solve_complex_f32(&mut values, columns, context)
                .map_err(|error| map_qr(error, context))?;
            let native = compact(values, rows, cols, columns, context)?;
            let output = map_values(&native, context, |value| ComplexElement(value.re, value.im))?;
            complex_tensor_into_value(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(output.into()),
                    vec![cols, columns],
                )
                .map_err(|error| {
                    errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
                })?,
            )
        }
        (SparseQrFactor::ComplexF64 { .. }, Value::ComplexTensor(tensor)) => {
            let mut values = map_values(
                tensor.as_f64_slice().expect("class checked"),
                context,
                |value| c64::new(value.0, value.1),
            )?;
            factor
                .solve_complex_f64(&mut values, columns, context)
                .map_err(|error| map_qr(error, context))?;
            let native = compact(values, rows, cols, columns, context)?;
            let output = map_values(&native, context, |value| ComplexElement(value.re, value.im))?;
            complex_tensor_into_value(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(output.into()),
                    vec![cols, columns],
                )
                .map_err(|error| {
                    errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
                })?,
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}
