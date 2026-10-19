//! General square CSC left solve, currently limited to an explicitly
//! Unbounded faer LU owner. This is not a finite factor-admission claim.

mod rectangular;
pub(in super::super) use rectangular::try_solve as try_rectangular;

use faer::{c32, c64};
use runmat_value::{ComplexElement, ComplexStorage, ComplexTensor, Tensor, Value};

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::tensor::tensor_into_value;
use crate::builtins::math::linalg::ops::sparse_factor::{SparseLuError, SparseLuFactor};
use crate::builtins::math::linalg::ops::sparse_operator::SparseOperatorError;
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};

use super::super::{errors, SolveOrientation};

fn rhs_columns(shape: &[usize], order: usize) -> Option<usize> {
    (shape.first().copied() == Some(order) && shape.iter().skip(2).all(|&dim| dim == 1))
        .then(|| shape.get(1).copied().unwrap_or(1))
}

fn map_factor(error: SparseLuError, context: &RuntimeContext) -> RuntimeError {
    match error {
        SparseLuError::Admission(error)
        | SparseLuError::Operator(SparseOperatorError::Admission(error)) => {
            crate::builtin::builtin_control_error(
                Some(&context.cancellation_context()),
                "mldivide",
                error,
            )
        }
        SparseLuError::Singular { column } => errors::invalid_input(
            SolveOrientation::Left,
            format!(
                "mldivide: sparse coefficient is singular at column {}",
                column + 1
            ),
        ),
        SparseLuError::NumericalBreakdown => errors::invalid_input(
            SolveOrientation::Left,
            "mldivide: sparse numeric LU solve has a zero or non-finite pivot",
        ),
        SparseLuError::NonFiniteInput => errors::invalid_input(
            SolveOrientation::Left,
            "mldivide: sparse LU coefficient values must be finite",
        ),
        SparseLuError::Library(source) => errors::internal(
            SolveOrientation::Left,
            format!("mldivide: sparse factorization failed: {source:?}"),
        ),
        other => errors::internal(SolveOrientation::Left, format!("mldivide: {other:?}")),
    }
}

fn require_unbounded(context: &RuntimeContext) -> BuiltinResult<()> {
    match HostAllocationReceipt::policy(Some(context))
        .map_err(|error| map_factor(SparseLuError::Admission(error), context))?
    {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(map_factor(
            SparseLuError::Admission(HostAllocationError::MissingManagedOwner),
            context,
        )),
    }
}

fn map_values<T: Copy, U>(
    values: &[T],
    context: &RuntimeContext,
    mut convert: impl FnMut(T) -> U,
) -> BuiltinResult<Vec<U>> {
    require_unbounded(context)?;
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|_| map_factor(SparseLuError::TemporaryAllocationFailed, context))?;
    for chunk in values.chunks(1024) {
        require_unbounded(context)?;
        copied.extend(chunk.iter().copied().map(&mut convert));
    }
    Ok(copied)
}

pub(in super::super) fn try_solve(
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
        Value::Tensor(tensor) => rhs_columns(tensor.shape(), order).filter(|_| {
            (sparse.as_f32_slice().is_some() && tensor.as_f32_slice().is_some())
                || (sparse.as_f64_slice().is_some() && tensor.as_f64_slice().is_some())
        }),
        Value::ComplexTensor(tensor) => rhs_columns(tensor.shape(), order).filter(|_| {
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

    // The factor owner checks the actual policy before any private faer
    // symbolic/fill allocation; no finite byte count is inferred from nnz.
    let factor = SparseLuFactor::factor_unbounded(sparse, context)
        .map_err(|error| map_factor(error, context))?;
    let result = match (&factor, rhs) {
        (SparseLuFactor::RealF32 { .. }, Value::Tensor(tensor)) => {
            let mut values = map_values(
                tensor
                    .as_f32_slice()
                    .expect("class checked before factorization"),
                context,
                std::convert::identity,
            )?;
            factor
                .solve_real_f32(&mut values, columns, context)
                .map_err(|error| map_factor(error, context))?;
            tensor_into_value(
                Tensor::from_f32(values, vec![order, columns]).map_err(|error| {
                    errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
                })?,
            )
        }
        (SparseLuFactor::RealF64 { .. }, Value::Tensor(tensor)) => {
            let mut values = map_values(
                tensor
                    .as_f64_slice()
                    .expect("class checked before factorization"),
                context,
                std::convert::identity,
            )?;
            factor
                .solve_real_f64(&mut values, columns, context)
                .map_err(|error| map_factor(error, context))?;
            tensor_into_value(Tensor::new(values, vec![order, columns]).map_err(|error| {
                errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
            })?)
        }
        (SparseLuFactor::ComplexF32 { .. }, Value::ComplexTensor(tensor)) => {
            let input = tensor
                .as_f32_slice()
                .expect("class checked before factorization");
            let mut values = map_values(input, context, |value| c32::new(value.0, value.1))?;
            factor
                .solve_complex_f32(&mut values, columns, context)
                .map_err(|error| map_factor(error, context))?;
            let output = map_values(&values, context, |value| ComplexElement(value.re, value.im))?;
            drop(values);
            complex_tensor_into_value(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F32(output.into()),
                    vec![order, columns],
                )
                .map_err(|error| {
                    errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
                })?,
            )
        }
        (SparseLuFactor::ComplexF64 { .. }, Value::ComplexTensor(tensor)) => {
            let input = tensor
                .as_f64_slice()
                .expect("class checked before factorization");
            let mut values = map_values(input, context, |value| c64::new(value.0, value.1))?;
            factor
                .solve_complex_f64(&mut values, columns, context)
                .map_err(|error| map_factor(error, context))?;
            let output = map_values(&values, context, |value| ComplexElement(value.re, value.im))?;
            drop(values);
            complex_tensor_into_value(
                ComplexTensor::from_complex_storage(
                    ComplexStorage::F64(output.into()),
                    vec![order, columns],
                )
                .map_err(|error| {
                    errors::internal(SolveOrientation::Left, format!("mldivide: {error}"))
                })?,
            )
        }
        (SparseLuFactor::RealF64 { .. }, Value::Num(value)) => {
            let mut values = [*value];
            factor
                .solve_real_f64(&mut values, 1, context)
                .map_err(|error| map_factor(error, context))?;
            Value::Num(values[0])
        }
        (SparseLuFactor::ComplexF64 { .. }, Value::Complex(real, imaginary)) => {
            let mut values = [c64::new(*real, *imaginary)];
            factor
                .solve_complex_f64(&mut values, 1, context)
                .map_err(|error| map_factor(error, context))?;
            Value::Complex(values[0].re, values[0].im)
        }
        _ => return Ok(None),
    };
    Ok(Some(result))
}
