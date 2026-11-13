//! Shared real CSC normalization; each public adapter owns its dtype policy.
use super::super::ScalarCooperation;
use crate::{BuiltinResult, RuntimeError};
use clarabel::algebra::CscMatrix;
use runmat_value::{NumericScalar, NumericStorage, Value};
mod validation;
pub(crate) use validation::validate_csc;

#[derive(Clone, Copy)]
pub(crate) enum IntegerPolicy {
    Exact,
    FloatingBoundary,
}

fn scalar(value: NumericScalar, policy: IntegerPolicy) -> Option<f64> {
    if let Some(integer) = value.into_int_value() {
        match policy {
            IntegerPolicy::Exact => integer.to_exact_f64(),
            IntegerPolicy::FloatingBoundary => Some(integer.to_f64()),
        }
    } else {
        Some(value.materialize_f64())
    }
}

/// Input is already gathered by its authentic public adapter. No device
/// operation, feature gate, class dispatch or inferred resource grant lives here.
pub(crate) async fn parse(
    value: Value,
    policy: IntegerPolicy,
    cooperation: &mut ScalarCooperation,
    invalid: impl Fn(&str) -> RuntimeError,
) -> BuiltinResult<Option<CscMatrix<f64>>> {
    if matches!(&value,Value::Tensor(t) if t.is_empty())
        || matches!(&value,Value::SparseTensor(t) if t.rows==0 || t.cols==0)
    {
        return Ok(None);
    }
    if let Value::SparseTensor(sparse) = value {
        if sparse.is_complex() || sparse.is_logical() {
            return Err(invalid("must be a real numeric matrix"));
        }
        sparse
            .cols
            .checked_add(1)
            .ok_or_else(|| invalid("column count overflow"))?;
        let mut validation = runmat_value::SparseCscValidation::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs.clone(),
            sparse.row_indices.clone(),
            sparse.nnz(),
        )
        .map_err(|message| invalid(&message))?;
        let quantum = std::num::NonZeroUsize::new(1024).expect("nonzero work");
        loop {
            let done = validation
                .advance(quantum)
                .map_err(|message| invalid(&message))?;
            cooperation.completed(1024).await?;
            if done {
                break;
            }
        }
        let storage = sparse
            .numeric_host_buffer()
            .ok_or_else(|| invalid("has no real numeric backing"))?;
        let mut values = runmat_value::exact_capacity_vec(sparse.nnz())
            .map_err(|source| allocation(source, &invalid))?;
        for first in (0..sparse.nnz()).step_by(1024) {
            let end = first.saturating_add(1024).min(sparse.nnz());
            for index in first..end {
                let value = storage
                    .value_at(index)
                    .and_then(|value| scalar(value, policy))
                    .ok_or_else(|| {
                        invalid("integer values must be exactly representable as double")
                    })?;
                if !value.is_finite() {
                    return Err(invalid("values must be finite"));
                }
                values.push(value);
            }
            cooperation.completed((end - first) as u64).await?;
        }
        return Ok(Some(CscMatrix::new(
            sparse.rows,
            sparse.cols,
            sparse.col_ptrs.to_vec(),
            sparse.row_indices.to_vec(),
            values,
        )));
    }
    let (rows, cols, storage) = match value {
        Value::Num(value) => (1, 1, NumericStorage::F64(vec![value])),
        Value::Int(value) => {
            let value = match policy {
                IntegerPolicy::Exact => value.to_exact_f64(),
                IntegerPolicy::FloatingBoundary => Some(value.to_f64()),
            }
            .ok_or_else(|| invalid("integer values must be exactly representable as double"))?;
            (1, 1, NumericStorage::F64(vec![value]))
        }
        Value::Tensor(tensor) if tensor.shape().len() <= 2 => {
            let shape = (tensor.rows(), tensor.cols());
            (
                shape.0,
                shape.1,
                tensor
                    .into_numeric_storage()
                    .map_err(|message| invalid(&message))?,
            )
        }
        _ => return Err(invalid("must be a real numeric matrix")),
    };
    if rows.checked_mul(cols) != Some(storage.len()) {
        return Err(invalid("matrix backing does not match its geometry"));
    }
    let mut pointers = runmat_value::exact_capacity_vec(
        cols.checked_add(1)
            .ok_or_else(|| invalid("column count overflow"))?,
    )
    .map_err(|source| allocation(source, &invalid))?;
    let mut indices = Vec::new();
    let mut nonzeros = Vec::new();
    pointers.push(0);
    for col in 0..cols {
        for first in (0..rows).step_by(1024) {
            let end = first.saturating_add(1024).min(rows);
            for row in first..end {
                let value = storage
                    .value_at(col * rows + row)
                    .and_then(|value| scalar(value, policy))
                    .ok_or_else(|| {
                        invalid("integer values must be exactly representable as double")
                    })?;
                if !value.is_finite() {
                    return Err(invalid("values must be finite"));
                }
                if value != 0.0 {
                    indices.push(row);
                    nonzeros.push(value);
                }
            }
            cooperation.completed((end - first) as u64).await?;
        }
        pointers.push(nonzeros.len());
    }
    Ok(Some(CscMatrix::new(
        rows, cols, pointers, indices, nonzeros,
    )))
}

fn allocation(
    source: runmat_value::ExactHostNumericAllocationError,
    invalid: &impl Fn(&str) -> RuntimeError,
) -> RuntimeError {
    let mut error = invalid("matrix storage allocation failed");
    error.source = Some(Box::new(source));
    error
}
