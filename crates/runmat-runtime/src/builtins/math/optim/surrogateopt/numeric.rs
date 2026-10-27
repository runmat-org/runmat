//! Primary-double callback storage projection after shared numeric admission.
//! Dense/sparse payloads stay owned in their original buffers; no dense mirror
//! or per-row materialized population is required for callback decoding.
use super::{invalid, IDENTITY, INVALID_ARGUMENT};
use crate::builtins::math::optim::common::{constrained_numeric as shared, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{SparseTensor, Tensor, Value};
mod archive;
pub(super) use archive::{ArchivedMatrix, BorrowedMatrix};
pub(super) use shared::Geometry;
const POLICY: shared::Policy = shared::Policy {
    owner: IDENTITY,
    invalid: Some(&INVALID_ARGUMENT),
    numeric: None,
    single: None,
    resident: None,
    sparse: None,
};
pub(super) fn policy() -> shared::Policy {
    POLICY
}
#[derive(Clone)]
pub(super) struct DoubleArray(Storage);
#[derive(Clone)]
enum Storage {
    Scalar(f64),
    Dense(Tensor),
    Sparse(SparseTensor),
}
impl DoubleArray {
    /// Transfer a solver-constructed host point header without gathering or
    /// mirroring its buffer. Untrusted inputs use prepare() instead.
    pub(super) fn owned_point(value: Tensor) -> BuiltinResult<Self> {
        if value.numeric_dtype() != runmat_value::NumericDType::F64
            || value.as_f64_slice().is_none()
        {
            return Err(invalid("constructed point lost its primary-double backing"));
        }
        Ok(Self(Storage::Dense(value)))
    }
    /// Visit only actual stored values when validating sparse output quality.
    /// Implicit zeros are finite and need neither a scan nor a dense mirror.
    pub(super) fn stored_len(&self) -> usize {
        match &self.0 {
            Storage::Scalar(_) => 1,
            Storage::Dense(value) => value.len(),
            Storage::Sparse(value) => value.nnz(),
        }
    }
    pub(super) fn stored_value(&self, index: usize) -> f64 {
        assert!(index < self.stored_len(), "validated stored-value ordinal");
        match &self.0 {
            Storage::Scalar(value) => *value,
            Storage::Dense(value) => value.as_f64_slice().expect("double backing")[index],
            Storage::Sparse(value) => value.as_f64_slice().expect("double backing")[index],
        }
    }
    pub(super) fn stored_row(&self, index: usize) -> usize {
        assert!(index < self.stored_len(), "validated stored-value ordinal");
        match &self.0 {
            Storage::Scalar(_) => 0,
            Storage::Dense(_) => index % self.rows(),
            Storage::Sparse(value) => value.row_indices[index],
        }
    }
    pub(super) fn matrix_matches(&self, rows: usize, columns: Option<usize>) -> bool {
        self.rows() == rows
            && columns.is_none_or(|n| self.columns() == n)
            && match &self.0 {
                Storage::Dense(value) => value.shape().iter().skip(2).all(|&n| n == 1),
                _ => true,
            }
    }
    pub(super) fn vector_like(&self) -> bool {
        match &self.0 {
            Storage::Scalar(_) => true,
            Storage::Dense(value) => {
                value.is_empty() || value.shape().iter().filter(|&&n| n != 1).count() <= 1
            }
            Storage::Sparse(value) => {
                value.rows == 1 || value.cols == 1 || value.rows == 0 || value.cols == 0
            }
        }
    }
    pub(super) fn rows(&self) -> usize {
        match &self.0 {
            Storage::Scalar(_) => 1,
            Storage::Dense(value) => value.shape().first().copied().unwrap_or(1),
            Storage::Sparse(value) => value.rows,
        }
    }
    pub(super) fn columns(&self) -> usize {
        match &self.0 {
            Storage::Scalar(_) => 1,
            Storage::Dense(value) => value.shape().get(1).copied().unwrap_or(1),
            Storage::Sparse(value) => value.cols,
        }
    }
    pub(super) fn len(&self) -> usize {
        match &self.0 {
            Storage::Scalar(_) => 1,
            Storage::Dense(value) => value.len(),
            Storage::Sparse(value) => value.rows * value.cols,
        }
    }
    pub(super) fn get(&self, index: usize) -> f64 {
        assert!(index < self.len(), "validated double-array index");
        match &self.0 {
            Storage::Scalar(value) => *value,
            Storage::Dense(value) => value.as_f64_slice().expect("primary-double backing")[index],
            Storage::Sparse(value) => {
                let (row, column) = (index % value.rows, index / value.rows);
                let first = value.col_ptrs[column];
                let end = value.col_ptrs[column + 1];
                match value.row_indices[first..end].binary_search(&row) {
                    Ok(offset) => {
                        value.as_f64_slice().expect("primary-double sparse backing")[first + offset]
                    }
                    Err(_) => 0.0,
                }
            }
        }
    }
    /// Actual dense entries or stored CSC entries in one original column.
    /// Dense/stored zeros are yielded too so the consumer charges their read;
    /// implicit sparse zeros need no scan or materialized matrix mirror.
    pub(super) fn column(&self, column: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        assert!(column < self.columns(), "validated double-array column");
        let (mut next, end) = match &self.0 {
            Storage::Scalar(_) => (0, 1),
            Storage::Dense(_) => (0, self.rows()),
            Storage::Sparse(value) => (value.col_ptrs[column], value.col_ptrs[column + 1]),
        };
        std::iter::from_fn(move || {
            if next == end {
                return None;
            }
            let index = next;
            next += 1;
            Some(match &self.0 {
                Storage::Scalar(value) => (0, *value),
                Storage::Dense(value) => (
                    index,
                    value.as_f64_slice().expect("double backing")[column * self.rows() + index],
                ),
                Storage::Sparse(value) => (
                    value.row_indices[index],
                    value.as_f64_slice().expect("double backing")[index],
                ),
            })
        })
    }
}
pub(super) fn preflight(
    value: &Value,
    role: &str,
    geometry: Geometry,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    if !shared::preflight(value, role, geometry, &POLICY, cooperation.runtime())? {
        return Err(invalid(format!("{role} must be a real double array")));
    }
    Ok(())
}
pub(super) async fn prepare(
    value: Value,
    role: &str,
    geometry: Geometry,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<DoubleArray> {
    let value = shared::prepare(value, role, geometry, &POLICY, cooperation)
        .await?
        .ok_or_else(|| invalid(format!("{role} must be a real double array")))?;
    match value {
        Value::Num(value) => Ok(DoubleArray(Storage::Scalar(value))),
        Value::Tensor(value) if value.as_f64_slice().is_some() => {
            let count = value
                .shape()
                .iter()
                .try_fold(1usize, |n, &axis| n.checked_mul(axis));
            if count != Some(value.len()) {
                return Err(invalid("invalid double-array storage extent"));
            }
            Ok(DoubleArray(Storage::Dense(value)))
        }
        Value::SparseTensor(value) if value.as_f64_slice().is_some() && !value.is_complex() => {
            value
                .rows
                .checked_mul(value.cols)
                .ok_or_else(|| invalid("sparse double-array extent overflow"))?;
            let mut proof = runmat_value::SparseCscValidation::new(
                value.rows,
                value.cols,
                value.col_ptrs.clone(),
                value.row_indices.clone(),
                value.nnz(),
            )
            .map_err(|detail| invalid(format!("{role}: {detail}")))?;
            let quantum = std::num::NonZeroUsize::new(1024).expect("positive CSC quantum");
            loop {
                let done = proof
                    .advance(quantum)
                    .map_err(|detail| invalid(format!("{role}: {detail}")))?;
                cooperation.completed(1024).await?;
                if done {
                    break;
                }
            }
            Ok(DoubleArray(Storage::Sparse(value)))
        }
        _ => Err(invalid(format!(
            "{role} changed its primary-double carrier during preparation"
        ))),
    }
}
