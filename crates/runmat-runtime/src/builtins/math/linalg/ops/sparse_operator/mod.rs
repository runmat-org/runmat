//! Borrowed, typed CSC operator for sparse iterative and direct solvers.
//!
//! The operator never densifies or owns coefficient values. Callers provide
//! their result/scratch vectors, so a cancelled multiply cannot publish a
//! partially computed Value.

mod product;
mod cooperative;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostNumericWorkspace,
};
use crate::context::RuntimeContext;
use runmat_value::{ComplexElement, NumericDType, SparseTensor};

#[derive(Debug, thiserror::Error)]
pub enum SparseOperatorError {
    #[error("sparse operator requires a supported numeric class")]
    UnsupportedClass,
    #[error("invalid sparse operator topology: {0}")]
    InvalidTopology(&'static str),
    #[error("sparse operator dimensions do not match")]
    DimensionMismatch,
    #[error("sparse operator extent overflows")]
    Overflow,
    #[error("sparse operator admission failed: {0}")]
    Admission(#[source] HostAllocationError),
}

impl From<HostAllocationError> for SparseOperatorError {
    fn from(error: HostAllocationError) -> Self {
        Self::Admission(error)
    }
}

/// A CSC view with validated column bounds and sorted, unique row indices.
/// The validation is O(nnz + columns), once per borrowed operator.
#[derive(Clone, Copy)]
pub struct CscOperator<'a, T> {
    rows: usize,
    cols: usize,
    col_ptrs: &'a [usize],
    row_indices: &'a [usize],
    values: &'a [T],
}

impl<'a, T> CscOperator<'a, T> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn as_slice(&self) -> &[T] {
        self.values
    }

    /// Borrow one validated column without copying CSC topology or values.
    pub(in crate::builtins::math) fn column(&self, column: usize) -> (&'a [usize], &'a [T]) {
        let range = self.col_ptrs[column]..self.col_ptrs[column + 1];
        (&self.row_indices[range.clone()], &self.values[range])
    }
}

pub enum SparseLinearOperator<'a> {
    RealF32(CscOperator<'a, f32>),
    RealF64(CscOperator<'a, f64>),
    ComplexF32(CscOperator<'a, ComplexElement<f32>>),
    ComplexF64(CscOperator<'a, ComplexElement<f64>>),
}

impl<'a> SparseLinearOperator<'a> {
    /// Validate and borrow one coefficient matrix under the selected current
    /// operation. No CSC index or coefficient buffer is copied.
    pub fn borrow_with_context(
        sparse: &'a SparseTensor,
        context: &RuntimeContext,
    ) -> Result<Self, SparseOperatorError> {
        Self::borrow_with_policy(sparse, Some(context))
    }

    /// Explicitly unbounded compatibility path for callers without a finite
    /// RuntimeContext. Managed callers use `borrow_with_context`.
    pub fn borrow_unbounded(sparse: &'a SparseTensor) -> Result<Self, SparseOperatorError> {
        Self::borrow_with_policy(sparse, None)
    }

    fn borrow_with_policy(
        sparse: &'a SparseTensor,
        context: Option<&RuntimeContext>,
    ) -> Result<Self, SparseOperatorError> {
        HostAllocationReceipt::policy(context)?;
        if sparse.nnz() != sparse.row_indices.len() {
            return Err(SparseOperatorError::InvalidTopology("value count"));
        }
        validate_topology(
            sparse.rows,
            sparse.cols,
            &sparse.col_ptrs,
            &sparse.row_indices,
            context,
        )?;
        HostAllocationReceipt::policy(context)?;
        if let Some(values) = sparse.as_f32_slice() {
            return Ok(Self::RealF32(make_view(sparse, values)));
        }
        if let Some(values) = sparse.as_f64_slice() {
            return Ok(Self::RealF64(make_view(sparse, values)));
        }
        if let Some(values) = sparse.as_complex_f32_slice() {
            return Ok(Self::ComplexF32(make_view(sparse, values)));
        }
        if let Some(values) = sparse.as_complex_f64_slice() {
            return Ok(Self::ComplexF64(make_view(sparse, values)));
        }
        Err(SparseOperatorError::UnsupportedClass)
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::RealF32(view) => view.rows(),
            Self::RealF64(view) => view.rows(),
            Self::ComplexF32(view) => view.rows(),
            Self::ComplexF64(view) => view.rows(),
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::RealF32(view) => view.cols(),
            Self::RealF64(view) => view.cols(),
            Self::ComplexF32(view) => view.cols(),
            Self::ComplexF64(view) => view.cols(),
        }
    }

    /// Quote one exact, interleaved numeric scratch allocation containing
    /// `vectors` result/work vectors of row length. Complex values use two
    /// lanes of their native precision; no hidden complex64 promotion.
    pub fn scratch_plan(&self, vectors: usize) -> Result<HostAllocationPlan, SparseOperatorError> {
        let (dtype, len) = self.scratch_geometry(vectors)?;
        let scratch = HostAllocationPlan::numeric_workspace_bytes(dtype, len)?;
        Ok(HostAllocationPlan::new(scratch, 0)?)
    }

    /// Allocate exactly the scratch described by `scratch_plan` from a live
    /// admitted receipt. The caller keeps the receipt through computation.
    pub fn allocate_scratch(
        &self,
        receipt: &HostAllocationReceipt,
        context: &RuntimeContext,
        vectors: usize,
    ) -> Result<HostNumericWorkspace, SparseOperatorError> {
        let (dtype, len) = self.scratch_geometry(vectors)?;
        Ok(receipt.temporary_numeric(context, dtype, len)?)
    }

    fn scratch_geometry(
        &self,
        vectors: usize,
    ) -> Result<(NumericDType, usize), SparseOperatorError> {
        if vectors == 0 {
            return Err(SparseOperatorError::DimensionMismatch);
        }
        let (dtype, lanes) = match self {
            Self::RealF32(_) => (NumericDType::F32, 1),
            Self::RealF64(_) => (NumericDType::F64, 1),
            Self::ComplexF32(_) => (NumericDType::F32, 2),
            Self::ComplexF64(_) => (NumericDType::F64, 2),
        };
        let len = self
            .rows()
            .checked_mul(vectors)
            .and_then(|len| len.checked_mul(lanes))
            .ok_or(SparseOperatorError::Overflow)?;
        Ok((dtype, len))
    }
}

fn make_view<'a, T>(sparse: &'a SparseTensor, values: &'a [T]) -> CscOperator<'a, T> {
    CscOperator {
        rows: sparse.rows,
        cols: sparse.cols,
        col_ptrs: &sparse.col_ptrs,
        row_indices: &sparse.row_indices,
        values,
    }
}

fn validate_topology(
    rows: usize,
    cols: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    context: Option<&RuntimeContext>,
) -> Result<(), SparseOperatorError> {
    if col_ptrs.len() != cols.checked_add(1).ok_or(SparseOperatorError::Overflow)?
        || col_ptrs.first() != Some(&0)
        || col_ptrs.last() != Some(&row_indices.len())
    {
        return Err(SparseOperatorError::InvalidTopology("column pointers"));
    }
    for (col, pair) in col_ptrs.windows(2).enumerate() {
        if col % 256 == 0 {
            HostAllocationReceipt::policy(context)?;
        }
        if pair[0] > pair[1] || pair[1] > row_indices.len() {
            return Err(SparseOperatorError::InvalidTopology("column order"));
        }
        let mut previous = None;
        for (offset, &row) in row_indices[pair[0]..pair[1]].iter().enumerate() {
            if offset % 1024 == 0 {
                HostAllocationReceipt::policy(context)?;
            }
            if row >= rows || previous.is_some_and(|prior| prior >= row) {
                return Err(SparseOperatorError::InvalidTopology("row order"));
            }
            previous = Some(row);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
