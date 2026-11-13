//! Public rectangular/singular LU, using the same native sparse reach and
//! exact-growth owners as the square nonsingular inverse. Zero pivots are
//! legitimate factor data here, never an instruction to densify or replay.
mod factor;
mod retained;
mod storage;
#[cfg(test)]
mod tests;
use super::{ClaimedVec, ManagedSparseLuError as Error, ManagedSparseScalar, Work};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;
use storage::Csc;

/// Empty elimination has an identity permutation without allocating or
/// traversing a virtual row axis. A nonempty factor borrows its native map.
#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg) struct PermutationView<'a> {
    values: &'a [usize],
    len: usize,
}
impl<'a> PermutationView<'a> {
    pub(in crate::builtins::math::linalg) fn len(self) -> usize {
        self.len
    }
    pub(in crate::builtins::math::linalg) fn at(self, index: usize) -> usize {
        assert!(index < self.len, "bounded factor permutation index");
        if self.values.is_empty() {
            index
        } else {
            self.values[index]
        }
    }
    pub(in crate::builtins::math::linalg) fn iter(
        self,
    ) -> impl ExactSizeIterator<Item = usize> + 'a {
        (0..self.len).map(move |index| self.at(index))
    }
}

#[derive(Clone, Copy)]
pub(in crate::builtins::math::linalg) struct SparseLuOptions {
    pub(in crate::builtins::math::linalg) column_ordering: bool,
    pub(in crate::builtins::math::linalg) row_scaling: bool,
    pub(in crate::builtins::math::linalg) unpermute_lower: bool,
    pub(in crate::builtins::math::linalg) threshold: [f64; 2],
}

pub(in crate::builtins::math::linalg) struct PublicSparseLu<T: ManagedSparseScalar> {
    rows: usize,
    cols: usize,
    lower: Csc<T>,
    upper: Csc<T>,
    row_perm: ClaimedVec<usize>,
    row_inverse: ClaimedVec<usize>,
    col_perm: ClaimedVec<usize>,
    scaling: ClaimedVec<T>,
    fence: Option<HostAllocationReceipt>,
    _metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}
impl<T: ManagedSparseScalar> PublicSparseLu<T> {
    pub(in crate::builtins::math::linalg) fn check_owner(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        self.fence
            .as_ref()
            .map_or(
                super::super::managed_storage::Fence::Unbounded,
                super::super::managed_storage::Fence::Managed,
            )
            .checkpoint(runtime)
    }
    pub(in crate::builtins::math::linalg) fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub(in crate::builtins::math::linalg) fn row_permutation(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<PermutationView<'_>, HostAllocationError> {
        self.check_owner(runtime)?;
        Ok(PermutationView {
            values: self.row_perm.as_slice(),
            len: self.rows,
        })
    }
    pub(in crate::builtins::math::linalg) fn row_inverse(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<PermutationView<'_>, HostAllocationError> {
        self.check_owner(runtime)?;
        Ok(PermutationView {
            values: self.row_inverse.as_slice(),
            len: self.rows,
        })
    }
    pub(in crate::builtins::math::linalg) fn column_permutation(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<PermutationView<'_>, HostAllocationError> {
        self.check_owner(runtime)?;
        Ok(PermutationView {
            values: self.col_perm.as_slice(),
            len: self.cols,
        })
    }
    pub(in crate::builtins::math::linalg) fn scaling(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<&[T], HostAllocationError> {
        self.check_owner(runtime)?;
        Ok(self.scaling.as_slice())
    }
    pub(in crate::builtins::math::linalg) fn lower_column(
        &self,
        column: usize,
        runtime: &RuntimeContext,
    ) -> Result<(&[usize], &[T]), Error> {
        self.check_owner(runtime)?;
        if column >= self.rows.min(self.cols) {
            return Err(Error::DimensionMismatch);
        }
        Ok(self.lower.column(column))
    }
    pub(in crate::builtins::math::linalg) fn upper_column(
        &self,
        column: usize,
        runtime: &RuntimeContext,
    ) -> Result<(&[usize], &[T]), Error> {
        self.check_owner(runtime)?;
        if column >= self.cols {
            return Err(Error::DimensionMismatch);
        }
        Ok(self.upper.column(column))
    }
}
