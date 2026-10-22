//! Invocation-owned incomplete LU factors with claimed sparse fill.

mod active_support;
mod numeric;
mod projection;

use super::managed_lu::ManagedSparseScalar;
use super::managed_storage::{ClaimedVec, SparseWorkError};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum IncompleteLuKind {
    NoFill,
    Crout,
    ThresholdPivoting,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum ModifiedIncompleteLu {
    Off,
    Row,
    Column,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::builtins::math::linalg) struct IncompleteLuOptions {
    pub kind: IncompleteLuKind,
    pub drop_tolerance: f64,
    pub modified: ModifiedIncompleteLu,
    pub replace_zero_diagonal: bool,
    pub pivot_threshold: f64,
    /// Factor the nonconjugating transpose. This is the canonical row-pivot
    /// route used by column-sum modified ILUTP.
    pub transpose_source: bool,
}

#[derive(Debug)]
pub(in crate::builtins::math::linalg) enum ManagedIncompleteLuError {
    Admission(HostAllocationError),
    Execution(ExecutionServiceError),
    InvalidTopology(String),
    NotSquare,
    UnsupportedClass,
    NonFiniteInput,
    Singular { column: usize },
    NumericalBreakdown,
}

impl From<HostAllocationError> for ManagedIncompleteLuError {
    fn from(value: HostAllocationError) -> Self {
        Self::Admission(value)
    }
}
impl From<ExecutionServiceError> for ManagedIncompleteLuError {
    fn from(value: ExecutionServiceError) -> Self {
        Self::Execution(value)
    }
}
impl From<SparseWorkError> for ManagedIncompleteLuError {
    fn from(value: SparseWorkError) -> Self {
        match value {
            SparseWorkError::Admission(value) => Self::Admission(value),
            SparseWorkError::Execution(value) => Self::Execution(value),
        }
    }
}
impl std::fmt::Display for ManagedIncompleteLuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(e) => write!(f, "incomplete LU admission failed: {e}"),
            Self::Execution(e) => write!(f, "incomplete LU execution failed: {e}"),
            Self::InvalidTopology(e) => write!(f, "incomplete LU requires valid CSC topology: {e}"),
            Self::NotSquare => f.write_str("incomplete LU requires a square matrix"),
            Self::UnsupportedClass => f.write_str("unsupported incomplete LU numerical class"),
            Self::NonFiniteInput => f.write_str("incomplete LU requires finite coefficients"),
            Self::Singular { column } => {
                write!(f, "incomplete LU has a singular pivot in column {column}")
            }
            Self::NumericalBreakdown => f.write_str("incomplete LU numerical breakdown"),
        }
    }
}
impl std::error::Error for ManagedIncompleteLuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(e) => Some(e),
            Self::Execution(e) => Some(e),
            _ => None,
        }
    }
}

/// Row-oriented factors. `column_permutation[new]` is the original column.
/// All backing blocks carry claims from the invocation that built the factor.
pub(in crate::builtins::math::linalg) struct ManagedIncompleteLu<T: ManagedSparseScalar> {
    order: usize,
    l_ptr: ClaimedVec<usize>,
    l_cols: ClaimedVec<usize>,
    l_values: ClaimedVec<T>,
    u_ptr: ClaimedVec<usize>,
    u_cols: ClaimedVec<usize>,
    u_values: ClaimedVec<T>,
    column_permutation: ClaimedVec<usize>,
    inverse_permutation: ClaimedVec<usize>,
    fence: Option<HostAllocationReceipt>,
    _metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

pub(in crate::builtins::math::linalg) enum IncompleteLuPart {
    Lower,
    Upper,
}

pub(in crate::builtins::math::linalg) struct IncompleteLuCsc<T: ManagedSparseScalar> {
    col_ptr: ClaimedVec<usize>,
    rows: ClaimedVec<usize>,
    values: ClaimedVec<T>,
}
impl<T: ManagedSparseScalar> IncompleteLuCsc<T> {
    pub fn col_ptrs(&self) -> &[usize] {
        self.col_ptr.as_slice()
    }
    pub fn rows(&self) -> &[usize] {
        self.rows.as_slice()
    }
    pub fn values(&self) -> &[T] {
        self.values.as_slice()
    }
}

impl<T: ManagedSparseScalar> ManagedIncompleteLu<T> {
    pub(in crate::builtins::math::linalg) fn check_owner(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), HostAllocationError> {
        self.fence
            .as_ref()
            .map_or(
                super::managed_storage::Fence::Unbounded,
                super::managed_storage::Fence::Managed,
            )
            .checkpoint(runtime)
    }
    pub(in crate::builtins::math::linalg) fn order(&self) -> usize {
        self.order
    }
    pub(in crate::builtins::math::linalg) fn column_permutation(&self) -> &[usize] {
        self.column_permutation.as_slice()
    }

    pub(in crate::builtins::math::linalg) fn inverse_permutation(&self) -> &[usize] {
        self.inverse_permutation.as_slice()
    }
}
