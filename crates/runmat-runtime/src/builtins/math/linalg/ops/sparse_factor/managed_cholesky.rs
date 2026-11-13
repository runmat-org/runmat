//! Invocation-owned simplicial sparse Cholesky with exact CSC fill claims.

mod incomplete;
mod numeric;
mod ordering;
mod projection;
mod retained;
mod solve;
pub(super) use solve::BorrowedCholesky;

use super::managed_lu::ManagedSparseScalar;
use super::managed_storage::{ClaimedVec, SparseWorkError};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

pub(in crate::builtins::math::linalg) trait ManagedCholeskyScalar:
    ManagedSparseScalar
{
    fn conjugate(self) -> Self;
    fn magnitude(self) -> f64 {
        let (real, imag) = self.parts();
        real.hypot(imag)
    }
    fn positive_sqrt(self) -> Option<Self> {
        let (real, imag) = self.parts();
        if imag == 0.0 && real > 0.0 && real.is_finite() {
            Self::from_parts(real.sqrt(), 0.0)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum IncompleteCholeskyKind {
    NoFill,
    Threshold,
}

#[derive(Clone, Copy, Debug)]
pub(in crate::builtins::math::linalg) struct IncompleteCholeskyOptions {
    pub kind: IncompleteCholeskyKind,
    pub drop_tolerance: f64,
    pub modified: bool,
    pub diagonal_compensation: f64,
}
impl ManagedCholeskyScalar for f32 {
    fn conjugate(self) -> Self {
        self
    }
}
impl ManagedCholeskyScalar for f64 {
    fn conjugate(self) -> Self {
        self
    }
}
impl ManagedCholeskyScalar for faer::c32 {
    fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }
}
impl ManagedCholeskyScalar for faer::c64 {
    fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }
}
impl ManagedCholeskyScalar for runmat_value::ComplexElement<f32> {
    fn conjugate(self) -> Self {
        Self(self.0, -self.1)
    }
}
impl ManagedCholeskyScalar for runmat_value::ComplexElement<f64> {
    fn conjugate(self) -> Self {
        Self(self.0, -self.1)
    }
}

#[derive(Debug)]
pub(in crate::builtins::math::linalg) enum ManagedSparseCholeskyError {
    Admission(HostAllocationError),
    Execution(ExecutionServiceError),
    Library(faer::sparse::FaerError),
    InvalidTopology(String),
    NotSquare,
    UnsupportedClass,
    NonFiniteInput,
    NumericalBreakdown,
    DimensionMismatch,
}

impl From<HostAllocationError> for ManagedSparseCholeskyError {
    fn from(value: HostAllocationError) -> Self {
        Self::Admission(value)
    }
}
impl From<ExecutionServiceError> for ManagedSparseCholeskyError {
    fn from(value: ExecutionServiceError) -> Self {
        Self::Execution(value)
    }
}
impl From<SparseWorkError> for ManagedSparseCholeskyError {
    fn from(value: SparseWorkError) -> Self {
        match value {
            SparseWorkError::Admission(value) => Self::Admission(value),
            SparseWorkError::Execution(value) => Self::Execution(value),
        }
    }
}
impl std::fmt::Display for ManagedSparseCholeskyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(e) => write!(f, "sparse Cholesky admission failed: {e}"),
            Self::Execution(e) => write!(f, "sparse Cholesky execution failed: {e}"),
            Self::Library(e) => write!(f, "sparse Cholesky ordering failed: {e}"),
            Self::InvalidTopology(e) => {
                write!(f, "sparse Cholesky requires valid CSC topology: {e}")
            }
            Self::NotSquare => f.write_str("sparse Cholesky requires a square matrix"),
            Self::UnsupportedClass => f.write_str("unsupported sparse Cholesky numerical class"),
            Self::NonFiniteInput => f.write_str("sparse Cholesky requires finite coefficients"),
            Self::NumericalBreakdown => f.write_str("sparse Cholesky numerical breakdown"),
            Self::DimensionMismatch => f.write_str("sparse Cholesky dimensions do not match"),
        }
    }
}
impl std::error::Error for ManagedSparseCholeskyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::Execution(source) => Some(source),
            Self::Library(source) => Some(source),
            _ => None,
        }
    }
}

pub(in crate::builtins::math::linalg) struct ManagedSparseCholesky<T: ManagedCholeskyScalar> {
    order: usize,
    completed: usize,
    lower_ptr: ClaimedVec<usize>,
    lower_rows: ClaimedVec<usize>,
    lower_values: ClaimedVec<T>,
    permutation: ClaimedVec<usize>,
    fence: Option<HostAllocationReceipt>,
    _metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

pub(in crate::builtins::math::linalg) struct ManagedSparseUpperProjection<T: ManagedCholeskyScalar>
{
    col_ptr: ClaimedVec<usize>,
    rows: ClaimedVec<usize>,
    values: ClaimedVec<T>,
}

impl<T: ManagedCholeskyScalar> ManagedSparseUpperProjection<T> {
    pub(in crate::builtins::math::linalg) fn col_ptrs(&self) -> &[usize] {
        self.col_ptr.as_slice()
    }
    pub(in crate::builtins::math::linalg) fn rows(&self) -> &[usize] {
        self.rows.as_slice()
    }
    pub(in crate::builtins::math::linalg) fn values(&self) -> &[T] {
        self.values.as_slice()
    }
}

impl<T: ManagedCholeskyScalar> ManagedSparseCholesky<T> {
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
    pub(in crate::builtins::math::linalg) fn completed_columns(&self) -> usize {
        self.completed
    }
    pub(in crate::builtins::math::linalg) fn permutation(&self) -> &[usize] {
        self.permutation.as_slice()
    }
    pub(in crate::builtins::math::linalg) fn lower_col_ptrs(&self) -> &[usize] {
        self.lower_ptr.as_slice()
    }
    pub(in crate::builtins::math::linalg) fn lower_rows(&self) -> &[usize] {
        self.lower_rows.as_slice()
    }
    pub(in crate::builtins::math::linalg) fn lower_values(&self) -> &[T] {
        self.lower_values.as_slice()
    }
}

#[cfg(test)]
mod tests;
