//! Invocation-owned sparse LU with exact, incrementally admitted CSC fill.
//!
//! The sparse reach / left-looking numerical scheme is adapted from faer
//! 0.24.4, sparse/linalg/lu.rs (simplicial); see managed_lu/FAER-LICENSE.txt.
//! No faer-owned factor or symbolic allocation enters this Managed route.
//! Factors are private until construction and the final authority fence pass.

mod decomposition;
mod numeric;
mod ordering;
mod reach;
mod scalar;
mod solve;
pub(super) use solve::BorrowedLu;
pub(in crate::builtins::math::linalg) use solve::FactorRhsGeometry;
pub(in crate::builtins::math::linalg) use solve::FactorSolveOptions;
pub(in crate::builtins::math::linalg) use solve::SingularSolvePolicy;
mod source;
pub(in crate::builtins::math::linalg) use decomposition::{PublicSparseLu, SparseLuOptions};

pub(in crate::builtins::math::linalg) use scalar::{cholesky_stage_scales, ManagedSparseScalar};

#[cfg(test)]
use super::managed_storage::admit;
use super::managed_storage::{ClaimedVec, SparseWorkError, Work};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_execution::resource::RetainedHostAllocationClaim;
use std::sync::Arc;

#[derive(Debug)]
pub(in crate::builtins::math::linalg) enum ManagedSparseLuError {
    Admission(HostAllocationError),
    Execution(ExecutionServiceError),
    Library(faer::sparse::FaerError),
    InvalidTopology(String),
    NotSquare,
    UnsupportedClass,
    DimensionMismatch,
    NonFiniteInput,
    Singular { column: usize },
    NumericalBreakdown,
}

impl From<HostAllocationError> for ManagedSparseLuError {
    fn from(error: HostAllocationError) -> Self {
        Self::Admission(error)
    }
}
impl From<ExecutionServiceError> for ManagedSparseLuError {
    fn from(error: ExecutionServiceError) -> Self {
        Self::Execution(error)
    }
}
impl From<SparseWorkError> for ManagedSparseLuError {
    fn from(error: SparseWorkError) -> Self {
        match error {
            SparseWorkError::Admission(error) => Self::Admission(error),
            SparseWorkError::Execution(error) => Self::Execution(error),
        }
    }
}
impl std::fmt::Display for ManagedSparseLuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(source) => write!(formatter, "sparse LU admission failed: {source}"),
            Self::Execution(source) => write!(formatter, "sparse LU execution failed: {source}"),
            Self::Library(source) => write!(formatter, "sparse LU ordering failed: {source}"),
            Self::InvalidTopology(detail) => {
                write!(formatter, "sparse LU requires valid CSC topology: {detail}")
            }
            Self::NotSquare => {
                formatter.write_str("sparse LU requires a square coefficient matrix")
            }
            Self::UnsupportedClass => formatter.write_str("unsupported sparse LU numerical class"),
            Self::DimensionMismatch => formatter.write_str("sparse LU dimensions do not match"),
            Self::NonFiniteInput => formatter.write_str("sparse LU requires finite coefficients"),
            Self::Singular { column } => {
                write!(
                    formatter,
                    "sparse LU has a singular pivot in column {column}"
                )
            }
            Self::NumericalBreakdown => formatter.write_str("sparse LU numerical breakdown"),
        }
    }
}
impl std::error::Error for ManagedSparseLuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::Execution(source) => Some(source),
            Self::Library(source) => Some(source),
            _ => None,
        }
    }
}

/// Reusable PAQ=LU factor, retaining its original invocation authority.
/// Arrays own their exact physical claims, independently of growth receipts.
/// One reusable native RHS is retained; no caller-visible Value is stored.
pub(in crate::builtins::math::linalg) struct ManagedSparseLu<T: ManagedSparseScalar> {
    order: usize,
    l_ptr: ClaimedVec<usize>,
    l_rows: ClaimedVec<usize>,
    l_values: ClaimedVec<T>,
    u_ptr: ClaimedVec<usize>,
    u_rows: ClaimedVec<usize>,
    u_values: ClaimedVec<T>,
    row_perm: ClaimedVec<usize>,
    col_perm: ClaimedVec<usize>,
    rhs: ClaimedVec<T>,
    fence: Option<HostAllocationReceipt>,
    // The factory quotes the stable Box control backing before allocation,
    // separately from numeric blocks; callers must not wrap an unfunded Self.
    _metadata_claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
    #[cfg(test)]
    work_units: u64,
}

impl<T: ManagedSparseScalar> ManagedSparseLu<T> {
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
}

#[cfg(test)]
mod tests;
