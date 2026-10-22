//! Immutable native factor DATA: construction, transfer, and fresh solve owners.
use super::managed_storage::SparseWorkError;
use super::{ManagedSparseCholeskyError, ManagedSparseLuError, SparseCompleteQrError};
use crate::{context::host_allocation::HostAllocationError, execution::ExecutionServiceError};
mod factor;
mod input;
mod publication;
mod scalar;
mod solve;
pub(in crate::builtins::math::linalg) use factor::factor_sparse;
pub(in crate::builtins::math::linalg) use input::{
    inspect_native_matrix, NativeMatrixFacts, SelectedTriangle,
};
pub(super) use publication::{transfer_csc, transfer_permutation, validate_data};
pub(in crate::builtins::math::linalg) use scalar::RetainedFactorScalar;
#[cfg(test)]
pub(in crate::builtins::math::linalg) use solve::solve_factor_native;
pub(in crate::builtins::math::linalg) use solve::{
    solve_factor_indexed, solve_factor_indexed_scaled, RetainedSolveStatus,
};
#[derive(Debug, thiserror::Error)]
pub(in crate::builtins::math::linalg) enum RetainedFactorError {
    #[error("retained factor allocation failed: {0}")]
    Admission(
        #[from]
        #[source]
        HostAllocationError,
    ),
    #[error("retained factor execution failed: {0}")]
    Execution(
        #[from]
        #[source]
        ExecutionServiceError,
    ),
    #[error("invalid retained numerical factor: {0}")]
    Invalid(String),
    #[error("Cholesky decomposition requires a positive definite coefficient matrix")]
    NotPositiveDefinite,
    #[error("retained factor host storage failed: {0}")]
    Storage(#[source] runmat_value::ExactHostNumericAllocationError),
    #[error("retained numerical factor does not match the requested native class")]
    UnsupportedClass,
    #[error("retained sparse LU failed: {0}")]
    Lu(
        #[from]
        #[source]
        ManagedSparseLuError,
    ),
    #[error("retained sparse Cholesky failed: {0}")]
    Cholesky(
        #[from]
        #[source]
        ManagedSparseCholeskyError,
    ),
    #[error("retained orthogonal factor failed: {0}")]
    Orthogonal(
        #[from]
        #[source]
        SparseCompleteQrError,
    ),
}
impl From<SparseWorkError> for RetainedFactorError {
    fn from(value: SparseWorkError) -> Self {
        match value {
            SparseWorkError::Admission(e) => Self::Admission(e),
            SparseWorkError::Execution(e) => Self::Execution(e),
        }
    }
}
