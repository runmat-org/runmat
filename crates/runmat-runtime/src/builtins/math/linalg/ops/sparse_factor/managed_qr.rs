//! Sparse rank-revealing Householder QR and complete orthogonal solve.
//! All numerical fill is owned by the shared sparse storage allocator.
mod csc;
mod decomposition;
mod factor;
mod householder;
mod norm;
mod retained;
mod scalar;
mod solve;
pub(in crate::builtins::math::linalg) use decomposition::SparseQrDecomposition;
pub(super) use solve::{BorrowedCompleteQr, BorrowedCsc, BorrowedQr};
#[cfg(test)]
mod tests;
use super::managed_storage::{ClaimedVec, SparseWorkError, Work};
use crate::context::host_allocation::{HostAllocationError, HostAllocationReceipt};
use crate::execution::ExecutionServiceError;
use csc::Csc;
pub(in crate::builtins::math::linalg) use scalar::SparseQrScalar;

#[derive(Debug, thiserror::Error)]
pub(in crate::builtins::math::linalg) enum SparseCompleteQrError {
    #[error("sparse QR allocation failed: {0}")]
    Admission(#[source] HostAllocationError),
    #[error("sparse QR execution failed: {0}")]
    Execution(#[source] ExecutionServiceError),
    #[error("sparse QR requires valid CSC topology: {0}")]
    InvalidTopology(String),
    #[error("unsupported sparse QR numerical class")]
    UnsupportedClass,
    #[error("sparse QR dimensions do not match")]
    DimensionMismatch,
    #[error("sparse QR requires finite coefficients and right-hand sides")]
    NonFiniteInput,
    #[error("sparse QR numerical breakdown")]
    NumericalBreakdown,
}
impl From<HostAllocationError> for SparseCompleteQrError {
    fn from(e: HostAllocationError) -> Self {
        Self::Admission(e)
    }
}
impl From<ExecutionServiceError> for SparseCompleteQrError {
    fn from(e: ExecutionServiceError) -> Self {
        Self::Execution(e)
    }
}
impl From<SparseWorkError> for SparseCompleteQrError {
    fn from(e: SparseWorkError) -> Self {
        match e {
            SparseWorkError::Admission(e) => Self::Admission(e),
            SparseWorkError::Execution(e) => Self::Execution(e),
        }
    }
}

struct Qr<T: SparseQrScalar> {
    rows: usize,
    cols: usize,
    rank: usize,
    r: Csc<T>,
    reflectors: Csc<T>,
    permutation: ClaimedVec<usize>,
}

/// Reusable original-issuer factor. No input Value or workspace snapshot is
/// retained; numerical fill and reflectors own their physical allocations.
pub(in crate::builtins::math::linalg) struct SparseCompleteQr<T: SparseQrScalar> {
    first: Qr<T>,
    cod: Option<Qr<T>>,
    fence: Option<HostAllocationReceipt>,
}
impl<T: SparseQrScalar> SparseCompleteQr<T> {
    pub(in crate::builtins::math::linalg) fn rank(&self) -> usize {
        self.first.rank
    }
}
