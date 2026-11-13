//! Borrowed-CSC sparse factors with allocator-owned LU and complete QR.
//!
//! LU and sparse QR/COD share actual-policy fill and cooperative workspace.

mod coefficient;
pub(crate) mod equality_factor;
pub(crate) mod least_squares;
mod managed_cholesky;
mod managed_ilu;
mod managed_lu;
mod managed_qr;
mod managed_storage;
pub(crate) mod orthogonal_frame;
use coefficient::coefficient_as;
mod publication;
mod retained;
pub(in crate::builtins::math::linalg) use managed_cholesky::{
    IncompleteCholeskyKind, IncompleteCholeskyOptions, ManagedCholeskyScalar,
    ManagedSparseCholesky, ManagedSparseCholeskyError,
};
pub(in crate::builtins::math::linalg) use managed_ilu::{
    IncompleteLuCsc, IncompleteLuKind, IncompleteLuOptions, IncompleteLuPart, ManagedIncompleteLu,
    ManagedIncompleteLuError, ModifiedIncompleteLu,
};
pub(in crate::builtins::math::linalg) use managed_lu::{
    cholesky_stage_scales, ManagedSparseLu, ManagedSparseLuError, ManagedSparseScalar,
    PublicSparseLu, SparseLuOptions,
};
pub(in crate::builtins::math::linalg) use managed_lu::{
    FactorRhsGeometry, FactorSolveOptions, SingularSolvePolicy,
};
pub(in crate::builtins::math::linalg) use managed_qr::{
    SparseCompleteQr, SparseCompleteQrError, SparseQrDecomposition, SparseQrScalar,
};
pub(in crate::builtins::math::linalg) use publication::{
    sparse_publication_bytes, SparsePublication, SparsePublicationError,
};
pub(in crate::builtins::math::linalg) use retained::{
    factor_sparse, inspect_native_matrix, solve_factor_indexed, solve_factor_indexed_scaled,
    NativeMatrixFacts, RetainedFactorError, RetainedFactorScalar, RetainedSolveStatus,
    SelectedTriangle,
};

#[cfg(test)]
mod tests;
