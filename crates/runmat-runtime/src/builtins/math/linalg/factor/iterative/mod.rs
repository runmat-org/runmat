//! Input and publication owners shared by iterative linear solvers.

/// Numerical inability to apply a preconditioner is a solver outcome, not
/// permission to swallow language, allocation or execution-control errors.
pub(in crate::builtins::math::linalg::factor) use runmat_sparse_solvers::krylov::ApplyStatus as PreconditionerStatus;

pub(in crate::builtins::math::linalg::factor) mod callback;
pub(in crate::builtins::math::linalg::factor) mod controls;
pub(in crate::builtins::math::linalg::factor) mod copy;
pub(in crate::builtins::math::linalg::factor) mod output;
pub(in crate::builtins::math::linalg::factor) mod value;
