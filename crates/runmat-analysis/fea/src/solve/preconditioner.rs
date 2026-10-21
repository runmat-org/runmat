//! SPD solve selection. The numerical preconditioners themselves live in the shared sparse
//! layer (`runmat_sparse_solvers::precond`).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpdPreconditionerKind {
    /// The shared policy decides: the coercivity-proving Cholesky factor when one exists,
    /// otherwise direct or AMG-preconditioned CG by size and memory.
    Auto,
    /// Sparse Cholesky (direct solve).
    Direct,
    /// CG with smoothed-aggregation algebraic multigrid.
    Amg,
    /// CG with Jacobi scaling.
    Jacobi,
    /// CG with incomplete Cholesky (ILU(0) on the symmetric pattern).
    Ilu0,
}

impl SpdPreconditionerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Direct => "direct",
            Self::Amg => "amg",
            Self::Jacobi => "jacobi",
            Self::Ilu0 => "ilu0",
        }
    }
}
