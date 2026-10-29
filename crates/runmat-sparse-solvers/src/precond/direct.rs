//! A direct factorization used as a preconditioner.

use std::sync::Arc;

use crate::direct::{DirectSolver, FactorScalar};
use crate::error::SolverResult;
use crate::operator::Preconditioner;

/// Applies `F⁻¹` for a factorization `F` (exact for the factored matrix; a lagged factor of a
/// nearby matrix makes this a strong preconditioner for Newton or frequency sweeps).
pub struct DirectPreconditioner<T: FactorScalar> {
    solver: Arc<DirectSolver<T>>,
    symmetric_positive_definite: bool,
}

impl<T: FactorScalar> DirectPreconditioner<T> {
    pub fn new(solver: Arc<DirectSolver<T>>) -> Self {
        let symmetric_positive_definite = solver.kind() == crate::direct::FactorKind::Cholesky;
        Self {
            solver,
            symmetric_positive_definite,
        }
    }
}

impl<T: FactorScalar> Preconditioner<T> for DirectPreconditioner<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> SolverResult<()> {
        z.copy_from_slice(r);
        self.solver.solve_in_place(z, 1)
    }
    fn label(&self) -> &'static str {
        "direct"
    }
    fn is_symmetric_positive_definite(&self) -> bool {
        self.symmetric_positive_definite
    }
}
