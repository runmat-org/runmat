//! Which Cholesky factorization a shared SPD solve uses, and under which workspace policy it
//! makes one when the coercivity proof left none.

use runmat_sparse_solvers::direct::{
    DirectOptions, DirectSolver, DirectWorkspacePolicy, FactorKind,
};

/// The Cholesky factorization of a shared SPD solve.
#[derive(Clone, Copy)]
pub(crate) enum CholeskyFactor<'a> {
    /// The coercivity proof's factorization of exactly the solved matrix.
    Reuse(&'a DirectSolver<f64>),
    /// Factor the matrix with this workspace policy.
    Make(DirectWorkspacePolicy),
}

impl<'a> CholeskyFactor<'a> {
    /// No plan admitted the solve: the backend may keep its own packing storage.
    pub(crate) const UNADMITTED: Self = Self::Make(DirectWorkspacePolicy::BackendManaged);

    /// Reuses the proof's factor when there is one. Otherwise an admitted solve factors with
    /// caller-owned workspace, because the plan quoted the factor's resident bytes from that
    /// policy's symbolic estimate. Backend-managed kernels keep unquoted storage: on x86_64 the
    /// dense supernodal update kernel retains a per-thread buffer of twice the L3 cache size.
    pub(crate) fn of<T>(factor: Option<&'a DirectSolver<f64>>, admission: Option<T>) -> Self {
        match (factor, admission) {
            (Some(factor), _) => Self::Reuse(factor),
            (None, Some(_)) => Self::Make(DirectWorkspacePolicy::CallerOwned),
            (None, None) => Self::UNADMITTED,
        }
    }

    /// Options of a factor the solve makes itself: Cholesky with two refinement steps.
    pub(crate) fn options(workspace: DirectWorkspacePolicy) -> DirectOptions {
        DirectOptions {
            refinement_steps: 2,
            workspace,
            ..DirectOptions::new(FactorKind::Cholesky)
        }
    }
}
