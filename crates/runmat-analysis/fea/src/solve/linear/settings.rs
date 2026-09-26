//! Shared scoped PCG execution, admission, and stopping policy.
use crate::planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy};

pub(super) const DEFAULT_RELATIVE_TOLERANCE: f64 = 1.0e-8;

pub(super) fn iteration_limit(
    operator: &crate::algebra::AssembledRealSystem,
    base_max_iters: usize,
) -> usize {
    base_max_iters
        .max(operator.dof_count().saturating_mul(2))
        .max(256)
}

/// Execution/admission and numerical stopping policy for the scoped shared PCG path.
pub(crate) struct ScopedLinearSolveSettings<'a> {
    pub plan: LocalExecutionPlan,
    pub admission: Option<&'a SolverAdmissionResourcePolicy>,
    pub relative_tolerance: f64,
}

impl<'a> ScopedLinearSolveSettings<'a> {
    pub(crate) fn with_default_tolerance(
        plan: LocalExecutionPlan,
        admission: Option<&'a SolverAdmissionResourcePolicy>,
    ) -> Self {
        Self {
            plan,
            admission,
            relative_tolerance: DEFAULT_RELATIVE_TOLERANCE,
        }
    }
}
