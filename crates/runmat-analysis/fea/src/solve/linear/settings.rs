//! Shared scoped PCG execution, admission, and stopping policy.
use crate::planning::resource::{LocalExecutionPlan, SolverAdmissionResourcePolicy};

pub(crate) const DEFAULT_RELATIVE_TOLERANCE: f64 = 1.0e-8;

pub(crate) fn iteration_limit(
    operator: &crate::algebra::AssembledRealSystem,
    base_max_iters: usize,
) -> usize {
    base_max_iters
        .max(operator.dof_count().saturating_mul(2))
        .max(256)
}

pub(crate) fn accelerator_iteration_limit(
    operator: &crate::algebra::AssembledRealSystem,
    base_max_iters: usize,
) -> u32 {
    u32::try_from(iteration_limit(operator, base_max_iters)).unwrap_or(u32::MAX)
}

/// Execution/admission and numerical stopping policy for the scoped shared PCG path.
pub(crate) struct ScopedLinearSolveSettings<'a> {
    pub plan: LocalExecutionPlan,
    pub admission: Option<&'a SolverAdmissionResourcePolicy>,
    pub relative_tolerance: f64,
    pub preparation: Option<&'a mut crate::algebra::LinearPreparationReuse>,
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
            preparation: None,
        }
    }
}
