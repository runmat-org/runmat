//! Invocation-owned CPU work cooperation for scalar callback solvers.

use runmat_builtins::OptimSolver;

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

pub(crate) struct ScalarCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    solver: OptimSolver,
}

impl ScalarCooperation {
    pub(crate) fn new(solver: OptimSolver, runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            solver,
        }
    }

    pub(crate) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units)
                .await
                .map_err(|source| match source {
                    source @ ExecutionServiceError::Cancelled => {
                        let namespace = self.namespace.as_deref().unwrap_or("RunMat");
                        build_runtime_error(format!(
                            "{}: execution was cancelled",
                            self.solver.name()
                        ))
                        .with_identifier(format!("{namespace}:Cancelled"))
                        .with_builtin(self.solver.name())
                        .with_source(source)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build()
                    }
                    other => {
                        let mut failure = other.into_runtime_error();
                        failure.context = failure.context.with_builtin(self.solver.name());
                        failure.gpu_gather_retry = GpuGatherRetry::Never;
                        failure
                    }
                })?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn with_budget(solver: OptimSolver, budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            solver,
        }
    }
}
