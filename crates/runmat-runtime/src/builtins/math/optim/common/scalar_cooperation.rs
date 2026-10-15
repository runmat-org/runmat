//! Invocation-owned CPU work cooperation for scalar callback solvers.

use runmat_builtins::{BuiltinCatalogIdentity, OptimSolver};

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};

pub(crate) struct ScalarCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: BuiltinCatalogIdentity,
}

impl ScalarCooperation {
    pub(crate) fn new(solver: OptimSolver, runtime: Option<&RuntimeContext>) -> Self {
        Self::for_builtin(
            BuiltinCatalogIdentity {
                name: solver.name(),
            },
            runtime,
        )
    }

    pub(crate) fn for_builtin(
        builtin: BuiltinCatalogIdentity,
        runtime: Option<&RuntimeContext>,
    ) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
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
                            self.builtin.name
                        ))
                        .with_identifier(format!("{namespace}:Cancelled"))
                        .with_builtin(self.builtin.name)
                        .with_source(source)
                        .with_gpu_gather_retry(GpuGatherRetry::Never)
                        .build()
                    }
                    other => {
                        let mut failure = other.into_runtime_error();
                        failure.context = failure.context.with_builtin(self.builtin.name);
                        failure.gpu_gather_retry = GpuGatherRetry::Never;
                        failure
                    }
                })?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn with_budget(solver: OptimSolver, budget: CpuCooperationBudget) -> Self {
        Self::with_builtin_budget(
            BuiltinCatalogIdentity {
                name: solver.name(),
            },
            budget,
        )
    }

    #[cfg(test)]
    pub(crate) fn with_builtin_budget(
        builtin: BuiltinCatalogIdentity,
        budget: CpuCooperationBudget,
    ) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
        }
    }
}
