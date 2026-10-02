//! One invocation-owned CPU work budget for clustering preparation and solver work.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::BuiltinResult;

use super::{build_runtime_error, with_kmeans_context, NAME};

pub(super) struct KmeansCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
}

impl KmeansCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
        }
    }

    pub(super) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units)
                .await
                .map_err(|error| match error {
                    ExecutionServiceError::Cancelled => {
                        let namespace = self.namespace.as_deref().unwrap_or("RunMat");
                        build_runtime_error("kmeans: execution was cancelled")
                            .with_identifier(format!("{namespace}:Cancelled"))
                            .with_builtin(NAME)
                            .build()
                    }
                    other => with_kmeans_context(other.into_runtime_error()),
                })?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".to_string()),
        }
    }
}
