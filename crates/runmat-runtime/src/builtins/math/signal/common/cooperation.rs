//! Signal-domain error context for invocation-owned CPU cooperation.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, GpuGatherRetry, RuntimeError};

pub(crate) struct SignalCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
}

impl SignalCooperation {
    pub(crate) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
        }
    }

    pub(crate) async fn completed(&mut self, units: u64) -> Result<(), RuntimeError> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        budget.checkpoint(units).await.map_err(|source| {
            let mut error = if source == ExecutionServiceError::Cancelled {
                build_runtime_error(format!("{}: execution was cancelled", self.builtin))
                    .with_builtin(self.builtin)
                    .with_identifier(format!(
                        "{}:Cancelled",
                        self.namespace.as_deref().unwrap_or("RunMat")
                    ))
                    .build()
            } else {
                let mut error = source.clone().into_runtime_error();
                error.context.builtin = Some(self.builtin.to_string());
                error
            };
            error.source = Some(Box::new(source));
            error.gpu_gather_retry = GpuGatherRetry::Never;
            error
        })
    }
}
