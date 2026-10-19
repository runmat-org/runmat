use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct TextscanCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
}

impl TextscanCooperation {
    pub(super) fn new(runtime: Option<&crate::context::RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(crate::context::RuntimeContext::cpu_cooperation),
            namespace: runtime.map(crate::context::RuntimeContext::error_namespace),
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".to_string()),
        }
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        budget
            .checkpoint(u64::try_from(units).unwrap_or(u64::MAX))
            .await
            .map_err(|error| self.error(error))
    }

    fn error(&self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            ExecutionServiceError::Cancelled => {
                build_runtime_error("textscan: execution was cancelled")
                    .with_builtin("textscan")
                    .with_identifier(format!(
                        "{}:Cancelled",
                        self.namespace.as_deref().unwrap_or("RunMat")
                    ))
                    .build()
            }
            other => other.into_runtime_error(),
        }
    }
}
