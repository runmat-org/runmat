//! Audio-owned work accounting; the runtime execution service owns yield policy.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};

pub(super) struct AudioCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
}

impl AudioCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
        }
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        budget
            .checkpoint(units as u64)
            .await
            .map_err(|error| match error {
                ExecutionServiceError::Cancelled => {
                    build_runtime_error(format!("{}: execution was cancelled", self.builtin))
                        .with_identifier(format!(
                            "{}:Cancelled",
                            self.namespace.as_deref().unwrap_or("RunMat")
                        ))
                        .with_builtin(self.builtin)
                        .build()
                }
                other => other.into_runtime_error(),
            })
    }
}
