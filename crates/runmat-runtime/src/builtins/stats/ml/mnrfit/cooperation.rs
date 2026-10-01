//! mnrfit's error boundary over the shared execution-owned CPU budget.

use super::{check_cancellation, BuiltinResult};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;

pub(super) struct FitCooperation<'a> {
    runtime: Option<&'a RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
}

impl<'a> FitCooperation<'a> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>) -> Self {
        Self {
            runtime,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(runtime: &'a RuntimeContext, budget: CpuCooperationBudget) -> Self {
        Self {
            runtime: Some(runtime),
            budget: Some(budget),
        }
    }

    pub(super) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        check_cancellation(self.runtime)?;
        if let Some(budget) = self.budget.as_mut() {
            let result = budget.checkpoint(units).await;
            check_cancellation(self.runtime)?;
            result.map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
        }
        Ok(())
    }
}
