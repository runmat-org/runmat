//! Tabular invocations use the installed execution service for suspension and
//! CPU work accounting; this module owns no scheduling policy.

use std::future::Future;

use crate::context::RuntimeContext;
use crate::execution::{staged_file, CpuCooperationBudget, ExecutionServiceError};
use crate::{BuiltinResult, RuntimeError};

pub(super) struct TabularCooperation<'a> {
    runtime: Option<&'a RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
    builtin: &'static str,
}

impl staged_file::StagedWriteCooperation for TabularCooperation<'_> {
    async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        TabularCooperation::completed(self, work).await
    }
}

impl<'a> TabularCooperation<'a> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            runtime,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            builtin,
        }
    }

    #[cfg(test)]
    pub(super) fn with_test_budget(
        runtime: &'a RuntimeContext,
        builtin: &'static str,
        budget: CpuCooperationBudget,
    ) -> Self {
        Self {
            runtime: Some(runtime),
            budget: Some(budget),
            builtin,
        }
    }

    pub(super) async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        let (Some(runtime), Some(budget)) = (self.runtime, self.budget.as_mut()) else {
            return Ok(());
        };
        budget
            .checkpoint(u64::try_from(work).unwrap_or(u64::MAX))
            .await
            .map_err(|source| execution_error(runtime, self.builtin, source))
    }
}

pub(super) async fn await_io<T, F>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    work: F,
) -> BuiltinResult<T>
where
    F: Future<Output = T>,
{
    staged_file::await_cancellable(runtime, builtin, work).await
}

fn execution_error(
    runtime: &RuntimeContext,
    builtin: &'static str,
    source: ExecutionServiceError,
) -> RuntimeError {
    staged_file::execution_error(runtime, builtin, source)
}
