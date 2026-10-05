//! Tabular invocations use the installed execution service for suspension and
//! CPU work accounting; this module owns no scheduling policy.

use std::future::Future;

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct TabularCooperation<'a> {
    runtime: Option<&'a RuntimeContext>,
    budget: Option<CpuCooperationBudget>,
    builtin: &'static str,
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
    let Some(runtime) = runtime else {
        return Ok(work.await);
    };
    runtime
        .cpu_cooperation()
        .await_cancellable(work)
        .await
        .map_err(|source| execution_error(runtime, builtin, source))
}

fn execution_error(
    runtime: &RuntimeContext,
    builtin: &'static str,
    source: ExecutionServiceError,
) -> RuntimeError {
    match source {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error(format!("{builtin}: execution was cancelled"))
                .with_builtin(builtin)
                .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}
