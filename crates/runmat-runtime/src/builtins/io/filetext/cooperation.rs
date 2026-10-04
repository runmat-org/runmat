//! Invocation-owned file I/O awaits use the execution service's cancellation
//! boundary. This supplies no file-local turn policy or scheduler.

use std::future::Future;

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};

/// Retains one invocation's work counter across resident gathers, file I/O,
/// and conversion. The execution service alone chooses the turn policy.
pub(crate) struct FileCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
}

impl FileCooperation {
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

    pub(crate) async fn await_io<T, E, F>(&self, work: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        let Some(budget) = self.budget.as_ref() else {
            return Ok(work.await);
        };
        budget.await_cancellable(work).await.map_err(|source| {
            execution_error(
                self.builtin,
                self.namespace.as_deref().unwrap_or("RunMat"),
                source,
            )
        })
    }

    pub(crate) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        let outcome = budget.checkpoint(units as u64).await;
        outcome.map_err(|source| {
            execution_error(
                self.builtin,
                self.namespace.as_deref().unwrap_or("RunMat"),
                source,
            )
        })
    }
}

pub(crate) async fn checkpoint(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
) -> BuiltinResult<()> {
    let Some(runtime) = runtime else {
        return Ok(());
    };
    runtime
        .cpu_cooperation()
        .checkpoint(0)
        .await
        .map_err(|source| execution_error(builtin, &runtime.error_namespace(), source))
}

pub(crate) async fn await_io<T, E, F>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    work: F,
) -> BuiltinResult<Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    let Some(runtime) = runtime else {
        return Ok(work.await);
    };
    runtime
        .cpu_cooperation()
        .await_cancellable(work)
        .await
        .map_err(|source| execution_error(builtin, &runtime.error_namespace(), source))
}

fn execution_error(
    builtin: &'static str,
    namespace: &str,
    source: ExecutionServiceError,
) -> crate::RuntimeError {
    match source {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error(format!("{builtin}: execution was cancelled"))
                .with_builtin(builtin)
                .with_identifier(format!("{namespace}:Cancelled"))
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}

#[cfg(test)]
#[path = "cooperation/full_call_tests.rs"]
mod full_call_tests;
