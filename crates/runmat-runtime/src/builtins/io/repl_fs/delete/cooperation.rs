//! Invocation-local cancellation at delete's host-effect boundaries.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};
use std::future::Future;

pub(super) struct DeleteCooperation {
    budget: Option<CpuCooperationBudget>,
}

impl DeleteCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
        }
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units as u64)
                .await
                .map_err(runtime_error)?;
        }
        Ok(())
    }

    pub(super) async fn await_io<T, E, F>(&mut self, future: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        self.completed(0).await?;
        let result = if let Some(budget) = self.budget.as_ref() {
            budget
                .await_cancellable(future)
                .await
                .map_err(runtime_error)?
        } else {
            future.await
        };
        if result.is_ok() {
            self.completed(0).await?;
        }
        Ok(result)
    }
}

fn runtime_error(error: ExecutionServiceError) -> crate::RuntimeError {
    match error {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error("delete: operation was cancelled")
                .with_identifier("RunMat:Cancelled")
                .with_builtin("delete")
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}
